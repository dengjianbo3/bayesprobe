from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

from harbor.constants import PACKAGE_CACHE_DIR
from harbor.models.task.config import TaskConfig as TaskDefinitionConfig

from bayesprobe_terminal_bench.config import TerminalBenchConfig
from bayesprobe_terminal_bench.experiment_lock import (
    CausalQualificationLock,
    FROZEN_GATE_TASK_IDS,
    FROZEN_GATE_TASK_REFS,
    LockedBudgets,
)
from bayesprobe_terminal_bench.planning import plan_contract_identity
from bayesprobe_terminal_bench.provider_contract import contract_identity
from bayesprobe_terminal_bench.runner_factory import collect_repository_git_identity
from capture_provider_identity import (
    STAGE0_BASE_URL,
    STAGE0_MODEL,
    load_provider_identity_artifact,
)
from write_benchmark_lock import write_lock_atomic
from write_paired_gate_lock import (
    _official_reward,
    _read_object,
    _trial_results_by_task,
    resolve_cached_image_digest,
)


@dataclass(frozen=True)
class CachedQualificationTask:
    image_digest: str
    agent_timeout_seconds: int


@dataclass(frozen=True)
class CausalQualificationRuntimeIdentity:
    harbor_version: str
    root_git_sha: str
    adapter_tree_sha: str
    adapter_dirty: bool


def build_causal_qualification_lock(
    *,
    job_dir: Path,
    config: TerminalBenchConfig,
    runtime_identity: CausalQualificationRuntimeIdentity,
    provider_identity_path: Path,
    task_identity_resolver: Callable[[str, str], CachedQualificationTask],
) -> dict[str, object]:
    if runtime_identity.harbor_version != "0.18.0":
        raise ValueError("Harbor version must be 0.18.0")
    if runtime_identity.adapter_dirty:
        raise ValueError("benchmark adapter worktree is dirty")
    if config.model != STAGE0_MODEL or config.base_url != STAGE0_BASE_URL:
        raise ValueError("Stage 0 model or base URL drift")

    root = Path(job_dir)
    job_config = _read_object(root / "config.json")
    job_lock = _read_object(root / "lock.json")
    _require_oracle_job_config(job_config)
    dataset = _qualification_dataset(job_config)
    resolved = _resolved_tasks(job_lock)
    results = _trial_results_by_task(root)

    tasks: list[dict[str, object]] = []
    for task_id, task_ref in resolved:
        result = results.get(task_id)
        if result is None or _official_reward(result) != 1.0:
            raise ValueError(f"Oracle reward must be 1 for {task_id}")
        result_task = result.get("task_id")
        if (
            not isinstance(result_task, Mapping)
            or result_task.get("ref") != task_ref
        ):
            raise ValueError("Oracle result task ref does not match the frozen ref")
        _require_oracle_result(result)
        cached = task_identity_resolver(task_id, task_ref)
        if not isinstance(cached, CachedQualificationTask):
            raise ValueError("cached qualification task identity is invalid")
        tasks.append(
            {
                "task_id": task_id,
                "task_ref": task_ref,
                "image_digest": cached.image_digest,
                "agent_timeout_seconds": cached.agent_timeout_seconds,
            }
        )

    provider = load_provider_identity_artifact(provider_identity_path)
    if (
        provider.configured_model != config.model
        or provider.base_url != config.base_url
        or provider.provider_protocol != "openai_chat_completions"
        or provider.temperature != 0
    ):
        raise ValueError("provider identity artifact configuration drift")

    lock = CausalQualificationLock(
        schema_version="terminal_bench_causal_qualification:v1",
        harbor_version="0.18.0",
        dataset_name="terminal-bench/terminal-bench-2",
        dataset_revision=_required_string(dataset.get("ref"), "dataset revision"),
        tasks=tuple(tasks),
        root_git_sha=runtime_identity.root_git_sha,
        adapter_tree_sha=runtime_identity.adapter_tree_sha,
        model=config.model,
        base_url=config.base_url,
        provider_protocol="openai_chat_completions",
        temperature=0,
        budgets=LockedBudgets(
            max_total_actions=config.max_total_actions,
            max_model_calls=config.max_model_calls,
            max_provider_tokens=config.max_provider_tokens,
            max_output_tokens=config.max_output_tokens,
            command_timeout_seconds=config.command_timeout_seconds,
            provider_timeout_seconds=config.provider_timeout_seconds,
            signal_output_bytes=config.signal_output_bytes,
        ),
        prompt_schema_hashes={
            **contract_identity(),
            **plan_contract_identity(),
        },
        expected_provider_model=provider.returned_model,
        provider_identity_sha256=provider.content_sha256,
        expected_system_fingerprint_available=provider.system_fingerprint_available,
        expected_system_fingerprint=provider.system_fingerprint,
    )
    return lock.model_dump(mode="json")


def _qualification_dataset(job_config: Mapping[str, object]) -> Mapping[str, object]:
    datasets = job_config.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != 1:
        raise ValueError("Oracle qualification must contain one dataset")
    dataset = datasets[0]
    if not isinstance(dataset, Mapping):
        raise ValueError("Oracle qualification dataset is invalid")
    if dataset.get("name") != "terminal-bench/terminal-bench-2":
        raise ValueError("Oracle qualification dataset is invalid")
    if tuple(dataset.get("task_names", ())) != FROZEN_GATE_TASK_IDS:
        raise ValueError("Oracle qualification tasks do not match the frozen order")
    if job_config.get("n_attempts", 1) != 1:
        raise ValueError("Oracle qualification must use one attempt")
    return dataset


def _require_oracle_job_config(job_config: Mapping[str, object]) -> None:
    agents = job_config.get("agents")
    if (
        not isinstance(agents, list)
        or len(agents) != 1
        or not isinstance(agents[0], Mapping)
        or agents[0].get("name") != "oracle"
    ):
        raise ValueError("Oracle agent provenance is invalid")


def _require_oracle_result(result: Mapping[str, object]) -> None:
    config = result.get("config")
    agent = config.get("agent") if isinstance(config, Mapping) else None
    agent_info = result.get("agent_info")
    if (
        not isinstance(agent, Mapping)
        or agent.get("name") != "oracle"
        or not isinstance(agent_info, Mapping)
        or agent_info.get("name") != "oracle"
    ):
        raise ValueError("Oracle agent provenance is invalid")


def _resolved_tasks(job_lock: Mapping[str, object]) -> list[tuple[str, str]]:
    harbor = job_lock.get("harbor")
    trials = job_lock.get("trials")
    if not isinstance(harbor, Mapping) or harbor.get("version") != "0.18.0":
        raise ValueError("Oracle qualification Harbor lock is invalid")
    if not isinstance(trials, list) or len(trials) != 3:
        raise ValueError("Oracle qualification must resolve three trials")
    by_id: dict[str, str] = {}
    for trial in trials:
        task = trial.get("task") if isinstance(trial, Mapping) else None
        agent = trial.get("agent") if isinstance(trial, Mapping) else None
        task_id = task.get("name") if isinstance(task, Mapping) else None
        task_ref = task.get("digest") if isinstance(task, Mapping) else None
        if (
            not isinstance(task_id, str)
            or not isinstance(task_ref, str)
            or not isinstance(agent, Mapping)
            or agent.get("name") != "oracle"
        ):
            raise ValueError("Oracle agent provenance is invalid")
        if task_id in by_id:
            raise ValueError("Oracle qualification contains a duplicate task")
        by_id[task_id] = task_ref
    if set(by_id) != set(FROZEN_GATE_TASK_IDS):
        raise ValueError("Oracle qualification tasks do not match the frozen set")
    resolved = [(task_id, by_id[task_id]) for task_id in FROZEN_GATE_TASK_IDS]
    if any(FROZEN_GATE_TASK_REFS[task_id] != ref for task_id, ref in resolved):
        raise ValueError("Oracle qualification refs do not match the frozen refs")
    return resolved


def collect_causal_qualification_runtime_identity(
    repository_root: Path,
) -> CausalQualificationRuntimeIdentity:
    harbor_version = version("harbor")
    if harbor_version != "0.18.0":
        raise ValueError(f"Harbor version must be 0.18.0; found {harbor_version}")
    identity = collect_repository_git_identity(Path(repository_root))
    if identity.adapter_dirty:
        raise ValueError("benchmark adapter worktree is dirty")
    return CausalQualificationRuntimeIdentity(
        harbor_version=harbor_version,
        root_git_sha=identity.root_git_sha,
        adapter_tree_sha=identity.adapter_tree_sha,
        adapter_dirty=False,
    )


def resolve_cached_qualification_task(
    task_id: str,
    task_ref: str,
) -> CachedQualificationTask:
    org, name = task_id.split("/", 1)
    task_dir = (
        Path(PACKAGE_CACHE_DIR)
        / org
        / name
        / task_ref.removeprefix("sha256:")
    )
    try:
        task = TaskDefinitionConfig.model_validate_toml(
            (task_dir / "task.toml").read_text(encoding="utf-8")
        )
    except Exception:
        raise ValueError(
            f"could not resolve cached qualification task for {task_id}"
        ) from None
    timeout = task.agent.timeout_sec
    if (
        type(timeout) not in (int, float)
        or isinstance(timeout, bool)
        or timeout <= 0
        or float(timeout).is_integer() is False
    ):
        raise ValueError(f"cached task has no integer agent timeout for {task_id}")
    return CachedQualificationTask(
        image_digest=resolve_cached_image_digest(task_id, task_ref),
        agent_timeout_seconds=int(timeout),
    )


def _required_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is required")
    return value.strip()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--oracle-job", required=True, type=Path)
    parser.add_argument("--provider-identity", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    config = TerminalBenchConfig(
        model=STAGE0_MODEL,
        base_url=STAGE0_BASE_URL,
        lock_path=args.output,
    )
    repository_root = Path(__file__).resolve().parents[3]
    lock = build_causal_qualification_lock(
        job_dir=args.oracle_job,
        config=config,
        runtime_identity=collect_causal_qualification_runtime_identity(
            repository_root
        ),
        provider_identity_path=args.provider_identity,
        task_identity_resolver=resolve_cached_qualification_task,
    )
    write_lock_atomic(args.output, lock)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
