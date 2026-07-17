from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

from harbor.constants import PACKAGE_CACHE_DIR
from harbor.environments.definition import (
    environment_content_hash,
    require_agent_environment_definition,
    should_use_prebuilt_docker_image,
)
from harbor.models.task.config import TaskConfig as TaskDefinitionConfig

from bayesprobe_terminal_bench.config import TerminalBenchConfig
from bayesprobe_terminal_bench.experiment_lock import (
    FROZEN_GATE_TASK_IDS,
    FROZEN_GATE_TASK_REFS,
    PAIRED_GATE_ARMS,
    PairedGateLock,
)
from bayesprobe_terminal_bench.runner_factory import collect_repository_git_identity
from write_benchmark_lock import _docker_image_digest, write_lock_atomic


@dataclass(frozen=True)
class PairedGateRuntimeIdentity:
    harbor_version: str
    root_git_sha: str
    adapter_tree_sha: str


def build_paired_gate_lock(
    *,
    job_dir: Path,
    config: TerminalBenchConfig,
    runtime_identity: object,
    image_digest_resolver: Callable[[str, str], str],
) -> dict[str, object]:
    root = Path(job_dir)
    job_config = _read_object(root / "config.json")
    job_lock = _read_object(root / "lock.json")
    if getattr(runtime_identity, "harbor_version", None) != "0.18.0":
        raise ValueError("Harbor version must be 0.18.0")

    datasets = job_config.get("datasets")
    if not isinstance(datasets, list) or len(datasets) != 1:
        raise ValueError("Oracle gate must contain one dataset")
    dataset = datasets[0]
    if not isinstance(dataset, Mapping):
        raise ValueError("Oracle gate dataset is invalid")
    if dataset.get("name") != "terminal-bench/terminal-bench-2":
        raise ValueError("Oracle gate dataset is invalid")
    if tuple(dataset.get("task_names", ())) != FROZEN_GATE_TASK_IDS:
        raise ValueError("Oracle gate tasks do not match the frozen order")
    if job_config.get("n_attempts", 1) != 1:
        raise ValueError("Oracle gate must use one attempt")

    harbor = job_lock.get("harbor")
    trials = job_lock.get("trials")
    if not isinstance(harbor, Mapping) or harbor.get("version") != "0.18.0":
        raise ValueError("Oracle gate Harbor lock is invalid")
    if not isinstance(trials, list) or len(trials) != 3:
        raise ValueError("Oracle gate must resolve three trials")
    resolved_by_id: dict[object, object] = {}
    for trial in trials:
        task = trial.get("task") if isinstance(trial, Mapping) else None
        if not isinstance(task, Mapping):
            raise ValueError("Oracle gate trial identity is invalid")
        task_id = task.get("name")
        if task_id in resolved_by_id:
            raise ValueError("Oracle gate contains a duplicate resolved task")
        resolved_by_id[task_id] = task.get("digest")
    if set(resolved_by_id) != set(FROZEN_GATE_TASK_IDS):
        raise ValueError("Oracle gate resolved tasks do not match the frozen set")
    resolved = [
        (task_id, resolved_by_id[task_id]) for task_id in FROZEN_GATE_TASK_IDS
    ]
    if any(FROZEN_GATE_TASK_REFS[task_id] != task_ref for task_id, task_ref in resolved):
        raise ValueError("Oracle gate resolved refs do not match the frozen refs")

    results = _trial_results_by_task(root)
    tasks: list[dict[str, str]] = []
    for task_id, task_ref in resolved:
        result = results.get(task_id)
        if result is None or _official_reward(result) != 1.0:
            raise ValueError(f"Oracle reward must be 1 for {task_id}")
        result_task_id = result.get("task_id")
        if not isinstance(result_task_id, Mapping) or result_task_id.get("ref") != task_ref:
            raise ValueError("Oracle result task ref does not match the frozen ref")
        tasks.append(
            {
                "task_id": task_id,
                "task_ref": task_ref,
                "image_digest": image_digest_resolver(task_id, task_ref),
            }
        )

    lock = PairedGateLock(
        schema_version="terminal_bench_paired_gate:v0.1",
        harbor_version="0.18.0",
        dataset_name="terminal-bench/terminal-bench-2",
        dataset_revision=_required_string(dataset.get("ref"), "dataset revision"),
        tasks=tuple(tasks),
        root_git_sha=getattr(runtime_identity, "root_git_sha"),
        adapter_tree_sha=getattr(runtime_identity, "adapter_tree_sha"),
        n_attempts=1,
        model=config.model,
        base_url=config.base_url,
        provider_protocol="openai_chat_completions",
        api_key_env=config.api_key_env,
        temperature=0,
        max_cycles=config.max_cycles,
        max_probes_per_cycle=config.max_probes_per_cycle,
        max_actions_per_probe=config.max_actions_per_probe,
        max_total_actions=config.max_total_actions,
        max_model_calls=config.max_model_calls,
        command_timeout_seconds=config.command_timeout_seconds,
        provider_timeout_seconds=config.provider_timeout_seconds,
        max_output_tokens=config.max_output_tokens,
        signal_output_bytes=config.signal_output_bytes,
        arms=PAIRED_GATE_ARMS,
    )
    return lock.model_dump(mode="json")


def collect_paired_gate_runtime_identity(repository_root: Path) -> PairedGateRuntimeIdentity:
    harbor_version = version("harbor")
    if harbor_version != "0.18.0":
        raise ValueError(f"Harbor version must be 0.18.0; found {harbor_version}")
    identity = collect_repository_git_identity(Path(repository_root))
    if identity.adapter_dirty:
        raise ValueError("benchmark adapter worktree is dirty")
    return PairedGateRuntimeIdentity(
        harbor_version=harbor_version,
        root_git_sha=identity.root_git_sha,
        adapter_tree_sha=identity.adapter_tree_sha,
    )


def resolve_cached_image_digest(task_id: str, task_ref: str) -> str:
    org, name = task_id.split("/", 1)
    task_dir = Path(PACKAGE_CACHE_DIR) / org / name / task_ref.removeprefix("sha256:")
    try:
        task_definition = TaskDefinitionConfig.model_validate_toml(
            (task_dir / "task.toml").read_text(encoding="utf-8")
        )
    except Exception:
        raise ValueError(f"could not resolve cached task environment for {task_id}") from None
    environment_dir = task_dir / "environment"
    docker_image = task_definition.environment.docker_image
    require_agent_environment_definition(environment_dir, docker_image=docker_image)
    if should_use_prebuilt_docker_image(
        environment_dir,
        docker_image=docker_image,
        force_build=False,
    ):
        if not isinstance(docker_image, str) or not docker_image.strip():
            raise ValueError(f"cached task has no usable image for {task_id}")
        image = docker_image.strip()
    else:
        image = f"hb__{environment_content_hash(environment_dir, docker_image=docker_image)}"
    return _docker_image_digest(image)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-job", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    model = os.environ.get("BAYESPROBE_BENCH_MODEL", "").strip()
    if not model:
        raise ValueError("BAYESPROBE_BENCH_MODEL is required")
    config = TerminalBenchConfig(
        model=model,
        base_url=os.environ.get("BAYESPROBE_BENCH_BASE_URL", "").strip() or None,
        lock_path=args.output,
    )
    repository_root = Path(__file__).resolve().parents[3]
    lock = build_paired_gate_lock(
        job_dir=args.oracle_job,
        config=config,
        runtime_identity=collect_paired_gate_runtime_identity(repository_root),
        image_digest_resolver=resolve_cached_image_digest,
    )
    write_lock_atomic(args.output, lock)
    return 0


def _trial_results_by_task(job_dir: Path) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    for path in sorted(job_dir.glob("*/result.json")):
        payload = _read_object(path)
        task_id = _result_task_id(payload)
        if task_id is not None and payload.get("finished_at") is not None:
            if task_id in results:
                raise ValueError(f"duplicate Oracle result for {task_id}")
            results[task_id] = payload
    return results


def _result_task_id(payload: Mapping[str, object]) -> str | None:
    identity = payload.get("task_id")
    if isinstance(identity, Mapping):
        org = identity.get("org")
        name = identity.get("name")
        if isinstance(org, str) and org.strip() and isinstance(name, str) and name.strip():
            return f"{org.strip()}/{name.strip()}"
    task_name = payload.get("task_name")
    if isinstance(task_name, str) and "/" in task_name and task_name.strip():
        return task_name.strip()
    return None


def _official_reward(result: Mapping[str, object]) -> float:
    verifier = result.get("verifier_result")
    rewards = verifier.get("rewards") if isinstance(verifier, Mapping) else None
    reward = rewards.get("reward") if isinstance(rewards, Mapping) else None
    if type(reward) not in (int, float):
        raise ValueError("completed trial has no official reward")
    return float(reward)


def _read_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise ValueError(f"invalid JSON artifact: {path}") from None
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _required_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} is required")
    return value.strip()


if __name__ == "__main__":
    raise SystemExit(main())
