from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

from harbor.constants import PACKAGE_CACHE_DIR
from harbor.environments.definition import (
    environment_content_hash,
    require_agent_environment_definition,
    should_use_prebuilt_docker_image,
)
from harbor.models.job.config import JobConfig
from harbor.models.job.lock import JobLock, TrialLock
from harbor.models.task.config import TaskConfig as TaskDefinitionConfig
from harbor.models.task.id import PackageTaskId
from harbor.models.trial.config import TrialConfig
from harbor.models.trial.result import TrialResult

from bayesprobe_terminal_bench.config import TerminalBenchConfig
from bayesprobe_terminal_bench.runner_factory import (
    terminal_bench_lock_schema_mismatches,
)


HARBOR_VERSION = "0.18.0"
DATASET_NAME = "terminal-bench/terminal-bench-2"
TASK_ID = "terminal-bench/break-filter-js-from-html"
LOCK_SCHEMA_VERSION = "terminal_bench_lock:v0.1"
TERMINAL_PLAN_VERSION = "terminal_probe_plan:v0.1"
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class RuntimeIdentity:
    harbor_version: str
    root_git_sha: str
    adapter_tree_sha: str
    container_image: str
    image_digest: str


@dataclass(frozen=True)
class OracleIdentity:
    dataset_name: str
    dataset_revision: str
    task_id: str
    task_checksum: str
    n_attempts: int
    package_org: str
    package_name: str
    package_ref: str
    download_dir: Path | None
    force_build: bool


def build_lock(
    *,
    job_dir: Path,
    config: TerminalBenchConfig,
    runtime_identity: RuntimeIdentity,
    restricted_values: tuple[str, ...] = (),
) -> dict[str, object]:
    oracle = _extract_oracle_identity(Path(job_dir))
    if runtime_identity.harbor_version != HARBOR_VERSION:
        raise ValueError(f"Harbor version must be {HARBOR_VERSION}")
    if not _DIGEST.fullmatch(runtime_identity.image_digest):
        raise ValueError("image digest must be a sha256 digest")

    lock: dict[str, object] = {
        "schema_version": LOCK_SCHEMA_VERSION,
        "harbor_version": runtime_identity.harbor_version,
        "dataset_name": oracle.dataset_name,
        "dataset_revision": oracle.dataset_revision,
        "task_id": oracle.task_id,
        "task_checksum": oracle.task_checksum,
        "container_image": runtime_identity.container_image,
        "image_digest": runtime_identity.image_digest,
        "root_git_sha": runtime_identity.root_git_sha,
        "adapter_tree_sha": runtime_identity.adapter_tree_sha,
        "n_attempts": oracle.n_attempts,
        "model": config.model,
        "base_url": config.base_url,
        "provider_protocol": "openai_chat_completions",
        "api_key_env": config.api_key_env,
        "temperature": 0,
        "max_cycles": config.max_cycles,
        "max_probes_per_cycle": config.max_probes_per_cycle,
        "max_actions_per_probe": config.max_actions_per_probe,
        "max_total_actions": config.max_total_actions,
        "max_model_calls": config.max_model_calls,
        "command_timeout_seconds": config.command_timeout_seconds,
        "provider_timeout_seconds": config.provider_timeout_seconds,
        "max_output_tokens": config.max_output_tokens,
        "signal_output_bytes": config.signal_output_bytes,
        "terminal_plan_version": TERMINAL_PLAN_VERSION,
    }
    mismatches = terminal_bench_lock_schema_mismatches(lock)
    if mismatches:
        raise ValueError(f"Terminal-Bench lock mismatch: {sorted(mismatches)}")
    _serialize_lock(lock, restricted_values=restricted_values)
    return lock


def collect_runtime_identity(
    *,
    repository_root: Path,
    job_dir: Path,
    package_cache_dir: Path = PACKAGE_CACHE_DIR,
) -> RuntimeIdentity:
    installed_harbor = version("harbor")
    if installed_harbor != HARBOR_VERSION:
        raise ValueError(
            f"Harbor version must be {HARBOR_VERSION}; found {installed_harbor}"
        )
    root = Path(repository_root).resolve()
    container_image = discover_container_image(
        job_dir=job_dir,
        package_cache_dir=package_cache_dir,
    )
    return RuntimeIdentity(
        harbor_version=installed_harbor,
        root_git_sha=_run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            error="could not resolve repository Git identity",
        ),
        adapter_tree_sha=_run(
            ["git", "rev-parse", "HEAD:benchmarks/terminal_bench"],
            cwd=root,
            error="could not resolve committed adapter tree identity",
        ),
        container_image=container_image,
        image_digest=_docker_image_digest(container_image),
    )


def discover_container_image(
    *,
    job_dir: Path,
    package_cache_dir: Path = PACKAGE_CACHE_DIR,
) -> str:
    identity = extract_job_identity(Path(job_dir), required_reward=None)
    base_dir = identity.download_dir or Path(package_cache_dir)
    task_dir = (
        base_dir
        / identity.package_org
        / identity.package_name
        / identity.package_ref.removeprefix("sha256:")
    )
    task_config_path = task_dir / "task.toml"
    try:
        task_definition = TaskDefinitionConfig.model_validate_toml(
            task_config_path.read_text(encoding="utf-8")
        )
    except Exception:
        raise ValueError(
            "could not resolve the exact cached Harbor package task environment"
        ) from None
    environment_dir = task_dir / "environment"
    docker_image = task_definition.environment.docker_image
    require_agent_environment_definition(
        environment_dir,
        docker_image=docker_image,
    )
    if should_use_prebuilt_docker_image(
        environment_dir,
        docker_image=docker_image,
        force_build=identity.force_build,
    ):
        if not isinstance(docker_image, str) or not docker_image.strip():
            raise ValueError("cached Harbor task has no usable prebuilt image")
        return docker_image.strip()
    return f"hb__{environment_content_hash(environment_dir, docker_image=docker_image)}"


def write_lock_atomic(
    output_path: Path,
    lock: Mapping[str, object],
    *,
    restricted_values: tuple[str, ...] = (),
) -> None:
    output = Path(output_path)
    serialized = _serialize_lock(lock, restricted_values=restricted_values)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write the locked Terminal-Bench engineering smoke identity."
    )
    parser.add_argument("--oracle-job", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    config, api_key = TerminalBenchConfig.from_sources()
    repository_root = Path(__file__).resolve().parents[3]
    runtime_identity = collect_runtime_identity(
        repository_root=repository_root,
        job_dir=args.oracle_job,
    )
    lock = build_lock(
        job_dir=args.oracle_job,
        config=config,
        runtime_identity=runtime_identity,
        restricted_values=(api_key,),
    )
    write_lock_atomic(args.output, lock, restricted_values=(api_key,))
    return 0


def _extract_oracle_identity(job_dir: Path) -> OracleIdentity:
    completed = _completed_trial_files(job_dir) if job_dir.is_dir() else []
    if not completed:
        root_result_path = job_dir / "result.json"
        if root_result_path.is_file():
            root_result = _read_object(root_result_path)
            if _official_reward(root_result) != 1.0:
                raise ValueError("oracle reward must be 1")
        raise ValueError("exactly one completed Oracle trial is required; found 0")
    if len(completed) != 1:
        raise ValueError(
            f"exactly one completed Oracle trial is required; found {len(completed)}"
        )
    identity = extract_job_identity(job_dir, required_reward=1.0)
    return identity


def extract_job_identity(
    job_dir: Path,
    *,
    required_reward: float | None,
) -> OracleIdentity:
    if not job_dir.is_dir():
        raise ValueError("Harbor job directory does not exist")
    completed = _completed_trial_files(job_dir)
    if len(completed) != 1:
        raise ValueError(
            f"exactly one completed Harbor trial is required; found {len(completed)}"
        )

    config_path, result_path = completed[0]
    trial_dir = result_path.parent
    job_config_payload = _read_object(job_dir / "config.json")
    job_lock_payload = _read_object(job_dir / "lock.json")
    trial_config_payload = _read_object(config_path)
    trial_lock_payload = _read_object(trial_dir / "lock.json")
    trial_result_payload = _read_object(result_path)
    try:
        job_config = JobConfig.model_validate(job_config_payload)
        job_lock = JobLock.model_validate(job_lock_payload)
        trial_config = TrialConfig.model_validate(trial_config_payload)
        trial_lock = TrialLock.model_validate(trial_lock_payload)
        trial_result = TrialResult.model_validate(trial_result_payload)
    except Exception:
        raise ValueError("Harbor identity artifacts do not match 0.18 models") from None

    if required_reward is not None and _official_reward(trial_result_payload) != required_reward:
        raise ValueError("oracle reward must be 1")
    if job_config.n_attempts != 1:
        raise ValueError("Oracle smoke must use exactly one attempt")
    if len(job_config.datasets) != 1:
        raise ValueError("Harbor job must contain exactly one fixed dataset")
    if len(job_lock.trials) != 1:
        raise ValueError("Harbor job lock must contain exactly one resolved trial")
    if job_lock.harbor.version != HARBOR_VERSION:
        raise ValueError(f"Harbor job lock version must be {HARBOR_VERSION}")
    if job_lock.trials[0].task.type != "package" or trial_lock.task.type != "package":
        raise ValueError("fixed Harbor task must be a package task")
    if not isinstance(trial_result.task_id, PackageTaskId):
        raise ValueError("completed Harbor trial must have a package task id")

    dataset = job_config.datasets[0]
    task_names = dataset.task_names
    if task_names is None or len(task_names) != 1:
        raise ValueError(
            f"Harbor job dataset task_names must contain exactly {TASK_ID}"
        )
    dataset_task_names = tuple(
        _required_string(value, f"job dataset task_names[{index}]")
        for index, value in enumerate(task_names)
    )
    if dataset_task_names != (TASK_ID,):
        raise ValueError(
            f"Harbor job dataset task_names must contain exactly {TASK_ID}"
        )
    dataset_name = _single_candidate(
        "dataset sources",
        (
            _required_string(dataset.name, "job dataset name"),
            _required_string(job_lock.trials[0].task.source, "job lock task source"),
            _required_string(trial_lock.task.source, "trial lock task source"),
            _required_string(trial_config.task.source, "trial config task source"),
            _required_string(trial_result.source, "trial result source"),
            _required_string(
                trial_result.config.task.source,
                "trial result config task source",
            ),
        ),
    )
    dataset_revision = _required_string(
        dataset.ref or dataset.version,
        "dataset revision",
    )
    package_task_id = (
        f"{trial_result.task_id.org}/{trial_result.task_id.name}"
    )
    task_id = _single_candidate(
        "task identities",
        (
            *dataset_task_names,
            _required_string(job_lock.trials[0].task.name, "job lock task name"),
            _required_string(trial_lock.task.name, "trial lock task name"),
            _required_string(trial_config.task.name, "trial config task name"),
            package_task_id,
            _required_string(
                trial_result.config.task.name,
                "trial result config task name",
            ),
        ),
    )
    if trial_result.task_name != trial_result.task_id.name:
        raise ValueError("completed Harbor trial contains conflicting task identities")
    task_checksum = _single_candidate(
        "task checksums or refs",
        (
            _required_string(job_lock.trials[0].task.digest, "job lock task digest"),
            _required_string(trial_lock.task.digest, "trial lock task digest"),
            _required_string(trial_config.task.ref, "trial config task ref"),
            _required_string(trial_result.task_id.ref, "trial result task ref"),
            _required_string(trial_result.task_checksum, "trial result task checksum"),
            _required_string(
                trial_result.config.task.ref,
                "trial result config task ref",
            ),
        ),
    )
    if dataset_name != DATASET_NAME:
        raise ValueError(f"Oracle dataset must be {DATASET_NAME}")
    if task_id != TASK_ID:
        raise ValueError(f"Oracle task must be {TASK_ID}")
    if not _DIGEST.fullmatch(dataset_revision):
        raise ValueError("dataset revision must be a sha256 digest")
    if not _DIGEST.fullmatch(task_checksum):
        raise ValueError("task checksum must be a sha256 digest")
    return OracleIdentity(
        dataset_name=dataset_name,
        dataset_revision=dataset_revision,
        task_id=task_id,
        task_checksum=task_checksum,
        n_attempts=job_config.n_attempts,
        package_org=trial_result.task_id.org,
        package_name=trial_result.task_id.name,
        package_ref=task_checksum,
        download_dir=trial_config.task.download_dir,
        force_build=trial_config.environment.force_build,
    )


def _completed_trial_files(job_dir: Path) -> list[tuple[Path, Path]]:
    completed: list[tuple[Path, Path]] = []
    for result_path in sorted(job_dir.rglob("result.json")):
        if result_path.parent == job_dir:
            continue
        config_path = result_path.with_name("config.json")
        if not config_path.is_file():
            continue
        result = _read_object(result_path)
        if result.get("finished_at") is not None:
            completed.append((config_path, result_path))
    return completed


def _official_reward(result: Mapping[str, Any]) -> float:
    verifier = result.get("verifier_result")
    if isinstance(verifier, Mapping):
        rewards = verifier.get("rewards")
        if isinstance(rewards, Mapping):
            values = [
                float(value)
                for value in rewards.values()
                if type(value) in (int, float)
            ]
            if len(values) == 1:
                return values[0]
            if "reward" in rewards and type(rewards["reward"]) in (int, float):
                return float(rewards["reward"])
    reward = result.get("reward")
    if type(reward) in (int, float):
        return float(reward)
    raise ValueError("completed Oracle trial has no official reward")


def _single_candidate(label: str, values: Sequence[str]) -> str:
    candidates = set(values)
    if len(candidates) != 1:
        raise ValueError(f"completed Harbor trial contains conflicting {label}")
    return candidates.pop()


def _docker_image_digest(container_image: str) -> str:
    output = _run(
        ["docker", "image", "inspect", container_image],
        cwd=None,
        error="could not inspect locked container image",
    )
    try:
        documents = json.loads(output)
    except json.JSONDecodeError:
        raise ValueError("Docker image inspect returned invalid JSON") from None
    if not isinstance(documents, list) or len(documents) != 1:
        raise ValueError("Docker image inspect must resolve exactly one image")
    document = documents[0]
    if not isinstance(document, Mapping):
        raise ValueError("Docker image inspect returned an invalid image record")
    matching_repo_digests: set[str] = set()
    repository = _image_repository(container_image)
    repo_digests = document.get("RepoDigests")
    if isinstance(repo_digests, list):
        for item in repo_digests:
            if not isinstance(item, str) or "@" not in item:
                continue
            repo, digest = item.rsplit("@", 1)
            if not _DIGEST.fullmatch(digest):
                continue
            if repo == repository:
                matching_repo_digests.add(digest)
    if len(matching_repo_digests) > 1:
        raise ValueError("Docker image resolves to conflicting repository digests")
    explicit_digest = (
        container_image.rsplit("@", 1)[-1] if "@" in container_image else None
    )
    if isinstance(explicit_digest, str) and _DIGEST.fullmatch(explicit_digest):
        image_id = document.get("Id")
        if matching_repo_digests:
            if matching_repo_digests != {explicit_digest}:
                raise ValueError(
                    "Docker image does not match the requested sha256 digest"
                )
            return explicit_digest
        if image_id != explicit_digest:
            raise ValueError("Docker image does not match the requested sha256 digest")
        return explicit_digest
    if matching_repo_digests:
        return matching_repo_digests.pop()
    image_id = document.get("Id")
    if isinstance(image_id, str) and _DIGEST.fullmatch(image_id):
        return image_id
    raise ValueError("Docker image has no unambiguous sha256 digest")


def _image_repository(reference: str) -> str:
    without_digest = reference.split("@", 1)[0]
    slash = without_digest.rfind("/")
    colon = without_digest.rfind(":")
    if colon > slash:
        return without_digest[:colon]
    return without_digest


def _run(
    command: list[str],
    *,
    cwd: Path | None,
    error: str,
) -> str:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        raise ValueError(error) from None
    output = completed.stdout.strip()
    if not output:
        raise ValueError(error)
    return output


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise ValueError(f"could not read Harbor JSON artifact: {path.name}") from None
    if not isinstance(value, dict):
        raise ValueError(f"Harbor JSON artifact must be an object: {path.name}")
    return value


def _required_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"completed Oracle trial has no {label}")
    return value.strip()


def _serialize_lock(
    lock: Mapping[str, object],
    *,
    restricted_values: tuple[str, ...],
) -> str:
    serialized = json.dumps(
        dict(lock),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    for restricted in restricted_values:
        if restricted and restricted in serialized:
            raise ValueError("benchmark lock contains a restricted value")
    return serialized


if __name__ == "__main__":
    raise SystemExit(main())
