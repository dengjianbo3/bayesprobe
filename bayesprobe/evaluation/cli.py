from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from bayesprobe.evaluation.arms import BayesProbePythonArm, DirectFlashArm
from bayesprobe.evaluation.artifacts import (
    CapabilityArtifactStore,
    _atomic_private_json,
    write_prepared_evaluation_set,
)
from bayesprobe.evaluation.config import (
    CapabilityExperimentConfig,
    capability_config_from_mapping,
    load_capability_config,
)
from bayesprobe.evaluation.contracts import EvaluationCase
from bayesprobe.evaluation.hle import (
    EvaluationGoldStore,
    HLEDatasetAdapter,
)
from bayesprobe.evaluation.python_probe import DockerPythonSandbox
from bayesprobe.evaluation.paradigm_checkpoint import (
    prepare_paradigm_checkpoint,
    run_paradigm_checkpoint,
    score_paradigm_checkpoint,
)
from bayesprobe.evaluation.runner import (
    CapabilityExperimentRunner,
    CapabilityPreflightResult,
    ExperimentIdentity,
    run_capability_preflight,
)
from bayesprobe.evaluation.scoring import (
    assert_shareable_payload_safe,
    score_and_write_experiment,
)
from bayesprobe.model_gateway import build_model_gateway


_SECRET_PATTERN = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b")


@dataclass(frozen=True)
class LoadedSelectionManifest:
    cases: tuple[EvaluationCase, ...]
    categories: dict[str, str]
    dataset_revision: str
    requested_sample_count: int
    manifest_sha256: str


def add_eval_subparser(subparsers: Any) -> None:
    eval_parser = subparsers.add_parser("eval")
    commands = eval_parser.add_subparsers(dest="eval_command", required=True)
    for command in ("prepare", "run"):
        command_parser = commands.add_parser(command)
        command_parser.add_argument(
            "--config",
            required=True,
            help="Path to the frozen capability experiment JSON config.",
        )
    for command in ("score", "report"):
        command_parser = commands.add_parser(command)
        command_parser.add_argument(
            "--experiment",
            required=True,
            help="Path to the restricted capability experiment directory.",
        )
    checkpoint_prepare = commands.add_parser("checkpoint-prepare")
    checkpoint_prepare.add_argument(
        "--config",
        required=True,
        help="Path to the frozen HLE v0.1 experiment JSON config.",
    )
    checkpoint_prepare.add_argument(
        "--source-experiment",
        required=True,
        help="Path to the completed restricted HLE v0.1 source experiment.",
    )
    for command in ("checkpoint-run", "checkpoint-score"):
        command_parser = commands.add_parser(command)
        command_parser.add_argument(
            "--experiment",
            required=True,
            help="Path to the prepared paradigm checkpoint directory.",
        )


def run_eval_command(args: argparse.Namespace) -> int:
    try:
        if args.eval_command == "prepare":
            message = prepare_capability_experiment(Path(args.config))
        elif args.eval_command == "run":
            message = run_capability_experiment(Path(args.config))
        elif args.eval_command == "score":
            message = score_capability_experiment(Path(args.experiment))
        elif args.eval_command == "report":
            message = report_capability_experiment(Path(args.experiment))
        elif args.eval_command == "checkpoint-prepare":
            message = prepare_paradigm_checkpoint(
                Path(args.config),
                Path(args.source_experiment),
            )
        elif args.eval_command == "checkpoint-run":
            message = run_paradigm_checkpoint(Path(args.experiment))
        elif args.eval_command == "checkpoint-score":
            message = score_paradigm_checkpoint(Path(args.experiment))
        else:
            raise ValueError(f"unsupported eval command: {args.eval_command}")
    except KeyError:
        print("error: invalid or incomplete checkpoint artifact", file=sys.stderr)
        return 1
    except (
        TypeError,
        ValueError,
        OSError,
        RuntimeError,
        subprocess.SubprocessError,
    ) as error:
        print(f"error: {_sanitized_error(error)}", file=sys.stderr)
        return 1
    print(message)
    return 0


def prepare_capability_experiment(config_path: Path) -> str:
    config = load_capability_config(config_path)
    prepared = HLEDatasetAdapter().prepare(config.selection)
    sandbox = DockerPythonSandbox(config.python_sandbox)
    preflight = run_capability_preflight(
        config,
        prepared,
        sandbox,
        repo_root=_repository_root(),
    )
    store = CapabilityArtifactStore(
        config.restricted_root,
        preflight.identity,
    )
    manifest_path = store.root / "selection_manifest.json"
    if manifest_path.exists():
        raise ValueError("capability experiment has already been prepared")
    paths = write_prepared_evaluation_set(store.root, prepared)
    _write_preparation_snapshots(
        store=store,
        config=config,
        preflight=preflight,
    )
    return (
        "BayesProbe capability preparation complete: "
        f"experiment={preflight.identity.experiment_id} "
        f"samples={len(prepared.runtime_cases)} "
        f"manifest={paths.selection_manifest}"
    )


def run_capability_experiment(config_path: Path) -> str:
    config = load_capability_config(config_path)
    experiment_path, selection, identity = _find_prepared_experiment(config)
    sandbox = DockerPythonSandbox(config.python_sandbox)
    preflight = run_capability_preflight(
        config,
        selection,
        sandbox,
        repo_root=_repository_root(),
    )
    if preflight.identity != identity or experiment_path.name != identity.experiment_id:
        raise ValueError("prepared experiment identity does not match current preflight")
    store = CapabilityArtifactStore(config.restricted_root, identity)
    observer = store.provider_observer()
    model_gateway = build_model_gateway(
        config.model_gateway,
        invocation_observer=observer,
    )
    direct = DirectFlashArm(
        model_gateway,
        invocation_metadata={"experiment_id": identity.experiment_id},
    )
    bayesprobe = BayesProbePythonArm(
        model_gateway,
        sandbox,
        image=preflight.image,
        invocation_metadata={"experiment_id": identity.experiment_id},
        ledger_factory=lambda case: store.ledger_for("bayesprobe_python", case),
        execution_observer_factory=lambda case: store.python_observer_for(
            "bayesprobe_python", case
        ),
    )
    summary = CapabilityExperimentRunner(
        identity=identity,
        cases=list(selection.cases),
        arms={"direct_flash": direct, "bayesprobe_python": bayesprobe},
        artifact_store=store,
        direct_concurrency=config.direct_concurrency,
        bayesprobe_concurrency=config.bayesprobe_concurrency,
    ).run()
    return (
        "BayesProbe capability run complete: "
        f"experiment={identity.experiment_id} "
        f"terminal={summary.terminal_count}/{summary.task_count} "
        f"completed={summary.completed_count} "
        f"terminal_failed={summary.terminal_failed_count}"
    )


def score_capability_experiment(experiment_path: Path) -> str:
    identity = _load_identity(experiment_path)
    config = _load_experiment_config_snapshot(experiment_path)
    selection = _load_selection_manifest(experiment_path / "selection_manifest.json")
    gold = _load_gold_store(experiment_path / "gold_store.json")
    store = CapabilityArtifactStore(experiment_path.parent, identity)
    provider_secret = os.environ.get(config.model_gateway.api_key_env)
    paths = score_and_write_experiment(
        artifact_store=store,
        cases=list(selection.cases),
        gold=gold,
        categories=selection.categories,
        report_root=config.report_root,
        provider_secrets=[provider_secret] if provider_secret else [],
        pricing_snapshot=config.pricing_snapshot,
    )
    return (
        "BayesProbe capability scoring complete: "
        f"experiment={identity.experiment_id} report={paths.summary_json}"
    )


def report_capability_experiment(experiment_path: Path) -> str:
    identity = _load_identity(experiment_path)
    config = _load_experiment_config_snapshot(experiment_path)
    selection = _load_selection_manifest(experiment_path / "selection_manifest.json")
    report_root = config.report_root / identity.experiment_id
    required_paths = (
        report_root / "summary.json",
        report_root / "summary.md",
        report_root / "paired_metrics.json",
        report_root / "provenance.json",
    )
    if not (experiment_path / "scoring_complete.json").exists():
        raise ValueError("capability experiment must be scored before report")
    if not all(path.exists() for path in required_paths):
        raise ValueError("shareable capability report is incomplete")
    restricted_values = [
        value
        for case in selection.cases
        for value in (case.sample_id, case.question, *case.choices.values())
    ]
    provider_secret = os.environ.get(config.model_gateway.api_key_env)
    for path in required_paths:
        payload: Any
        if path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            payload = path.read_text(encoding="utf-8")
        assert_shareable_payload_safe(
            payload,
            restricted_values=restricted_values,
            canaries=(),
            provider_secrets=[provider_secret] if provider_secret else [],
        )
    return (
        "BayesProbe capability report verified: "
        f"experiment={identity.experiment_id} report={required_paths[0]}"
    )


def _find_prepared_experiment(
    config: CapabilityExperimentConfig,
) -> tuple[Path, LoadedSelectionManifest, ExperimentIdentity]:
    if not config.restricted_root.exists():
        raise ValueError("no prepared capability experiment was found")
    for candidate in sorted(config.restricted_root.iterdir()):
        if not candidate.is_dir():
            continue
        identity_path = candidate / "experiment_identity.json"
        manifest_path = candidate / "selection_manifest.json"
        config_path = candidate / "config_snapshot.json"
        if not (identity_path.exists() and manifest_path.exists() and config_path.exists()):
            continue
        identity = _load_identity(candidate)
        if (
            identity.dataset_revision_sha != config.selection.revision
            or identity.config_sha256 != config.config_sha256
            or identity.prompt_registry_sha256 != config.prompt_registry_sha256
        ):
            continue
        return candidate, _load_selection_manifest(manifest_path), identity
    raise ValueError("no prepared capability experiment matches the frozen config")


def _load_selection_manifest(path: Path) -> LoadedSelectionManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("selection manifest must be an object")
    claimed_hash = payload.get("manifest_sha256")
    unsigned = dict(payload)
    unsigned.pop("manifest_sha256", None)
    actual_hash = _canonical_sha256(unsigned)
    if claimed_hash != actual_hash:
        raise ValueError("selection manifest hash does not match content")
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("selection manifest items must be an array")
    cases: list[EvaluationCase] = []
    categories: dict[str, str] = {}
    for item in items:
        if not isinstance(item, Mapping):
            raise ValueError("selection manifest item must be an object")
        case = EvaluationCase(
            sample_id=item["sample_id"],
            question=item["question"],
            choices=item["choices"],
        )
        cases.append(case)
        categories[case.sample_id] = str(item["category"])
    requested_sample_count = int(payload["requested_sample_count"])
    if len(cases) != requested_sample_count:
        raise ValueError("selection manifest item count is incomplete")
    return LoadedSelectionManifest(
        cases=tuple(cases),
        categories=categories,
        dataset_revision=str(payload["dataset_revision"]),
        requested_sample_count=requested_sample_count,
        manifest_sha256=actual_hash,
    )


def _load_gold_store(path: Path) -> EvaluationGoldStore:
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("items") if isinstance(payload, Mapping) else None
    if not isinstance(items, list):
        raise ValueError("gold store items must be an array")
    labels: dict[str, str] = {}
    for item in items:
        if not isinstance(item, Mapping) or set(item) != {"sample_id", "gold_label"}:
            raise ValueError("gold store item has an invalid schema")
        labels[str(item["sample_id"])] = str(item["gold_label"])
    return EvaluationGoldStore(
        manifest_sha256=str(payload["manifest_sha256"]),
        labels=labels,
    )


def _load_identity(path: Path) -> ExperimentIdentity:
    payload = json.loads((path / "experiment_identity.json").read_text(encoding="utf-8"))
    return ExperimentIdentity(**payload)


def _load_experiment_config_snapshot(path: Path) -> CapabilityExperimentConfig:
    payload = json.loads((path / "config_snapshot.json").read_text(encoding="utf-8"))
    return capability_config_from_mapping(payload)


def _write_preparation_snapshots(
    *,
    store: CapabilityArtifactStore,
    config: CapabilityExperimentConfig,
    preflight: CapabilityPreflightResult,
) -> None:
    _atomic_private_json(store.root / "config_snapshot.json", config.snapshot())
    _atomic_private_json(
        store.root / "prompt_registry_snapshot.json",
        config.prompt_registry,
    )
    _atomic_private_json(
        store.root / "pricing_snapshot.json",
        config.pricing_snapshot,
    )
    _atomic_private_json(
        store.root / "dataset_revision.json",
        {
            "dataset": "cais/hle",
            "revision": config.selection.revision,
        },
    )
    _atomic_private_json(store.root / "preflight.json", asdict(preflight))


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _sanitized_error(error: Exception) -> str:
    message = str(error)
    message = _SECRET_PATTERN.sub("<redacted>", message)
    for key, value in os.environ.items():
        normalized_key = key.lower()
        if (
            any(marker in normalized_key for marker in ("key", "secret", "token"))
            and len(value) >= 8
        ):
            message = message.replace(value, "<redacted>")
    return message


__all__ = [
    "add_eval_subparser",
    "prepare_capability_experiment",
    "report_capability_experiment",
    "run_capability_experiment",
    "run_eval_command",
    "score_capability_experiment",
]
