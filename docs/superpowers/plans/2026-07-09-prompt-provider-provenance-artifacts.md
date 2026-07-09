# Prompt Provider Provenance Artifacts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add experiment-level prompt/provider provenance artifacts derived from existing ledger model traces.

**Architecture:** Keep the evidence ledger as the source of truth. Extend `experiment_artifacts.py` so artifact packaging reads copied `ledger.jsonl`, aggregates non-empty `evidence_event.payload.model_trace` records into `model_invocations.json`, and mirrors that summary in `manifest.json`. Live provider testing stays opt-in and default-skipped.

**Tech Stack:** Python dataclasses, pathlib, JSON/JSONL files, pytest, existing BayesProbe benchmark/config/model gateway modules.

## Global Constraints

- Do not change `BayesProbeCore`, evidence integration, posterior updates, or probe control flow.
- `model_invocations.json` is derived only from `ledger.jsonl`.
- Include only ledger records where `record_type == "evidence_event"` and `payload.model_trace` is a non-empty object.
- `model_invocations.json` uses `artifact_version == "0.1"`, `invocation_count`, and `invocations`.
- `invocations` is a stable sorted list of unique invocation signatures with `occurrence_count`.
- Invocation metadata is sanitized with the existing artifact metadata secret-key redaction behavior.
- `manifest.json` includes `model_invocations_path`, `model_invocation_count`, and `model_invocation_summary`.
- Tests must be deterministic/offline except the explicitly opt-in OpenAI live smoke.
- OpenAI live smoke must skip unless `BAYESPROBE_RUN_OPENAI_LIVE=1` and `OPENAI_API_KEY` is set.

---

## File Structure

- Modify `bayesprobe/experiment_artifacts.py`: add model invocation aggregation, write `model_invocations.json`, extend `ExperimentArtifactBundle`, and add manifest provenance fields.
- Modify `tests/test_experiment_runner.py`: assert normal benchmark artifact runs write model invocation provenance.
- Create `tests/test_experiment_artifacts.py`: focused tests for aggregation, duplicate counting, repair attempt preservation, empty ledger behavior, and metadata redaction.
- Modify `tests/test_openai_live.py`: add default-skipped provider-backed benchmark artifact provenance smoke.
- Modify `docs/ARCHITECTURE.md`: record prompt/model invocation artifact summary v0.1.

---

### Task 1: Artifact Writer Produces Model Invocation Summary

**Files:**
- Modify: `bayesprobe/experiment_artifacts.py`
- Modify: `tests/test_experiment_runner.py`

**Interfaces:**
- Consumes:
  - `write_experiment_artifact_bundle(...)`
  - copied artifact `ledger.jsonl`
  - existing `EvidenceEvent.model_trace` payloads in ledger records
- Produces:
  - `ExperimentArtifactBundle.model_invocations_path: Path`
  - `model_invocations.json`
  - manifest keys `model_invocations_path`, `model_invocation_count`, `model_invocation_summary`

- [ ] **Step 1: Write failing runner artifact test**

Update `tests/test_experiment_runner.py::test_run_benchmark_experiment_writes_artifact_bundle` with these assertions:

```python
model_invocations_path = artifact_dir / "model_invocations.json"

assert result.artifact_manifest_path == manifest_path
assert result.artifact_dir == artifact_dir
assert model_invocations_path.exists()

model_invocations = json.loads(model_invocations_path.read_text(encoding="utf-8"))

assert manifest["model_invocations_path"] == str(model_invocations_path)
assert manifest["model_invocation_count"] == 3
assert manifest["model_invocation_summary"] == [
    {
        "task": "judge_evidence",
        "adapter_kind": "scripted",
        "prompt_id": "evidence_judgment",
        "prompt_version": "v0.1",
        "schema_name": "EvidenceJudgment",
        "schema_version": "v0.1",
        "repair_attempt_index": None,
        "metadata": {},
        "occurrence_count": 3,
    }
]
assert model_invocations == {
    "artifact_version": "0.1",
    "invocation_count": 3,
    "invocations": manifest["model_invocation_summary"],
}
```

- [ ] **Step 2: Run the new assertion to verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_experiment_runner.py::test_run_benchmark_experiment_writes_artifact_bundle -q -p no:cacheprovider
```

Expected: FAIL because `model_invocations.json` and manifest provenance fields do not exist.

- [ ] **Step 3: Implement minimal artifact integration**

In `bayesprobe/experiment_artifacts.py`, update the bundle dataclass:

```python
@dataclass(frozen=True)
class ExperimentArtifactBundle:
    artifact_dir: Path
    manifest_path: Path
    report_path: Path
    ledger_path: Path | None
    config_snapshot_path: Path
    dataset_snapshot_path: Path
    model_invocations_path: Path
```

In `write_experiment_artifact_bundle(...)`, define and write the new artifact before manifest:

```python
model_invocations_path = target_dir / "model_invocations.json"
model_invocations = _model_invocation_artifact(artifact_ledger_path)
_write_json(model_invocations_path, model_invocations)
```

Pass `model_invocations_path` and `model_invocations` into `_manifest_payload(...)`, then return `model_invocations_path` in `ExperimentArtifactBundle`.

Add these helpers:

```python
def _model_invocation_artifact(ledger_path: Path | None) -> dict[str, Any]:
    traces = _model_traces_from_ledger(ledger_path)
    invocations = _aggregate_model_traces(traces)
    return {
        "artifact_version": "0.1",
        "invocation_count": len(traces),
        "invocations": invocations,
    }


def _model_traces_from_ledger(ledger_path: Path | None) -> list[Mapping[str, Any]]:
    if ledger_path is None or not ledger_path.exists():
        return []
    traces: list[Mapping[str, Any]] = []
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        envelope = json.loads(line)
        if envelope.get("record_type") != "evidence_event":
            continue
        payload = envelope.get("payload", {})
        if not isinstance(payload, Mapping):
            continue
        model_trace = payload.get("model_trace", {})
        if isinstance(model_trace, Mapping) and model_trace:
            traces.append(model_trace)
    return traces


def _aggregate_model_traces(traces: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[str, dict[str, Any]] = {}
    for trace in traces:
        invocation = _model_invocation_signature(trace)
        key = json.dumps(invocation, ensure_ascii=False, sort_keys=True)
        if key not in counts:
            counts[key] = {**invocation, "occurrence_count": 0}
        counts[key]["occurrence_count"] += 1
    return sorted(
        counts.values(),
        key=lambda item: json.dumps(item, ensure_ascii=False, sort_keys=True),
    )


def _model_invocation_signature(trace: Mapping[str, Any]) -> dict[str, Any]:
    metadata = trace.get("metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    return {
        "task": trace.get("task"),
        "adapter_kind": trace.get("adapter_kind"),
        "prompt_id": trace.get("prompt_id"),
        "prompt_version": trace.get("prompt_version"),
        "schema_name": trace.get("schema_name"),
        "schema_version": trace.get("schema_version"),
        "repair_attempt_index": trace.get("repair_attempt_index"),
        "metadata": _sanitize_metadata(metadata),
    }
```

Update `_manifest_payload(...)` signature and payload:

```python
"model_invocations_path": str(model_invocations_path),
"model_invocation_count": model_invocations["invocation_count"],
"model_invocation_summary": model_invocations["invocations"],
```

- [ ] **Step 4: Verify Task 1 GREEN**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_experiment_runner.py::test_run_benchmark_experiment_writes_artifact_bundle -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 5: Commit Task 1**

```bash
git add bayesprobe/experiment_artifacts.py tests/test_experiment_runner.py
git commit -m "feat: write model invocation artifacts"
```

---

### Task 2: Focused Provenance Aggregation Edge Cases

**Files:**
- Create: `tests/test_experiment_artifacts.py`
- Modify: `bayesprobe/experiment_artifacts.py`

**Interfaces:**
- Consumes: `write_experiment_artifact_bundle(...)`
- Produces: robust aggregation for duplicates, repair traces, empty ledgers, and sanitized metadata.

- [ ] **Step 1: Write focused failing aggregation tests**

Create `tests/test_experiment_artifacts.py`:

```python
import json
from pathlib import Path

from bayesprobe.benchmark_io import BenchmarkDataset
from bayesprobe.experiment_artifacts import write_experiment_artifact_bundle
from bayesprobe.experiment_runner import ExperimentRunConfig


FIXTURE_PATH = Path("fixtures/benchmarks/toy_belief_revision.json")


def write_report(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")


def append_ledger_record(path: Path, record_type: str, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"record_type": record_type, "payload": payload}) + "\n")


def test_model_invocation_artifact_aggregates_duplicate_and_repair_traces(
    tmp_path: Path,
):
    artifact_dir = tmp_path / "artifacts"
    report_path = tmp_path / "report.json"
    ledger_path = tmp_path / "ledger.jsonl"
    write_report(report_path)
    for signal_id in ["S1", "S2"]:
        append_ledger_record(
            ledger_path,
            "evidence_event",
            {
                "id": f"E_{signal_id}",
                "model_trace": {
                    "task": "judge_evidence",
                    "adapter_kind": "scripted",
                    "prompt_id": "evidence_judgment",
                    "prompt_version": "v0.1",
                    "schema_name": "EvidenceJudgment",
                    "schema_version": "v0.1",
                    "metadata": {"safe": "kept", "apiKey": "hidden"},
                },
            },
        )
    append_ledger_record(
        ledger_path,
        "evidence_event",
        {
            "id": "E_repair",
            "model_trace": {
                "task": "repair_evidence_judgment",
                "adapter_kind": "scripted",
                "prompt_id": "evidence_judgment_repair",
                "prompt_version": "v0.1",
                "schema_name": "EvidenceJudgment",
                "schema_version": "v0.1",
                "repair_attempt_index": 1,
                "metadata": {"safe": "repair"},
            },
        },
    )
    append_ledger_record(ledger_path, "evidence_event", {"id": "E_empty", "model_trace": {}})
    append_ledger_record(ledger_path, "belief_update", {"model_trace": {"task": "ignored"}})

    bundle = write_experiment_artifact_bundle(
        artifact_dir=artifact_dir,
        config=ExperimentRunConfig(
            dataset_path=FIXTURE_PATH,
            report_path=report_path,
            ledger_path=ledger_path,
            artifact_dir=artifact_dir,
        ),
        dataset=BenchmarkDataset(dataset_name="toy", samples=[]),
        report_path=report_path,
        ledger_path=ledger_path,
        sample_count=0,
    )

    payload = json.loads(bundle.model_invocations_path.read_text(encoding="utf-8"))
    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    text = bundle.model_invocations_path.read_text(encoding="utf-8")

    assert payload["invocation_count"] == 3
    assert payload["invocations"] == [
        {
            "task": "judge_evidence",
            "adapter_kind": "scripted",
            "prompt_id": "evidence_judgment",
            "prompt_version": "v0.1",
            "schema_name": "EvidenceJudgment",
            "schema_version": "v0.1",
            "repair_attempt_index": None,
            "metadata": {"safe": "kept"},
            "occurrence_count": 2,
        },
        {
            "task": "repair_evidence_judgment",
            "adapter_kind": "scripted",
            "prompt_id": "evidence_judgment_repair",
            "prompt_version": "v0.1",
            "schema_name": "EvidenceJudgment",
            "schema_version": "v0.1",
            "repair_attempt_index": 1,
            "metadata": {"safe": "repair"},
            "occurrence_count": 1,
        },
    ]
    assert manifest["model_invocation_count"] == 3
    assert manifest["model_invocation_summary"] == payload["invocations"]
    assert "hidden" not in text
    assert "apiKey" not in text


def test_model_invocation_artifact_is_empty_for_missing_or_empty_ledger(
    tmp_path: Path,
):
    artifact_dir = tmp_path / "artifacts"
    report_path = tmp_path / "report.json"
    write_report(report_path)

    bundle = write_experiment_artifact_bundle(
        artifact_dir=artifact_dir,
        config=ExperimentRunConfig(
            dataset_path=FIXTURE_PATH,
            report_path=report_path,
            artifact_dir=artifact_dir,
        ),
        dataset=BenchmarkDataset(dataset_name="toy", samples=[]),
        report_path=report_path,
        ledger_path=None,
        sample_count=0,
    )

    payload = json.loads(bundle.model_invocations_path.read_text(encoding="utf-8"))
    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    assert payload == {"artifact_version": "0.1", "invocation_count": 0, "invocations": []}
    assert manifest["model_invocation_count"] == 0
    assert manifest["model_invocation_summary"] == []
```

- [ ] **Step 2: Run focused tests to verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_experiment_artifacts.py -q -p no:cacheprovider
```

Expected: FAIL until Task 1 implementation exists, or PASS if Task 1 already implemented the complete behavior. If it passes immediately, add the missing edge assertion before proceeding.

- [ ] **Step 3: Tighten implementation if needed**

If ordering or sanitization fails, adjust `_aggregate_model_traces(...)`,
`_model_invocation_signature(...)`, or `_model_traces_from_ledger(...)` only.

- [ ] **Step 4: Verify Task 2 GREEN**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_experiment_artifacts.py tests/test_experiment_runner.py::test_run_benchmark_experiment_writes_artifact_bundle -q -p no:cacheprovider
```

Expected: PASS.

- [ ] **Step 5: Commit Task 2**

```bash
git add bayesprobe/experiment_artifacts.py tests/test_experiment_artifacts.py tests/test_experiment_runner.py
git commit -m "test: cover model invocation artifact aggregation"
```

---

### Task 3: Default-Skipped OpenAI Benchmark Artifact Smoke

**Files:**
- Modify: `tests/test_openai_live.py`

**Interfaces:**
- Consumes:
  - `run_benchmark_experiment(...)`
  - `ExperimentRunConfig`
  - OpenAI `kind="openai"` model gateway config
- Produces:
  - default-skipped live benchmark artifact provenance smoke

- [ ] **Step 1: Write default-skip smoke test**

Append to `tests/test_openai_live.py`:

```python
import json
from pathlib import Path

from bayesprobe.experiment_runner import ExperimentRunConfig, run_benchmark_experiment
```

Add:

```python
def test_openai_live_benchmark_writes_model_invocation_artifacts_when_enabled(
    tmp_path: Path,
):
    if os.environ.get("BAYESPROBE_RUN_OPENAI_LIVE") != "1":
        pytest.skip("set BAYESPROBE_RUN_OPENAI_LIVE=1 to run OpenAI live smoke")
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("set OPENAI_API_KEY to run OpenAI live smoke")

    dataset_path = tmp_path / "one-sample-openai.json"
    dataset_path.write_text(
        json.dumps(
            {
                "dataset_name": "openai_live_artifact_smoke",
                "samples": [
                    {
                        "sample_id": "openai_live_passive",
                        "question_or_claim": "Can OpenAI live smoke produce provenance artifacts?",
                        "signal_shape": "passive_only",
                        "gold_best_hypothesis": "H1",
                        "passive_signals": [
                            {
                                "signal_id": "S_openai_live",
                                "source_type": "live_smoke",
                                "source": "pytest",
                                "raw_content": "SUPPORTS: this smoke fixture supports H1 more than H2.",
                                "target_hypotheses": ["H1", "H2"],
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    artifact_dir = tmp_path / "artifacts" / "openai-live"

    result = run_benchmark_experiment(
        ExperimentRunConfig(
            dataset_path=dataset_path,
            report_path=tmp_path / "report.json",
            artifact_dir=artifact_dir,
            model_gateway={
                "kind": "openai",
                "model": os.environ.get("BAYESPROBE_OPENAI_MODEL", "gpt-5.5"),
                "max_output_tokens": 256,
            },
        )
    )

    manifest = json.loads((artifact_dir / "manifest.json").read_text(encoding="utf-8"))
    model_invocations = json.loads(
        (artifact_dir / "model_invocations.json").read_text(encoding="utf-8")
    )
    assert result.suite_result.sample_count == 1
    assert manifest["model_invocations_path"] == str(artifact_dir / "model_invocations.json")
    assert manifest["model_invocation_count"] >= 1
    assert model_invocations["invocations"][0]["adapter_kind"] == "openai"
```

- [ ] **Step 2: Verify default skip**

Run:

```bash
BAYESPROBE_RUN_OPENAI_LIVE=0 PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_openai_live.py -q -p no:cacheprovider
```

Expected: both live tests skipped.

- [ ] **Step 3: Commit Task 3**

```bash
git add tests/test_openai_live.py
git commit -m "test: add openai provenance artifact live smoke"
```

---

### Task 4: Architecture Docs and Full Verification

**Files:**
- Modify: `docs/ARCHITECTURE.md`

**Interfaces:**
- Consumes: completed model invocation artifact behavior.
- Produces: updated roadmap/status docs and full verification evidence.

- [ ] **Step 1: Update architecture docs**

In `docs/ARCHITECTURE.md`, update Phase 2 status:

```markdown
Status: OpenAI Responses adapter implemented as v0.1, and prompt/model
invocation artifact summaries implemented as v0.1; broader provider registry,
prompt registry snapshots, and provider observability remain future work.
```

Update Phase 5 status:

```markdown
Status: stable artifact directory and model invocation provenance summaries
implemented as v0.1; SQLite persistence, dataset split filters, and full prompt
registry snapshots remain future work.
```

Add a sentence near the existing Artifact v0.1 note:

```markdown
Model invocation provenance v0.1 summarizes existing ledger `model_trace`
records into `model_invocations.json` and the manifest without changing
BayesProbe core control flow.
```

- [ ] **Step 2: Run focused verification**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_experiment_artifacts.py tests/test_experiment_runner.py tests/test_openai_live.py -q -p no:cacheprovider
```

Expected: offline tests pass and live tests skip unless explicitly enabled.

- [ ] **Step 3: Run full verification**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
git diff --check
```

Expected: all tests pass except default-skipped live smoke, and `git diff --check` emits no output.

- [ ] **Step 4: Commit Task 4**

```bash
git add docs/ARCHITECTURE.md
git commit -m "docs: record model invocation artifact provenance"
```

---

## Self-Review

- Spec coverage: `model_invocations.json`, manifest provenance fields, ledger-derived source of truth, aggregation, duplicate counts, repair attempt preservation, metadata redaction, default-skipped OpenAI smoke, and architecture docs are all covered.
- Placeholder scan: no deferred implementation markers are used. Follow-up items remain outside this plan.
- Type consistency: `model_invocations_path`, `model_invocation_count`, `model_invocation_summary`, and `ExperimentArtifactBundle.model_invocations_path` are named consistently across tasks.
- Scope check: the plan touches artifact packaging, tests, live smoke, and docs only. It does not require changing BayesProbe core/evidence/posterior/probe control flow.
