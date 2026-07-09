# Benchmark Expansion and Provider Fixtures Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add offline provider replay fixtures and a v0.2 methodology benchmark dataset so BayesProbe can evaluate final-answer utility and belief-state quality without live network calls.

**Architecture:** `RecordedModelGateway` becomes a provider-shaped offline `ModelGateway` beside deterministic/scripted/OpenAI adapters. Benchmark quality metrics are computed in `bayesprobe/benchmark.py` from existing core outputs, so no belief update rule changes are needed. New fixtures enter through existing dataset/config/experiment/report/artifact seams.

**Tech Stack:** Python 3.11+, stdlib JSON/path handling, existing dataclasses/Pydantic domain models, existing pytest suite, JSON benchmark/provider fixtures.

## Global Constraints

- Do not build a ReAct/ReWOO baseline in this slice.
- Do not add a new internal agent control-flow.
- Do not require live provider calls in normal tests.
- Do not store API keys in fixtures, configs, reports, ledgers, or docs.
- Do not replace JSONL ledger storage yet.
- Do not build a full statistical significance framework yet.
- External provider output remains model-shaped judgment, not control flow.
- Recorded provider data enters only through `ModelGateway`.
- Passive multi-agent projections remain signals, not imported belief states.
- Final answers remain evaluated while belief-state quality becomes observable.

---

## File Structure

- Create `bayesprobe/recorded_gateway.py`: load and replay recorded provider fixtures through the `ModelGateway` protocol.
- Modify `bayesprobe/model_gateway.py`: add `fixture_path` to `ModelGatewayConfig`, add `kind="recorded"` factory routing, export `RecordedModelGateway`.
- Modify `bayesprobe/config.py`: parse `model_gateway.kind == "recorded"` and resolve `fixture_path` relative to the experiment config file.
- Modify `bayesprobe/experiment_artifacts.py`: include sanitized `fixture_path` for recorded gateway snapshots.
- Modify `bayesprobe/__init__.py`: export `RecordedModelGateway`.
- Modify `bayesprobe/benchmark.py`: add benchmark quality fields and compute them from result evidence events, belief updates, and final belief state.
- Modify `bayesprobe/benchmark_io.py`: no schema change required for v0.2 dataset loading; report writing automatically includes new dataclass fields via `asdict`.
- Create `fixtures/providers/deepseek_chat_evidence_v0_1.json`: offline recorded provider fixture with no secrets.
- Create `fixtures/benchmarks/bayesprobe_v0_2_methodology.json`: richer methodology dataset.
- Modify tests:
  - `tests/test_recorded_model_gateway.py`
  - `tests/test_model_gateway.py`
  - `tests/test_public_api_and_config.py`
  - `tests/test_experiment_artifacts.py`
  - `tests/test_benchmark_harness.py`
  - `tests/test_benchmark_io.py`
  - `tests/test_experiment_runner.py`

---

### Task 1: Recorded Model Gateway Core

**Files:**
- Create: `bayesprobe/recorded_gateway.py`
- Test: `tests/test_recorded_model_gateway.py`

**Interfaces:**
- Consumes: `StructuredModelRequest`, `ModelGatewayValidationError`, and `evidence_judgment_from_mapping(...)` from `bayesprobe.model_gateway`.
- Produces:
  - `RecordedModelGateway.from_json(path: str | Path) -> RecordedModelGateway`
  - `RecordedModelGateway.complete_structured(request: StructuredModelRequest) -> dict[str, Any]`
  - `RecordedModelGateway.adapter_kind = "recorded"`
  - `RecordedModelGateway.requests: list[StructuredModelRequest]`

- [ ] **Step 1: Write failing recorded gateway tests**

Add `tests/test_recorded_model_gateway.py`:

```python
import json
from pathlib import Path

import pytest

from bayesprobe.model_gateway import ModelGatewayValidationError, StructuredModelRequest
from bayesprobe.recorded_gateway import RecordedModelGateway


def write_fixture(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def make_request(signal_id: str = "S_chem_constant_volume") -> StructuredModelRequest:
    return StructuredModelRequest(
        task="judge_evidence",
        input={
            "signal_id": signal_id,
            "raw_content": "Constant-volume inert gas evidence.",
            "target_hypotheses": ["H1", "H2"],
        },
        prompt_id="evidence_judgment",
        prompt_version="v0.1",
        schema_name="EvidenceJudgment",
        schema_version="v0.1",
    )


def recorded_fixture_payload() -> dict:
    return {
        "fixture_name": "deepseek_chat_evidence_v0_1",
        "metadata": {
            "provider_kind": "openai_chat_completions",
            "model": "deepseek-v4-flash",
            "recorded_at": "2026-07-10",
        },
        "responses": [
            {
                "match": {
                    "task": "judge_evidence",
                    "signal_id": "S_chem_constant_volume",
                },
                "response": {
                    "evidence_type": "supporting",
                    "likelihoods": {
                        "H1": "moderately_confirming",
                        "H2": "moderately_disconfirming",
                    },
                    "interpretation": "Recorded provider judgment.",
                    "quality_overrides": {},
                },
            }
        ],
    }


def test_recorded_model_gateway_replays_response_by_task_and_signal_id(tmp_path: Path):
    path = tmp_path / "recorded.json"
    write_fixture(path, recorded_fixture_payload())
    gateway = RecordedModelGateway.from_json(path)

    result = gateway.complete_structured(make_request())

    assert gateway.adapter_kind == "recorded"
    assert gateway.fixture_name == "deepseek_chat_evidence_v0_1"
    assert gateway.metadata["model"] == "deepseek-v4-flash"
    assert result["evidence_type"] == "supporting"
    assert result["likelihoods"]["H1"] == "moderately_confirming"
    assert gateway.requests[0].input["signal_id"] == "S_chem_constant_volume"


def test_recorded_model_gateway_raises_clear_error_when_no_entry_matches(tmp_path: Path):
    path = tmp_path / "recorded.json"
    write_fixture(path, recorded_fixture_payload())
    gateway = RecordedModelGateway.from_json(path)

    with pytest.raises(
        ModelGatewayValidationError,
        match="no recorded model response for task=judge_evidence signal_id=S_unknown",
    ):
        gateway.complete_structured(make_request("S_unknown"))


def test_recorded_model_gateway_rejects_fixture_with_api_key(tmp_path: Path):
    path = tmp_path / "unsafe.json"
    payload = recorded_fixture_payload()
    payload["metadata"]["api_key"] = "sk-unsafe"
    write_fixture(path, payload)

    with pytest.raises(ValueError, match="recorded model fixture must not contain secrets"):
        RecordedModelGateway.from_json(path)


def test_recorded_model_gateway_validates_recorded_response(tmp_path: Path):
    path = tmp_path / "invalid.json"
    payload = recorded_fixture_payload()
    payload["responses"][0]["response"] = {"likelihoods": {}}
    write_fixture(path, payload)

    with pytest.raises(
        ModelGatewayValidationError,
        match="evidence judgment missing field: evidence_type",
    ):
        RecordedModelGateway.from_json(path)
```

- [ ] **Step 2: Run RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_recorded_model_gateway.py -q -p no:cacheprovider
```

Expected: import failure because `bayesprobe.recorded_gateway` does not exist.

- [ ] **Step 3: Implement `RecordedModelGateway`**

Create `bayesprobe/recorded_gateway.py`:

```python
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from bayesprobe.model_gateway import (
    ModelGatewayValidationError,
    StructuredModelRequest,
    evidence_judgment_from_mapping,
)

_SECRET_KEY_PARTS = ("api_key", "apikey", "authorization", "token", "secret")


class RecordedModelGateway:
    adapter_kind = "recorded"

    def __init__(
        self,
        *,
        fixture_name: str,
        responses: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
        fixture_path: str | Path | None = None,
    ) -> None:
        if not isinstance(fixture_name, str) or not fixture_name.strip():
            raise ValueError("recorded model fixture_name must not be empty")
        self.fixture_name = fixture_name.strip()
        self.responses = list(responses)
        self.metadata = dict(metadata or {})
        self.fixture_path = Path(fixture_path) if fixture_path is not None else None
        self.requests: list[StructuredModelRequest] = []

    @classmethod
    def from_json(cls, path: str | Path) -> "RecordedModelGateway":
        fixture_path = Path(path)
        payload = json.loads(fixture_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("recorded model fixture must be an object")
        _reject_secrets(payload)
        responses = payload.get("responses")
        if not isinstance(responses, list):
            raise ValueError("recorded model fixture responses must be an array")
        for entry in responses:
            _validate_entry(entry)
        fixture_name = payload.get("fixture_name", fixture_path.stem)
        metadata = payload.get("metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("recorded model fixture metadata must be an object")
        return cls(
            fixture_name=str(fixture_name),
            responses=[dict(entry) for entry in responses],
            metadata=dict(metadata),
            fixture_path=fixture_path,
        )

    def complete_structured(self, request: StructuredModelRequest) -> dict[str, Any]:
        self.requests.append(request)
        signal_id = str(request.input.get("signal_id", ""))
        for entry in self.responses:
            match = entry["match"]
            if _matches_request(match, request):
                return dict(entry["response"])
        raise ModelGatewayValidationError(
            f"no recorded model response for task={request.task} signal_id={signal_id}"
        )


def _matches_request(match: Mapping[str, Any], request: StructuredModelRequest) -> bool:
    task = match.get("task")
    if task is not None and task != request.task:
        return False
    signal_id = match.get("signal_id")
    if signal_id is not None and signal_id != request.input.get("signal_id"):
        return False
    return True


def _validate_entry(entry: Any) -> None:
    if not isinstance(entry, Mapping):
        raise ValueError("recorded model response entry must be an object")
    match = entry.get("match")
    if not isinstance(match, Mapping):
        raise ValueError("recorded model response entry match must be an object")
    if "task" not in match:
        raise ValueError("recorded model response match must include task")
    response = entry.get("response")
    if not isinstance(response, Mapping):
        raise ValueError("recorded model response must be an object")
    evidence_judgment_from_mapping(dict(response))


def _reject_secrets(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).replace("_", "").replace("-", "").lower()
            if any(secret_part in key_text for secret_part in _SECRET_KEY_PARTS):
                raise ValueError("recorded model fixture must not contain secrets")
            _reject_secrets(item)
    elif isinstance(value, list):
        for item in value:
            _reject_secrets(item)


__all__ = ["RecordedModelGateway"]
```

- [ ] **Step 4: Run GREEN**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_recorded_model_gateway.py -q -p no:cacheprovider
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add bayesprobe/recorded_gateway.py tests/test_recorded_model_gateway.py
git commit -m "feat: add recorded model gateway"
```

---

### Task 2: Recorded Gateway Config, SDK, and Artifact Wiring

**Files:**
- Modify: `bayesprobe/model_gateway.py`
- Modify: `bayesprobe/config.py`
- Modify: `bayesprobe/experiment_artifacts.py`
- Modify: `bayesprobe/__init__.py`
- Test: `tests/test_model_gateway.py`
- Test: `tests/test_public_api_and_config.py`
- Test: `tests/test_experiment_artifacts.py`

**Interfaces:**
- Consumes: `RecordedModelGateway.from_json(path)` from Task 1.
- Produces:
  - `ModelGatewayConfig.fixture_path: str | Path | None`
  - `build_model_gateway({"kind": "recorded", "fixture_path": "..."})`
  - experiment config parsing for recorded gateways
  - artifact snapshots with sanitized `fixture_path`
  - public import `from bayesprobe import RecordedModelGateway`

- [ ] **Step 1: Write failing factory test**

Add to `tests/test_model_gateway.py`:

```python
def test_build_model_gateway_creates_recorded_gateway(tmp_path: Path):
    fixture_path = tmp_path / "recorded.json"
    fixture_path.write_text(
        json.dumps(
            {
                "fixture_name": "recorded_factory",
                "responses": [
                    {
                        "match": {"task": "judge_evidence", "signal_id": "S1"},
                        "response": {
                            "evidence_type": "supporting",
                            "likelihoods": {"H1": "moderately_confirming"},
                            "interpretation": "Recorded factory response.",
                            "quality_overrides": {},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    gateway = build_model_gateway({"kind": "recorded", "fixture_path": str(fixture_path)})

    assert isinstance(gateway, RecordedModelGateway)
    assert gateway.fixture_name == "recorded_factory"
```

Also import `json`, `Path`, and `RecordedModelGateway` where needed.

- [ ] **Step 2: Write failing config and public API tests**

Add to `tests/test_public_api_and_config.py`:

```python
def test_experiment_config_from_mapping_parses_recorded_gateway(tmp_path: Path):
    dataset_path = tmp_path / "dataset.json"
    report_path = tmp_path / "report.json"
    fixture_path = tmp_path / "recorded.json"
    dataset_path.write_text('{"dataset_name":"empty","samples":[]}', encoding="utf-8")
    fixture_path.write_text('{"fixture_name":"recorded","responses":[]}', encoding="utf-8")

    config = experiment_config_from_mapping(
        {
            "dataset_path": "dataset.json",
            "report_path": "report.json",
            "model_gateway": {
                "kind": "recorded",
                "fixture_path": "recorded.json",
            },
        },
        base_dir=tmp_path,
    )

    assert config.model_gateway.kind == "recorded"
    assert config.model_gateway.fixture_path == tmp_path / "recorded.json"


def test_public_api_exports_recorded_model_gateway():
    from bayesprobe import RecordedModelGateway

    assert RecordedModelGateway.adapter_kind == "recorded"
```

- [ ] **Step 3: Write failing artifact snapshot test**

Add to `tests/test_experiment_artifacts.py`:

```python
def test_artifact_snapshot_includes_recorded_gateway_fixture_path(tmp_path: Path):
    from bayesprobe.experiment_runner import ExperimentRunConfig

    report_path = tmp_path / "report.json"
    report_path.write_text(
        '{"dataset_name":"toy","metadata":{},"sample_count":0,"final_accuracy":0.0,"update_direction_accuracy":null,"results":[]}',
        encoding="utf-8",
    )
    config = ExperimentRunConfig(
        dataset_path=FIXTURE_PATH,
        report_path=report_path,
        model_gateway={
            "kind": "recorded",
            "fixture_path": "fixtures/providers/deepseek_chat_evidence_v0_1.json",
        },
    )

    bundle = write_experiment_artifact_bundle(
        artifact_dir=tmp_path / "artifacts",
        config=config,
        dataset=BenchmarkDataset(dataset_name="toy", samples=[]),
        report_path=report_path,
        ledger_path=None,
        sample_count=0,
    )

    config_snapshot = json.loads(bundle.config_snapshot_path.read_text(encoding="utf-8"))
    assert config_snapshot["model_gateway"] == {
        "kind": "recorded",
        "fixture_path": "fixtures/providers/deepseek_chat_evidence_v0_1.json",
    }
```

- [ ] **Step 4: Run RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tests/test_model_gateway.py::test_build_model_gateway_creates_recorded_gateway \
  tests/test_public_api_and_config.py::test_experiment_config_from_mapping_parses_recorded_gateway \
  tests/test_public_api_and_config.py::test_public_api_exports_recorded_model_gateway \
  tests/test_experiment_artifacts.py::test_artifact_snapshot_includes_recorded_gateway_fixture_path \
  -q -p no:cacheprovider
```

Expected: failures because `fixture_path` and public export are not wired.

- [ ] **Step 5: Implement config/factory/artifact wiring**

In `bayesprobe/model_gateway.py`:

```python
from pathlib import Path
```

Extend `ModelGatewayConfig`:

```python
fixture_path: str | Path | None = None
```

Add factory branch:

```python
if gateway_config.kind == "recorded":
    if gateway_config.fixture_path is None:
        raise ValueError("recorded model gateway requires fixture_path")
    from bayesprobe.recorded_gateway import RecordedModelGateway

    return RecordedModelGateway.from_json(gateway_config.fixture_path)
```

Extend `_model_gateway_config_from_input(...)`:

```python
fixture_path = config.get("fixture_path")
if fixture_path is not None and not isinstance(fixture_path, (str, Path)):
    raise ValueError("recorded model gateway fixture_path must be a path")
```

Pass `fixture_path=fixture_path` into `ModelGatewayConfig(...)`.

In `bayesprobe/config.py`, parse recorded kind:

```python
fixture_path = value.get("fixture_path")
if kind == "recorded":
    if fixture_path is None:
        raise ValueError("recorded model gateway requires fixture_path")
    if not isinstance(fixture_path, str):
        raise ValueError("recorded model gateway fixture_path must be a string")
    fixture_path = _resolve_path(fixture_path, base_dir=base_dir)
```

Pass `fixture_path=fixture_path`.

In `bayesprobe/experiment_artifacts.py`, include:

```python
"fixture_path": str(Path(config.fixture_path)) if config.fixture_path is not None else None,
```

for both dataclass and mapping config snapshots.

In `bayesprobe/__init__.py`, import and export `RecordedModelGateway`.

- [ ] **Step 6: Run GREEN**

Run the focused command from Step 4.

Expected: all focused tests pass.

- [ ] **Step 7: Commit**

```bash
git add bayesprobe/model_gateway.py bayesprobe/config.py bayesprobe/experiment_artifacts.py bayesprobe/__init__.py tests/test_model_gateway.py tests/test_public_api_and_config.py tests/test_experiment_artifacts.py
git commit -m "feat: wire recorded model gateway config"
```

---

### Task 3: Benchmark Belief-Quality Metrics

**Files:**
- Modify: `bayesprobe/benchmark.py`
- Test: `tests/test_benchmark_harness.py`
- Test: `tests/test_benchmark_io.py`

**Interfaces:**
- Consumes: `BenchmarkSampleResult`, `CycleResult`, `ControllerResult`, `BeliefState`, `EvidenceEvent`, and `BeliefUpdate`.
- Produces new `BenchmarkSampleResult` fields:
  - `discarded_evidence_count: int`
  - `schema_violation_count: int`
  - `dominant_hypothesis_margin: float`
  - `belief_revision_efficiency: float`

- [ ] **Step 1: Write failing benchmark metric tests**

Add to `tests/test_benchmark_harness.py`:

```python
def test_benchmark_harness_reports_belief_quality_metrics():
    sample = BenchmarkSample(
        sample_id="quality_active",
        question_or_claim="Does active-only quality metric work?",
        signal_shape=BenchmarkSignalShape.ACTIVE_ONLY,
        gold_best_hypothesis="H1",
        gold_update_directions={"H1": "strengthened"},
    )

    result = BenchmarkHarness().run_sample(sample)

    assert result.discarded_evidence_count == 0
    assert result.schema_violation_count == 0
    assert result.dominant_hypothesis_margin > 0
    assert result.belief_revision_efficiency == 1.0


def test_benchmark_harness_counts_schema_violations_as_discarded_evidence():
    gateway = ScriptedModelGateway(responses={"judge_evidence": {"likelihoods": {}}})
    sample = BenchmarkSample(
        sample_id="quality_schema_violation",
        question_or_claim="Does schema violation quality metric work?",
        signal_shape=BenchmarkSignalShape.PASSIVE_ONLY,
        gold_best_hypothesis="H1",
        passive_signals=[
            BenchmarkSignal(
                signal_id="S_quality_schema",
                source_type="benchmark_stream",
                source="fixture",
                raw_content="No valid schema payload.",
                target_hypotheses=["H1", "H2"],
            )
        ],
    )

    result = BenchmarkHarness(model_gateway=gateway).run_sample(sample)

    assert result.discarded_evidence_count == 1
    assert result.schema_violation_count == 1
    assert result.belief_revision_efficiency == 0.0
```

Add to `tests/test_benchmark_io.py`:

```python
def test_write_benchmark_report_includes_belief_quality_metrics(tmp_path: Path):
    dataset_path = tmp_path / "quality-report-suite.json"
    write_json(dataset_path, [active_sample_payload("quality_report_active")])
    dataset = load_benchmark_dataset(dataset_path)
    suite_result = BenchmarkHarness().run_suite(dataset.samples)
    report_path = tmp_path / "report.json"

    write_benchmark_report(report_path, suite_result, dataset_name="quality_report")

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    result = payload["results"][0]
    assert result["discarded_evidence_count"] == 0
    assert result["schema_violation_count"] == 0
    assert result["dominant_hypothesis_margin"] > 0
    assert result["belief_revision_efficiency"] == 1.0
```

- [ ] **Step 2: Run RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tests/test_benchmark_harness.py::test_benchmark_harness_reports_belief_quality_metrics \
  tests/test_benchmark_harness.py::test_benchmark_harness_counts_schema_violations_as_discarded_evidence \
  tests/test_benchmark_io.py::test_write_benchmark_report_includes_belief_quality_metrics \
  -q -p no:cacheprovider
```

Expected: attribute/key failures for missing metrics.

- [ ] **Step 3: Implement metric fields and helpers**

In `BenchmarkSampleResult`, add:

```python
discarded_evidence_count: int
schema_violation_count: int
dominant_hypothesis_margin: float
belief_revision_efficiency: float
```

Add helpers:

```python
def _discarded_evidence_count(evidence_events: list[EvidenceEvent]) -> int:
    return sum(1 for event in evidence_events if event.discard_reason is not None)


def _schema_violation_count(evidence_events: list[EvidenceEvent]) -> int:
    return sum(
        1
        for event in evidence_events
        if isinstance(event.discard_reason, str)
        and event.discard_reason.startswith("schema_violation:")
    )


def _dominant_hypothesis_margin(belief_state: BeliefState) -> float:
    posteriors = sorted(
        (hypothesis.posterior for hypothesis in belief_state.hypotheses),
        reverse=True,
    )
    if not posteriors:
        return 0.0
    if len(posteriors) == 1:
        return posteriors[0]
    return posteriors[0] - posteriors[1]


def _belief_revision_efficiency(
    *,
    belief_updates: list[BeliefUpdate],
    evidence_events: list[EvidenceEvent],
) -> float:
    if not evidence_events:
        return 0.0
    return len(belief_updates) / len(evidence_events)
```

Pass these fields from `_sample_result_from_question_run(...)` and
`_sample_result_from_controller_result(...)` using each result's evidence events,
belief updates, and final belief state.

- [ ] **Step 4: Run GREEN**

Run the focused command from Step 2.

Expected: all focused tests pass.

- [ ] **Step 5: Commit**

```bash
git add bayesprobe/benchmark.py tests/test_benchmark_harness.py tests/test_benchmark_io.py
git commit -m "feat: add benchmark belief quality metrics"
```

---

### Task 4: v0.2 Benchmark Dataset and Recorded Provider Fixture

**Files:**
- Create: `fixtures/providers/deepseek_chat_evidence_v0_1.json`
- Create: `fixtures/benchmarks/bayesprobe_v0_2_methodology.json`
- Test: `tests/test_benchmark_io.py`
- Test: `tests/test_experiment_runner.py`

**Interfaces:**
- Consumes: `load_benchmark_dataset(...)`, `BenchmarkHarness`, `run_benchmark_experiment(...)`, `RecordedModelGateway`.
- Produces:
  - a richer methodology dataset with at least 8 samples;
  - an offline recorded provider fixture with no secrets;
  - an experiment smoke proving the v0.2 dataset can run offline.

- [ ] **Step 1: Write failing dataset coverage test**

Add to `tests/test_benchmark_io.py`:

```python
def test_v0_2_methodology_fixture_covers_required_scenarios():
    dataset = load_benchmark_dataset("fixtures/benchmarks/bayesprobe_v0_2_methodology.json")

    assert dataset.dataset_name == "bayesprobe_v0_2_methodology"
    assert len(dataset.samples) >= 8
    assert {sample.signal_shape for sample in dataset.samples} == {
        BenchmarkSignalShape.ACTIVE_ONLY,
        BenchmarkSignalShape.PASSIVE_ONLY,
        BenchmarkSignalShape.ACTIVE_PLUS_PASSIVE,
    }
    source_types = {
        signal.source_type
        for sample in dataset.samples
        for signal in sample.passive_signals
    }
    assert {"system_log", "agent_projection", "noisy_stream"} <= source_types
```

- [ ] **Step 2: Write failing recorded experiment smoke test**

Add to `tests/test_experiment_runner.py`:

```python
def test_experiment_runner_runs_v0_2_dataset_with_recorded_gateway(tmp_path: Path):
    config = ExperimentRunConfig(
        dataset_path=Path("fixtures/benchmarks/bayesprobe_v0_2_methodology.json"),
        report_path=tmp_path / "report.json",
        ledger_path=tmp_path / "ledger.jsonl",
        model_gateway={
            "kind": "recorded",
            "fixture_path": "fixtures/providers/deepseek_chat_evidence_v0_1.json",
        },
        max_cycles=1,
        max_probes_per_cycle=1,
    )

    result = run_benchmark_experiment(config)

    assert result.suite_result.sample_count >= 8
    assert result.report_path.exists()
    assert result.ledger_path.exists()
    assert result.suite_result.final_accuracy >= 0.5
    assert result.suite_result.update_direction_accuracy is not None
```

- [ ] **Step 3: Run RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tests/test_benchmark_io.py::test_v0_2_methodology_fixture_covers_required_scenarios \
  tests/test_experiment_runner.py::test_experiment_runner_runs_v0_2_dataset_with_recorded_gateway \
  -q -p no:cacheprovider
```

Expected: file-not-found failures for new fixtures.

- [ ] **Step 4: Create recorded provider fixture**

Create `fixtures/providers/deepseek_chat_evidence_v0_1.json` with entries for
the passive signal IDs in the v0.2 dataset and a final task-only fallback entry
for active deterministic probe signals. Use valid `EvidenceJudgment` objects
only. Do not include `api_key`, `Authorization`, `token`, or raw request headers.

Minimum fixture structure:

```json
{
  "fixture_name": "deepseek_chat_evidence_v0_1",
  "metadata": {
    "provider_kind": "openai_chat_completions",
    "model": "deepseek-v4-flash",
    "recorded_at": "2026-07-10",
    "notes": "Offline replay fixture. No credentials or raw headers are stored."
  },
  "responses": [
    {
      "match": {
        "task": "judge_evidence",
        "signal_id": "S_v02_expert_constant_volume"
      },
      "response": {
        "evidence_type": "supporting",
        "likelihoods": {
          "H1": "moderately_confirming",
          "H2": "moderately_disconfirming"
        },
        "interpretation": "At constant volume, adding inert gas does not change reacting-gas partial pressures, supporting the no-shift hypothesis.",
        "quality_overrides": {
          "reliability": 0.85,
          "relevance": 0.9,
          "specificity": 0.85
        }
      }
    }
  ]
}
```

- [ ] **Step 5: Create v0.2 methodology dataset**

Create `fixtures/benchmarks/bayesprobe_v0_2_methodology.json` with at least 8
samples. Use sample IDs and signal IDs that are readable and stable:

- `v02_active_basic_support`
- `v02_passive_expert_constant_volume`
- `v02_mixed_conflict_resolution`
- `v02_passive_noisy_stream`
- `v02_passive_system_log`
- `v02_passive_agent_projection`
- `v02_mixed_schema_repair`
- `v02_passive_ambiguous_boundary`

Each passive or mixed sample must include `passive_signals`; active-only samples
must not include passive signals. Include `gold_update_directions` for H1/H2
where the expected update is clear.

- [ ] **Step 6: Run GREEN**

Run the focused command from Step 3.

Expected: both tests pass.

- [ ] **Step 7: Commit**

```bash
git add fixtures/providers/deepseek_chat_evidence_v0_1.json fixtures/benchmarks/bayesprobe_v0_2_methodology.json tests/test_benchmark_io.py tests/test_experiment_runner.py
git commit -m "test: add v0.2 benchmark fixtures"
```

---

### Task 5: Documentation and Full Verification

**Files:**
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/superpowers/specs/2026-07-10-benchmark-expansion-provider-fixtures-design.md`
- Test: full repository

**Interfaces:**
- Consumes: completed Tasks 1-4.
- Produces: documented implemented status and final verification evidence.

- [ ] **Step 1: Update docs**

In `docs/ARCHITECTURE.md`:

- update benchmark harness row to mention v0.2 methodology fixture;
- update Model gateway row to mention recorded provider fixture adapter;
- update progress estimate if the implementation materially changes the status;
- keep provider registry/observability as future work.

In `docs/superpowers/specs/2026-07-10-benchmark-expansion-provider-fixtures-design.md`, change:

```markdown
Status: Proposed for implementation
```

to:

```markdown
Status: Implemented
```

- [ ] **Step 2: Run full verification**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
git diff --check
rg -n "sk-|Authorization|api_key|token" fixtures/providers fixtures/benchmarks
```

Expected:

- pytest passes;
- `git diff --check` prints nothing;
- secret scan prints no API keys and no Authorization header in fixtures.

- [ ] **Step 3: Commit docs**

```bash
git add docs/ARCHITECTURE.md docs/superpowers/specs/2026-07-10-benchmark-expansion-provider-fixtures-design.md
git commit -m "docs: mark benchmark fixtures implemented"
```

- [ ] **Step 4: Final push**

```bash
git status --short
git log --oneline -5
git push origin main
```

Expected:

- worktree clean before push;
- push succeeds.
