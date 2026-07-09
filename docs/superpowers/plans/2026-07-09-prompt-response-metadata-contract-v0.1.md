# Prompt / Response Metadata Contract v0.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add minimal prompt/schema/model-call metadata to the `ModelGateway` seam and persist that trace on evidence events without adding provider adapters or changing model payload behavior.

**Architecture:** Keep `ModelGateway.complete_structured(...) -> dict[str, Any]` stable and deepen the existing request object with prompt/schema metadata. Add `ModelInvocationTrace` as a small audit helper, attach its dictionary form to `EvidenceEvent.model_trace`, and generate traces inside `EvidenceIntegrationGate`, where model output already becomes BayesProbe evidence.

**Tech Stack:** Python 3.11+, dataclasses, `collections.abc.Mapping`, existing Pydantic schemas, pytest, JSONL ledger fixtures.

## Global Constraints

- No live provider adapter.
- No network calls.
- No prompt template rendering system.
- No provider token, latency, cost, or rate-limit tracking.
- No transport retry policy.
- No changes to posterior update math.
- No changes to projection decomposition.
- No changes to probe planning or probe execution.
- No new `model_call` ledger record type in this slice.
- No prompt registry or dynamic plugin registry.
- No breaking change for callers that construct `StructuredModelRequest(task=..., input=...)`.
- Existing deterministic and scripted gateway payload behavior is unchanged.
- Direct evidence requests carry prompt/schema metadata.
- Repair requests carry prompt/schema metadata and repair attempt index.
- `EvidenceEvent` can persist model invocation trace data.
- Discarded schema-violation evidence keeps model trace data and remains belief-neutral.
- Public SDK exports `ModelInvocationTrace`.

---

## File Structure

- Modify `bayesprobe/model_gateway.py`: extend `StructuredModelRequest`, add `ModelInvocationTrace`, add adapter-kind helper, and add built-in adapter identities.
- Modify `bayesprobe/__init__.py`: export `ModelInvocationTrace`.
- Modify `bayesprobe/schemas.py`: add `EvidenceEvent.model_trace`.
- Modify `bayesprobe/evidence.py`: set prompt/schema defaults, build invocation traces, and attach traces to direct evidence and schema-violation events.
- Modify `tests/test_model_gateway.py`: request metadata, invocation trace, adapter-kind, and export-adjacent module tests.
- Modify `tests/test_public_api_and_config.py`: public SDK export test for `ModelInvocationTrace`.
- Modify `tests/test_schemas.py`: `EvidenceEvent.model_trace` defaults and JSON round-trip tests.
- Modify `tests/test_core_cycles.py`: evidence-gate trace propagation tests.
- Modify `tests/test_benchmark_harness.py`: ledger visibility test for `model_trace`.
- Modify `docs/ARCHITECTURE.md`: mark prompt/response metadata contract implemented.

### Task 1: Model Gateway Metadata Types And SDK Export

**Files:**
- Modify: `tests/test_model_gateway.py`
- Modify: `tests/test_public_api_and_config.py`
- Modify: `bayesprobe/model_gateway.py`
- Modify: `bayesprobe/__init__.py`

**Interfaces:**
- Consumes existing:
  - `StructuredModelRequest(task: str, input: dict[str, Any])`
  - `DeterministicModelGateway`
  - `ScriptedModelGateway`
- Produces:
  - `StructuredModelRequest(task, input, prompt_id=None, prompt_version=None, schema_name=None, schema_version=None, metadata={})`
  - `ModelInvocationTrace.from_request(request: StructuredModelRequest, *, adapter_kind: str) -> ModelInvocationTrace`
  - `ModelInvocationTrace.to_dict() -> dict[str, Any]`
  - `model_gateway_adapter_kind(gateway: object) -> str`
  - public package export `ModelInvocationTrace`

- [ ] **Step 1: Write failing model gateway metadata tests**

Update imports in `tests/test_model_gateway.py`:

```python
from dataclasses import FrozenInstanceError

import pytest

from bayesprobe.model_gateway import (
    DeterministicModelGateway,
    EvidenceJudgmentRepairPolicy,
    ModelGatewayConfig,
    ModelGatewayValidationError,
    ModelInvocationTrace,
    ScriptedModelGateway,
    StructuredModelRequest,
    build_model_gateway,
    evidence_judgment_from_mapping,
    model_gateway_adapter_kind,
)
```

Add these tests after `make_request(...)`:

```python
def test_structured_model_request_accepts_minimal_call():
    request = StructuredModelRequest(
        task="judge_evidence",
        input={"raw_content": "SUPPORTS: fixture"},
    )

    assert request.task == "judge_evidence"
    assert request.input == {"raw_content": "SUPPORTS: fixture"}
    assert request.prompt_id is None
    assert request.prompt_version is None
    assert request.schema_name is None
    assert request.schema_version is None
    assert request.metadata == {}


def test_structured_model_request_stores_metadata_and_is_frozen():
    request = StructuredModelRequest(
        task="judge_evidence",
        input={"raw_content": "SUPPORTS: fixture"},
        prompt_id="evidence_judgment",
        prompt_version="v0.1",
        schema_name="EvidenceJudgment",
        schema_version="v0.1",
        metadata={"run_id": "run_1"},
    )

    assert request.prompt_id == "evidence_judgment"
    assert request.prompt_version == "v0.1"
    assert request.schema_name == "EvidenceJudgment"
    assert request.schema_version == "v0.1"
    assert request.metadata == {"run_id": "run_1"}
    with pytest.raises(FrozenInstanceError):
        request.task = "other"
```

Add validation tests:

```python
@pytest.mark.parametrize(
    ("kwargs", "expected_message"),
    [
        (
            {"task": 1, "input": {}},
            "structured model request task must be a string",
        ),
        (
            {"task": "", "input": {}},
            "structured model request task must not be empty",
        ),
        (
            {"task": "   ", "input": {}},
            "structured model request task must not be empty",
        ),
        (
            {"task": "judge_evidence", "input": []},
            "structured model request input must be an object",
        ),
        (
            {"task": "judge_evidence", "input": {}, "prompt_id": ""},
            "structured model request prompt_id must not be empty",
        ),
        (
            {"task": "judge_evidence", "input": {}, "prompt_version": " "},
            "structured model request prompt_version must not be empty",
        ),
        (
            {"task": "judge_evidence", "input": {}, "schema_name": 1},
            "structured model request schema_name must be a string",
        ),
        (
            {"task": "judge_evidence", "input": {}, "schema_version": ""},
            "structured model request schema_version must not be empty",
        ),
        (
            {"task": "judge_evidence", "input": {}, "metadata": []},
            "structured model request metadata must be an object",
        ),
    ],
)
def test_structured_model_request_rejects_invalid_metadata(kwargs, expected_message):
    with pytest.raises(ValueError, match=expected_message):
        StructuredModelRequest(**kwargs)
```

Add invocation trace tests:

```python
def test_model_invocation_trace_from_request_copies_prompt_schema_metadata():
    request = StructuredModelRequest(
        task="judge_evidence",
        input={"raw_content": "SUPPORTS: fixture"},
        prompt_id="evidence_judgment",
        prompt_version="v0.1",
        schema_name="EvidenceJudgment",
        schema_version="v0.1",
        metadata={"run_id": "run_1", "repair_attempt_index": 1},
    )

    trace = ModelInvocationTrace.from_request(request, adapter_kind="scripted")

    assert trace.task == "judge_evidence"
    assert trace.adapter_kind == "scripted"
    assert trace.prompt_id == "evidence_judgment"
    assert trace.prompt_version == "v0.1"
    assert trace.schema_name == "EvidenceJudgment"
    assert trace.schema_version == "v0.1"
    assert trace.repair_attempt_index == 1
    assert trace.metadata == {"run_id": "run_1"}
    assert trace.to_dict() == {
        "task": "judge_evidence",
        "adapter_kind": "scripted",
        "prompt_id": "evidence_judgment",
        "prompt_version": "v0.1",
        "schema_name": "EvidenceJudgment",
        "schema_version": "v0.1",
        "repair_attempt_index": 1,
        "metadata": {"run_id": "run_1"},
    }
```

Add invalid trace tests:

```python
@pytest.mark.parametrize(
    "repair_attempt_index",
    [0, -1, "1"],
)
def test_model_invocation_trace_rejects_invalid_repair_attempt_index(repair_attempt_index):
    request = StructuredModelRequest(
        task="repair_evidence_judgment",
        input={},
        metadata={"repair_attempt_index": repair_attempt_index},
    )

    with pytest.raises(ValueError, match="model invocation repair_attempt_index must be a positive integer"):
        ModelInvocationTrace.from_request(request, adapter_kind="scripted")


@pytest.mark.parametrize(
    ("kwargs", "expected_message"),
    [
        ({"task": "", "adapter_kind": "scripted"}, "model invocation task must not be empty"),
        ({"task": "judge_evidence", "adapter_kind": ""}, "model invocation adapter_kind must not be empty"),
        (
            {"task": "judge_evidence", "adapter_kind": "scripted", "prompt_id": ""},
            "model invocation prompt_id must not be empty",
        ),
        (
            {"task": "judge_evidence", "adapter_kind": "scripted", "metadata": []},
            "model invocation metadata must be an object",
        ),
    ],
)
def test_model_invocation_trace_rejects_invalid_fields(kwargs, expected_message):
    with pytest.raises(ValueError, match=expected_message):
        ModelInvocationTrace(**kwargs)
```

Add adapter-kind test:

```python
def test_model_gateway_adapter_kind_uses_stable_adapter_identities():
    class CustomGateway:
        def complete_structured(self, request):
            return {}

    assert DeterministicModelGateway.adapter_kind == "deterministic"
    assert ScriptedModelGateway(responses={}).adapter_kind == "scripted"
    assert model_gateway_adapter_kind(DeterministicModelGateway()) == "deterministic"
    assert model_gateway_adapter_kind(ScriptedModelGateway(responses={})) == "scripted"
    assert model_gateway_adapter_kind(CustomGateway()) == "CustomGateway"
```

- [ ] **Step 2: Write failing public SDK export test**

Update imports in `tests/test_public_api_and_config.py`:

```python
from bayesprobe import (
    BenchmarkDataset,
    BenchmarkHarness,
    BenchmarkSample,
    BenchmarkSampleResult,
    BenchmarkSignal,
    BenchmarkSignalShape,
    BenchmarkSuiteResult,
    DeterministicModelGateway,
    EvidenceJudgment,
    EvidenceJudgmentRepairPolicy,
    ExperimentRunConfig,
    ExperimentRunResult,
    ModelGateway,
    ModelGatewayConfig,
    ModelGatewayValidationError,
    ModelInvocationTrace,
    ScriptedModelGateway,
    StructuredModelRequest,
    build_model_gateway,
    evidence_judgment_from_mapping,
    load_benchmark_dataset,
    load_experiment_config,
    run_benchmark_experiment,
    write_benchmark_report,
)
```

Add `"ModelInvocationTrace"` to `expected_names` and add:

```python
assert ModelInvocationTrace is not None
```

- [ ] **Step 3: Run tests to verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_model_gateway.py::test_structured_model_request_accepts_minimal_call tests/test_public_api_and_config.py::test_public_sdk_exports_supported_names -q -p no:cacheprovider
```

Expected: failure because `ModelInvocationTrace` is not defined or not exported.

- [ ] **Step 4: Implement request metadata and invocation trace**

In `bayesprobe/model_gateway.py`, replace `StructuredModelRequest` with:

```python
@dataclass(frozen=True)
class StructuredModelRequest:
    task: str
    input: dict[str, Any]
    prompt_id: str | None = None
    prompt_version: str | None = None
    schema_name: str | None = None
    schema_version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        task = _required_nonempty_string(
            self.task,
            "task",
            owner="structured model request",
        )
        input_payload = _required_mapping(
            self.input,
            "input",
            owner="structured model request",
        )
        metadata = _required_mapping(
            self.metadata,
            "metadata",
            owner="structured model request",
        )
        object.__setattr__(self, "task", task)
        object.__setattr__(self, "input", input_payload)
        object.__setattr__(
            self,
            "prompt_id",
            _optional_nonempty_string(
                self.prompt_id,
                "prompt_id",
                owner="structured model request",
            ),
        )
        object.__setattr__(
            self,
            "prompt_version",
            _optional_nonempty_string(
                self.prompt_version,
                "prompt_version",
                owner="structured model request",
            ),
        )
        object.__setattr__(
            self,
            "schema_name",
            _optional_nonempty_string(
                self.schema_name,
                "schema_name",
                owner="structured model request",
            ),
        )
        object.__setattr__(
            self,
            "schema_version",
            _optional_nonempty_string(
                self.schema_version,
                "schema_version",
                owner="structured model request",
            ),
        )
        object.__setattr__(self, "metadata", metadata)
```

Add `ModelInvocationTrace` after `StructuredModelRequest`:

```python
@dataclass(frozen=True)
class ModelInvocationTrace:
    task: str
    adapter_kind: str
    prompt_id: str | None = None
    prompt_version: str | None = None
    schema_name: str | None = None
    schema_version: str | None = None
    repair_attempt_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        task = _required_nonempty_string(
            self.task,
            "task",
            owner="model invocation",
        )
        adapter_kind = _required_nonempty_string(
            self.adapter_kind,
            "adapter_kind",
            owner="model invocation",
        )
        metadata = _required_mapping(
            self.metadata,
            "metadata",
            owner="model invocation",
        )
        object.__setattr__(self, "task", task)
        object.__setattr__(self, "adapter_kind", adapter_kind)
        object.__setattr__(
            self,
            "prompt_id",
            _optional_nonempty_string(
                self.prompt_id,
                "prompt_id",
                owner="model invocation",
            ),
        )
        object.__setattr__(
            self,
            "prompt_version",
            _optional_nonempty_string(
                self.prompt_version,
                "prompt_version",
                owner="model invocation",
            ),
        )
        object.__setattr__(
            self,
            "schema_name",
            _optional_nonempty_string(
                self.schema_name,
                "schema_name",
                owner="model invocation",
            ),
        )
        object.__setattr__(
            self,
            "schema_version",
            _optional_nonempty_string(
                self.schema_version,
                "schema_version",
                owner="model invocation",
            ),
        )
        if self.repair_attempt_index is not None:
            if type(self.repair_attempt_index) is not int or self.repair_attempt_index < 1:
                raise ValueError("model invocation repair_attempt_index must be a positive integer")
        object.__setattr__(self, "metadata", metadata)

    @classmethod
    def from_request(
        cls,
        request: StructuredModelRequest,
        *,
        adapter_kind: str,
    ) -> "ModelInvocationTrace":
        metadata = dict(request.metadata)
        repair_attempt_index = metadata.pop("repair_attempt_index", None)
        return cls(
            task=request.task,
            adapter_kind=adapter_kind,
            prompt_id=request.prompt_id,
            prompt_version=request.prompt_version,
            schema_name=request.schema_name,
            schema_version=request.schema_version,
            repair_attempt_index=repair_attempt_index,
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "task": self.task,
            "adapter_kind": self.adapter_kind,
            "metadata": dict(self.metadata),
        }
        if self.prompt_id is not None:
            payload["prompt_id"] = self.prompt_id
        if self.prompt_version is not None:
            payload["prompt_version"] = self.prompt_version
        if self.schema_name is not None:
            payload["schema_name"] = self.schema_name
        if self.schema_version is not None:
            payload["schema_version"] = self.schema_version
        if self.repair_attempt_index is not None:
            payload["repair_attempt_index"] = self.repair_attempt_index
        return payload
```

Add helpers near `_model_gateway_config_from_input(...)`:

```python
def model_gateway_adapter_kind(gateway: object) -> str:
    adapter_kind = getattr(gateway, "adapter_kind", None)
    if isinstance(adapter_kind, str) and adapter_kind.strip():
        return adapter_kind.strip()
    return gateway.__class__.__name__


def _required_nonempty_string(value: Any, field_name: str, *, owner: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{owner} {field_name} must be a string")
    cleaned = value.strip()
    if not cleaned:
        raise ValueError(f"{owner} {field_name} must not be empty")
    return cleaned


def _optional_nonempty_string(
    value: Any,
    field_name: str,
    *,
    owner: str,
) -> str | None:
    if value is None:
        return None
    return _required_nonempty_string(value, field_name, owner=owner)


def _required_mapping(value: Any, field_name: str, *, owner: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{owner} {field_name} must be an object")
    return dict(value)
```

Add adapter kind attributes:

```python
class DeterministicModelGateway:
    adapter_kind = "deterministic"

    def complete_structured(self, request: StructuredModelRequest) -> dict[str, Any]:
        ...
```

```python
class ScriptedModelGateway:
    adapter_kind = "scripted"

    def __init__(self, responses: dict[str, dict[str, Any]]) -> None:
        ...
```

Update `__all__`:

```python
__all__ = [
    "DeterministicModelGateway",
    "EvidenceJudgment",
    "EvidenceJudgmentRepairPolicy",
    "ModelGateway",
    "ModelGatewayConfig",
    "ModelGatewayValidationError",
    "ModelInvocationTrace",
    "ScriptedModelGateway",
    "StructuredModelRequest",
    "build_model_gateway",
    "evidence_judgment_from_mapping",
    "model_gateway_adapter_kind",
]
```

- [ ] **Step 5: Export from package root**

In `bayesprobe/__init__.py`, update imports:

```python
from bayesprobe.model_gateway import (
    DeterministicModelGateway,
    EvidenceJudgment,
    EvidenceJudgmentRepairPolicy,
    ModelGateway,
    ModelGatewayConfig,
    ModelGatewayValidationError,
    ModelInvocationTrace,
    ScriptedModelGateway,
    StructuredModelRequest,
    build_model_gateway,
    evidence_judgment_from_mapping,
)
```

Update `__all__`:

```python
"ModelInvocationTrace",
```

- [ ] **Step 6: Verify GREEN for gateway metadata tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_model_gateway.py tests/test_public_api_and_config.py::test_public_sdk_exports_supported_names -q -p no:cacheprovider
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit**

Run:

```bash
git add bayesprobe/model_gateway.py bayesprobe/__init__.py tests/test_model_gateway.py tests/test_public_api_and_config.py
git commit -m "feat: add model invocation metadata types"
```

### Task 2: EvidenceEvent Model Trace Schema

**Files:**
- Modify: `tests/test_schemas.py`
- Modify: `bayesprobe/schemas.py`

**Interfaces:**
- Consumes:
  - `EvidenceEvent`
  - `EvidenceType`
  - `LikelihoodBand`
- Produces:
  - `EvidenceEvent.model_trace: dict[str, Any] = Field(default_factory=dict)`

- [ ] **Step 1: Write failing schema tests**

Update `tests/test_schemas.py` imports:

```python
from bayesprobe.schemas import (
    BeliefState,
    ChangeMyMindCondition,
    CycleRecord,
    CycleSignalShape,
    EvidenceEvent,
    EvidenceType,
    ExternalSignal,
    Hypothesis,
    HypothesisStatus,
    LikelihoodBand,
    ProbeCandidate,
    ProbeDesign,
    ProbeSet,
    RunRecord,
    RunRegime,
    SignalKind,
)
```

Add:

```python
def test_evidence_event_model_trace_defaults_to_empty_dict():
    event = EvidenceEvent(
        id="E1",
        derived_from_signal="S1",
        target_hypotheses=["H1"],
        evidence_type=EvidenceType.SUPPORTING,
        content="SUPPORTS: evidence.",
        likelihoods={"H1": LikelihoodBand.MODERATELY_CONFIRMING},
    )

    assert event.model_trace == {}


def test_evidence_event_model_trace_round_trips_through_json():
    event = EvidenceEvent(
        id="E1",
        derived_from_signal="S1",
        target_hypotheses=["H1"],
        evidence_type=EvidenceType.SUPPORTING,
        content="SUPPORTS: evidence.",
        likelihoods={"H1": LikelihoodBand.MODERATELY_CONFIRMING},
        model_trace={
            "task": "judge_evidence",
            "adapter_kind": "scripted",
            "prompt_id": "evidence_judgment",
            "prompt_version": "v0.1",
            "schema_name": "EvidenceJudgment",
            "schema_version": "v0.1",
            "metadata": {},
        },
    )

    loaded = EvidenceEvent.model_validate_json(event.model_dump_json())

    assert loaded.model_trace == {
        "task": "judge_evidence",
        "adapter_kind": "scripted",
        "prompt_id": "evidence_judgment",
        "prompt_version": "v0.1",
        "schema_name": "EvidenceJudgment",
        "schema_version": "v0.1",
        "metadata": {},
    }
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_schemas.py::test_evidence_event_model_trace_defaults_to_empty_dict tests/test_schemas.py::test_evidence_event_model_trace_round_trips_through_json -q -p no:cacheprovider
```

Expected: failure because `EvidenceEvent` has no `model_trace`.

- [ ] **Step 3: Implement schema field**

In `bayesprobe/schemas.py`, update `EvidenceEvent`:

```python
class EvidenceEvent(BaseModel):
    id: str
    derived_from_signal: str
    target_hypotheses: list[str]
    evidence_type: EvidenceType
    content: str
    reliability: float = 0.5
    independence: float = 0.5
    relevance: float = 0.5
    novelty: float = 0.5
    specificity: float = 0.5
    verifiability: float = 0.5
    likelihoods: dict[str, LikelihoodBand] = Field(default_factory=dict)
    interpretation: str = ""
    discard_reason: str | None = None
    model_trace: dict[str, Any] = Field(default_factory=dict)
```

- [ ] **Step 4: Verify GREEN for schema tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_schemas.py -q -p no:cacheprovider
```

Expected: all schema tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add bayesprobe/schemas.py tests/test_schemas.py
git commit -m "feat: add model trace to evidence events"
```

### Task 3: Evidence Gate Invocation Trace Propagation

**Files:**
- Modify: `tests/test_core_cycles.py`
- Modify: `bayesprobe/evidence.py`

**Interfaces:**
- Consumes:
  - `ModelInvocationTrace`
  - `model_gateway_adapter_kind(...)`
  - `EvidenceEvent.model_trace`
  - `EvidenceIntegrationGate(..., model_gateway=..., judgment_repair_policy=...)`
- Produces:
  - direct judge requests with prompt/schema defaults
  - repair requests with prompt/schema defaults and `metadata["repair_attempt_index"]`
  - model trace on normal direct evidence events
  - model trace on schema-violation events
  - repair trace on repaired evidence events

- [ ] **Step 1: Write failing valid direct judgment trace test**

Add to `tests/test_core_cycles.py` after the repair tests:

```python
def test_direct_signal_valid_judgment_records_model_trace():
    gateway = ScriptedModelGateway(
        responses={
            "judge_evidence": {
                "evidence_type": "supporting",
                "likelihoods": {
                    "H1": "moderately_confirming",
                    "H2": "moderately_disconfirming",
                },
                "interpretation": "Scripted supporting judgment.",
            }
        }
    )
    gate = EvidenceIntegrationGate(model_gateway=gateway)

    result = gate.integrate(
        cycle=make_cycle("cycle_model_trace_valid"),
        belief_state=make_belief_state(cycle_id="cycle_0"),
        probe_set=make_empty_probe_set("cycle_model_trace_valid"),
        signals=[make_active_signal()],
    )

    request = gateway.requests[0]
    event = result.evidence_events[0]
    assert request.prompt_id == "evidence_judgment"
    assert request.prompt_version == "v0.1"
    assert request.schema_name == "EvidenceJudgment"
    assert request.schema_version == "v0.1"
    assert request.metadata == {}
    assert event.model_trace == {
        "task": "judge_evidence",
        "adapter_kind": "scripted",
        "prompt_id": "evidence_judgment",
        "prompt_version": "v0.1",
        "schema_name": "EvidenceJudgment",
        "schema_version": "v0.1",
        "metadata": {},
    }
```

- [ ] **Step 2: Write failing schema-violation trace test**

Add:

```python
def test_direct_signal_schema_violation_records_judge_model_trace():
    gateway = ScriptedModelGateway(
        responses={
            "judge_evidence": {
                "evidence_type": "not_a_type",
                "likelihoods": {"H1": "neutral", "H2": "neutral"},
                "interpretation": "Invalid evidence type.",
            }
        }
    )
    gate = EvidenceIntegrationGate(model_gateway=gateway)

    result = gate.integrate(
        cycle=make_cycle("cycle_model_trace_violation"),
        belief_state=make_belief_state(cycle_id="cycle_0"),
        probe_set=make_empty_probe_set("cycle_model_trace_violation"),
        signals=[make_active_signal()],
    )

    event = result.evidence_events[0]
    assert event.discard_reason.startswith("schema_violation:")
    assert event.model_trace["task"] == "judge_evidence"
    assert event.model_trace["adapter_kind"] == "scripted"
    assert event.model_trace["prompt_id"] == "evidence_judgment"
    assert event.model_trace["schema_name"] == "EvidenceJudgment"
```

- [ ] **Step 3: Write failing repaired evidence trace test**

Add:

```python
def test_direct_signal_repaired_judgment_records_repair_model_trace():
    gateway = ScriptedModelGateway(
        responses={
            "judge_evidence": {
                "evidence_type": "not_a_type",
                "likelihoods": {"H1": "neutral", "H2": "neutral"},
                "interpretation": "Invalid evidence type.",
            },
            "repair_evidence_judgment": {
                "evidence_type": "supporting",
                "likelihoods": {
                    "H1": "moderately_confirming",
                    "H2": "moderately_disconfirming",
                },
                "interpretation": "Repaired supporting judgment.",
            },
        }
    )
    gate = EvidenceIntegrationGate(
        model_gateway=gateway,
        judgment_repair_policy=EvidenceJudgmentRepairPolicy(max_attempts=1),
    )

    result = gate.integrate(
        cycle=make_cycle("cycle_model_trace_repair"),
        belief_state=make_belief_state(cycle_id="cycle_0"),
        probe_set=make_empty_probe_set("cycle_model_trace_repair"),
        signals=[make_active_signal()],
    )

    repair_request = gateway.requests[1]
    event = result.evidence_events[0]
    assert repair_request.prompt_id == "evidence_judgment_repair"
    assert repair_request.prompt_version == "v0.1"
    assert repair_request.schema_name == "EvidenceJudgment"
    assert repair_request.schema_version == "v0.1"
    assert repair_request.metadata == {"repair_attempt_index": 1}
    assert event.discard_reason is None
    assert event.model_trace == {
        "task": "repair_evidence_judgment",
        "adapter_kind": "scripted",
        "prompt_id": "evidence_judgment_repair",
        "prompt_version": "v0.1",
        "schema_name": "EvidenceJudgment",
        "schema_version": "v0.1",
        "repair_attempt_index": 1,
        "metadata": {},
    }
```

- [ ] **Step 4: Write failing projection no-trace test**

Add:

```python
def test_projection_decomposition_events_keep_empty_model_trace():
    gate = EvidenceIntegrationGate()
    signal = ExternalSignal(
        id="S_projection_trace",
        cycle_id="pending",
        signal_kind=SignalKind.PASSIVE,
        source_type="external_agent_projection",
        source="agent_a",
        raw_content="Agent A believes H2 because Source A refutes the claim.",
    )

    result = gate.integrate(
        cycle=make_cycle("cycle_projection_trace"),
        belief_state=make_belief_state(cycle_id="cycle_0"),
        probe_set=make_empty_probe_set("cycle_projection_trace"),
        signals=[signal],
    )

    assert [event.evidence_type for event in result.evidence_events] == [
        EvidenceType.SENDER_JUDGMENT,
        EvidenceType.SOURCE_CLAIM,
    ]
    assert [event.model_trace for event in result.evidence_events] == [{}, {}]
```

- [ ] **Step 5: Run tests to verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_core_cycles.py::test_direct_signal_valid_judgment_records_model_trace tests/test_core_cycles.py::test_direct_signal_schema_violation_records_judge_model_trace tests/test_core_cycles.py::test_direct_signal_repaired_judgment_records_repair_model_trace tests/test_core_cycles.py::test_projection_decomposition_events_keep_empty_model_trace -q -p no:cacheprovider
```

Expected: failures because direct requests do not set prompt/schema metadata and events do not set `model_trace`.

- [ ] **Step 6: Implement evidence gate trace support**

Update imports in `bayesprobe/evidence.py`:

```python
from bayesprobe.model_gateway import (
    DeterministicModelGateway,
    EvidenceJudgment,
    EvidenceJudgmentRepairPolicy,
    ModelGateway,
    ModelGatewayValidationError,
    ModelInvocationTrace,
    StructuredModelRequest,
    evidence_judgment_from_mapping,
    model_gateway_adapter_kind,
)
```

Add a private failure wrapper near the top of `bayesprobe/evidence.py`, after `SignalQuality`:

```python
class _EvidenceJudgmentFailure(Exception):
    def __init__(
        self,
        *,
        error: ModelGatewayValidationError,
        model_trace: ModelInvocationTrace,
    ) -> None:
        super().__init__(str(error))
        self.error = error
        self.model_trace = model_trace
```

Add a helper method on `EvidenceIntegrationGate`:

```python
def _model_trace_for_request(self, request: StructuredModelRequest) -> ModelInvocationTrace:
    return ModelInvocationTrace.from_request(
        request,
        adapter_kind=model_gateway_adapter_kind(self._model_gateway),
    )
```

Update `_build_judge_evidence_request(...)`:

```python
return StructuredModelRequest(
    task="judge_evidence",
    input={
        "signal_id": signal.id,
        "source_type": signal.source_type,
        "source": signal.source,
        "raw_content": signal.raw_content,
        "target_hypotheses": hypothesis_ids,
        "cycle_id": cycle.cycle_id,
        "probe_ids": [probe.id for probe in probe_set.probes],
    },
    prompt_id="evidence_judgment",
    prompt_version="v0.1",
    schema_name="EvidenceJudgment",
    schema_version="v0.1",
)
```

Replace `_evidence_judgment_with_repair(...)` with:

```python
def _evidence_judgment_with_repair(
    self,
    *,
    request: StructuredModelRequest,
) -> tuple[EvidenceJudgment, ModelInvocationTrace]:
    model_trace = self._model_trace_for_request(request)
    payload = self._model_gateway.complete_structured(request)
    try:
        return evidence_judgment_from_mapping(payload), model_trace
    except ModelGatewayValidationError as error:
        if self._judgment_repair_policy.max_attempts == 0:
            raise _EvidenceJudgmentFailure(error=error, model_trace=model_trace) from error
        return self._repair_evidence_judgment(
            original_request=request,
            invalid_payload=payload,
            validation_error=error,
        )
```

Replace `_repair_evidence_judgment(...)` with:

```python
def _repair_evidence_judgment(
    self,
    *,
    original_request: StructuredModelRequest,
    invalid_payload: Any,
    validation_error: ModelGatewayValidationError,
) -> tuple[EvidenceJudgment, ModelInvocationTrace]:
    latest_invalid_payload = _repair_payload_from(invalid_payload)
    latest_error = validation_error
    latest_trace = self._model_trace_for_request(original_request)
    max_attempts = self._judgment_repair_policy.max_attempts

    for attempt_index in range(1, max_attempts + 1):
        repair_request = StructuredModelRequest(
            task=self._judgment_repair_policy.repair_task,
            input={
                "original_request": {
                    "task": original_request.task,
                    "input": dict(original_request.input),
                },
                "invalid_payload": latest_invalid_payload,
                "validation_error": str(latest_error),
                "attempt_index": attempt_index,
                "allowed_evidence_types": [evidence_type.value for evidence_type in EvidenceType],
                "allowed_likelihood_bands": [band.value for band in LikelihoodBand],
                "required_fields": [
                    "evidence_type",
                    "likelihoods",
                    "interpretation",
                ],
            },
            prompt_id="evidence_judgment_repair",
            prompt_version="v0.1",
            schema_name="EvidenceJudgment",
            schema_version="v0.1",
            metadata={"repair_attempt_index": attempt_index},
        )
        repair_trace = self._model_trace_for_request(repair_request)
        repair_payload = self._model_gateway.complete_structured(repair_request)
        try:
            return evidence_judgment_from_mapping(repair_payload), repair_trace
        except ModelGatewayValidationError as error:
            latest_invalid_payload = _repair_payload_from(repair_payload)
            latest_error = error
            latest_trace = repair_trace

    failure = ModelGatewayValidationError(
        f"repair failed after {max_attempts} attempt(s): {latest_error}"
    )
    raise _EvidenceJudgmentFailure(error=failure, model_trace=latest_trace) from latest_error
```

Update `_build_direct_evidence_event(...)`:

```python
try:
    judgment, model_trace = self._evidence_judgment_with_repair(request=request)
except _EvidenceJudgmentFailure as failure:
    return self._schema_violation_event(
        index=index,
        signal=signal,
        cycle=cycle,
        hypothesis_ids=hypothesis_ids,
        is_duplicate=is_duplicate,
        error=failure.error,
        model_trace=failure.model_trace,
    )
```

Then pass `model_trace` into `_event(...)`:

```python
return self._event(
    event_id=f"{_scoped_cycle_key(cycle.run_id, cycle.cycle_id)}_E{index}",
    signal=signal,
    hypothesis_ids=hypothesis_ids,
    evidence_type=judgment.evidence_type,
    likelihoods=judgment.likelihoods,
    interpretation=judgment.interpretation,
    is_duplicate=is_duplicate,
    quality_overrides=judgment.quality_overrides,
    model_trace=model_trace,
)
```

Update `_schema_violation_event(...)` signature:

```python
def _schema_violation_event(
    self,
    *,
    index: int,
    signal: ExternalSignal,
    cycle: CycleRecord,
    hypothesis_ids: list[str],
    is_duplicate: bool,
    error: ModelGatewayValidationError,
    model_trace: ModelInvocationTrace | None = None,
) -> EvidenceEvent:
```

and pass:

```python
model_trace=model_trace,
```

Update `_event(...)` signature:

```python
model_trace: ModelInvocationTrace | None = None,
```

and include in `EvidenceEvent(...)`:

```python
model_trace=model_trace.to_dict() if model_trace is not None else {},
```

- [ ] **Step 7: Verify GREEN for evidence-gate trace tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_core_cycles.py -q -p no:cacheprovider
```

Expected: all core cycle tests pass.

- [ ] **Step 8: Commit**

Run:

```bash
git add bayesprobe/evidence.py tests/test_core_cycles.py
git commit -m "feat: trace model invocations on evidence events"
```

### Task 4: Ledger And Benchmark Trace Visibility

**Files:**
- Modify: `tests/test_benchmark_harness.py`
- Modify only if the test reveals a serialization gap: `bayesprobe/ledger.py`

**Interfaces:**
- Consumes:
  - `EvidenceEvent.model_trace`
  - `JsonlLedgerStore.append(...)`
  - `BenchmarkHarness(..., model_gateway=...)`
- Produces:
  - JSONL evidence records containing `payload["model_trace"]` for model-judged direct evidence

- [ ] **Step 1: Write failing benchmark ledger trace test**

Add to `tests/test_benchmark_harness.py` near the model gateway ledger tests:

```python
def test_benchmark_harness_records_model_trace_in_evidence_ledger(tmp_path: Path):
    ledger = JsonlLedgerStore(tmp_path / "model-trace-ledger.jsonl")
    gateway = ScriptedModelGateway(
        responses={
            "judge_evidence": {
                "evidence_type": "supporting",
                "likelihoods": {
                    "H1": "moderately_confirming",
                    "H2": "moderately_disconfirming",
                },
                "interpretation": "Harness trace judgment.",
            }
        }
    )
    harness = BenchmarkHarness(ledger=ledger, model_gateway=gateway)
    sample = BenchmarkSample(
        sample_id="model_trace_passive",
        question_or_claim="Can benchmark ledger preserve model trace?",
        signal_shape=BenchmarkSignalShape.PASSIVE_ONLY,
        gold_best_hypothesis="H1",
        passive_signals=[
            BenchmarkSignal(
                signal_id="S_model_trace_passive",
                source_type="user_feedback",
                source="user",
                raw_content="Model trace fixture.",
                target_hypotheses=["H1", "H2"],
            )
        ],
    )

    harness.run_sample(sample)

    evidence_payload = ledger.read_all("evidence_event")[0]["payload"]
    assert evidence_payload["model_trace"] == {
        "task": "judge_evidence",
        "adapter_kind": "scripted",
        "prompt_id": "evidence_judgment",
        "prompt_version": "v0.1",
        "schema_name": "EvidenceJudgment",
        "schema_version": "v0.1",
        "metadata": {},
    }
```

- [ ] **Step 2: Run test to verify RED or GREEN**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_benchmark_harness.py::test_benchmark_harness_records_model_trace_in_evidence_ledger -q -p no:cacheprovider
```

Expected after Tasks 2 and 3: this may pass immediately because `JsonlLedgerStore` serializes Pydantic models with `model_dump(mode="json")`. If it fails, fix only the serialization gap shown by the test.

- [ ] **Step 3: Run benchmark and ledger test group**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_benchmark_harness.py tests/test_benchmark_io.py tests/test_inbox_and_ledger.py -q -p no:cacheprovider
```

Expected: all selected tests pass.

- [ ] **Step 4: Commit**

Run:

```bash
git add tests/test_benchmark_harness.py bayesprobe/ledger.py
git commit -m "test: cover model trace ledger visibility"
```

If `bayesprobe/ledger.py` was not changed, run:

```bash
git add tests/test_benchmark_harness.py
git commit -m "test: cover model trace ledger visibility"
```

### Task 5: Regression, Docs Consistency, And Push Prep

**Files:**
- Modify: `docs/ARCHITECTURE.md`

**Interfaces:**
- Consumes all prior tasks.
- Produces docs that mark the prompt/response metadata contract as implemented.

- [ ] **Step 1: Update architecture status**

In `docs/ARCHITECTURE.md`, change:

```markdown
| Prompt/version metadata | Missing | Needed before serious provider-based experiments. |
```

to:

```markdown
| Prompt/version metadata | Good MVP | StructuredModelRequest metadata and EvidenceEvent model_trace are implemented. |
```

In Phase 2, change:

```markdown
### Phase 2: Provider Adapter and Prompt Metadata
```

to:

```markdown
### Phase 2: Provider Adapter and Prompt Metadata

Status: prompt/response metadata contract implemented as MVP; provider adapter remains future work.
```

Also add this bullet under `ModelGateway` future/current extension wording if not already present:

```markdown
- `ModelInvocationTrace` persists prompt/schema adapter metadata on evidence events.
```

- [ ] **Step 2: Run full regression**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
```

Expected:

```text
all tests pass
```

The exact test count should be higher than 191 after these tasks.

- [ ] **Step 3: Check whitespace**

Run:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 4: Check metadata references**

Run:

```bash
rg -n "ModelInvocationTrace|model_trace|prompt/version metadata|Prompt/version metadata|StructuredModelRequest" docs/ARCHITECTURE.md bayesprobe tests
```

Expected:

- `ModelInvocationTrace` appears in model gateway exports/tests.
- `model_trace` appears in `EvidenceEvent`, evidence-gate tests, and benchmark ledger tests.
- architecture status does not still call prompt/version metadata missing.

- [ ] **Step 5: Commit docs update**

Run:

```bash
git add docs/ARCHITECTURE.md
git commit -m "docs: mark prompt metadata contract implemented"
```

- [ ] **Step 6: Review git status and recent commits**

Run:

```bash
git status --short
git log --oneline -8
```

Expected:

- `git status --short` is empty.
- recent commits include model invocation metadata, evidence event model trace, evidence-gate trace propagation, ledger visibility, and docs status update.

- [ ] **Step 7: Push**

Run:

```bash
git push origin main
```

Expected: push succeeds.

## Self-Review Checklist

- Spec coverage: tasks cover request metadata, invocation trace, adapter kind, public SDK export, EvidenceEvent trace schema, direct evidence trace, schema-violation trace, repair trace, projection no-trace behavior, ledger visibility, docs status, and full regression.
- Placeholder scan: this plan contains no placeholder task bodies.
- Type consistency: `StructuredModelRequest`, `ModelInvocationTrace`, `model_gateway_adapter_kind`, `model_trace`, `prompt_id`, `prompt_version`, `schema_name`, `schema_version`, and `repair_attempt_index` are used consistently.
- Scope check: the plan does not add provider adapters, network calls, prompt registries, transport retries, model-call ledger records, or posterior math changes.
