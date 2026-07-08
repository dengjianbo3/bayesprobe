# Evidence Judgment Repair / Retry Policy v0.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an opt-in repair/retry policy for malformed structured evidence judgments while preserving belief-neutral schema failure behavior by default.

**Architecture:** Introduce `EvidenceJudgmentRepairPolicy` beside the existing `ModelGateway` contract, then thread it through `EvidenceIntegrationGate`, `BayesProbeCore`, `BenchmarkHarness`, `ExperimentRunConfig`, JSON config parsing, and package exports. The Evidence Integration Gate remains the only place where model judgment becomes evidence: it validates initial judgment, optionally asks the same `ModelGateway` to repair malformed structure, validates the repaired payload, and falls back to the existing discarded neutral schema-violation event when repair fails.

**Tech Stack:** Python 3.11+, dataclasses, `collections.abc.Mapping`, existing Pydantic schemas, pytest, JSONL ledger fixtures.

## Global Constraints

- No live provider adapter.
- No network calls.
- No prompt template system.
- No automatic retry for transport errors or rate limits.
- No broad exception swallowing.
- No changes to posterior update math.
- No changes to projection decomposition.
- No changes to probe planning or probe execution.
- No manual human review queue.
- No hidden default behavior change.
- Default repair behavior must be `max_attempts=0`.
- Repair attempts must use the existing `ModelGateway.complete_structured(...)` seam.
- Valid repaired judgment becomes ordinary evidence.
- Invalid repaired judgment falls back to discarded neutral schema-violation evidence.
- Discarded schema-violation evidence still produces no belief updates.

---

## File Structure

- Modify `bayesprobe/model_gateway.py`: add `EvidenceJudgmentRepairPolicy`, mapping conversion helper, validation, and exports.
- Modify `bayesprobe/evidence.py`: add repair policy dependency, direct judgment repair loop, repair request shape, and repair-failure schema violation text.
- Modify `bayesprobe/core.py`: accept and pass repair policy to `EvidenceIntegrationGate`.
- Modify `bayesprobe/benchmark.py`: accept and pass repair policy when creating a core.
- Modify `bayesprobe/experiment_runner.py`: accept mapping/dataclass repair policy in `ExperimentRunConfig` and pass a normalized policy into `BenchmarkHarness`.
- Modify `bayesprobe/config.py`: parse optional JSON `judgment_repair_policy`.
- Modify `bayesprobe/__init__.py`: export `EvidenceJudgmentRepairPolicy`.
- Modify `tests/test_model_gateway.py`: policy validation and mapping conversion tests.
- Modify `tests/test_core_cycles.py`: evidence gate repair/default/failure/core propagation tests.
- Modify `tests/test_benchmark_harness.py`: benchmark repair policy propagation test.
- Modify `tests/test_experiment_runner.py`: experiment runner repair policy propagation test.
- Modify `tests/test_public_api_and_config.py`: SDK export and JSON config parsing/validation tests.

### Task 1: Repair Policy Type And SDK Export

**Files:**
- Modify: `tests/test_model_gateway.py`
- Modify: `tests/test_public_api_and_config.py`
- Modify: `bayesprobe/model_gateway.py`
- Modify: `bayesprobe/__init__.py`

**Interfaces:**
- Consumes existing:
  - `ModelGatewayValidationError`
  - `ModelGatewayConfig`
- Produces:
  - `EvidenceJudgmentRepairPolicy(max_attempts: int = 0, repair_task: str = "repair_evidence_judgment")`
  - `EvidenceJudgmentRepairPolicy.from_config(config: EvidenceJudgmentRepairPolicy | Mapping[str, Any] | None = None) -> EvidenceJudgmentRepairPolicy`

- [ ] **Step 1: Write failing model-gateway policy tests**

Update `tests/test_model_gateway.py` imports:

```python
from bayesprobe.model_gateway import (
    DeterministicModelGateway,
    EvidenceJudgmentRepairPolicy,
    ModelGatewayConfig,
    ModelGatewayValidationError,
    ScriptedModelGateway,
    StructuredModelRequest,
    build_model_gateway,
    evidence_judgment_from_mapping,
)
```

Add these tests after the factory tests:

```python
def test_evidence_judgment_repair_policy_defaults_to_disabled():
    policy = EvidenceJudgmentRepairPolicy()

    assert policy.max_attempts == 0
    assert policy.repair_task == "repair_evidence_judgment"


def test_evidence_judgment_repair_policy_from_config_accepts_mapping():
    policy = EvidenceJudgmentRepairPolicy.from_config(
        {"max_attempts": 2, "repair_task": "repair_evidence_judgment"}
    )

    assert policy.max_attempts == 2
    assert policy.repair_task == "repair_evidence_judgment"


def test_evidence_judgment_repair_policy_from_config_accepts_existing_policy():
    existing = EvidenceJudgmentRepairPolicy(max_attempts=1)

    assert EvidenceJudgmentRepairPolicy.from_config(existing) is existing
```

Add validation tests:

```python
@pytest.mark.parametrize(
    ("config", "expected_message"),
    [
        ([], "judgment repair policy config must be an object"),
        ({"max_attempts": "1"}, "judgment repair max_attempts must be an integer"),
        ({"max_attempts": -1}, "judgment repair max_attempts must be non-negative"),
        ({"repair_task": 1}, "judgment repair task must be a string"),
        ({"repair_task": ""}, "judgment repair task must not be empty"),
        ({"repair_task": "   "}, "judgment repair task must not be empty"),
    ],
)
def test_evidence_judgment_repair_policy_rejects_invalid_config(config, expected_message):
    with pytest.raises(ValueError, match=expected_message):
        EvidenceJudgmentRepairPolicy.from_config(config)
```

- [ ] **Step 2: Write failing public SDK export test**

Update `tests/test_public_api_and_config.py` imports:

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

Update `expected_names`:

```python
expected_names = {
    "BenchmarkDataset",
    "BenchmarkHarness",
    "BenchmarkSample",
    "BenchmarkSampleResult",
    "BenchmarkSignal",
    "BenchmarkSignalShape",
    "BenchmarkSuiteResult",
    "DeterministicModelGateway",
    "EvidenceJudgment",
    "EvidenceJudgmentRepairPolicy",
    "ExperimentRunConfig",
    "ExperimentRunResult",
    "ModelGateway",
    "ModelGatewayConfig",
    "ModelGatewayValidationError",
    "ScriptedModelGateway",
    "StructuredModelRequest",
    "build_model_gateway",
    "evidence_judgment_from_mapping",
    "load_benchmark_dataset",
    "load_experiment_config",
    "run_benchmark_experiment",
    "write_benchmark_report",
}
```

Add the assertion:

```python
assert EvidenceJudgmentRepairPolicy is not None
```

- [ ] **Step 3: Run tests to verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_model_gateway.py::test_evidence_judgment_repair_policy_defaults_to_disabled tests/test_public_api_and_config.py::test_public_sdk_exports_supported_names -q -p no:cacheprovider
```

Expected: failure because `EvidenceJudgmentRepairPolicy` is not defined or not exported.

- [ ] **Step 4: Implement repair policy type**

In `bayesprobe/model_gateway.py`, keep the existing imports and add no new module dependency because `Mapping` and `Any` are already imported.

Add after `ModelGatewayConfig`:

```python
@dataclass(frozen=True)
class EvidenceJudgmentRepairPolicy:
    max_attempts: int = 0
    repair_task: str = "repair_evidence_judgment"

    def __post_init__(self) -> None:
        if type(self.max_attempts) is not int:
            raise ValueError("judgment repair max_attempts must be an integer")
        if self.max_attempts < 0:
            raise ValueError("judgment repair max_attempts must be non-negative")
        if not isinstance(self.repair_task, str):
            raise ValueError("judgment repair task must be a string")
        if not self.repair_task.strip():
            raise ValueError("judgment repair task must not be empty")

    @classmethod
    def from_config(
        cls,
        config: "EvidenceJudgmentRepairPolicy | Mapping[str, Any] | None" = None,
    ) -> "EvidenceJudgmentRepairPolicy":
        if config is None:
            return cls()
        if isinstance(config, cls):
            return config
        if not isinstance(config, Mapping):
            raise ValueError("judgment repair policy config must be an object")
        max_attempts = config.get("max_attempts", 0)
        repair_task = config.get("repair_task", "repair_evidence_judgment")
        return cls(max_attempts=max_attempts, repair_task=repair_task)
```

Update `__all__` in `bayesprobe/model_gateway.py`:

```python
__all__ = [
    "DeterministicModelGateway",
    "EvidenceJudgment",
    "EvidenceJudgmentRepairPolicy",
    "ModelGateway",
    "ModelGatewayConfig",
    "ModelGatewayValidationError",
    "ScriptedModelGateway",
    "StructuredModelRequest",
    "build_model_gateway",
    "evidence_judgment_from_mapping",
]
```

- [ ] **Step 5: Export from package root**

In `bayesprobe/__init__.py`, update the model gateway import:

```python
from bayesprobe.model_gateway import (
    DeterministicModelGateway,
    EvidenceJudgment,
    EvidenceJudgmentRepairPolicy,
    ModelGateway,
    ModelGatewayConfig,
    ModelGatewayValidationError,
    ScriptedModelGateway,
    StructuredModelRequest,
    build_model_gateway,
    evidence_judgment_from_mapping,
)
```

Update `__all__`:

```python
"EvidenceJudgmentRepairPolicy",
```

- [ ] **Step 6: Verify GREEN for policy and SDK tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_model_gateway.py tests/test_public_api_and_config.py::test_public_sdk_exports_supported_names -q -p no:cacheprovider
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit**

Run:

```bash
git add bayesprobe/model_gateway.py bayesprobe/__init__.py tests/test_model_gateway.py tests/test_public_api_and_config.py
git commit -m "feat: add evidence judgment repair policy"
```

### Task 2: Evidence Gate Repair Success And Default-Off Behavior

**Files:**
- Modify: `tests/test_core_cycles.py`
- Modify: `bayesprobe/evidence.py`

**Interfaces:**
- Consumes:
  - `EvidenceJudgmentRepairPolicy`
  - `ScriptedModelGateway.requests`
  - `EvidenceIntegrationGate.integrate(...)`
- Produces:
  - `EvidenceIntegrationGate(..., judgment_repair_policy=...)`
  - repair request task `repair_evidence_judgment`
  - normal `EvidenceEvent` from valid repaired judgment

- [ ] **Step 1: Write failing default-off and repair-success tests**

Update imports in `tests/test_core_cycles.py`:

```python
from bayesprobe.model_gateway import EvidenceJudgmentRepairPolicy, ScriptedModelGateway
```

Add these helper functions near `make_belief_state(...)`:

```python
def make_cycle(cycle_id: str = "cycle_repair") -> CycleRecord:
    return CycleRecord(
        cycle_id=cycle_id,
        run_id="run_1",
        cycle_index=1,
        signal_shape=CycleSignalShape.ACTIVE_ONLY,
    )


def make_empty_probe_set(cycle_id: str = "cycle_repair") -> ProbeSet:
    return ProbeSet(
        probe_set_id=f"ps_{cycle_id}",
        cycle_id=cycle_id,
        probes=[],
        selection_reason="Repair policy fixture.",
        may_be_empty=True,
    )


def make_active_signal(cycle_id: str = "pending") -> ExternalSignal:
    return ExternalSignal(
        id="S_repair",
        cycle_id=cycle_id,
        signal_kind=SignalKind.ACTIVE,
        source_type="benchmark_stream",
        source="fixture",
        raw_content="Malformed judgment fixture.",
        initial_target_hypotheses=["H1", "H2"],
    )
```

Add default-off test:

```python
def test_direct_signal_schema_violation_does_not_attempt_repair_by_default():
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
                "interpretation": "This repair should not be called.",
            },
        }
    )
    gate = EvidenceIntegrationGate(model_gateway=gateway)

    result = gate.integrate(
        cycle=make_cycle("cycle_repair_default"),
        belief_state=make_belief_state(cycle_id="cycle_0"),
        probe_set=make_empty_probe_set("cycle_repair_default"),
        signals=[make_active_signal()],
    )

    event = result.evidence_events[0]
    assert [request.task for request in gateway.requests] == ["judge_evidence"]
    assert event.evidence_type == EvidenceType.NEUTRAL
    assert event.discard_reason.startswith("schema_violation:")
```

Add repair-success test:

```python
def test_direct_signal_repair_success_produces_normal_evidence():
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
                "quality_overrides": {"reliability": 0.91},
            },
        }
    )
    gate = EvidenceIntegrationGate(
        model_gateway=gateway,
        judgment_repair_policy=EvidenceJudgmentRepairPolicy(max_attempts=1),
    )

    result = gate.integrate(
        cycle=make_cycle("cycle_repair_success"),
        belief_state=make_belief_state(cycle_id="cycle_0"),
        probe_set=make_empty_probe_set("cycle_repair_success"),
        signals=[make_active_signal()],
    )

    event = result.evidence_events[0]
    repair_input = gateway.requests[1].input
    assert [request.task for request in gateway.requests] == [
        "judge_evidence",
        "repair_evidence_judgment",
    ]
    assert repair_input["original_request"]["task"] == "judge_evidence"
    assert repair_input["original_request"]["input"]["signal_id"] == "S_repair"
    assert repair_input["invalid_payload"]["evidence_type"] == "not_a_type"
    assert repair_input["validation_error"].startswith("invalid evidence_type")
    assert repair_input["attempt_index"] == 1
    assert "boundary_condition" in repair_input["allowed_evidence_types"]
    assert "moderately_confirming" in repair_input["allowed_likelihood_bands"]
    assert repair_input["required_fields"] == [
        "evidence_type",
        "likelihoods",
        "interpretation",
    ]
    assert event.evidence_type == EvidenceType.SUPPORTING
    assert event.likelihoods["H1"] == LikelihoodBand.MODERATELY_CONFIRMING
    assert event.discard_reason is None
    assert event.interpretation == "Repaired supporting judgment."
    assert event.reliability == 0.91
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_core_cycles.py::test_direct_signal_schema_violation_does_not_attempt_repair_by_default tests/test_core_cycles.py::test_direct_signal_repair_success_produces_normal_evidence -q -p no:cacheprovider
```

Expected: failure because `EvidenceIntegrationGate.__init__` does not accept `judgment_repair_policy`.

- [ ] **Step 3: Implement repair policy dependency and request helpers**

Update imports in `bayesprobe/evidence.py`:

```python
from collections.abc import Mapping
from typing import Any
```

Update the model-gateway imports:

```python
from bayesprobe.model_gateway import (
    DeterministicModelGateway,
    EvidenceJudgment,
    EvidenceJudgmentRepairPolicy,
    ModelGateway,
    ModelGatewayValidationError,
    StructuredModelRequest,
    evidence_judgment_from_mapping,
)
```

Update `EvidenceIntegrationGate.__init__`:

```python
class EvidenceIntegrationGate:
    def __init__(
        self,
        *,
        quality_assessor: SignalQualityAssessor | None = None,
        projection_decomposer: ProjectionDecomposer | None = None,
        model_gateway: ModelGateway | None = None,
        judgment_repair_policy: EvidenceJudgmentRepairPolicy | None = None,
    ) -> None:
        self._quality_assessor = quality_assessor or SignalQualityAssessor()
        self._projection_decomposer = projection_decomposer or ProjectionDecomposer()
        self._model_gateway = model_gateway or DeterministicModelGateway()
        self._judgment_repair_policy = judgment_repair_policy or EvidenceJudgmentRepairPolicy()
```

Update `_ensure_helpers(...)`:

```python
def _ensure_helpers(self) -> None:
    if not hasattr(self, "_quality_assessor"):
        self._quality_assessor = SignalQualityAssessor()
    if not hasattr(self, "_projection_decomposer"):
        self._projection_decomposer = ProjectionDecomposer()
    if not hasattr(self, "_model_gateway"):
        self._model_gateway = DeterministicModelGateway()
    if not hasattr(self, "_judgment_repair_policy"):
        self._judgment_repair_policy = EvidenceJudgmentRepairPolicy()
```

Add helper methods inside `EvidenceIntegrationGate` before `_build_direct_evidence_event(...)`:

```python
def _build_judge_evidence_request(
    self,
    *,
    signal: ExternalSignal,
    hypothesis_ids: list[str],
    cycle: CycleRecord,
    probe_set: ProbeSet,
) -> StructuredModelRequest:
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
    )

def _evidence_judgment_with_repair(
    self,
    *,
    request: StructuredModelRequest,
) -> EvidenceJudgment:
    payload = self._model_gateway.complete_structured(request)
    try:
        return evidence_judgment_from_mapping(payload)
    except ModelGatewayValidationError as error:
        if self._judgment_repair_policy.max_attempts == 0:
            raise
        return self._repair_evidence_judgment(
            original_request=request,
            invalid_payload=payload,
            validation_error=error,
        )

def _repair_evidence_judgment(
    self,
    *,
    original_request: StructuredModelRequest,
    invalid_payload: Any,
    validation_error: ModelGatewayValidationError,
) -> EvidenceJudgment:
    latest_invalid_payload = _repair_payload_from(invalid_payload)
    latest_error = validation_error
    max_attempts = self._judgment_repair_policy.max_attempts

    for attempt_index in range(1, max_attempts + 1):
        repair_payload = self._model_gateway.complete_structured(
            StructuredModelRequest(
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
            )
        )
        try:
            return evidence_judgment_from_mapping(repair_payload)
        except ModelGatewayValidationError as error:
            latest_invalid_payload = _repair_payload_from(repair_payload)
            latest_error = error

    raise ModelGatewayValidationError(
        f"repair failed after {max_attempts} attempt(s): {latest_error}"
    )
```

Add module-level helper near `_scoped_cycle_key(...)`:

```python
def _repair_payload_from(payload: Any) -> dict[str, Any]:
    if isinstance(payload, Mapping):
        return dict(payload)
    return {"_raw_payload": payload}
```

- [ ] **Step 4: Route direct evidence through the repair helper**

Replace the `try` block in `_build_direct_evidence_event(...)` with:

```python
request = self._build_judge_evidence_request(
    signal=signal,
    hypothesis_ids=hypothesis_ids,
    cycle=cycle,
    probe_set=probe_set,
)
try:
    judgment = self._evidence_judgment_with_repair(request=request)
except ModelGatewayValidationError as error:
    return self._schema_violation_event(
        index=index,
        signal=signal,
        cycle=cycle,
        hypothesis_ids=hypothesis_ids,
        is_duplicate=is_duplicate,
        error=error,
    )
```

Keep the valid event construction unchanged:

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
)
```

- [ ] **Step 5: Verify GREEN for evidence gate repair tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_core_cycles.py::test_direct_signal_schema_violation_does_not_attempt_repair_by_default tests/test_core_cycles.py::test_direct_signal_repair_success_produces_normal_evidence -q -p no:cacheprovider
```

Expected: both tests pass.

- [ ] **Step 6: Commit**

Run:

```bash
git add bayesprobe/evidence.py tests/test_core_cycles.py
git commit -m "feat: repair malformed evidence judgments"
```

### Task 3: Repair Failure And Adapter Error Semantics

**Files:**
- Modify: `tests/test_core_cycles.py`
- Modify: `bayesprobe/evidence.py`

**Interfaces:**
- Consumes:
  - `EvidenceIntegrationGate(..., judgment_repair_policy=...)`
  - existing schema-violation event helper
- Produces:
  - failed repair discard reason beginning with `schema_violation: repair failed after`
  - non-validation repair adapter errors propagate

- [ ] **Step 1: Write failing repair-failure test**

Add to `tests/test_core_cycles.py`:

```python
def test_direct_signal_invalid_repair_becomes_schema_violation():
    gateway = ScriptedModelGateway(
        responses={
            "judge_evidence": {
                "evidence_type": "not_a_type",
                "likelihoods": {"H1": "neutral", "H2": "neutral"},
                "interpretation": "Invalid evidence type.",
            },
            "repair_evidence_judgment": {
                "evidence_type": "still_not_a_type",
                "likelihoods": {"H1": "neutral", "H2": "neutral"},
                "interpretation": "Still invalid.",
            },
        }
    )
    gate = EvidenceIntegrationGate(
        model_gateway=gateway,
        judgment_repair_policy=EvidenceJudgmentRepairPolicy(max_attempts=1),
    )

    result = gate.integrate(
        cycle=make_cycle("cycle_repair_failure"),
        belief_state=make_belief_state(cycle_id="cycle_0"),
        probe_set=make_empty_probe_set("cycle_repair_failure"),
        signals=[make_active_signal()],
    )

    event = result.evidence_events[0]
    assert [request.task for request in gateway.requests] == [
        "judge_evidence",
        "repair_evidence_judgment",
    ]
    assert event.evidence_type == EvidenceType.NEUTRAL
    assert event.discard_reason.startswith(
        "schema_violation: repair failed after 1 attempt(s): invalid evidence_type"
    )
    assert event.reliability == 0.0
    assert event.independence == 0.0
    assert event.relevance == 0.0
    assert event.novelty == 0.0
    assert event.specificity == 0.0
    assert event.verifiability == 0.0
```

- [ ] **Step 2: Write failing missing repair task test**

Add:

```python
def test_direct_signal_missing_repair_task_raises_when_repair_enabled():
    gateway = ScriptedModelGateway(
        responses={
            "judge_evidence": {
                "evidence_type": "not_a_type",
                "likelihoods": {"H1": "neutral", "H2": "neutral"},
                "interpretation": "Invalid evidence type.",
            }
        }
    )
    gate = EvidenceIntegrationGate(
        model_gateway=gateway,
        judgment_repair_policy=EvidenceJudgmentRepairPolicy(max_attempts=1),
    )

    with pytest.raises(ValueError, match="no scripted response for task: repair_evidence_judgment"):
        gate.integrate(
            cycle=make_cycle("cycle_repair_missing_task"),
            belief_state=make_belief_state(cycle_id="cycle_0"),
            probe_set=make_empty_probe_set("cycle_repair_missing_task"),
            signals=[make_active_signal()],
        )
```

- [ ] **Step 3: Run tests to verify RED or GREEN**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_core_cycles.py::test_direct_signal_invalid_repair_becomes_schema_violation tests/test_core_cycles.py::test_direct_signal_missing_repair_task_raises_when_repair_enabled -q -p no:cacheprovider
```

Expected before implementation: at least one failure if Task 2 did not already implement exact failure text and propagation. Expected after Task 2 implementation matches the snippets: both tests pass.

- [ ] **Step 4: Adjust failure text only if needed**

If `test_direct_signal_invalid_repair_becomes_schema_violation` fails because the discard reason does not include the repair-attempt text, update the final raise in `_repair_evidence_judgment(...)` to exactly:

```python
raise ModelGatewayValidationError(
    f"repair failed after {max_attempts} attempt(s): {latest_error}"
)
```

Do not catch `ValueError` from `ScriptedModelGateway`; it must propagate.

- [ ] **Step 5: Verify GREEN for core cycle repair tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_core_cycles.py -q -p no:cacheprovider
```

Expected: all core cycle tests pass.

- [ ] **Step 6: Commit**

Run:

```bash
git add bayesprobe/evidence.py tests/test_core_cycles.py
git commit -m "test: cover repair failure semantics"
```

### Task 4: Core Propagation And Belief Update

**Files:**
- Modify: `tests/test_core_cycles.py`
- Modify: `bayesprobe/core.py`

**Interfaces:**
- Consumes:
  - `BayesProbeCore(..., model_gateway=...)`
  - `EvidenceJudgmentRepairPolicy`
- Produces:
  - `BayesProbeCore(..., judgment_repair_policy=...)`
  - repaired evidence can produce `BeliefUpdate`

- [ ] **Step 1: Write failing core propagation test**

Add to `tests/test_core_cycles.py`:

```python
def test_core_passes_judgment_repair_policy_to_evidence_gate():
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
                "interpretation": "Core repaired judgment.",
            },
        }
    )
    core = BayesProbeCore(
        model_gateway=gateway,
        judgment_repair_policy=EvidenceJudgmentRepairPolicy(max_attempts=1),
    )
    cycle = make_cycle("cycle_core_repair")

    result = core.integrate_cycle(
        cycle=cycle,
        belief_state=make_belief_state(cycle_id="cycle_0"),
        probe_set=make_empty_probe_set("cycle_core_repair"),
        signals=[make_active_signal()],
    )

    h1 = result.belief_state.hypotheses_by_id()["H1"]
    h2 = result.belief_state.hypotheses_by_id()["H2"]
    assert [request.task for request in gateway.requests] == [
        "judge_evidence",
        "repair_evidence_judgment",
    ]
    assert result.evidence_events[0].discard_reason is None
    assert result.evidence_events[0].evidence_type == EvidenceType.SUPPORTING
    assert len(result.belief_updates) == 2
    assert h1.posterior > 0.5
    assert h2.posterior < 0.5
```

- [ ] **Step 2: Run test to verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_core_cycles.py::test_core_passes_judgment_repair_policy_to_evidence_gate -q -p no:cacheprovider
```

Expected: failure because `BayesProbeCore.__init__` does not accept `judgment_repair_policy`.

- [ ] **Step 3: Implement core propagation**

Update imports in `bayesprobe/core.py`:

```python
from bayesprobe.model_gateway import EvidenceJudgmentRepairPolicy, ModelGateway
```

Update `BayesProbeCore.__init__(...)`:

```python
class BayesProbeCore:
    def __init__(
        self,
        ledger: JsonlLedgerStore | None = None,
        model_gateway: ModelGateway | None = None,
        judgment_repair_policy: EvidenceJudgmentRepairPolicy | None = None,
    ) -> None:
        self._ledger = ledger
        self._model_gateway = model_gateway
        self._judgment_repair_policy = judgment_repair_policy
        self._cycle_allocations: dict[str, int] = {}
        self._evidence_gate = self._create_evidence_integration_gate()
        self._evolution_policy = self._create_hypothesis_evolution_policy()
```

Update `_create_evidence_integration_gate(...)`:

```python
def _create_evidence_integration_gate(self) -> EvidenceIntegrationGate:
    return EvidenceIntegrationGate(
        model_gateway=self._model_gateway,
        judgment_repair_policy=self._judgment_repair_policy,
    )
```

- [ ] **Step 4: Verify GREEN for core propagation**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_core_cycles.py::test_core_passes_judgment_repair_policy_to_evidence_gate -q -p no:cacheprovider
```

Expected: test passes.

- [ ] **Step 5: Run related core tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_core_cycles.py tests/test_controllers.py tests/test_autonomous_runner.py tests/test_synchronized_runner.py -q -p no:cacheprovider
```

Expected: all selected tests pass.

- [ ] **Step 6: Commit**

Run:

```bash
git add bayesprobe/core.py tests/test_core_cycles.py
git commit -m "feat: pass repair policy through core"
```

### Task 5: Benchmark And Experiment Configuration Chain

**Files:**
- Modify: `tests/test_benchmark_harness.py`
- Modify: `tests/test_experiment_runner.py`
- Modify: `tests/test_public_api_and_config.py`
- Modify: `bayesprobe/benchmark.py`
- Modify: `bayesprobe/experiment_runner.py`
- Modify: `bayesprobe/config.py`

**Interfaces:**
- Consumes:
  - `EvidenceJudgmentRepairPolicy`
  - `BenchmarkHarness(..., model_gateway=...)`
  - `ExperimentRunConfig(..., model_gateway=...)`
  - `experiment_config_from_mapping(...)`
- Produces:
  - `BenchmarkHarness(..., judgment_repair_policy=...)`
  - `ExperimentRunConfig(..., judgment_repair_policy=...)`
  - JSON field `judgment_repair_policy`

- [ ] **Step 1: Write failing benchmark propagation test**

Update `tests/test_benchmark_harness.py` import:

```python
from bayesprobe.model_gateway import EvidenceJudgmentRepairPolicy, ScriptedModelGateway
```

Add:

```python
def test_benchmark_harness_passes_judgment_repair_policy_to_created_core(tmp_path: Path):
    ledger = JsonlLedgerStore(tmp_path / "repair-ledger.jsonl")
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
                "interpretation": "Harness repaired judgment.",
            },
        }
    )
    harness = BenchmarkHarness(
        ledger=ledger,
        model_gateway=gateway,
        judgment_repair_policy=EvidenceJudgmentRepairPolicy(max_attempts=1),
    )
    sample = BenchmarkSample(
        sample_id="repair_passive",
        question_or_claim="Can benchmark configure repair policy?",
        signal_shape=BenchmarkSignalShape.PASSIVE_ONLY,
        gold_best_hypothesis="H1",
        passive_signals=[
            BenchmarkSignal(
                signal_id="S_repair_passive",
                source_type="user_feedback",
                source="user",
                raw_content="Malformed judgment fixture.",
                target_hypotheses=["H1", "H2"],
            )
        ],
    )

    result = harness.run_sample(sample)

    evidence_payloads = [
        record["payload"]
        for record in ledger.read_all("evidence_event")
    ]
    assert result.belief_update_count == 2
    assert evidence_payloads[0]["evidence_type"] == "supporting"
    assert evidence_payloads[0]["discard_reason"] is None
    assert [request.task for request in gateway.requests] == [
        "judge_evidence",
        "repair_evidence_judgment",
    ]
```

- [ ] **Step 2: Write failing experiment runner test**

Add to `tests/test_experiment_runner.py`:

```python
def test_run_benchmark_experiment_uses_judgment_repair_policy_config(tmp_path: Path):
    report_path = tmp_path / "toy-report.json"
    ledger_path = tmp_path / "toy-ledger.jsonl"

    result = run_benchmark_experiment(
        ExperimentRunConfig(
            dataset_path=FIXTURE_PATH,
            report_path=report_path,
            ledger_path=ledger_path,
            model_gateway={
                "kind": "scripted",
                "responses": {
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
                        "interpretation": "Experiment repaired judgment.",
                    },
                },
            },
            judgment_repair_policy={"max_attempts": 1},
        )
    )

    evidence_payloads = [
        record["payload"]
        for record in JsonlLedgerStore(ledger_path).read_all("evidence_event")
    ]
    assert result.ledger_path == ledger_path
    assert evidence_payloads[0]["evidence_type"] == "supporting"
    assert evidence_payloads[0]["discard_reason"] is None
```

- [ ] **Step 3: Write failing JSON config parsing tests**

Update `tests/test_public_api_and_config.py` model-gateway config parsing area with:

```python
def test_experiment_config_from_mapping_parses_judgment_repair_policy(tmp_path: Path):
    config = experiment_config_from_mapping(
        {
            "dataset_path": "datasets/toy.json",
            "report_path": "outputs/toy-report.json",
            "judgment_repair_policy": {
                "max_attempts": 1,
                "repair_task": "repair_evidence_judgment",
            },
        },
        base_dir=tmp_path,
    )

    assert isinstance(config.judgment_repair_policy, EvidenceJudgmentRepairPolicy)
    assert config.judgment_repair_policy.max_attempts == 1
    assert config.judgment_repair_policy.repair_task == "repair_evidence_judgment"
```

Extend the invalid config parametrization in `tests/test_public_api_and_config.py`:

```python
(
    "non_object_judgment_repair_policy.json",
    json.dumps(
        {
            "dataset_path": "dataset.json",
            "report_path": "report.json",
            "judgment_repair_policy": [],
        }
    ),
    "experiment config field judgment_repair_policy must be an object",
),
(
    "non_integer_judgment_repair_attempts.json",
    json.dumps(
        {
            "dataset_path": "dataset.json",
            "report_path": "report.json",
            "judgment_repair_policy": {"max_attempts": "1"},
        }
    ),
    "judgment repair max_attempts must be an integer",
),
(
    "negative_judgment_repair_attempts.json",
    json.dumps(
        {
            "dataset_path": "dataset.json",
            "report_path": "report.json",
            "judgment_repair_policy": {"max_attempts": -1},
        }
    ),
    "judgment repair max_attempts must be non-negative",
),
(
    "empty_judgment_repair_task.json",
    json.dumps(
        {
            "dataset_path": "dataset.json",
            "report_path": "report.json",
            "judgment_repair_policy": {"repair_task": ""},
        }
    ),
    "judgment repair task must not be empty",
),
```

- [ ] **Step 4: Run tests to verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_benchmark_harness.py::test_benchmark_harness_passes_judgment_repair_policy_to_created_core tests/test_experiment_runner.py::test_run_benchmark_experiment_uses_judgment_repair_policy_config tests/test_public_api_and_config.py::test_experiment_config_from_mapping_parses_judgment_repair_policy -q -p no:cacheprovider
```

Expected: failure because benchmark, experiment, and config objects do not accept or parse `judgment_repair_policy`.

- [ ] **Step 5: Implement BenchmarkHarness wiring**

Update imports in `bayesprobe/benchmark.py`:

```python
from bayesprobe.model_gateway import EvidenceJudgmentRepairPolicy, ModelGateway
```

Update `BenchmarkHarness.__init__(...)`:

```python
class BenchmarkHarness:
    def __init__(
        self,
        *,
        core: BayesProbeCore | None = None,
        ledger: JsonlLedgerStore | None = None,
        model_gateway: ModelGateway | None = None,
        judgment_repair_policy: EvidenceJudgmentRepairPolicy | None = None,
        max_cycles: int = 1,
        max_probes_per_cycle: int = 1,
    ) -> None:
        self.core = core or BayesProbeCore(
            ledger=ledger,
            model_gateway=model_gateway,
            judgment_repair_policy=judgment_repair_policy,
        )
        self.ledger = self.core.ledger
        self.max_cycles = max_cycles
        self.max_probes_per_cycle = max_probes_per_cycle
```

- [ ] **Step 6: Implement ExperimentRunConfig wiring**

Update imports in `bayesprobe/experiment_runner.py`:

```python
from bayesprobe.model_gateway import (
    EvidenceJudgmentRepairPolicy,
    ModelGatewayConfig,
    build_model_gateway,
)
```

Update `ExperimentRunConfig`:

```python
@dataclass(frozen=True)
class ExperimentRunConfig:
    dataset_path: str | Path
    report_path: str | Path
    ledger_path: str | Path | None = None
    max_cycles: int = 1
    max_probes_per_cycle: int = 1
    model_gateway: ModelGatewayConfig | Mapping[str, Any] | None = None
    judgment_repair_policy: EvidenceJudgmentRepairPolicy | Mapping[str, Any] | None = None
```

Update `run_benchmark_experiment(...)`:

```python
model_gateway = build_model_gateway(config.model_gateway)
judgment_repair_policy = EvidenceJudgmentRepairPolicy.from_config(
    config.judgment_repair_policy
)
harness = BenchmarkHarness(
    ledger=ledger,
    model_gateway=model_gateway,
    judgment_repair_policy=judgment_repair_policy,
    max_cycles=config.max_cycles,
    max_probes_per_cycle=config.max_probes_per_cycle,
)
```

- [ ] **Step 7: Implement JSON config parsing**

Update imports in `bayesprobe/config.py`:

```python
from bayesprobe.model_gateway import EvidenceJudgmentRepairPolicy, ModelGatewayConfig
```

Update `experiment_config_from_mapping(...)`:

```python
return ExperimentRunConfig(
    dataset_path=_required_path(data, "dataset_path", base_dir=base_dir),
    report_path=_required_path(data, "report_path", base_dir=base_dir),
    ledger_path=_optional_path(data, "ledger_path", base_dir=base_dir),
    max_cycles=_optional_int(data, "max_cycles", default=1),
    max_probes_per_cycle=_optional_int(data, "max_probes_per_cycle", default=1),
    model_gateway=_optional_model_gateway_config(data),
    judgment_repair_policy=_optional_judgment_repair_policy(data),
)
```

Add helper:

```python
def _optional_judgment_repair_policy(
    data: Mapping[str, Any],
) -> EvidenceJudgmentRepairPolicy | None:
    if "judgment_repair_policy" not in data or data["judgment_repair_policy"] is None:
        return None
    value = data["judgment_repair_policy"]
    if not isinstance(value, Mapping):
        raise ValueError("experiment config field judgment_repair_policy must be an object")
    return EvidenceJudgmentRepairPolicy.from_config(value)
```

- [ ] **Step 8: Verify GREEN for benchmark/config chain**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_benchmark_harness.py tests/test_experiment_runner.py tests/test_public_api_and_config.py -q -p no:cacheprovider
```

Expected: all selected tests pass.

- [ ] **Step 9: Commit**

Run:

```bash
git add bayesprobe/benchmark.py bayesprobe/experiment_runner.py bayesprobe/config.py tests/test_benchmark_harness.py tests/test_experiment_runner.py tests/test_public_api_and_config.py
git commit -m "feat: configure judgment repair policy"
```

### Task 6: Regression, Docs Consistency, And Push

**Files:**
- Modify only if needed after verification:
  - `docs/ARCHITECTURE.md`
  - `docs/superpowers/specs/2026-07-08-evidence-judgment-repair-retry-policy-v0.1-design.md`

**Interfaces:**
- Consumes all prior tasks.
- Produces a clean pushed branch with the full repair/retry policy implemented and tested.

- [ ] **Step 1: Run full regression**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
```

Expected:

```text
all tests pass
```

The exact test count may increase above 170 after adding repair policy tests.

- [ ] **Step 2: Check formatting and whitespace**

Run:

```bash
git diff --check
```

Expected: no output.

- [ ] **Step 3: Check docs still mention the correct next state**

Run:

```bash
rg -n "repair|EvidenceJudgmentRepairPolicy|judgment_repair_policy|schema_violation" docs/ARCHITECTURE.md docs/superpowers/specs/2026-07-08-evidence-judgment-repair-retry-policy-v0.1-design.md
```

Expected: references show repair policy as implemented or planned consistently. If the architecture still says repair is not implemented, change that sentence to:

```markdown
Current repair support:

- `EvidenceJudgmentRepairPolicy` can opt into model-gateway repair attempts before schema-violation fallback.
```

- [ ] **Step 4: Commit docs consistency update if changed**

If Step 3 changed docs, run:

```bash
git add docs/ARCHITECTURE.md docs/superpowers/specs/2026-07-08-evidence-judgment-repair-retry-policy-v0.1-design.md
git commit -m "docs: mark repair policy implemented"
```

If no docs changed, skip this commit.

- [ ] **Step 5: Review git history and status**

Run:

```bash
git log --oneline -6
git status --short
```

Expected:

- recent commits include the repair policy implementation commits;
- `git status --short` is empty.

- [ ] **Step 6: Push**

Run:

```bash
git push origin main
```

Expected: push succeeds.

## Self-Review Checklist

- Spec coverage: tasks cover policy type, evidence-gate repair flow, failure semantics, core propagation, benchmark propagation, experiment config, JSON parsing, SDK export, regression, and docs consistency.
- Placeholder scan: plan contains no placeholder task bodies.
- Type consistency: `EvidenceJudgmentRepairPolicy`, `judgment_repair_policy`, `repair_evidence_judgment`, `ModelGatewayValidationError`, and `EvidenceIntegrationGate` names are used consistently.
- Scope check: plan is a single coherent provider-readiness slice and does not implement provider adapters, prompt templates, transport retry, or benchmark expansion.
