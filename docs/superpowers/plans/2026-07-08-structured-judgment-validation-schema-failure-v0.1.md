# Structured Judgment Validation / Schema Failure Handling v0.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert malformed structured model judgments into auditable schema-violation evidence events that do not mutate belief state.

**Architecture:** Add a focused `ModelGatewayValidationError` at the model-gateway seam and make `evidence_judgment_from_mapping(...)` raise it for malformed judgment payloads. `EvidenceIntegrationGate` catches only that validation error and emits a discarded neutral evidence event with zero quality scores. `solve_updates(...)` skips discarded evidence so invalid judgments are ledger-visible but belief-neutral.

**Tech Stack:** Python 3.11+, dataclasses, `collections.abc.Mapping`, existing Pydantic schemas, pytest.

## Global Constraints

- No live model provider adapter.
- No network calls.
- No prompt templates.
- No schema-repair retry loop.
- No manual review queue.
- No changes to likelihood-band math.
- No changes to projection decomposition.
- No changes to Hypothesis Evolution Engine.
- No broad error swallowing for unrelated bugs.
- Do not attempt git commits because this workspace is not currently a git repository.

---

## File Structure

- Modify `tests/test_model_gateway.py`: gateway judgment validation error tests.
- Modify `tests/test_core_cycles.py`: evidence-gate schema violation and belief-skip integration tests.
- Modify `tests/test_benchmark_harness.py`: benchmark ledger schema-violation replay test.
- Modify `tests/test_public_api_and_config.py`: public SDK export test for the validation error.
- Modify `bayesprobe/model_gateway.py`: `ModelGatewayValidationError` and focused parsing failures.
- Modify `bayesprobe/evidence.py`: schema-violation evidence event conversion.
- Modify `bayesprobe/belief.py`: skip discarded evidence events.
- Modify `bayesprobe/__init__.py`: export `ModelGatewayValidationError` from the supported SDK facade.

### Task 1: ModelGateway Judgment Validation Tests

**Files:**
- Modify: `tests/test_model_gateway.py`

**Interfaces:**
- Consumes existing:
  - `evidence_judgment_from_mapping(payload: dict[str, Any]) -> EvidenceJudgment`
- Produces expected:
  - `ModelGatewayValidationError`

- [x] **Step 1: Write failing validation-error import and tests**

Update import:

```python
from bayesprobe.model_gateway import (
    DeterministicModelGateway,
    ModelGatewayConfig,
    ModelGatewayValidationError,
    ScriptedModelGateway,
    StructuredModelRequest,
    build_model_gateway,
    evidence_judgment_from_mapping,
)
```

Add tests:

```python
@pytest.mark.parametrize(
    ("payload", "expected_message"),
    [
        ({}, "evidence judgment missing field: evidence_type"),
        ({"evidence_type": "not_a_type"}, "invalid evidence_type"),
        (
            {"evidence_type": "neutral", "likelihoods": []},
            "evidence judgment likelihoods must be an object",
        ),
        (
            {"evidence_type": "neutral", "likelihoods": {"H1": "not_a_band"}},
            "invalid likelihood band for H1",
        ),
        (
            {"evidence_type": "neutral", "quality_overrides": []},
            "evidence judgment quality_overrides must be an object",
        ),
        (
            {"evidence_type": "neutral", "quality_overrides": {"reliability": "high"}},
            "invalid quality override for reliability",
        ),
    ],
)
def test_evidence_judgment_from_mapping_raises_validation_error(payload, expected_message):
    with pytest.raises(ModelGatewayValidationError, match=expected_message):
        evidence_judgment_from_mapping(payload)
```

- [x] **Step 2: Verify RED**

Run:

```bash
python3 -m pytest tests/test_model_gateway.py::test_evidence_judgment_from_mapping_raises_validation_error -q
```

Expected: failure because `ModelGatewayValidationError` does not exist yet.

### Task 2: ModelGateway Judgment Validation Implementation

**Files:**
- Modify: `bayesprobe/model_gateway.py`
- Test: `tests/test_model_gateway.py`

**Interfaces:**
- Produces:
  - `class ModelGatewayValidationError(ValueError)`
  - `evidence_judgment_from_mapping(...)` raises `ModelGatewayValidationError` for schema-like failures.

- [x] **Step 1: Implement focused validation error**

Add after imports:

```python
class ModelGatewayValidationError(ValueError):
    pass
```

Replace `evidence_judgment_from_mapping(...)` with:

```python
def evidence_judgment_from_mapping(payload: dict[str, Any]) -> EvidenceJudgment:
    if not isinstance(payload, Mapping):
        raise ModelGatewayValidationError("evidence judgment payload must be an object")
    if "evidence_type" not in payload:
        raise ModelGatewayValidationError("evidence judgment missing field: evidence_type")

    raw_evidence_type = payload["evidence_type"]
    try:
        evidence_type = EvidenceType(raw_evidence_type)
    except ValueError as error:
        raise ModelGatewayValidationError(f"invalid evidence_type: {raw_evidence_type}") from error

    likelihoods_payload = payload.get("likelihoods", {})
    if not isinstance(likelihoods_payload, Mapping):
        raise ModelGatewayValidationError("evidence judgment likelihoods must be an object")

    likelihoods: dict[str, LikelihoodBand] = {}
    for hypothesis_id, likelihood in likelihoods_payload.items():
        try:
            likelihoods[str(hypothesis_id)] = LikelihoodBand(likelihood)
        except ValueError as error:
            raise ModelGatewayValidationError(
                f"invalid likelihood band for {hypothesis_id}: {likelihood}"
            ) from error

    quality_overrides_payload = payload.get("quality_overrides", {})
    if quality_overrides_payload is None:
        quality_overrides_payload = {}
    if not isinstance(quality_overrides_payload, Mapping):
        raise ModelGatewayValidationError("evidence judgment quality_overrides must be an object")

    quality_overrides: dict[str, float] = {}
    for metric, value in quality_overrides_payload.items():
        try:
            quality_overrides[str(metric)] = float(value)
        except (TypeError, ValueError) as error:
            raise ModelGatewayValidationError(
                f"invalid quality override for {metric}: {value}"
            ) from error

    return EvidenceJudgment(
        evidence_type=evidence_type,
        likelihoods=likelihoods,
        interpretation=str(payload.get("interpretation", "")),
        quality_overrides=quality_overrides,
    )
```

Update `__all__`:

```python
"ModelGatewayValidationError",
```

- [x] **Step 2: Verify GREEN for model gateway tests**

Run:

```bash
python3 -m pytest tests/test_model_gateway.py -q
```

Expected: all model gateway tests pass.

### Task 3: EvidenceGate Schema-Violation Conversion

**Files:**
- Modify: `tests/test_core_cycles.py`
- Modify: `bayesprobe/evidence.py`

**Interfaces:**
- Consumes:
  - `ModelGatewayValidationError`
  - `ScriptedModelGateway`
- Produces:
  - Direct evidence schema failures become discarded `EvidenceEvent`s.

- [x] **Step 1: Write failing evidence-gate schema-violation test**

Add test near `test_direct_signal_judgment_uses_model_gateway`:

```python
def test_direct_signal_schema_violation_becomes_discarded_evidence():
    gateway = ScriptedModelGateway(
        responses={
            "judge_evidence": {
                "likelihoods": {"H1": "neutral", "H2": "neutral"},
                "interpretation": "Missing evidence type.",
            }
        }
    )
    gate = EvidenceIntegrationGate(model_gateway=gateway)
    cycle = CycleRecord(
        cycle_id="cycle_schema_violation",
        run_id="run_1",
        cycle_index=1,
        signal_shape=CycleSignalShape.ACTIVE_ONLY,
    )
    signal = ExternalSignal(
        id="S_schema_violation",
        cycle_id="pending",
        signal_kind=SignalKind.ACTIVE,
        source_type="benchmark_stream",
        source="fixture",
        raw_content="Malformed judgment fixture.",
    )

    result = gate.integrate(
        cycle=cycle,
        belief_state=make_belief_state(cycle_id="cycle_schema_violation"),
        probe_set=ProbeSet(
            probe_set_id="ps_schema_violation",
            cycle_id="cycle_schema_violation",
            probes=[],
            selection_reason="Schema violation evidence test.",
            may_be_empty=True,
        ),
        signals=[signal],
    )

    event = result.evidence_events[0]
    assert event.evidence_type == EvidenceType.NEUTRAL
    assert event.likelihoods == {
        "H1": LikelihoodBand.NEUTRAL,
        "H2": LikelihoodBand.NEUTRAL,
    }
    assert event.discard_reason.startswith("schema_violation:")
    assert "evidence judgment missing field: evidence_type" in event.discard_reason
    assert event.interpretation == "Model gateway judgment failed schema validation."
    assert event.reliability == 0.0
    assert event.independence == 0.0
    assert event.relevance == 0.0
    assert event.novelty == 0.0
    assert event.specificity == 0.0
    assert event.verifiability == 0.0
```

- [x] **Step 2: Verify RED**

Run:

```bash
python3 -m pytest tests/test_core_cycles.py::test_direct_signal_schema_violation_becomes_discarded_evidence -q
```

Expected: failure because `EvidenceIntegrationGate` still lets validation errors propagate.

- [x] **Step 3: Implement schema-violation event conversion**

Update import in `bayesprobe/evidence.py`:

```python
from bayesprobe.model_gateway import (
    DeterministicModelGateway,
    ModelGateway,
    ModelGatewayValidationError,
    StructuredModelRequest,
    evidence_judgment_from_mapping,
)
```

In `_build_direct_evidence_event(...)`, wrap judgment construction:

```python
        try:
            judgment = evidence_judgment_from_mapping(
                self._model_gateway.complete_structured(
                    StructuredModelRequest(
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
                )
            )
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

Add method:

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
    ) -> EvidenceEvent:
        return self._event(
            event_id=f"{_scoped_cycle_key(cycle.run_id, cycle.cycle_id)}_E{index}",
            signal=signal,
            hypothesis_ids=hypothesis_ids,
            evidence_type=EvidenceType.NEUTRAL,
            likelihoods={hypothesis_id: LikelihoodBand.NEUTRAL for hypothesis_id in hypothesis_ids},
            interpretation="Model gateway judgment failed schema validation.",
            is_duplicate=is_duplicate,
            quality_overrides=_ZERO_QUALITY_OVERRIDES,
            discard_reason=f"schema_violation: {error}",
        )
```

Add module constant:

```python
_ZERO_QUALITY_OVERRIDES = {
    "reliability": 0.0,
    "independence": 0.0,
    "relevance": 0.0,
    "novelty": 0.0,
    "specificity": 0.0,
    "verifiability": 0.0,
}
```

Update `_event(...)` signature:

```python
discard_reason: str | None = None,
```

and set:

```python
discard_reason=discard_reason,
```

inside `EvidenceEvent(...)`.

- [x] **Step 4: Verify GREEN**

Run:

```bash
python3 -m pytest tests/test_core_cycles.py::test_direct_signal_schema_violation_becomes_discarded_evidence -q
```

Expected: test passes.

### Task 4: Belief Solver Discard Skip

**Files:**
- Modify: `tests/test_core_cycles.py`
- Modify: `bayesprobe/belief.py`

**Interfaces:**
- Consumes:
  - `EvidenceEvent.discard_reason`
- Produces:
  - `solve_updates(...)` skips discarded evidence.

- [x] **Step 1: Write failing core belief-skip test**

Add test near the schema-violation evidence test:

```python
def test_core_schema_violation_does_not_update_belief_state():
    gateway = ScriptedModelGateway(
        responses={
            "judge_evidence": {
                "likelihoods": {"H1": "moderately_confirming", "H2": "moderately_disconfirming"},
                "interpretation": "Missing evidence type.",
            }
        }
    )
    core = BayesProbeCore(model_gateway=gateway)
    cycle = CycleRecord(
        cycle_id="cycle_schema_violation_core",
        run_id="run_1",
        cycle_index=1,
        signal_shape=CycleSignalShape.PASSIVE_ONLY,
    )

    result = core.integrate_cycle(
        cycle=cycle,
        belief_state=make_belief_state(cycle_id="cycle_schema_violation_core"),
        probe_set=ProbeSet(
            probe_set_id="ps_schema_violation_core",
            cycle_id="cycle_schema_violation_core",
            probes=[],
            selection_reason="Schema violation skip update test.",
            may_be_empty=True,
        ),
        signals=[
            ExternalSignal(
                id="S_schema_violation_core",
                cycle_id="pending",
                signal_kind=SignalKind.PASSIVE,
                source_type="user_feedback",
                source="user",
                raw_content="Malformed judgment fixture.",
            )
        ],
    )

    assert result.evidence_events[0].discard_reason.startswith("schema_violation:")
    assert result.belief_updates == []
    assert result.belief_state.hypotheses_by_id()["H1"].posterior == 0.5
    assert result.belief_state.hypotheses_by_id()["H2"].posterior == 0.5
```

- [x] **Step 2: Verify RED**

Run:

```bash
python3 -m pytest tests/test_core_cycles.py::test_core_schema_violation_does_not_update_belief_state -q
```

Expected: failure because discarded neutral evidence still produces neutral belief updates.

- [x] **Step 3: Implement belief-solver skip rule**

In `bayesprobe/belief.py`, update:

```python
    for event_index, event in enumerate(events, start=1):
        if event.discard_reason is not None:
            continue
        for hypothesis_id, band in event.likelihoods.items():
```

- [x] **Step 4: Verify GREEN**

Run:

```bash
python3 -m pytest tests/test_core_cycles.py::test_core_schema_violation_does_not_update_belief_state tests/test_core_cycles.py::test_active_only_signal_updates_belief_through_evidence_gate -q
```

Expected: schema-violation event produces no belief updates, and existing active-only update behavior still passes.

### Task 5: Benchmark Ledger Replay For Schema Violations

**Files:**
- Modify: `tests/test_benchmark_harness.py`

**Interfaces:**
- Consumes:
  - `BenchmarkHarness(model_gateway=...)`
  - `JsonlLedgerStore.read_all(...)`
- Produces:
  - Benchmark run records schema-violation evidence and no belief updates for discarded event.

- [x] **Step 1: Write failing benchmark ledger test**

Add test:

```python
def test_benchmark_harness_records_schema_violation_without_belief_update(tmp_path: Path):
    ledger = JsonlLedgerStore(tmp_path / "schema-violation-ledger.jsonl")
    gateway = ScriptedModelGateway(
        responses={
            "judge_evidence": {
                "likelihoods": {"H1": "moderately_confirming", "H2": "moderately_disconfirming"},
                "interpretation": "Missing evidence type.",
            }
        }
    )
    harness = BenchmarkHarness(ledger=ledger, model_gateway=gateway)
    sample = BenchmarkSample(
        sample_id="schema_violation_passive",
        question_or_claim="Can benchmark replay schema violations?",
        signal_shape=BenchmarkSignalShape.PASSIVE_ONLY,
        gold_best_hypothesis="H1",
        passive_signals=[
            BenchmarkSignal(
                signal_id="S_schema_violation_passive",
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
    assert result.evidence_event_count == 1
    assert result.belief_update_count == 0
    assert evidence_payloads[0]["discard_reason"].startswith("schema_violation:")
    assert evidence_payloads[0]["evidence_type"] == "neutral"
    assert ledger.read_all("belief_update") == []
```

- [x] **Step 2: Verify RED Or GREEN Based On Prior Tasks**

Run:

```bash
python3 -m pytest tests/test_benchmark_harness.py::test_benchmark_harness_records_schema_violation_without_belief_update -q
```

Expected after Tasks 3-4: pass. If run before Tasks 3-4, it fails because schema violations are not converted or discarded evidence is not skipped.

### Task 6: Public SDK Export And Regression Verification

**Files:**
- Modify: `tests/test_public_api_and_config.py`
- Modify: `bayesprobe/__init__.py`
- Test: all pytest files

**Interfaces:**
- Produces:
  - `ModelGatewayValidationError` exported from package root.

- [x] **Step 1: Write failing public SDK export test**

Update import in `tests/test_public_api_and_config.py`:

```python
    ModelGatewayValidationError,
```

Update expected names:

```python
"ModelGatewayValidationError",
```

Update assertions:

```python
assert ModelGatewayValidationError is not None
```

- [x] **Step 2: Verify RED**

Run:

```bash
python3 -m pytest tests/test_public_api_and_config.py::test_public_sdk_exports_supported_names -q
```

Expected: failure because package root does not export `ModelGatewayValidationError`.

- [x] **Step 3: Export validation error**

Update `bayesprobe/__init__.py` import list:

```python
from bayesprobe.model_gateway import (
    DeterministicModelGateway,
    EvidenceJudgment,
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
"ModelGatewayValidationError",
```

- [x] **Step 4: Run focused tests**

Run:

```bash
python3 -m pytest tests/test_model_gateway.py tests/test_core_cycles.py tests/test_benchmark_harness.py tests/test_public_api_and_config.py -q
```

Expected: focused gateway/evidence/belief/benchmark/public SDK tests pass.

- [x] **Step 5: Run full regression**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
```

Expected: all tests pass.

- [x] **Step 6: Remove generated caches**

Run:

```bash
find . -type d \( -name __pycache__ -o -name .pytest_cache \) -prune -exec rm -rf {} +
```

Expected: no generated cache directories remain.

## Self-Review

- Spec coverage: The plan covers model-gateway validation failures, evidence-gate schema-violation conversion, discarded evidence skip behavior, benchmark ledger replay, public SDK export, focused tests, full regression, and cache cleanup.
- No unresolved markers: The plan contains concrete tests, implementation snippets, commands, and expected outcomes for each task.
- Type consistency: `ModelGatewayValidationError`, `discard_reason`, `EvidenceType.NEUTRAL`, `LikelihoodBand.NEUTRAL`, `ScriptedModelGateway`, and the existing helper names are used consistently across tasks.
