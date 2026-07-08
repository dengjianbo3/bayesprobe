# ModelGateway Structured Judgment v0.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a structured model judgment seam and route direct evidence judgment through it while preserving deterministic offline behavior.

**Architecture:** Create `bayesprobe/model_gateway.py` with generic structured request and adapter interfaces plus evidence-judgment helpers. Update `EvidenceIntegrationGate` to depend on this seam for direct non-projection signal judgment; projection decomposition remains deterministic for this slice.

**Tech Stack:** Python 3.11+, dataclasses, Protocol, existing Pydantic schemas, pytest.

## Global Constraints

- No OpenAI or external model adapter in this slice.
- No network calls.
- No prompt templates.
- No retries, rate limits, or cost tracking.
- No changes to projection decomposition behavior.
- No changes to Hypothesis Evolution Engine.
- No changes to Belief Solver likelihood math.
- No changes to public CLI/config behavior.
- Default deterministic behavior must preserve current `SUPPORTS`, `REFUTES`, `CONTRADICTS`, `ANOMALY`, and neutral outputs.
- Do not attempt git commits because this workspace is not currently a git repository.

---

## File Structure

- Create `tests/test_model_gateway.py`: direct tests for deterministic and scripted gateways.
- Modify `tests/test_core_cycles.py`: evidence gate seam tests.
- Create `bayesprobe/model_gateway.py`: structured request, gateway protocol, evidence judgment, deterministic adapter, scripted adapter.
- Modify `bayesprobe/evidence.py`: inject model gateway and use it for direct evidence judgment.
- Modify `bayesprobe/__init__.py` and `tests/test_public_api_and_config.py`: expose the gateway seam through the public SDK for external configuration.

### Task 1: ModelGateway Module Tests

**Files:**
- Create: `tests/test_model_gateway.py`

**Interfaces:**
- Consumes planned:
  - `StructuredModelRequest`
  - `DeterministicModelGateway`
  - `ScriptedModelGateway`
  - `evidence_judgment_from_mapping`

- [x] **Step 1: Write failing module tests**

Create tests covering:

```python
def test_deterministic_gateway_judges_refuting_signal():
    gateway = DeterministicModelGateway()
    response = gateway.complete_structured(
        StructuredModelRequest(
            task="judge_evidence",
            input={
                "raw_content": "REFUTES: passage contradicts H1.",
                "target_hypotheses": ["H1", "H2"],
                "source_type": "benchmark_stream",
            },
        )
    )

    judgment = evidence_judgment_from_mapping(response)
    assert judgment.evidence_type == EvidenceType.COUNTEREVIDENCE
    assert judgment.likelihoods["H1"] == LikelihoodBand.MODERATELY_DISCONFIRMING
    assert judgment.likelihoods["H2"] == LikelihoodBand.MODERATELY_CONFIRMING
```

Also cover:

- `SUPPORTS`
- `ANOMALY`
- neutral
- scripted gateway request recording
- scripted missing task raises `ValueError`

- [x] **Step 2: Verify RED**

Run:

```bash
python3 -m pytest tests/test_model_gateway.py -q
```

Expected: failure because `bayesprobe.model_gateway` does not exist yet.

### Task 2: ModelGateway Implementation

**Files:**
- Create: `bayesprobe/model_gateway.py`
- Test: `tests/test_model_gateway.py`

**Interfaces:**
- Produces:
  - `StructuredModelRequest`
  - `ModelGateway`
  - `EvidenceJudgment`
  - `evidence_judgment_from_mapping`
  - `DeterministicModelGateway`
  - `ScriptedModelGateway`

- [x] **Step 1: Implement structured request and evidence judgment parsing**

Implement:

```python
@dataclass(frozen=True)
class StructuredModelRequest:
    task: str
    input: dict[str, Any]


class ModelGateway(Protocol):
    def complete_structured(self, request: StructuredModelRequest) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class EvidenceJudgment:
    evidence_type: EvidenceType
    likelihoods: dict[str, LikelihoodBand]
    interpretation: str
    quality_overrides: dict[str, float] = field(default_factory=dict)
```

- [x] **Step 2: Implement deterministic gateway**

Implement keyword behavior matching existing Evidence Gate direct signal logic.

- [x] **Step 3: Implement scripted gateway**

Implement fixture adapter:

```python
class ScriptedModelGateway:
    def __init__(self, responses: dict[str, dict[str, Any]]) -> None:
        self.responses = responses
        self.requests = []

    def complete_structured(self, request: StructuredModelRequest) -> dict[str, Any]:
        self.requests.append(request)
        if request.task not in self.responses:
            raise ValueError(f"no scripted response for task: {request.task}")
        return self.responses[request.task]
```

- [x] **Step 4: Run focused module tests**

Run:

```bash
python3 -m pytest tests/test_model_gateway.py -q
```

Expected: all model gateway tests pass.

### Task 3: Evidence Gate Integration Tests

**Files:**
- Modify: `tests/test_core_cycles.py`

**Interfaces:**
- Consumes:
  - `EvidenceIntegrationGate(model_gateway=...)`
  - `ScriptedModelGateway`

- [x] **Step 1: Add failing Evidence Gate seam tests**

Add tests equivalent to:

```python
def test_direct_signal_judgment_uses_model_gateway():
    gateway = ScriptedModelGateway(
        responses={
            "judge_evidence": {
                "evidence_type": "boundary_condition",
                "likelihoods": {"H1": "weakly_disconfirming", "H2": "neutral"},
                "interpretation": "Scripted boundary judgment.",
                "quality_overrides": {"reliability": 0.62},
            }
        }
    )
    gate = EvidenceIntegrationGate(model_gateway=gateway)
    result = gate.integrate(...)
    event = result.evidence_events[0]
    assert event.evidence_type == EvidenceType.BOUNDARY_CONDITION
    assert event.likelihoods["H1"] == LikelihoodBand.WEAKLY_DISCONFIRMING
    assert event.interpretation == "Scripted boundary judgment."
    assert event.reliability == 0.62
    assert gateway.requests[0].task == "judge_evidence"
```

Also assert request input includes `signal_id`, `raw_content`, `target_hypotheses`, `cycle_id`, and `probe_ids`.

- [x] **Step 2: Verify RED**

Run:

```bash
python3 -m pytest tests/test_core_cycles.py::test_direct_signal_judgment_uses_model_gateway -q
```

Expected: failure because `EvidenceIntegrationGate` does not accept `model_gateway`.

### Task 4: Evidence Gate Integration Implementation

**Files:**
- Modify: `bayesprobe/evidence.py`
- Test: `tests/test_core_cycles.py`

**Interfaces:**
- `EvidenceIntegrationGate.__init__(..., model_gateway: ModelGateway | None = None)`

- [x] **Step 1: Add gateway dependency**

Import `DeterministicModelGateway`, `ModelGateway`, `StructuredModelRequest`, and `evidence_judgment_from_mapping`.

Update constructor and `_ensure_helpers`.

- [x] **Step 2: Replace direct signal keyword judgment with gateway judgment**

In `_build_direct_evidence_event`:

- resolve target hypotheses
- build `StructuredModelRequest(task="judge_evidence", input={...})`
- call gateway
- parse `EvidenceJudgment`
- pass judgment fields into `_event`

- [x] **Step 3: Apply quality overrides**

Update `_event(...)` to accept `quality_overrides: dict[str, float] | None = None` and apply overrides after `SignalQualityAssessor.assess(...)`.

- [x] **Step 4: Run focused evidence/model tests**

Run:

```bash
python3 -m pytest tests/test_model_gateway.py tests/test_core_cycles.py -q
```

Expected: model gateway and core/evidence tests pass.

### Task 5: Regression Verification

**Files:**
- Test: all pytest files

**Interfaces:**
- Confirms gateway seam preserves deterministic default behavior across the project.

- [x] **Step 1: Run full test suite**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
```

Expected: all tests pass with no failures.

- [x] **Step 2: Remove generated caches**

Run:

```bash
find . -type d \( -name __pycache__ -o -name .pytest_cache \) -prune -exec rm -rf {} +
```

Expected: no generated cache directories remain.

### Task 6: Public SDK Exposure

**Files:**
- Modify: `bayesprobe/__init__.py`
- Modify: `tests/test_public_api_and_config.py`

- [x] **Step 1: Expose gateway seam for external code**

Export `StructuredModelRequest`, `ModelGateway`, `EvidenceJudgment`, `DeterministicModelGateway`, `ScriptedModelGateway`, and `evidence_judgment_from_mapping` from the package root.

- [x] **Step 2: Verify public SDK import contract**

Run:

```bash
python3 -m pytest tests/test_public_api_and_config.py::test_public_sdk_exports_supported_names tests/test_model_gateway.py tests/test_core_cycles.py -q
```

Expected: public SDK, gateway, and core cycle tests pass.

## Self-Review

- Spec coverage: The plan covers the gateway seam, deterministic adapter, scripted adapter, evidence judgment parsing, direct evidence integration, quality overrides, focused tests, and full regression verification.
- Placeholder scan: No unspecified implementation placeholders remain.
- Type consistency: Public names and function signatures match the design spec.
