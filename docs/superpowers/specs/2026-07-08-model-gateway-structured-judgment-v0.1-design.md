# ModelGateway Structured Judgment v0.1 Design

Date: 2026-07-08
Status: Approved from core-depth roadmap

## Goal

Create the first structured model judgment seam so BayesProbe can move from hard-coded heuristic judgments toward model-backed evidence construction without destabilizing tests or changing core behavior.

The v0.1 seam must preserve deterministic offline behavior while making the future model adapter slot explicit.

## Scope

Create `bayesprobe/model_gateway.py` with:

- `StructuredModelRequest`
- `ModelGateway`
- `EvidenceJudgment`
- `DeterministicModelGateway`
- `ScriptedModelGateway`

Update `EvidenceIntegrationGate` so direct non-projection signals call the gateway for evidence type, likelihood bands, interpretation, and optional quality overrides.

## Non-Goals

- No OpenAI or external model adapter in this slice.
- No network calls.
- No prompt templates.
- No retries, rate limits, or cost tracking.
- No changes to projection decomposition behavior.
- No changes to Hypothesis Evolution Engine.
- No changes to Belief Solver likelihood math.
- No changes to public CLI/config behavior.

## Module Interface

```python
@dataclass(frozen=True)
class StructuredModelRequest:
    task: str
    input: dict[str, Any]


class ModelGateway(Protocol):
    def complete_structured(self, request: StructuredModelRequest) -> dict[str, Any]:
        ...
```

The interface is intentionally generic because future uses include:

- evidence judgment
- likelihood judgment
- hypothesis generation
- projection generation
- hypothesis evolution judgment

v0.1 only uses the `judge_evidence` task.

## Evidence Judgment

`EvidenceJudgment` should represent:

- `evidence_type: EvidenceType`
- `likelihoods: dict[str, LikelihoodBand]`
- `interpretation: str`
- `quality_overrides: dict[str, float]`

The gateway returns dictionaries so adapters can remain transport-friendly. The Evidence Gate converts dictionaries into typed `EvidenceJudgment` objects.

## Deterministic Adapter

`DeterministicModelGateway` must reproduce the existing keyword behavior:

- `REFUTES` or `CONTRADICTS` -> `COUNTEREVIDENCE`
  - `H1`: `MODERATELY_DISCONFIRMING`
  - `H2`: `MODERATELY_CONFIRMING`
- `SUPPORTS` -> `SUPPORTING`
  - `H1`: `MODERATELY_CONFIRMING`
  - `H2`: `MODERATELY_DISCONFIRMING`
- `ANOMALY` -> `ANOMALY`
  - all targets: `MODERATELY_DISCONFIRMING`
- otherwise -> `NEUTRAL`
  - all targets: `NEUTRAL`

It must return the same default interpretation string currently used by the Evidence Gate:

```text
Deterministic v0.2 interpretation for <source_type>.
```

## Scripted Adapter

`ScriptedModelGateway` is a test and fixture adapter:

- Constructed with `responses: dict[str, dict[str, Any]]`.
- `complete_structured(request)` returns `responses[request.task]`.
- It records all requests in `requests`.
- It raises `ValueError` if no scripted response exists for the task.

This allows tests to prove that Evidence Gate uses the model seam rather than local keyword logic.

## Evidence Gate Integration

`EvidenceIntegrationGate.__init__` gains:

```python
model_gateway: ModelGateway | None = None
```

Default:

```python
model_gateway or DeterministicModelGateway()
```

For direct evidence events:

1. Resolve target hypotheses.
2. Build `StructuredModelRequest(task="judge_evidence", input={...})`.
3. Call `model_gateway.complete_structured(...)`.
4. Convert response to `EvidenceJudgment`.
5. Build `EvidenceEvent`.

Request input should include:

- `signal_id`
- `source_type`
- `source`
- `raw_content`
- `target_hypotheses`
- `cycle_id`
- `probe_ids`

Projection sender judgment and source claim decomposition remain deterministic in v0.1.

## Quality Overrides

`SignalQualityAssessor` remains the default source of quality scores.

If `EvidenceJudgment.quality_overrides` includes any of:

- `reliability`
- `independence`
- `relevance`
- `novelty`
- `specificity`
- `verifiability`

then the final `EvidenceEvent` should use the override value for that field after normal duplicate/low-reliability adjustments.

Overrides must be clamped by existing Pydantic validation through `EvidenceEvent`.

## Test Strategy

Add `tests/test_model_gateway.py` covering:

- deterministic gateway preserves `REFUTES`, `SUPPORTS`, `ANOMALY`, and neutral behavior.
- scripted gateway records requests and returns configured responses.

Add or update `tests/test_core_cycles.py` covering:

- `EvidenceIntegrationGate` sends direct evidence judgment through gateway.
- scripted evidence judgment can force `BOUNDARY_CONDITION` and custom likelihoods.
- quality overrides appear on the resulting `EvidenceEvent`.

Run focused model/evidence/core tests first, then full pytest.
