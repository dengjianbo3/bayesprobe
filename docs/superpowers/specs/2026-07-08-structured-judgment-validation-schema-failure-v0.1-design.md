# Structured Judgment Validation / Schema Failure Handling v0.1 Design

## Goal

Make model-gateway evidence judgment failures auditable and non-destructive.

When a `ModelGateway` adapter returns malformed structured judgment, BayesProbe should record a schema-violation evidence event, skip belief updates for that event, and keep the cycle running. This preserves benchmark stability and future model-provider readiness without introducing prompt repair or live model calls.

## Context

BayesProbe currently has:

- `ModelGateway`, `ModelGatewayConfig`, and `build_model_gateway(...)`.
- `DeterministicModelGateway` and `ScriptedModelGateway`.
- `EvidenceIntegrationGate` routing direct signals through `evidence_judgment_from_mapping(...)`.
- `EvidenceEvent.discard_reason`, already present in schema but not yet used by the belief solver.

The current failure mode is too sharp: malformed gateway output can raise during evidence integration and abort the run. That is acceptable for early deterministic fixtures, but not for the final direction where model-backed judgment must be reproducible, auditable, and safely isolated from belief-state mutation.

## Non-Goals

- No live model provider adapter.
- No network calls.
- No prompt templates.
- No schema-repair retry loop.
- No manual review queue.
- No changes to likelihood-band math.
- No changes to projection decomposition.
- No changes to Hypothesis Evolution Engine.
- No broad error swallowing for unrelated bugs.

## Design

### Judgment Validation Error

Add a focused exception in `bayesprobe/model_gateway.py`:

```python
class ModelGatewayValidationError(ValueError):
    pass
```

`evidence_judgment_from_mapping(...)` should raise `ModelGatewayValidationError` for malformed payloads:

- missing `evidence_type`
- invalid `evidence_type`
- non-object `likelihoods`
- invalid likelihood band
- non-object `quality_overrides`
- non-numeric quality override value

This keeps the validation failure local to the model-gateway seam. Callers can catch exactly schema-like gateway failures without catching unrelated programming errors.

### Schema Violation Evidence Event

`EvidenceIntegrationGate._build_direct_evidence_event(...)` should catch `ModelGatewayValidationError` around the gateway call and judgment parsing.

On validation failure, it should create an `EvidenceEvent` with:

- `evidence_type = EvidenceType.NEUTRAL`
- likelihoods for all target hypotheses set to `LikelihoodBand.NEUTRAL`
- `interpretation = "Model gateway judgment failed schema validation."`
- `discard_reason = "schema_violation: <message>"`
- low quality scores, using a dedicated quality override:
  - `reliability = 0.0`
  - `independence = 0.0`
  - `relevance = 0.0`
  - `novelty = 0.0`
  - `specificity = 0.0`
  - `verifiability = 0.0`

The raw signal content remains in the event. The point is not to erase the signal; it is to record that BayesProbe could not promote it into usable evidence.

### EvidenceEvent Helper

Extend `EvidenceIntegrationGate._event(...)` with:

```python
discard_reason: str | None = None
```

and pass that into the `EvidenceEvent`.

Existing valid events keep `discard_reason=None`.

### Belief Solver Skip Rule

Update `solve_updates(...)` in `bayesprobe/belief.py`:

```python
for event_index, event in enumerate(events, start=1):
    if event.discard_reason is not None:
        continue
```

A discarded evidence event is still recorded in the ledger and available for audit, but it must not produce a `BeliefUpdate` and must not alter hypothesis posterior values.

This is the critical BayesProbe distinction:

- signal enters the system
- failed evidence construction is recorded
- belief state remains unchanged by invalid evidence

### Ledger And Benchmark Behavior

No new ledger record type is needed. Schema violations are ordinary `evidence_event` records with `discard_reason`.

Benchmark runs should not crash when a scripted gateway returns malformed judgment. If a benchmark uses a malformed scripted judgment, its ledger should contain an evidence event with `discard_reason` beginning with `"schema_violation:"` and no belief updates for that event.

## Data Flow

Valid judgment:

```text
ExternalSignal
→ ModelGateway.complete_structured(...)
→ evidence_judgment_from_mapping(...)
→ EvidenceEvent(discard_reason=None)
→ solve_updates(...)
→ BeliefUpdate
```

Malformed judgment:

```text
ExternalSignal
→ ModelGateway.complete_structured(...)
→ evidence_judgment_from_mapping(...) raises ModelGatewayValidationError
→ EvidenceEvent(discard_reason="schema_violation: ...")
→ solve_updates(...) skips event
→ no BeliefUpdate
```

## Error Handling

Only `ModelGatewayValidationError` is converted into schema-violation evidence.

Other exceptions should continue to propagate. For example:

- missing scripted task response still raises `ValueError`
- programming errors inside adapters still raise
- unexpected runtime errors still fail fast

This keeps schema failure handling narrow and auditable rather than becoming a general exception sink.

## Testing

Add focused tests for:

- `evidence_judgment_from_mapping(...)` raises `ModelGatewayValidationError` for missing `evidence_type`.
- invalid evidence type raises `ModelGatewayValidationError`.
- invalid likelihood band raises `ModelGatewayValidationError`.
- non-object likelihoods raises `ModelGatewayValidationError`.
- non-object quality overrides raises `ModelGatewayValidationError`.
- non-numeric quality override raises `ModelGatewayValidationError`.
- `EvidenceIntegrationGate` converts malformed judgment into a schema-violation evidence event.
- schema-violation evidence event has all target likelihoods neutral and zero quality scores.
- `BayesProbeCore.integrate_cycle(...)` records schema-violation evidence but produces no belief updates and leaves posteriors unchanged.
- `BenchmarkHarness` with ledger records schema-violation evidence instead of crashing.
- full regression passes with deterministic defaults.

## Acceptance Criteria

- Malformed structured model judgment no longer aborts normal evidence integration when the failure is a schema-validation failure.
- Schema violations are visible in `EvidenceEvent.discard_reason`.
- Discarded evidence events do not produce belief updates.
- Hypothesis posteriors remain unchanged when all evidence events in a cycle are discarded.
- Existing deterministic behavior remains unchanged.
- Full pytest suite passes.

## Future Work

A later slice can add retry and schema-repair behavior:

1. call model once
2. validate
3. if invalid, call repair prompt
4. validate repaired output
5. if still invalid, emit schema-violation evidence

That future work should reuse `ModelGatewayValidationError` and the schema-violation evidence path introduced here.
