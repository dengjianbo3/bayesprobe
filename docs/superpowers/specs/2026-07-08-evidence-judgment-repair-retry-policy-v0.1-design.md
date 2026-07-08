# Evidence Judgment Repair / Retry Policy v0.1 Design

## Goal

Add an opt-in repair/retry policy for malformed structured evidence judgments.

The current implementation already protects belief state quality when a
`ModelGateway` returns malformed `judge_evidence` output: the Evidence
Integration Gate emits a discarded neutral `EvidenceEvent`, and the Belief
Solver skips it. This slice adds one controlled step before that fallback:

```text
judge_evidence
-> validate
-> if invalid and repair is enabled, ask ModelGateway to repair the payload
-> validate repaired payload
-> if valid, build normal EvidenceEvent
-> if invalid, emit schema-violation EvidenceEvent
```

Default behavior must remain unchanged.

## Architectural Context

The new architecture document defines the relevant invariant:

> Model output must be parsed, validated, and converted into BayesProbe domain
> objects before it can influence belief.

This design preserves that invariant. Repair is not a shortcut around the
Evidence Integration Gate. It is an internal model-gateway recovery path inside
the gate.

Relevant existing modules:

- `bayesprobe/model_gateway.py`
  - `ModelGateway`
  - `StructuredModelRequest`
  - `ModelGatewayConfig`
  - `ModelGatewayValidationError`
  - `evidence_judgment_from_mapping(...)`
- `bayesprobe/evidence.py`
  - `EvidenceIntegrationGate`
  - schema-violation neutral event path
- `bayesprobe/core.py`
  - `BayesProbeCore(..., model_gateway=...)`
- `bayesprobe/benchmark.py`
  - `BenchmarkHarness(..., model_gateway=...)`
- `bayesprobe/experiment_runner.py`
  - `ExperimentRunConfig(..., model_gateway=...)`
- `bayesprobe/config.py`
  - JSON config parsing for experiment runs

## Non-Goals

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

## Design Decision

Use **opt-in repair through `ModelGateway`**.

Rejected alternatives:

- **Always repair by default**: too surprising for deterministic fixtures and
  existing users.
- **Let `ScriptedModelGateway` hardcode repair behavior**: shallow; repair would
  live in the test adapter instead of the evidence-gate flow.
- **Provider-level retry now**: too early; provider errors, schema repair,
  prompt versioning, and network retry should remain separate concerns.

Chosen design:

- add `EvidenceJudgmentRepairPolicy`;
- default `max_attempts=0`;
- pass the policy through the same configuration chain as `ModelGateway`;
- when enabled, call `ModelGateway.complete_structured(...)` with task
  `repair_evidence_judgment`;
- validate the repaired payload with `evidence_judgment_from_mapping(...)`;
- produce normal evidence only from a valid repaired payload;
- fall back to the existing schema-violation neutral event if repair fails
  validation.

## Public Types

Add this dataclass in `bayesprobe/model_gateway.py`:

```python
@dataclass(frozen=True)
class EvidenceJudgmentRepairPolicy:
    max_attempts: int = 0
    repair_task: str = "repair_evidence_judgment"
```

Validation:

- `max_attempts` must be `>= 0`.
- `repair_task` must be a non-empty string.

The policy belongs near `ModelGateway` because it describes structured model
judgment recovery. The Evidence Integration Gate consumes it, but the policy is
part of the model-gateway contract.

Export it from:

- `bayesprobe/model_gateway.py`
- package root `bayesprobe/__init__.py`

## Evidence Integration Flow

`EvidenceIntegrationGate.__init__(...)` gains:

```python
judgment_repair_policy: EvidenceJudgmentRepairPolicy | None = None
```

Default:

```python
EvidenceJudgmentRepairPolicy(max_attempts=0)
```

Direct evidence handling changes from:

```text
gateway judge -> parse -> event
gateway judge -> parse raises ModelGatewayValidationError -> schema violation
```

to:

```text
gateway judge -> parse -> event
gateway judge -> parse raises ModelGatewayValidationError
  -> if repair disabled: schema violation
  -> if repair enabled:
       repair request 1 -> parse -> event
       repair request 1 -> parse raises ModelGatewayValidationError
         -> repeat until attempts exhausted
         -> schema violation
```

Only `ModelGatewayValidationError` enters the repair path. Other exceptions
continue to propagate.

## Repair Request Shape

Repair must be ledger-reproducible and adapter-testable. The repair task should
receive enough context to repair structure without needing hidden gate state.

Task:

```text
repair_evidence_judgment
```

Input:

```python
{
    "original_request": {
        "task": "judge_evidence",
        "input": {
            "signal_id": "...",
            "source_type": "...",
            "source": "...",
            "raw_content": "...",
            "target_hypotheses": ["H1", "H2"],
            "cycle_id": "...",
            "probe_ids": ["P1"],
        },
    },
    "invalid_payload": {...},
    "validation_error": "invalid evidence_type: ...",
    "attempt_index": 1,
    "allowed_evidence_types": [
        "supporting",
        "counterevidence",
        "boundary_condition",
        "neutral",
        "anomaly",
        "sender_judgment",
        "source_claim",
    ],
    "allowed_likelihood_bands": [
        "strongly_disconfirming",
        "moderately_disconfirming",
        "weakly_disconfirming",
        "neutral",
        "weakly_confirming",
        "moderately_confirming",
        "strongly_confirming",
    ],
    "required_fields": [
        "evidence_type",
        "likelihoods",
        "interpretation",
    ],
}
```

`invalid_payload` should preserve the original malformed payload when it is a
mapping. If the returned payload is not a mapping, store it under:

```python
{"_raw_payload": payload}
```

This keeps the repair request structured while preserving the failure for
adapter logging and scripted tests.

## Event Semantics

### Repair succeeds

If a repaired payload validates, BayesProbe produces a normal `EvidenceEvent`:

- `discard_reason = None`
- evidence type comes from repaired judgment
- likelihoods come from repaired judgment
- interpretation comes from repaired judgment
- quality overrides come from repaired judgment
- the event participates in belief update normally

The event id remains the original signal event id, for example:

```text
<run_cycle>_E1
```

No extra evidence event is emitted for the failed first attempt.

### Repair fails

If all repair attempts fail validation, BayesProbe emits the existing
schema-violation event:

- `evidence_type = EvidenceType.NEUTRAL`
- likelihoods are neutral for all target hypotheses
- quality scores are zero
- `discard_reason` begins with `"schema_violation:"`
- the Belief Solver skips it

The discard reason should include the last validation error and indicate repair
was attempted, for example:

```text
schema_violation: repair failed after 1 attempt(s): invalid evidence_type: ...
```

### Repair adapter errors

If repair is enabled but the model gateway raises a non-validation exception
during repair, that exception propagates.

Example:

- `ScriptedModelGateway` has `judge_evidence` but no
  `repair_evidence_judgment` response.

This should fail fast as a configuration error. It is not a schema violation.

## Configuration Chain

The repair policy must be configurable anywhere external users can configure a
model gateway.

### BayesProbeCore

Add:

```python
BayesProbeCore(
    ledger: JsonlLedgerStore | None = None,
    model_gateway: ModelGateway | None = None,
    judgment_repair_policy: EvidenceJudgmentRepairPolicy | None = None,
)
```

`_create_evidence_integration_gate(...)` passes both `model_gateway` and
`judgment_repair_policy`.

### BenchmarkHarness

Add:

```python
BenchmarkHarness(
    *,
    core: BayesProbeCore | None = None,
    ledger: JsonlLedgerStore | None = None,
    model_gateway: ModelGateway | None = None,
    judgment_repair_policy: EvidenceJudgmentRepairPolicy | None = None,
    max_cycles: int = 1,
    max_probes_per_cycle: int = 1,
)
```

If `core` is provided, `BenchmarkHarness` uses that core as-is. If not, it
constructs `BayesProbeCore(...)` with both model gateway and repair policy.

### ExperimentRunConfig

Add:

```python
judgment_repair_policy: EvidenceJudgmentRepairPolicy | Mapping[str, Any] | None = None
```

`run_benchmark_experiment(...)` passes it to `BenchmarkHarness`.

### JSON Config

Add optional field:

```json
{
  "judgment_repair_policy": {
    "max_attempts": 1
  }
}
```

Validation:

- field must be an object if present;
- `max_attempts` must be an integer;
- `repair_task` may be supplied as a non-empty string, but defaults to
  `"repair_evidence_judgment"`.

## Scripted Gateway Behavior

`ScriptedModelGateway` currently maps one response per task. For the MVP repair
tests, that is enough:

```python
ScriptedModelGateway(
    responses={
        "judge_evidence": {"evidence_type": "bad"},
        "repair_evidence_judgment": {
            "evidence_type": "supporting",
            "likelihoods": {"H1": "moderately_confirming"},
            "interpretation": "Repaired judgment.",
        },
    }
)
```

This design does not require sequential scripted responses yet. Multiple repair
attempt tests can use `max_attempts=1` for v0.1. A future recorded fixture
gateway can support richer call sequences.

## Data Flow

Repair disabled:

```text
ExternalSignal
-> ModelGateway(task="judge_evidence")
-> invalid payload
-> evidence_judgment_from_mapping raises ModelGatewayValidationError
-> EvidenceEvent(discard_reason="schema_violation: ...")
-> no BeliefUpdate
```

Repair enabled and successful:

```text
ExternalSignal
-> ModelGateway(task="judge_evidence")
-> invalid payload
-> validation error
-> ModelGateway(task="repair_evidence_judgment")
-> repaired valid payload
-> EvidenceEvent(discard_reason=None)
-> BeliefUpdate
```

Repair enabled and unsuccessful:

```text
ExternalSignal
-> ModelGateway(task="judge_evidence")
-> invalid payload
-> validation error
-> ModelGateway(task="repair_evidence_judgment")
-> invalid repaired payload
-> EvidenceEvent(discard_reason="schema_violation: repair failed after ...")
-> no BeliefUpdate
```

## Testing

Add focused tests.

### Model Gateway Tests

- `EvidenceJudgmentRepairPolicy(max_attempts=-1)` raises `ValueError`.
- empty `repair_task` raises `ValueError`.
- package root exports `EvidenceJudgmentRepairPolicy`.

### Evidence Gate Tests

- default repair policy does not call `repair_evidence_judgment`.
- repair-enabled gate calls `judge_evidence` then `repair_evidence_judgment`
  after validation failure.
- valid repaired payload produces normal evidence with `discard_reason=None`.
- invalid repaired payload produces schema-violation evidence.
- missing scripted repair task raises `ValueError` when repair is enabled.
- non-validation exceptions still propagate.

### Core Tests

- `BayesProbeCore(judgment_repair_policy=...)` propagates the policy to the
  Evidence Integration Gate.
- repaired evidence can produce a `BeliefUpdate`.
- unrepaired schema violation still produces no `BeliefUpdate`.

### Benchmark / Config Tests

- `BenchmarkHarness(judgment_repair_policy=...)` can repair malformed scripted
  judgment and score through the normal run path.
- `ExperimentRunConfig(judgment_repair_policy=...)` passes policy into the
  benchmark harness.
- `experiment_config_from_mapping(...)` parses
  `judgment_repair_policy.max_attempts`.
- invalid JSON repair policy shapes raise `ValueError`.

### Regression

- full deterministic default test suite continues to pass.

## Acceptance Criteria

- Default behavior is unchanged.
- Repair is opt-in through `EvidenceJudgmentRepairPolicy`.
- Repair attempts use the existing `ModelGateway` seam.
- Valid repaired judgment becomes ordinary evidence.
- Invalid repaired judgment falls back to discarded neutral schema-violation
  evidence.
- Discarded schema-violation evidence still produces no belief updates.
- External code can configure repair policy through core, benchmark, experiment
  config, JSON config, and package exports.
- Full pytest suite passes.

## Future Work

After this slice, the next provider-readiness work should be:

1. provider-backed `ModelGateway` adapter;
2. prompt templates keyed by task and version;
3. response metadata on structured model requests/results;
4. recorded fixture gateway for reproducible provider experiments;
5. benchmark samples covering repair success and repair failure.
