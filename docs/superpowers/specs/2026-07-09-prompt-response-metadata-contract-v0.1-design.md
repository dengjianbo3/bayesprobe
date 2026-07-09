# Prompt / Response Metadata Contract v0.1 Design

## Goal

Add a minimal prompt/schema/model-call metadata contract to the `ModelGateway`
seam before adding real provider adapters.

BayesProbe now has:

- configurable `ModelGateway`;
- structured judgment validation;
- schema-violation fallback;
- opt-in evidence judgment repair/retry.

The next missing provider-readiness layer is reproducibility metadata. Benchmark
and ledger records should be able to answer:

- Which task produced this judgment?
- Which prompt identity and version was intended?
- Which schema identity and version was expected?
- Which adapter kind handled the request?
- Was this a repair attempt?

This slice should add those answers without network calls, prompt templates, or
real provider adapters.

## Architectural Context

BayesProbe's architecture document defines `ModelGateway` as the single seam for
model-shaped structured decisions. This slice deepens that seam. It does not
move model judgment outside the Evidence Integration Gate and does not let raw
model output influence belief directly.

The key rule remains:

```text
Model output
-> parse / validate / repair if enabled
-> BayesProbe domain object
-> EvidenceEvent
-> BeliefUpdate
```

Metadata is audit context. It is not evidence, and it must not affect posterior
math.

## Non-Goals

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
- No breaking change for callers that construct
  `StructuredModelRequest(task=..., input=...)`.

## Design Decision

Use **request metadata plus evidence-level invocation trace**.

Rejected alternatives:

- **Provider adapter first**: too much at once. Provider calls would force prompt
  versioning, schema metadata, retries, and reproducibility decisions into the
  same patch.
- **Full prompt registry now**: useful later, but too heavy before a provider
  exists.
- **New model-call ledger stream now**: eventually desirable, but this first
  slice can preserve useful audit metadata on the `EvidenceEvent` already
  emitted by the core.
- **Change `ModelGateway.complete_structured(...)` to return a response wrapper
  now**: this would risk breaking current callers that expect a dict. Keep the
  return value stable for v0.1.

Chosen design:

- Extend `StructuredModelRequest` with optional prompt/schema metadata fields.
- Add a small `ModelInvocationTrace` helper in `bayesprobe/model_gateway.py`.
- Give built-in gateways stable `adapter_kind` values.
- Add `EvidenceEvent.model_trace: dict[str, Any]`.
- Attach `model_trace` to direct evidence events and schema-violation events
  produced by model-gateway judgment.
- Keep deterministic/scripted gateway payload behavior unchanged.

## Public Types

### StructuredModelRequest

Extend the existing dataclass:

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
```

Validation:

- `task` must be a non-empty string.
- `input` must be a mapping-like dict.
- optional string fields must be non-empty when supplied.
- `metadata` must be a dict.

Backward compatibility:

```python
StructuredModelRequest(task="judge_evidence", input={...})
```

must continue to work.

### ModelInvocationTrace

Add:

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

    @classmethod
    def from_request(
        cls,
        request: StructuredModelRequest,
        *,
        adapter_kind: str,
    ) -> "ModelInvocationTrace":
        ...

    def to_dict(self) -> dict[str, Any]:
        ...
```

`repair_attempt_index` is read from:

```python
request.metadata.get("repair_attempt_index")
```

if present.

Validation:

- `task` and `adapter_kind` must be non-empty strings.
- optional string fields must be non-empty when supplied.
- `repair_attempt_index` must be a positive integer when supplied.
- `metadata` must be a dict.

The trace is intentionally generic. A future provider adapter can later enrich
it with provider/model/token/cost metadata by adding fields or by using
`metadata` for provider-specific details.

### Adapter Kind

Built-in gateway adapters should expose stable adapter identities:

```python
class DeterministicModelGateway:
    adapter_kind = "deterministic"

class ScriptedModelGateway:
    adapter_kind = "scripted"
```

For external gateway objects that do not expose `adapter_kind`, BayesProbe
should derive a readable fallback:

```python
gateway.__class__.__name__
```

This keeps custom adapters usable without forcing immediate interface changes.

## Evidence Event Trace

Extend `EvidenceEvent` in `bayesprobe/schemas.py`:

```python
model_trace: dict[str, Any] = Field(default_factory=dict)
```

Rules:

- Empty for evidence events that did not come from `ModelGateway`.
- Present for direct evidence events created from `judge_evidence`.
- Present for schema-violation events caused by malformed model output.
- Present for repaired evidence events, using the trace of the repair request
  that produced the valid judgment.

This field is audit-only. `solve_updates(...)` must not read it.

## Prompt / Schema Defaults

Evidence judgment requests should use stable defaults:

```text
task = "judge_evidence"
prompt_id = "evidence_judgment"
prompt_version = "v0.1"
schema_name = "EvidenceJudgment"
schema_version = "v0.1"
```

Repair requests should use:

```text
task = "repair_evidence_judgment"
prompt_id = "evidence_judgment_repair"
prompt_version = "v0.1"
schema_name = "EvidenceJudgment"
schema_version = "v0.1"
metadata.repair_attempt_index = <attempt_index>
```

The repair request `input` already contains `attempt_index`; the metadata field
duplicates it intentionally for trace queries.

## Evidence Integration Flow

### Valid direct judgment

```text
ExternalSignal
-> StructuredModelRequest(task="judge_evidence", prompt_id="evidence_judgment", ...)
-> ModelGateway.complete_structured(...)
-> evidence_judgment_from_mapping(...)
-> EvidenceEvent(model_trace={task, adapter_kind, prompt/schema versions, ...})
-> BeliefUpdate
```

### Invalid judgment, repair disabled

```text
ExternalSignal
-> judge request with trace metadata
-> malformed payload
-> schema-violation EvidenceEvent(model_trace=<judge trace>)
-> no BeliefUpdate
```

### Invalid judgment, repair enabled and successful

```text
ExternalSignal
-> judge request
-> malformed payload
-> repair request with repair_attempt_index
-> valid repaired payload
-> EvidenceEvent(model_trace=<repair trace>)
-> BeliefUpdate
```

### Invalid judgment, repair enabled and unsuccessful

```text
ExternalSignal
-> judge request
-> malformed payload
-> repair request with repair_attempt_index
-> invalid repaired payload
-> schema-violation EvidenceEvent(model_trace=<last repair trace>)
-> no BeliefUpdate
```

## Implementation Shape

### Model Gateway Module

Modify `bayesprobe/model_gateway.py`:

- extend `StructuredModelRequest`;
- add `ModelInvocationTrace`;
- add helper:

```python
def model_gateway_adapter_kind(gateway: ModelGateway) -> str:
    ...
```

- export `ModelInvocationTrace`.

Do not change `ModelGateway.complete_structured(...)` return type in v0.1.

### Evidence Gate

Modify `bayesprobe/evidence.py`:

- `_build_judge_evidence_request(...)` sets prompt/schema defaults.
- `_repair_evidence_judgment(...)` sets repair prompt/schema defaults.
- repair request metadata includes `repair_attempt_index`.
- `_evidence_judgment_with_repair(...)` returns both:

```python
tuple[EvidenceJudgment, ModelInvocationTrace]
```

- `_schema_violation_event(...)` accepts optional `model_trace`.
- `_event(...)` accepts optional `model_trace` and passes it to `EvidenceEvent`.

This keeps trace generation local to the same module that owns model judgment.

### Schemas

Modify `bayesprobe/schemas.py`:

- add `model_trace` to `EvidenceEvent`;
- default to `{}` for backward compatibility.

### Built-in Adapters

Modify `DeterministicModelGateway` and `ScriptedModelGateway`:

- add `adapter_kind` class attribute.
- keep return payloads unchanged.
- keep `ScriptedModelGateway.requests` as `list[StructuredModelRequest]`, now
  capturing request metadata naturally.

## Configuration

No new experiment config field is required in this slice.

Reason:

- evidence judgment and repair defaults are stable enough for v0.1;
- provider-specific model selection and prompt template versions belong in the
  provider adapter / prompt registry slice;
- avoiding config now keeps this patch focused on the metadata contract.

Future config can add:

```json
{
  "model_gateway": {
    "kind": "provider",
    "provider": "openai",
    "model": "...",
    "prompt_version": "v0.2",
    "schema_version": "v0.1"
  }
}
```

## Testing

Add focused tests for:

### Model Gateway Metadata

- `StructuredModelRequest(task="judge_evidence", input={})` remains valid.
- request metadata fields are stored and immutable.
- empty `task` raises `ValueError`.
- empty optional string metadata field raises `ValueError`.
- non-dict metadata raises `ValueError`.
- `ModelInvocationTrace.from_request(...)` copies task/prompt/schema metadata.
- `ModelInvocationTrace.from_request(...)` captures
  `metadata["repair_attempt_index"]`.
- invalid repair attempt index raises `ValueError`.
- built-in gateways expose adapter kinds.

### Evidence Gate Trace

- direct valid judgment evidence includes:
  - `task = "judge_evidence"`
  - `adapter_kind = "scripted"` or `"deterministic"`
  - `prompt_id = "evidence_judgment"`
  - `prompt_version = "v0.1"`
  - `schema_name = "EvidenceJudgment"`
  - `schema_version = "v0.1"`
- default schema-violation evidence includes judge trace metadata.
- repaired evidence includes repair trace metadata and
  `repair_attempt_index = 1`.
- projection-decomposition events that do not call `ModelGateway` keep
  `model_trace == {}`.

### Ledger / Benchmark

- benchmark ledger evidence records include `model_trace` for model-judged
  direct evidence.
- existing deterministic benchmark tests continue to pass.

## Acceptance Criteria

- Existing callers can still construct `StructuredModelRequest(task, input)`.
- Existing deterministic and scripted gateway payload behavior is unchanged.
- Direct evidence requests carry prompt/schema metadata.
- Repair requests carry prompt/schema metadata and repair attempt index.
- `EvidenceEvent` can persist model invocation trace data.
- Discarded schema-violation evidence keeps model trace data and remains
  belief-neutral.
- Public SDK exports `ModelInvocationTrace`.
- No provider adapter, network call, prompt registry, or posterior math change is
  introduced.
- Full pytest suite passes.

## Future Work

This slice prepares but does not implement:

1. provider-backed `ModelGateway` adapters;
2. prompt template registry keyed by `prompt_id` and `prompt_version`;
3. response token/cost/latency metadata;
4. dedicated `model_call` ledger records;
5. recorded provider fixture adapter for reproducible benchmark replay.
