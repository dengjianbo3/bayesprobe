# OpenAI ModelGateway v0.1 Design

Status: Approved for spec by user on 2026-07-09.

## 1. Context

BayesProbe already has a stable local `ModelGateway` seam:

- `StructuredModelRequest` carries task, input, prompt/schema metadata, and free
  metadata.
- `ModelInvocationTrace` persists request-level model metadata on
  `EvidenceEvent.model_trace`.
- Deterministic and scripted gateways support offline tests and reproducible
  benchmark fixtures.
- Evidence judgment validation, neutral schema violation, and opt-in repair
  policy are already implemented.

The next provider-readiness step is a first real model adapter. The adapter must
not change the BayesProbe control flow. It should connect OpenAI to the existing
gateway seam and keep provider details out of the core.

Official OpenAI docs used for direction:

- [Text generation](https://developers.openai.com/api/docs/guides/text) covers
  model requests through the Responses API.
- [Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
  covers schema-constrained JSON output.
- [Latest model guidance](https://developers.openai.com/api/docs/guides/latest-model)
  informs model selection, but BayesProbe keeps model choice explicit for
  reproducible experiments.

## 2. Goal

Add an OpenAI-first provider-backed `ModelGateway` adapter that can judge
BayesProbe evidence through the existing structured request/response seam.

The v0.1 adapter should make a real provider usable without spreading OpenAI
details into:

- `BayesProbeCore`;
- `EvidenceIntegrationGate`;
- belief update;
- hypothesis evolution;
- probe planning;
- synchronized or autonomous control flow.

## 3. Non-Goals

- No generic provider registry.
- No multi-provider abstraction.
- No prompt template registry.
- No provider token, cost, latency, or rate-limit accounting.
- No transport retry policy.
- No default live-network test.
- No API key in JSON config, ledger records, or model trace.
- No direct OpenAI response object stored as evidence.
- No bypass of `evidence_judgment_from_mapping(...)`.
- No changes to posterior update math.
- No changes to projection decomposition.
- No changes to probe planning or probe execution.
- No hidden chain-of-thought storage.

## 4. Chosen Approach

Use an independent OpenAI adapter module:

```text
bayesprobe/openai_gateway.py
```

The module implements the existing `ModelGateway` protocol:

```python
class OpenAIResponsesModelGateway:
    adapter_kind = "openai"

    def complete_structured(self, request: StructuredModelRequest) -> dict[str, Any]:
        ...
```

Why this approach:

- It keeps `bayesprobe/model_gateway.py` as the common gateway contract.
- It avoids making OpenAI a core dependency of the BayesProbe loop.
- It lets tests use fake clients and recorded fixtures without network calls.
- It gives future provider adapters a clear example without adding a registry
  too early.

Rejected alternatives:

- Put OpenAI logic directly in `model_gateway.py`: fastest, but provider details
  would make the common contract file too heavy.
- Build a provider registry now: more general, but premature for a first real
  adapter and likely to obscure the current experiment goal.
- Expand benchmark first: useful soon, but the current metadata/repair seam is
  ready for a first real provider integration.

## 5. Configuration

Add an OpenAI-specific config object:

```python
@dataclass(frozen=True)
class OpenAIModelGatewayConfig:
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    timeout_seconds: float = 30.0
    max_output_tokens: int | None = None
```

Validation:

- `model` is required and must be a non-empty string.
- `api_key_env` must be a non-empty environment variable name.
- `timeout_seconds` must be positive.
- `max_output_tokens`, when supplied, must be a positive integer.

No default model is provided. Model choice is an experimental condition and must
be explicit.

JSON experiment config shape:

```json
{
  "model_gateway": {
    "kind": "openai",
    "model": "gpt-5.5",
    "api_key_env": "OPENAI_API_KEY",
    "timeout_seconds": 30
  }
}
```

`ModelGatewayConfig` should be extended narrowly enough to preserve existing
callers:

- existing `ModelGatewayConfig(kind="deterministic")` remains valid;
- existing scripted config remains valid;
- mapping config with `kind="openai"` must include `model`.

## 6. Runtime Construction

`build_model_gateway(...)` should support:

| kind | Result |
|---|---|
| `deterministic` | existing `DeterministicModelGateway` |
| `scripted` | existing `ScriptedModelGateway` |
| `openai` | new `OpenAIResponsesModelGateway` |

The OpenAI adapter should read the API key from the environment variable named
by `api_key_env` at construction time or first request time. If the key is
missing, it should raise a clear runtime error. It must not store the key in
ledger records or traces.

To keep tests offline, the adapter accepts an injectable client:

```python
OpenAIResponsesModelGateway(
    config=OpenAIModelGatewayConfig(model="gpt-5.5"),
    client=fake_client,
)
```

The production path constructs an OpenAI client when no client is provided.

## 7. Supported Tasks

v0.1 supports only:

| task | prompt_id | Purpose |
|---|---|---|
| `judge_evidence` | `evidence_judgment` | Convert an external signal into a BayesProbe evidence judgment. |
| `repair_evidence_judgment` | `evidence_judgment_repair` | Convert malformed judgment output into valid evidence judgment shape. |

Unknown tasks raise `ValueError`, matching the existing deterministic and
scripted gateway behavior.

## 8. Prompt And Schema Assembly

Add a small request assembly helper:

```python
def build_openai_request_payload(
    request: StructuredModelRequest,
    *,
    model: str,
    max_output_tokens: int | None = None,
) -> dict[str, Any]:
    ...
```

Responsibilities:

- choose a task-specific instruction from `request.task`;
- include `request.input` as structured context;
- include `request.task`, `prompt_id`, `prompt_version`, `schema_name`,
  `schema_version`, and safe metadata in provider metadata;
- set the explicit `model`;
- configure Structured Outputs for the `EvidenceJudgment` JSON object.

The helper should not call the network and should be unit-tested directly.

### EvidenceJudgment JSON Schema

The provider-facing schema requires an object with:

```json
{
  "evidence_type": "string",
  "likelihoods": {
    "type": "object",
    "additionalProperties": { "type": "string" }
  },
  "interpretation": "string",
  "quality_overrides": {
    "type": "object",
    "additionalProperties": { "type": "number" }
  }
}
```

The schema may enumerate the currently valid BayesProbe values:

- `EvidenceType` values;
- `LikelihoodBand` values.

Even with provider-side schema constraints, the returned object must still pass
through `evidence_judgment_from_mapping(...)`. The adapter does not become the
source of truth for evidence validity.

## 9. Response Parsing

The adapter must normalize OpenAI responses into `dict[str, Any]`.

Accepted v0.1 response forms:

- a direct `dict[str, Any]` from a fake client;
- a JSON string containing an object;
- a small fake/recorded response object exposing a text field compatible with
  Responses API test fixtures.

If parsing fails, or if the parsed value is not an object, raise
`ModelGatewayValidationError`.

Provider or network exceptions should not be converted into
`ModelGatewayValidationError`. A provider outage is not evidence schema failure.
It should fail the run visibly so experiment infrastructure can decide how to
handle it later.

## 10. Trace Metadata

The existing `ModelInvocationTrace` remains the evidence-level audit object.

The adapter can add safe provider metadata to the request metadata before
building a trace, or include provider metadata in the outgoing payload. The v0.1
safe metadata is:

```json
{
  "provider": "openai",
  "model": "<explicit model>"
}
```

Do not record:

- API key;
- raw provider response;
- hidden reasoning;
- token counts;
- cost;
- latency.

Provider observability can be added later as a separate explicit slice.

## 11. Testing Strategy

### Unit Tests

Add `tests/test_openai_gateway.py` covering:

- `OpenAIModelGatewayConfig` validation;
- request payload assembly for `judge_evidence`;
- request payload assembly for `repair_evidence_judgment`;
- unknown task rejection;
- fake client receives a payload with explicit model and structured output
  schema;
- fake dict response returns a dict;
- fake JSON string response returns a dict;
- malformed JSON response raises `ModelGatewayValidationError`;
- non-object JSON response raises `ModelGatewayValidationError`;
- provider exception propagates and is not converted into
  `ModelGatewayValidationError`.

### Config And SDK Tests

Extend existing tests so:

- `experiment_config_from_mapping(...)` parses `kind="openai"`;
- `build_model_gateway(...)` constructs an OpenAI gateway for OpenAI config;
- missing `model` under `kind="openai"` raises a clear error;
- package root exports the OpenAI config and adapter.

### Integration Smoke Test

Add an opt-in live smoke test or script.

It runs only when both are true:

```text
OPENAI_API_KEY is set
BAYESPROBE_RUN_OPENAI_LIVE=1
```

Default `pytest` must not make network calls.

The live smoke path should:

- construct `OpenAIResponsesModelGateway`;
- send one `judge_evidence` request;
- parse a dict response;
- validate it with `evidence_judgment_from_mapping(...)`.

## 12. Acceptance Criteria

- `kind="openai"` config is supported.
- `model` is required for OpenAI config.
- Existing deterministic/scripted configs are unchanged.
- OpenAI adapter implements `ModelGateway`.
- OpenAI adapter supports `judge_evidence`.
- OpenAI adapter supports `repair_evidence_judgment`.
- Payload assembly is unit-testable without network calls.
- Fake client tests prove payload shape and response parsing.
- Malformed provider output raises `ModelGatewayValidationError`.
- Provider/network exceptions propagate.
- JSON experiment config can declare OpenAI gateway.
- Public SDK exports OpenAI adapter/config.
- Default test suite performs no network calls.
- Live smoke is explicit opt-in.
- Full test suite passes.

## 13. Out-Of-Scope Future Work

- provider retry/backoff;
- token/cost/latency accounting;
- prompt template registry;
- prompt version artifact packaging;
- provider response fixture recorder;
- multi-provider registry;
- provider-side tool calling;
- streaming responses;
- model-call ledger event type;
- experiment artifact bundle with provider metadata.

## 14. Expected Progress Impact

After this slice, the overall final-goal progress should move from roughly
58%-62% to roughly 63%-67%.

The offline MVP percentage does not change much, because this slice targets
real-provider readiness rather than offline deterministic behavior.
