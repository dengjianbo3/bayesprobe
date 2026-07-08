# ModelGateway Configuration Contract v0.1 Design

## Goal

Make the existing `ModelGateway` seam externally configurable and experiment-friendly without adding network calls or real model providers.

This slice should let tests, benchmark runs, and external Python callers choose between deterministic and scripted evidence judgment while preserving BayesProbe's current default behavior.

## Context

BayesProbe now has a structured judgment seam:

- `StructuredModelRequest`
- `ModelGateway`
- `EvidenceJudgment`
- `DeterministicModelGateway`
- `ScriptedModelGateway`

`EvidenceIntegrationGate` already routes direct non-projection signals through this seam. The missing engineering capability is configuration and propagation: benchmark and experiment entrypoints still create `BayesProbeCore()` without a model gateway choice, so external users cannot configure the seam through the supported SDK path.

This matters for BayesProbe's engineering goal because benchmark reproducibility depends on explicit model/gateway settings. It also keeps the paradigm clean: model judgment is an adapter behind BayesProbe's own evidence gate, not a replacement control flow.

## Non-Goals

- No OpenAI, Anthropic, local LLM, or other live model adapter.
- No network calls.
- No prompt templates.
- No schema-repair retry loop.
- No cost, token, latency, or rate-limit tracking.
- No changes to belief update math.
- No changes to projection decomposition.
- No changes to Hypothesis Evolution Engine.
- No dynamic plugin registry.

## Design

### ModelGatewayConfig

Add a small configuration object in `bayesprobe/model_gateway.py`:

```python
@dataclass(frozen=True)
class ModelGatewayConfig:
    kind: str = "deterministic"
    responses: dict[str, dict[str, Any]] | None = None
```

Supported `kind` values:

- `"deterministic"`: builds `DeterministicModelGateway()`.
- `"scripted"`: builds `ScriptedModelGateway(responses=responses)`.

For `"scripted"`, `responses` is required and must be a mapping from task name to structured response payload.

Unknown kinds raise `ValueError`.

### Gateway Factory

Add:

```python
def build_model_gateway(config: ModelGatewayConfig | Mapping[str, Any] | None = None) -> ModelGateway:
    ...
```

Behavior:

- `None` means deterministic.
- `ModelGatewayConfig(kind="deterministic")` means deterministic.
- `{"kind": "deterministic"}` means deterministic.
- `{"kind": "scripted", "responses": {...}}` means scripted.
- Invalid mapping types raise `ValueError` with focused messages.

The factory returns the `ModelGateway` protocol type. Callers should not need to know concrete gateway classes unless they are writing tests or fixtures.

### Core Injection

Update `BayesProbeCore`:

```python
class BayesProbeCore:
    def __init__(
        self,
        ledger: JsonlLedgerStore | None = None,
        model_gateway: ModelGateway | None = None,
    ) -> None:
        ...
```

`_create_evidence_integration_gate()` passes the model gateway to `EvidenceIntegrationGate`.

Default behavior remains unchanged because `None` resolves to the deterministic gateway already used by the evidence gate.

### Benchmark Harness Injection

Update `BenchmarkHarness`:

```python
class BenchmarkHarness:
    def __init__(
        self,
        *,
        core: BayesProbeCore | None = None,
        ledger: JsonlLedgerStore | None = None,
        model_gateway: ModelGateway | None = None,
        max_cycles: int = 1,
        max_probes_per_cycle: int = 1,
    ) -> None:
        ...
```

If `core` is provided, the harness uses it as-is. If `core` is not provided, the harness creates:

```python
BayesProbeCore(ledger=ledger, model_gateway=model_gateway)
```

This preserves existing tests that pass a core directly while enabling supported gateway configuration for benchmark entrypoints.

### Experiment Config

Extend `ExperimentRunConfig`:

```python
@dataclass(frozen=True)
class ExperimentRunConfig:
    ...
    model_gateway: ModelGatewayConfig | Mapping[str, Any] | None = None
```

`run_benchmark_experiment(config)` builds a gateway with:

```python
gateway = build_model_gateway(config.model_gateway)
```

and passes it to `BenchmarkHarness`.

### JSON Config Parsing

Extend `experiment_config_from_mapping(...)` to accept optional:

```json
{
  "model_gateway": {
    "kind": "scripted",
    "responses": {
      "judge_evidence": {
        "evidence_type": "boundary_condition",
        "likelihoods": {
          "H1": "weakly_disconfirming",
          "H2": "neutral"
        },
        "interpretation": "Scripted benchmark judgment.",
        "quality_overrides": {
          "reliability": 0.62
        }
      }
    }
  }
}
```

If `model_gateway` is missing or null, deterministic behavior is used.

If `model_gateway` is not an object, raise `ValueError("experiment config field model_gateway must be an object")`.

## Data Flow

Default benchmark run:

```text
ExperimentRunConfig(model_gateway=None)
→ build_model_gateway(None)
→ DeterministicModelGateway
→ BenchmarkHarness(model_gateway=...)
→ BayesProbeCore(model_gateway=...)
→ EvidenceIntegrationGate(model_gateway=...)
→ EvidenceEvent
```

Scripted benchmark run:

```text
JSON config model_gateway.kind = "scripted"
→ ModelGatewayConfig(kind="scripted", responses={...})
→ ScriptedModelGateway
→ same BayesProbe control flow
```

## Error Handling

- Unknown gateway kind raises `ValueError("unsupported model gateway kind: <kind>")`.
- Scripted gateway without responses raises `ValueError("scripted model gateway requires responses")`.
- Non-object JSON `model_gateway` raises `ValueError("experiment config field model_gateway must be an object")`.
- Non-object `responses` raises `ValueError("model gateway responses must be an object")`.

This slice deliberately does not swallow gateway errors. If a scripted gateway lacks a response for a task, the existing `ScriptedModelGateway` raises `ValueError`. That is desirable for fixtures because missing scripted judgments should fail loudly.

## Testing

Add focused tests for:

- `build_model_gateway(None)` returns a deterministic gateway.
- `build_model_gateway({"kind": "deterministic"})` preserves deterministic judgment.
- `build_model_gateway({"kind": "scripted", "responses": ...})` returns a scripted gateway that records requests and returns configured judgment.
- invalid kind raises `ValueError`.
- scripted without responses raises `ValueError`.
- `BayesProbeCore(model_gateway=...)` propagates scripted judgment to `EvidenceIntegrationGate`.
- `BenchmarkHarness(model_gateway=...)` affects benchmark evidence judgment without requiring a custom core.
- `ExperimentRunConfig(model_gateway=...)` and JSON config parsing pass scripted judgment into benchmark runs.
- public SDK exports `ModelGatewayConfig` and `build_model_gateway`.

Full regression must still pass with deterministic defaults.

## Acceptance Criteria

- Existing callers that do not pass `model_gateway` keep current behavior.
- External Python callers can import `ModelGatewayConfig` and `build_model_gateway` from `bayesprobe`.
- Benchmark/experiment runs can choose deterministic or scripted gateway through supported config.
- Scripted benchmark config can force a non-keyword judgment such as `boundary_condition`.
- No network dependency is introduced.
- Full pytest suite passes.

## Future Work

This spec creates the configuration path for future real model adapters. A later slice can add a provider-backed adapter with prompt versioning, schema repair, and reproducibility metadata. That adapter should satisfy the same `ModelGateway` interface rather than changing BayesProbe core control flow.
