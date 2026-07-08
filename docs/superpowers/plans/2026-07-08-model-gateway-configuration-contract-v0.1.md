# ModelGateway Configuration Contract v0.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing `ModelGateway` seam configurable from public SDK, core, benchmark harness, and experiment JSON config while preserving deterministic defaults.

**Architecture:** Add a small `ModelGatewayConfig` plus `build_model_gateway(...)` factory in `bayesprobe/model_gateway.py`. Propagate an optional `ModelGateway` through `BayesProbeCore`, `BenchmarkHarness`, and `ExperimentRunConfig`; JSON config parsing converts `model_gateway` objects into the same config shape. Tests prove default behavior is unchanged and scripted gateway settings can force evidence judgment through the normal BayesProbe control flow.

**Tech Stack:** Python 3.11+, dataclasses, `collections.abc.Mapping`, existing Pydantic schemas, pytest.

## Global Constraints

- No OpenAI, Anthropic, local LLM, or other live model adapter.
- No network calls.
- No prompt templates.
- No schema-repair retry loop.
- No cost, token, latency, or rate-limit tracking.
- No changes to belief update math.
- No changes to projection decomposition.
- No changes to Hypothesis Evolution Engine.
- No dynamic plugin registry.
- Existing callers that do not pass `model_gateway` must keep current deterministic behavior.
- Do not attempt git commits because this workspace is not currently a git repository.

---

## File Structure

- Modify `tests/test_model_gateway.py`: factory and config parsing tests.
- Modify `tests/test_core_cycles.py`: `BayesProbeCore(model_gateway=...)` propagation test.
- Modify `tests/test_benchmark_harness.py`: `BenchmarkHarness(model_gateway=...)` propagation test using ledger evidence records.
- Modify `tests/test_experiment_runner.py`: `ExperimentRunConfig(model_gateway=...)` propagation test.
- Modify `tests/test_public_api_and_config.py`: JSON config parsing and public SDK export tests.
- Modify `bayesprobe/model_gateway.py`: `ModelGatewayConfig`, mapping validation, factory.
- Modify `bayesprobe/core.py`: optional model gateway injection.
- Modify `bayesprobe/benchmark.py`: optional model gateway injection.
- Modify `bayesprobe/experiment_runner.py`: config field and harness wiring.
- Modify `bayesprobe/config.py`: parse optional JSON `model_gateway` object.
- Modify `bayesprobe/__init__.py`: export `ModelGatewayConfig` and `build_model_gateway`.

### Task 1: ModelGateway Config And Factory Tests

**Files:**
- Modify: `tests/test_model_gateway.py`

**Interfaces:**
- Consumes existing: `StructuredModelRequest`, `evidence_judgment_from_mapping`
- Produces expected: `ModelGatewayConfig`, `build_model_gateway`

- [x] **Step 1: Write failing factory tests**

Update imports:

```python
from bayesprobe.model_gateway import (
    DeterministicModelGateway,
    ModelGatewayConfig,
    ScriptedModelGateway,
    StructuredModelRequest,
    build_model_gateway,
    evidence_judgment_from_mapping,
)
```

Add tests:

```python
def test_build_model_gateway_defaults_to_deterministic():
    gateway = build_model_gateway()

    judgment = evidence_judgment_from_mapping(
        gateway.complete_structured(make_request("SUPPORTS: evidence supports H1."))
    )

    assert isinstance(gateway, DeterministicModelGateway)
    assert judgment.evidence_type == EvidenceType.SUPPORTING
    assert judgment.likelihoods["H1"] == LikelihoodBand.MODERATELY_CONFIRMING


def test_build_model_gateway_accepts_deterministic_mapping():
    gateway = build_model_gateway({"kind": "deterministic"})

    judgment = evidence_judgment_from_mapping(
        gateway.complete_structured(make_request("This signal has no deterministic cue."))
    )

    assert isinstance(gateway, DeterministicModelGateway)
    assert judgment.evidence_type == EvidenceType.NEUTRAL


def test_build_model_gateway_accepts_scripted_config_and_records_requests():
    gateway = build_model_gateway(
        ModelGatewayConfig(
            kind="scripted",
            responses={
                "judge_evidence": {
                    "evidence_type": "boundary_condition",
                    "likelihoods": {"H1": "weakly_disconfirming", "H2": "neutral"},
                    "interpretation": "Configured scripted judgment.",
                    "quality_overrides": {"reliability": 0.62},
                }
            },
        )
    )

    request = make_request("No keyword cue.")
    judgment = evidence_judgment_from_mapping(gateway.complete_structured(request))

    assert isinstance(gateway, ScriptedModelGateway)
    assert gateway.requests == [request]
    assert judgment.evidence_type == EvidenceType.BOUNDARY_CONDITION
    assert judgment.likelihoods["H1"] == LikelihoodBand.WEAKLY_DISCONFIRMING
    assert judgment.quality_overrides == {"reliability": 0.62}
```

- [x] **Step 2: Write failing validation tests**

Add:

```python
def test_build_model_gateway_rejects_unknown_kind():
    with pytest.raises(ValueError, match="unsupported model gateway kind"):
        build_model_gateway({"kind": "unknown"})


def test_build_model_gateway_rejects_scripted_without_responses():
    with pytest.raises(ValueError, match="scripted model gateway requires responses"):
        build_model_gateway({"kind": "scripted"})


def test_build_model_gateway_rejects_non_object_responses():
    with pytest.raises(ValueError, match="model gateway responses must be an object"):
        build_model_gateway({"kind": "scripted", "responses": []})
```

- [x] **Step 3: Verify RED**

Run:

```bash
python3 -m pytest tests/test_model_gateway.py -q
```

Expected: failure because `ModelGatewayConfig` and `build_model_gateway` do not exist.

### Task 2: ModelGateway Config And Factory Implementation

**Files:**
- Modify: `bayesprobe/model_gateway.py`
- Test: `tests/test_model_gateway.py`

**Interfaces:**
- Produces:
  - `ModelGatewayConfig`
  - `build_model_gateway(config: ModelGatewayConfig | Mapping[str, Any] | None = None) -> ModelGateway`

- [x] **Step 1: Implement config dataclass and factory**

Add imports:

```python
from collections.abc import Mapping
```

Add:

```python
@dataclass(frozen=True)
class ModelGatewayConfig:
    kind: str = "deterministic"
    responses: dict[str, dict[str, Any]] | None = None
```

Add:

```python
def build_model_gateway(
    config: ModelGatewayConfig | Mapping[str, Any] | None = None,
) -> ModelGateway:
    gateway_config = _model_gateway_config_from_input(config)
    if gateway_config.kind == "deterministic":
        return DeterministicModelGateway()
    if gateway_config.kind == "scripted":
        if gateway_config.responses is None:
            raise ValueError("scripted model gateway requires responses")
        return ScriptedModelGateway(responses=gateway_config.responses)
    raise ValueError(f"unsupported model gateway kind: {gateway_config.kind}")
```

Add helper:

```python
def _model_gateway_config_from_input(
    config: ModelGatewayConfig | Mapping[str, Any] | None,
) -> ModelGatewayConfig:
    if config is None:
        return ModelGatewayConfig()
    if isinstance(config, ModelGatewayConfig):
        return config
    if not isinstance(config, Mapping):
        raise ValueError("model gateway config must be an object")
    kind = str(config.get("kind", "deterministic"))
    responses = config.get("responses")
    if responses is not None and not isinstance(responses, Mapping):
        raise ValueError("model gateway responses must be an object")
    return ModelGatewayConfig(
        kind=kind,
        responses=dict(responses) if responses is not None else None,
    )
```

Update `__all__` with `ModelGatewayConfig` and `build_model_gateway`.

- [x] **Step 2: Verify GREEN for gateway tests**

Run:

```bash
python3 -m pytest tests/test_model_gateway.py -q
```

Expected: all model gateway tests pass.

### Task 3: Core Gateway Propagation

**Files:**
- Modify: `tests/test_core_cycles.py`
- Modify: `bayesprobe/core.py`

**Interfaces:**
- Consumes: `ModelGateway`
- Produces: `BayesProbeCore(ledger: JsonlLedgerStore | None = None, model_gateway: ModelGateway | None = None)`

- [x] **Step 1: Write failing core propagation test**

Add:

```python
def test_core_accepts_model_gateway_for_evidence_gate():
    gateway = ScriptedModelGateway(
        responses={
            "judge_evidence": {
                "evidence_type": "boundary_condition",
                "likelihoods": {"H1": "weakly_disconfirming", "H2": "neutral"},
                "interpretation": "Core configured scripted judgment.",
            }
        }
    )
    core = BayesProbeCore(model_gateway=gateway)
    cycle = CycleRecord(
        cycle_id="cycle_core_model_gateway",
        run_id="run_1",
        cycle_index=1,
        signal_shape=CycleSignalShape.PASSIVE_ONLY,
    )

    result = core.integrate_cycle(
        cycle=cycle,
        belief_state=make_belief_state(cycle_id="cycle_core_model_gateway"),
        probe_set=ProbeSet(
            probe_set_id="ps_core_model_gateway",
            cycle_id="cycle_core_model_gateway",
            probes=[],
            selection_reason="Core gateway propagation.",
            may_be_empty=True,
        ),
        signals=[
            ExternalSignal(
                id="S_core_model_gateway",
                cycle_id="pending",
                signal_kind=SignalKind.PASSIVE,
                source_type="user_feedback",
                source="user",
                raw_content="No keyword cue.",
            )
        ],
    )

    assert result.evidence_events[0].evidence_type == EvidenceType.BOUNDARY_CONDITION
    assert gateway.requests[0].input["signal_id"] == "S_core_model_gateway"
```

- [x] **Step 2: Verify RED**

Run:

```bash
python3 -m pytest tests/test_core_cycles.py::test_core_accepts_model_gateway_for_evidence_gate -q
```

Expected: failure because `BayesProbeCore.__init__` does not accept `model_gateway`.

- [x] **Step 3: Implement core injection**

Update imports:

```python
from bayesprobe.model_gateway import ModelGateway
```

Update constructor:

```python
class BayesProbeCore:
    def __init__(
        self,
        ledger: JsonlLedgerStore | None = None,
        model_gateway: ModelGateway | None = None,
    ) -> None:
        self._ledger = ledger
        self._model_gateway = model_gateway
        ...
```

Update:

```python
def _create_evidence_integration_gate(self) -> EvidenceIntegrationGate:
    return EvidenceIntegrationGate(model_gateway=self._model_gateway)
```

- [x] **Step 4: Verify GREEN**

Run:

```bash
python3 -m pytest tests/test_core_cycles.py::test_core_accepts_model_gateway_for_evidence_gate tests/test_core_cycles.py::test_active_only_signal_updates_belief_through_evidence_gate -q
```

Expected: both tests pass.

### Task 4: Benchmark Harness Gateway Propagation

**Files:**
- Modify: `tests/test_benchmark_harness.py`
- Modify: `bayesprobe/benchmark.py`

**Interfaces:**
- Consumes: `ModelGateway`
- Produces: `BenchmarkHarness(model_gateway: ModelGateway | None = None, ...)`

- [x] **Step 1: Write failing benchmark harness test**

Add imports:

```python
from bayesprobe.model_gateway import ScriptedModelGateway
```

Add:

```python
def test_benchmark_harness_passes_model_gateway_to_created_core(tmp_path: Path):
    ledger = JsonlLedgerStore(tmp_path / "gateway-ledger.jsonl")
    gateway = ScriptedModelGateway(
        responses={
            "judge_evidence": {
                "evidence_type": "boundary_condition",
                "likelihoods": {"H1": "weakly_disconfirming", "H2": "neutral"},
                "interpretation": "Harness configured scripted judgment.",
                "quality_overrides": {"reliability": 0.62},
            }
        }
    )
    harness = BenchmarkHarness(ledger=ledger, model_gateway=gateway)
    sample = BenchmarkSample(
        sample_id="gateway_passive",
        question_or_claim="Can benchmark configure model gateway?",
        signal_shape=BenchmarkSignalShape.PASSIVE_ONLY,
        gold_best_hypothesis="H1",
        passive_signals=[
            BenchmarkSignal(
                signal_id="S_gateway_passive",
                source_type="user_feedback",
                source="user",
                raw_content="No keyword cue.",
                target_hypotheses=["H1", "H2"],
            )
        ],
    )

    harness.run_sample(sample)

    evidence_payloads = [
        record["payload"]
        for record in ledger.read_all("evidence_event")
    ]
    assert evidence_payloads[0]["evidence_type"] == "boundary_condition"
    assert evidence_payloads[0]["reliability"] == 0.62
    assert gateway.requests[0].input["signal_id"] == "S_gateway_passive"
```

- [x] **Step 2: Verify RED**

Run:

```bash
python3 -m pytest tests/test_benchmark_harness.py::test_benchmark_harness_passes_model_gateway_to_created_core -q
```

Expected: failure because `BenchmarkHarness.__init__` does not accept `model_gateway`.

- [x] **Step 3: Implement harness injection**

Update imports:

```python
from bayesprobe.model_gateway import ModelGateway
```

Update constructor:

```python
def __init__(
    self,
    *,
    core: BayesProbeCore | None = None,
    ledger: JsonlLedgerStore | None = None,
    model_gateway: ModelGateway | None = None,
    max_cycles: int = 1,
    max_probes_per_cycle: int = 1,
) -> None:
    self.core = core or BayesProbeCore(ledger=ledger, model_gateway=model_gateway)
    ...
```

- [x] **Step 4: Verify GREEN**

Run:

```bash
python3 -m pytest tests/test_benchmark_harness.py::test_benchmark_harness_passes_model_gateway_to_created_core tests/test_benchmark_harness.py -q
```

Expected: benchmark harness tests pass.

### Task 5: Experiment Config And JSON Parsing

**Files:**
- Modify: `tests/test_experiment_runner.py`
- Modify: `tests/test_public_api_and_config.py`
- Modify: `bayesprobe/experiment_runner.py`
- Modify: `bayesprobe/config.py`

**Interfaces:**
- Consumes: `ModelGatewayConfig | Mapping[str, Any] | None`
- Produces:
  - `ExperimentRunConfig(..., model_gateway: ModelGatewayConfig | Mapping[str, Any] | None = None)`
  - JSON field `model_gateway`

- [x] **Step 1: Write failing experiment runner propagation test**

In `tests/test_experiment_runner.py`, add:

```python
def test_run_benchmark_experiment_uses_model_gateway_config(tmp_path: Path):
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
                        "evidence_type": "boundary_condition",
                        "likelihoods": {"H1": "weakly_disconfirming", "H2": "neutral"},
                        "interpretation": "Experiment configured scripted judgment.",
                        "quality_overrides": {"reliability": 0.62},
                    }
                },
            },
        )
    )

    evidence_payloads = [
        record["payload"]
        for record in JsonlLedgerStore(ledger_path).read_all("evidence_event")
    ]
    assert result.ledger_path == ledger_path
    assert evidence_payloads[0]["evidence_type"] == "boundary_condition"
    assert evidence_payloads[0]["reliability"] == 0.62
```

- [x] **Step 2: Write failing JSON config parsing tests**

In `tests/test_public_api_and_config.py`, update imports to include:

```python
    ModelGatewayConfig,
    build_model_gateway,
```

Update expected public names and non-null assertions.

Add:

```python
def test_experiment_config_from_mapping_parses_model_gateway_object(tmp_path: Path):
    config = experiment_config_from_mapping(
        {
            "dataset_path": "datasets/toy.json",
            "report_path": "outputs/toy-report.json",
            "model_gateway": {
                "kind": "scripted",
                "responses": {
                    "judge_evidence": {
                        "evidence_type": "boundary_condition",
                        "likelihoods": {"H1": "weakly_disconfirming"},
                        "interpretation": "JSON configured judgment.",
                    }
                },
            },
        },
        base_dir=tmp_path,
    )

    assert isinstance(config.model_gateway, ModelGatewayConfig)
    assert config.model_gateway.kind == "scripted"
    assert config.model_gateway.responses["judge_evidence"]["evidence_type"] == "boundary_condition"
```

Add invalid config case:

```python
(
    "non_object_model_gateway.json",
    json.dumps({"dataset_path": "dataset.json", "report_path": "report.json", "model_gateway": []}),
    "experiment config field model_gateway must be an object",
),
```

- [x] **Step 3: Verify RED**

Run:

```bash
python3 -m pytest tests/test_experiment_runner.py::test_run_benchmark_experiment_uses_model_gateway_config tests/test_public_api_and_config.py::test_experiment_config_from_mapping_parses_model_gateway_object -q
```

Expected: failure because config fields and exports do not exist.

- [x] **Step 4: Implement experiment runner and JSON parsing**

In `bayesprobe/experiment_runner.py`, import:

```python
from collections.abc import Mapping
from typing import Any
from bayesprobe.model_gateway import ModelGatewayConfig, build_model_gateway
```

Update dataclass:

```python
model_gateway: ModelGatewayConfig | Mapping[str, Any] | None = None
```

Update runner:

```python
model_gateway = build_model_gateway(config.model_gateway)
harness = BenchmarkHarness(
    ledger=ledger,
    model_gateway=model_gateway,
    max_cycles=config.max_cycles,
    max_probes_per_cycle=config.max_probes_per_cycle,
)
```

In `bayesprobe/config.py`, import `ModelGatewayConfig` and add:

```python
model_gateway=_optional_model_gateway_config(data),
```

Add helper:

```python
def _optional_model_gateway_config(data: Mapping[str, Any]) -> ModelGatewayConfig | None:
    if "model_gateway" not in data or data["model_gateway"] is None:
        return None
    value = data["model_gateway"]
    if not isinstance(value, Mapping):
        raise ValueError("experiment config field model_gateway must be an object")
    kind = str(value.get("kind", "deterministic"))
    responses = value.get("responses")
    if responses is not None and not isinstance(responses, Mapping):
        raise ValueError("model gateway responses must be an object")
    return ModelGatewayConfig(
        kind=kind,
        responses=dict(responses) if responses is not None else None,
    )
```

- [x] **Step 5: Verify GREEN**

Run:

```bash
python3 -m pytest tests/test_experiment_runner.py tests/test_public_api_and_config.py -q
```

Expected: experiment runner and config tests pass.

### Task 6: Public SDK Exports And Regression Verification

**Files:**
- Modify: `bayesprobe/__init__.py`
- Test: all pytest files

**Interfaces:**
- Produces package root exports:
  - `ModelGatewayConfig`
  - `build_model_gateway`

- [x] **Step 1: Export public config names**

Update `bayesprobe/__init__.py` import list:

```python
from bayesprobe.model_gateway import (
    DeterministicModelGateway,
    EvidenceJudgment,
    ModelGateway,
    ModelGatewayConfig,
    ScriptedModelGateway,
    StructuredModelRequest,
    build_model_gateway,
    evidence_judgment_from_mapping,
)
```

Update `__all__` with `"ModelGatewayConfig"` and `"build_model_gateway"`.

- [x] **Step 2: Run focused seam tests**

Run:

```bash
python3 -m pytest tests/test_model_gateway.py tests/test_core_cycles.py tests/test_benchmark_harness.py tests/test_experiment_runner.py tests/test_public_api_and_config.py -q
```

Expected: focused gateway/config/benchmark tests pass.

- [x] **Step 3: Run full regression**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
```

Expected: all tests pass.

- [x] **Step 4: Remove generated caches**

Run:

```bash
find . -type d \( -name __pycache__ -o -name .pytest_cache \) -prune -exec rm -rf {} +
```

Expected: no generated cache directories remain.

## Self-Review

- Spec coverage: The plan covers factory/config, validation errors, core injection, benchmark harness injection, experiment config propagation, JSON config parsing, public SDK exports, focused tests, full regression, and cache cleanup.
- Placeholder scan: No unspecified implementation placeholders remain.
- Type consistency: `ModelGatewayConfig`, `build_model_gateway`, `model_gateway`, and existing gateway class names are used consistently across tests and implementation steps.
