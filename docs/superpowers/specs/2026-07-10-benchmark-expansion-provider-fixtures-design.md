# Benchmark Expansion and Provider Fixtures Design

Date: 2026-07-10
Status: Proposed for implementation

## Context

BayesProbe already has a working engineering kernel, a toy benchmark fixture,
and real provider-backed model gateways. The next mainline step is not adding
more provider endpoints. The next step is making the methodology testable:
BayesProbe must be evaluated on both final-answer utility and belief-state
revision quality.

The current benchmark fixture proves the harness paths work, but it is too
small to evaluate methodology. It has active-only, passive-only, and mixed
samples, but it does not yet cover noisy signals, system-log signals,
multi-agent projection intake, schema-repair behavior, or provider-backed
reproducibility.

## Goal

Build a v0.2 benchmark slice that gives BayesProbe a repeatable experimental
surface:

- a richer deterministic benchmark dataset;
- recorded provider-backed fixtures that replay real-model-shaped judgments
  without network access;
- report fields that make belief-state quality visible;
- tests that prove all new fixtures run through the existing BayesProbe core.

This is still an MVP slice. It should validate the experimental direction, not
pretend to be a full HLE-scale benchmark suite.

## Non-Goals

- Do not build a ReAct/ReWOO baseline in this slice.
- Do not add a new internal agent control-flow.
- Do not require live provider calls in normal tests.
- Do not store API keys in fixtures, configs, reports, ledgers, or docs.
- Do not replace JSONL ledger storage yet.
- Do not build a full statistical significance framework yet.

## Approach Options

### Option A: Dataset-Only Expansion

Add more deterministic benchmark samples to `fixtures/benchmarks`.

Pros:

- Fastest to implement.
- Low risk.
- Exercises active/passive/mixed paths.

Cons:

- Does not prove provider-backed reproducibility.
- Does not answer whether real model judgment behavior is stable enough.
- Still weak on methodology validation.

### Option B: Recorded Provider Fixture First

Create a replayable model gateway fixture from provider-shaped requests and
responses, then add a small benchmark using it.

Pros:

- Establishes offline reproducibility for provider-backed experiments.
- Protects against live provider drift.
- Gives prompt/schema behavior a testable contract.

Cons:

- Needs one new fixture format.
- Does not by itself broaden benchmark coverage much.

### Option C: Combined Thin Slice

Add both a richer deterministic v0.2 dataset and a minimal recorded provider
fixture adapter.

Pros:

- Directly supports methodology validation.
- Keeps live provider calls optional.
- Produces useful reports immediately.
- Preserves the existing core and harness seams.

Cons:

- Slightly more implementation work than either single path.
- Requires careful scope control.

Recommendation: Option C.

## Design

### 1. Benchmark Dataset v0.2

Create a new fixture:

```text
fixtures/benchmarks/bayesprobe_v0_2_methodology.json
```

The dataset should include 8-10 samples, grouped by scenario:

- active-only factual probe;
- passive-only expert correction;
- active-plus-passive conflict;
- noisy external signal;
- system-log signal;
- multi-agent projection signal;
- schema-failure/repair fixture;
- ambiguous evidence where belief should remain cautious.

The dataset remains compatible with existing `BenchmarkSample` and
`BenchmarkSignal` objects. Any additional metadata should live in dataset
`metadata` or sample-level optional fields only if the loader is extended in a
tested way.

### 2. Belief-State Quality Metrics

The existing benchmark result already records:

- final correctness;
- update direction accuracy;
- cycle count;
- active/passive signal counts;
- evidence event count;
- belief update count;
- projection kind.

This slice should add a small quality summary without overfitting:

- `discarded_evidence_count`;
- `schema_violation_count`;
- `dominant_hypothesis_margin`;
- `belief_revision_efficiency`.

Definitions:

- `discarded_evidence_count`: evidence events that did not produce belief
  updates because quality or schema failed.
- `schema_violation_count`: evidence events whose evidence type is
  `schema_violation`.
- `dominant_hypothesis_margin`: top posterior minus second posterior at the
  final belief state.
- `belief_revision_efficiency`: belief update count divided by evidence event
  count, or `0.0` when there are no evidence events.

These fields should be stored on `BenchmarkSampleResult` and written into JSON
reports. They should not change the core update rules.

### 3. Recorded Provider Fixture

Add a fixture-backed provider gateway for offline reproducibility:

```python
RecordedModelGateway.from_json(path: str | Path) -> RecordedModelGateway
```

Fixture shape:

```json
{
  "fixture_name": "deepseek_chat_evidence_v0_1",
  "metadata": {
    "provider_kind": "openai_chat_completions",
    "model": "deepseek-v4-flash",
    "recorded_at": "2026-07-10",
    "notes": "No API key or raw provider request headers are stored."
  },
  "responses": [
    {
      "match": {
        "task": "judge_evidence",
        "signal_id": "S_chem_constant_volume"
      },
      "response": {
        "evidence_type": "supporting",
        "likelihoods": {
          "H1": "moderately_confirming",
          "H2": "moderately_disconfirming"
        },
        "interpretation": "The recorded provider judgment supports H1 under constant volume.",
        "quality_overrides": {}
      }
    }
  ]
}
```

The adapter should match by `task` and selected input fields, starting with
`signal_id`. If no entry matches, it should raise a clear validation error. It
must expose `adapter_kind = "recorded"`, store requests for tests, and return
plain structured payloads through the existing Evidence Integration Gate.

### 4. Experiment Runner Integration

The immediate integration path is config-driven:

```json
{
  "model_gateway": {
    "kind": "recorded",
    "fixture_path": "fixtures/providers/deepseek_chat_evidence_v0_1.json"
  }
}
```

`build_model_gateway(...)` should construct the recorded gateway from this
config. Artifact snapshots should include `kind` and `fixture_path`, but no
provider credentials.

### 5. Live Provider Capture Boundary

This slice does not need an automated live capture command. Capturing live
provider responses can come later. For now, recorded fixtures may be authored
from known-good live outputs and committed only after:

- the response contains no API key;
- the response contains no raw Authorization header;
- the response validates as `EvidenceJudgment`;
- a test proves the fixture runs offline.

### 6. Testing Strategy

Add tests in layers:

- `tests/test_recorded_model_gateway.py` for fixture loading, matching, request
  recording, validation errors, and no-key fixture safety.
- `tests/test_model_gateway.py` for `kind="recorded"` factory wiring.
- `tests/test_public_api_and_config.py` for JSON experiment config parsing.
- `tests/test_experiment_artifacts.py` for sanitized artifact snapshots.
- `tests/test_benchmark_io.py` for loading the v0.2 dataset.
- `tests/test_benchmark_harness.py` for new quality metrics.
- `tests/test_experiment_runner.py` for running the v0.2 dataset with recorded
  provider fixture.

Normal tests must remain offline and deterministic.

## Acceptance Criteria

- `fixtures/benchmarks/bayesprobe_v0_2_methodology.json` loads successfully.
- The v0.2 dataset has at least 8 samples and covers all three signal shapes.
- The v0.2 dataset includes noisy, system-log, and multi-agent projection style
  passive signals.
- `RecordedModelGateway.from_json(...)` can replay provider-shaped judgments
  offline.
- `kind="recorded"` works through `build_model_gateway(...)` and experiment
  config.
- Benchmark reports include the added belief-quality summary fields.
- Artifact snapshots record recorded fixture metadata without secrets.
- Full tests pass without network access.

## Direction Check

This work reinforces the BayesProbe paradigm instead of drifting toward another
agent architecture:

- external provider output remains model-shaped judgment, not control flow;
- recorded provider data enters only through `ModelGateway`;
- passive multi-agent projections remain signals, not imported belief states;
- final answers remain evaluated, while belief-state quality becomes more
  observable.
