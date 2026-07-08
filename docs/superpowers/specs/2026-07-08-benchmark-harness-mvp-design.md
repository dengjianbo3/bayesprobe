# Benchmark Harness MVP Design

Date: 2026-07-08
Status: Approved by continuation from v0.2 engineering and benchmark roadmap

## Goal

Build a minimal benchmark harness that can run BayesProbe on small, deterministic signal-stream samples and report basic dual-objective metrics. This slice proves the current engineering core can be evaluated as an experiment target before adding real FEVER/PubMedQA datasets, LLM evaluators, or external retrieval.

## Scope

The MVP covers three cycle shapes:

- `active_only`: run through `AutonomousQuestionRunner`.
- `passive_only`: initialize a run, pass benchmark-provided passive signals through `SynchronizedController`, and score the resulting belief-state projection.
- `active_plus_passive`: initialize a run, plan and execute one active probe, combine returned active signals with benchmark passive signals, and integrate the mixed cycle through `BayesProbeCore`.

The MVP reports:

- final best-hypothesis correctness against a sample gold hypothesis id.
- update-direction accuracy against optional gold update directions.
- per-sample counts for cycles, signals, evidence events, belief updates, and projections.
- suite-level accuracy and update-direction accuracy.

## Non-Goals

- No real FEVER/PubMedQA ingestion.
- No web search, document retrieval, or model calls.
- No baseline implementations for ReAct/ReWOO/Direct Answer.
- No human evaluation rubric execution.
- No statistical significance testing.
- No projection decomposition scoring beyond preserving projection-as-signal behavior through the existing core.

## Public API

Create `bayesprobe/benchmark.py` with these public types:

- `BenchmarkSample`
- `BenchmarkSignal`
- `BenchmarkSignalShape`
- `BenchmarkSampleResult`
- `BenchmarkSuiteResult`
- `BenchmarkHarness`

`BenchmarkHarness.run_sample(sample)` returns a `BenchmarkSampleResult`.

`BenchmarkHarness.run_suite(samples)` returns a `BenchmarkSuiteResult`.

## Data Flow

### Active-Only

```text
BenchmarkSample
→ InitializeRunInput
→ AutonomousQuestionRunner.run_question
→ final BeliefState + AnswerProjection
→ BenchmarkSampleResult
```

### Passive-Only

```text
BenchmarkSample
→ BayesProbeInitializer.initialize
→ benchmark passive ExternalSignals
→ SynchronizedController.process_round
→ final BeliefState + BeliefStateProjection
→ BenchmarkSampleResult
```

### Active-Plus-Passive

```text
BenchmarkSample
→ BayesProbeInitializer.initialize
→ ProbePlanner.design_probe_set
→ ProbeExecutor.execute_probe_set
→ benchmark passive ExternalSignals
→ CycleRecord(signal_shape=ACTIVE_PLUS_PASSIVE)
→ BayesProbeCore.integrate_cycle
→ AnswerProjection
→ BenchmarkSampleResult
```

The harness must not create `EvidenceEvent`s, update posterior values, or evolve hypotheses directly.

## Scoring Rules

`final_correct` is true when the top posterior hypothesis id in the final belief state equals `gold_best_hypothesis`.

`update_direction_accuracy` is computed over `gold_update_directions`, keyed by hypothesis id. A hypothesis is correct if any belief update for that hypothesis in the run has the expected direction. If a sample has no gold update directions, its update-direction accuracy is `None`.

Suite-level `final_accuracy` is the mean of `final_correct`.

Suite-level `update_direction_accuracy` is the mean over samples that have a non-`None` update-direction accuracy.

## Validation

`BenchmarkSample` rejects:

- empty `sample_id`
- empty `question_or_claim`
- empty `gold_best_hypothesis`
- passive-only samples without passive signals
- active-plus-passive samples without passive signals
- unknown signal shapes

## Test Strategy

Add `tests/test_benchmark_harness.py` covering:

- active-only samples run through the autonomous question runner and score final correctness.
- passive-only samples run through the synchronized controller and produce a belief-state projection.
- active-plus-passive samples integrate both active and passive signals into the same mixed cycle.
- suite aggregation computes sample count, final accuracy, and update-direction accuracy.
- invalid samples raise `ValueError`.

Run focused tests first, then full pytest.
