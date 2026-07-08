# Evidence Integration Gate v0.2 Design

Date: 2026-07-08
Status: Proposed design; awaiting review before implementation plan

## Goal

Upgrade the deterministic MVP Evidence Integration Gate so BayesProbe can treat passive projections, cited source claims, low-quality signals, duplicate signals, and authority-biased signals in a way that better matches the BayesProbe paradigm.

The change keeps the public `BayesProbeCore.integrate_cycle(...)` call signature stable. Controllers and runners still pass `CycleRecord`, prior `BeliefState`, `ProbeSet`, and raw `ExternalSignal`s. The core remains the only owner of evidence construction, signal quality assessment, likelihood judgment, posterior updates, and hypothesis evolution.

## Why This Comes Next

Autonomous runner, synchronized runner, and benchmark harness now work. The current bottleneck is inside `EvidenceIntegrationGate`: it still relies on simple keyword rules and treats `external_agent_projection` too shallowly. If we expand benchmarks before strengthening this gate, the experiments mostly measure toy keyword behavior rather than BayesProbe's signal-grounded belief revision mechanism.

## Options Considered

### Option A: Add Rules Directly In `core.py`

This is the smallest patch. It keeps all logic inside `EvidenceIntegrationGate._build_evidence_event`.

Tradeoff: quick but makes `core.py` harder to understand and makes quality/decomposition difficult to test in isolation.

### Option B: Extract `evidence.py` But Keep Event-Only Output

Move the gate and helpers into `bayesprobe/evidence.py`, but keep returning only `list[EvidenceEvent]`.

Tradeoff: cleaner code, but Projection Decomposition cannot expose direct verification probe candidates except by hiding them in event text.

### Option C: Extract `evidence.py` And Return Integration Result

Create an internal `EvidenceIntegrationResult` containing:

- `evidence_events`
- `probe_candidates`

`BayesProbeCore.integrate_cycle(...)` normalizes the gate output, so old test gates returning `list[EvidenceEvent]` still work. `CycleResult` gains a `probe_candidates` field, defaulting to `[]`.

This is the recommended path. It is still small, but it makes source verification candidates first-class and keeps future benchmark metrics possible.

## Scope

Implement deterministic v0.2 behavior for:

- Projection-as-Signal Rule.
- Projection Decomposition Rule.
- Signal quality assessment.
- duplicate-source and duplicate-content downweighting within a cycle.
- generated verification `ProbeCandidate`s from cited source claims.
- ledger refs and optional ledger writes for generated probe candidates.

## Non-Goals

- No LLM evidence builder.
- No real citation parser.
- No cross-run source memory or global duplicate database.
- No full natural-language decomposition.
- No runner-level automatic use of core-generated probe candidates in this slice.
- No changes to `BayesProbeCore.integrate_cycle(...)` arguments.
- No new external dependency.

## Public And Internal API Changes

### New Module

Create `bayesprobe/evidence.py`.

It owns:

- `EvidenceIntegrationGate`
- `EvidenceIntegrationResult`
- `SignalQualityAssessor`
- `ProjectionDecomposer`

`bayesprobe/core.py` imports and re-exports `EvidenceIntegrationGate` so existing tests and imports continue to work.

### `EvidenceIntegrationResult`

```python
@dataclass(frozen=True)
class EvidenceIntegrationResult:
    evidence_events: list[EvidenceEvent]
    probe_candidates: list[ProbeCandidate]
```

### `CycleResult`

Add:

```python
probe_candidates: list[ProbeCandidate] = field(default_factory=list)
```

Existing callers that ignore this field remain valid.

### Core Normalization

`BayesProbeCore.integrate_cycle(...)` accepts either return shape from the gate:

- `list[EvidenceEvent]` for legacy/custom test gates.
- `EvidenceIntegrationResult` for v0.2 gate.

The normalized events feed `solve_updates(...)`.

The normalized probe candidates are:

- added to `CycleResult.probe_candidates`.
- appended to `belief_state.ledger_refs["probe_candidates"]`.
- written to ledger as `probe_candidate` records when a ledger exists.

## Projection-as-Signal Rule

For `ExternalSignal.source_type == "external_agent_projection"`:

- The sender's belief is never accepted as direct evidence.
- The gate creates a `SENDER_JUDGMENT` evidence event.
- The sender judgment uses weaker quality scores than direct benchmark/document/tool evidence.
- The likelihood direction is weak and based on the sender's stated best hypothesis, not every hypothesis id mentioned in cited-source text.

Example:

```text
Agent A believes H2 because Source X refutes H1.
```

Expected decomposition:

- sender judgment: weak support for H2.
- source claim: neutral evidence event recording that a cited source was claimed.
- verification probe candidate: source-tracing probe to verify Source X directly.

## Projection Decomposition Rule

When an external projection contains source-citation cues such as:

- `because`
- `source`
- `cites`
- `according to`
- `passage`
- `paper`
- `evidence`

The gate creates a second `SOURCE_CLAIM` event.

`SOURCE_CLAIM` events are not direct support/counterevidence in v0.2. Their likelihoods are neutral by default. They exist to:

- preserve the audit trail.
- avoid double-counting the sender's authority and the claimed source.
- generate a direct verification `ProbeCandidate`.

## Generated Verification Probe Candidate

For each decomposed `SOURCE_CLAIM`, generate one `ProbeCandidate`:

- `source="passive_signal"`
- `candidate_id=f"pc_{event.id}_verify_source"`
- `candidate_probe.id=f"P_{event.id}_verify_source"`
- `candidate_probe.cycle_id=cycle.cycle_id`
- `candidate_probe.method="source_tracing"`
- `candidate_probe.target_hypotheses=event.target_hypotheses`
- `candidate_probe.inquiry_goal="Verify the cited source behind external projection ..."`
- `expected_information_gain=0.75`
- `decision_relevance=0.85`
- `cost_estimate=0.45`
- `priority=0.8`

The probe candidate does not execute automatically. Runners may consume it in later slices.

## Signal Quality Assessment

Create `SignalQualityAssessor.assess(...)` returning the six existing quality fields:

- reliability
- independence
- relevance
- novelty
- specificity
- verifiability

Initial deterministic rules:

### Direct Active Or Benchmark Signals

Signals from active probes, benchmark streams, deterministic tool gateways, direct documents, or simulations receive the current strong-ish default:

- reliability `0.8`
- independence `0.8`
- relevance `0.9`
- novelty `0.8`
- specificity `0.7`
- verifiability `0.7`

### External Agent Projection

Sender judgment quality:

- reliability `0.55`
- independence `0.45`
- relevance `0.75`
- novelty `0.6`
- specificity `0.6`
- verifiability `0.4`

This prevents authority bias from producing the same update strength as direct evidence.

### Source Claim

Claimed source event quality:

- reliability `0.5`
- independence `0.5`
- relevance `0.7`
- novelty `0.7`
- specificity `0.6`
- verifiability `0.65`

Likelihoods remain neutral in v0.2.

### Low Reliability Text

If raw content contains cues such as `rumor`, `unverified`, `hearsay`, `maybe`, or `unclear`, cap reliability at `0.35` and verifiability at `0.4`.

### Duplicate Within Cycle

Within the same cycle, if two normalized signals have the same `(source, raw_content)` or the same `source` and highly similar content cue, downweight later events:

- independence `0.25`
- novelty `0.25`

They may still become EvidenceEvents, but update strength is lower because `solve_updates(...)` already weights likelihoods by quality.

## Likelihood Judgment Rules

Keep existing deterministic interpretation for direct signals:

- `REFUTES` or `CONTRADICTS`: `COUNTEREVIDENCE`
  - H1 moderately disconfirming
  - H2 moderately confirming
- `SUPPORTS`: `SUPPORTING`
  - H1 moderately confirming
  - H2 moderately disconfirming
- `ANOMALY`: `ANOMALY`
  - all targets moderately disconfirming

For external projections:

- Sender judgment event type is `SENDER_JUDGMENT`.
- Prefer explicit patterns like `believes H2`, `best hypothesis H2`, or `current_best_hypothesis H2`.
- If a sender explicitly endorses one hypothesis, that hypothesis is `WEAKLY_CONFIRMING`.
- Other hypotheses remain neutral unless the sender explicitly rejects them outside cited-source text.

For source claims:

- Event type is `SOURCE_CLAIM`.
- All target likelihoods are neutral in v0.2.

## IDs

When one signal decomposes into multiple events, event ids use a suffix:

- first event: `{scoped_cycle_key}_E{index}`
- second event from same signal: `{scoped_cycle_key}_E{index}_source`

Generated probe candidates derive from the source-claim event id.

## Ledger Policy

Core ledger writes remain centralized in `_append_ledger_records`.

Add generated `probe_candidate` records after hypothesis evolutions and before final `belief_state`.

Update `BeliefState.ledger_refs` with:

- `"probe_candidates": [candidate ids generated this cycle]`

Existing `probe_sets`, `evidence_events`, `belief_updates`, and `hypothesis_evolutions` refs stay unchanged.

## Backward Compatibility

- `BayesProbeCore.integrate_cycle(...)` call signature does not change.
- Existing runners/controllers do not need immediate changes.
- Existing custom gates returning `list[EvidenceEvent]` continue to work through core normalization.
- Existing tests that ignore `CycleResult.probe_candidates` remain valid.
- `EvidenceIntegrationGate` remains importable from `bayesprobe.core`.

## Tests

Add or update tests for:

- external projection with cited source decomposes into `SENDER_JUDGMENT` and `SOURCE_CLAIM`.
- sender judgment weakly supports only the endorsed hypothesis.
- source claim has neutral likelihoods and generates a verification probe candidate.
- direct benchmark `REFUTES` still produces counterevidence and strong enough update.
- low reliability text caps reliability/verifiability.
- duplicate signal downweights later event independence/novelty.
- generated probe candidates appear in `CycleResult.probe_candidates`.
- generated probe candidates are written to ledger and `BeliefState.ledger_refs`.
- legacy gate subclass returning a plain list still works.

## Implementation Boundaries

This slice improves internal evidence semantics. It should not update:

- autonomous runner candidate pooling.
- synchronized runner candidate pooling.
- benchmark harness schema.
- projection generation text.

Those can consume `CycleResult.probe_candidates` in a later slice once the core behavior is stable.
