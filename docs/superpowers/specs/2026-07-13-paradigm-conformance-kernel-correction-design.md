# BayesProbe Paradigm-Conformance Kernel Correction

Date: 2026-07-13

Status: Written design for user review

Scope: P0 methodology-conformance correction for the runnable MVP

## 1. Purpose

This design corrects implementation behavior that causes BayesProbe to amplify
an initial model preference instead of revising belief from genuinely new or
revised information.

The goal is not to turn BayesProbe into a complete general-purpose agent. The
goal is to make the current MVP faithful enough to the original paradigm that a
benchmark measures BayesProbe rather than an anchored self-consistency loop.

The frozen atomic paradigm remains:

```text
Belief State_t
-> Probe
-> Signal
-> Evidence
-> Belief Update
-> Belief State_t+1
```

There is no second path by which model reasoning, tool output, a projection, or
an answer can affect posterior belief.

## 2. Decision Summary

The correction uses **root contribution reconciliation**:

- every accepted Evidence Event belongs to one canonical Evidence Root;
- an Evidence Root owns one current likelihood contribution to the Belief
  State;
- a later assessment from the same root may strengthen, weaken, reverse, or
  retract that root's current position;
- the Belief Solver applies only the difference between the root's new and old
  contributions;
- separate roots may accumulate only when provenance policy considers them
  independent;
- repeated text or repeated conclusions from one root cannot compound merely
  because they arrived in different cycles.

For model-only reasoning, one provider/model/run session is one
`ModelReasoningRoot`. Later reasoning revises that root; it does not create a
new independent root per Probe or cycle.

## 3. Why This Is a P0 Correction

The current HLE pilot exposed a characteristic failure:

- cycle-one accuracy on 77 completed paired cases was 22.1 percent;
- final cycle accuracy was 18.2 percent;
- only 4 of 60 initially wrong cases were corrected;
- 7 of 17 initially correct cases degraded to wrong;
- 50 of 51 stable-wrong cases became more confident;
- three cycle-10 retries preserved the wrong answer while posterior rose to
  0.9892, 0.8305, and 0.7226.

The observed posterior trajectory is reproduced exactly by the configured
likelihood update formula. The numerical solver is therefore behaving as
implemented. The failure lies in what the implementation authorizes as fresh
evidence.

The current implementation allows this loop:

```text
small first-cycle preference
-> posterior-aware Probe execution
-> same model produces supporting text
-> posterior-aware Evidence judgment
-> same correlation group receives more directional credit
-> preference becomes stronger
```

This violates the intended rule that low-independence information must not be
repeatedly weighted.

## 4. Frozen Paradigm Commitments

The correction must preserve all of the following:

1. Belief State is the primitive runtime state.
2. Probe is the agent's hypothesis-conditioned inquiry plan.
3. Every raw return is a Signal before it can influence belief.
4. Active and Passive Signals use one Evidence Integration Gate.
5. Evidence is a quality- and provenance-assessed interpretation of Signal.
6. Only Evidence can authorize a Belief Update.
7. Hypotheses remain revisable and falsifiable.
8. Answer Projection remains downstream of Belief State.
9. Autonomous and Synchronized regimes share the same epistemic core.
10. Model inference does not receive an alternative update channel.

`ModelReasoningSignal` remains a Signal. It is a raw return from a reasoning
Probe and is external to the current Belief State, but it is not independent
external-world evidence.

## 5. Current Architecture Divergence

### 5.1 Probe execution receives the answer anchor

`ProbeExecutionContext` currently contains the complete `BeliefState`, and the
model execution request includes each hypothesis posterior. The executor is
therefore not merely carrying out an inquiry; it is told which answer currently
leads.

### 5.2 Evidence judgment receives the answer anchor

The Evidence Judge currently receives:

- hypothesis posteriors;
- Probe support and weaken conditions;
- directional correlation-credit state;
- accepted evidence count.

This makes semantic judgment conditional on the current winner and leaks
server-side update policy into the model's assessment.

### 5.3 Correlation is capped accumulation, not revision

`EvidenceMemoryManager` marks different text from the same provider/session as
`correlated_novel`. It then permits repeated additive movement until a
per-hypothesis, per-direction effective-weight cap is consumed.

The cap prevents mathematically unbounded movement but still lets one model
session turn repeated conclusions into a large Bayes factor. Existing tests
encode and protect this behavior, so this is a contract error rather than an
isolated defect.

### 5.4 Evidence Event and update authorization are conflated

`EvidenceEvent` currently combines:

- semantic interpretation;
- quality dimensions;
- provenance and correlation status;
- likelihood bands;
- final effective update weight.

This makes the model-assisted semantic result appear to be the update
authorization itself. The missing concept is the current contribution owned by
an Evidence Root and the delta created by revising it.

### 5.5 Probe selection mistakes targeting for falsification

The planner considers a Probe to attack the top hypothesis whenever the top
hypothesis appears in `target_hypotheses`. A Probe that asks for general support
for every answer therefore satisfies the current rule even when it does not
attempt to falsify the leader.

### 5.6 Stopping observes posterior without observing information

Confidence and posterior-stability stop conditions do not require a new
Evidence Root, a nonzero root revision, or an executed falsification Probe.
Consequently, confidence created by correlated repetition appears to be valid
epistemic progress.

## 6. Alternatives Considered

### 6.1 Root contribution reconciliation, selected

Each root stores one current log-likelihood contribution vector. A later event
from the same root replaces the vector, and the solver applies only the delta.

This preserves continuous belief revision while preventing repeated counting.
It directly represents the intended distinction between a new thought and new
independent information.

### 6.2 Zero credit after the first same-root event, rejected

This is simpler but freezes the first model opinion. A later correction from the
same reasoning process could not weaken or reverse the first contribution. It
would prevent self-amplification by also preventing genuine self-correction.

### 6.3 Full proposition and dependency graph, deferred as a non-goal

A complete graph could model claim entailment, partial dependence, premise
lineage, and logical contradiction. It would be a different engineering
milestone and is unnecessary to test the core paradigm. The MVP uses
conservative root families and replacement semantics instead.

## 7. Corrected Domain Model

### 7.1 Signal

Signal remains the sole raw-information envelope. Two properties are
orthogonal:

```text
acquisition_mode: active | passive
epistemic_origin:
  model_reasoning
  tool_result
  retrieved_source
  external_observation
  human_input
  agent_message
  derived_summary
```

Examples:

| Situation | Acquisition | Origin |
| --- | --- | --- |
| LLM executes a reasoning Probe | active | model_reasoning |
| Python computes a requested value | active | tool_result |
| Human supplies a correction | passive | human_input |
| Another agent sends a conclusion | passive | agent_message |
| A system log arrives during a cycle | passive | external_observation |

Signal does not claim truth, independence, or update eligibility.

### 7.2 Evidence Event

An Evidence Event remains the auditable semantic assessment of a Signal. It
records:

- atomic interpretation relevant to the current hypothesis frame;
- likelihood band per target hypothesis;
- quality dimensions and deterministic origin caps;
- provenance and canonical contribution-root identity;
- discard, defer, or assessment-failure status;
- the model trace used for semantic judgment, when applicable.

It does not itself decide how much new information is added to the run.

### 7.3 Evidence Root

An Evidence Root is the canonical unit of potentially independent information.
It identifies the information family whose current position can be revised but
must not be counted repeatedly.

Initial conservative root policy:

- all model reasoning from the same provider/model/run session shares one root;
- an exact source artifact or retrieved document version shares one root across
  excerpts and summaries;
- a deterministic tool result is rooted in its tool identity and canonical
  inputs;
- a derived summary inherits its parent root and adds no independent root;
- a human or agent assertion is rooted in its sender/session identity unless it
  exposes a separately verifiable source;
- any shared parent root causes the MVP to treat the result as correlated rather
  than partially independent.

The last rule is deliberately conservative and avoids a full dependency graph.

### 7.4 Root Contribution

Each Evidence Root owns one current contribution:

```text
RootContribution
  contribution_root_id
  revision
  assessment_event_ids
  epistemic_origin
  per_hypothesis_log_likelihood
  unresolved_log_likelihood | null
  active
```

Likelihood bands remain human-auditable on `EvidenceEvent`. For each Event, the
deterministic kernel multiplies the band's log-likelihood value by the product
of capped reliability, independence, relevance, and novelty. Root
reconciliation combines those per-Event candidate vectors before storing one
current root contribution.

### 7.5 Evidence Contribution Delta

The reconciliation output is:

```text
EvidenceContributionDelta
  contribution_root_id
  mode: new_root | revise_root | retract_root | no_change
  previous_contribution
  current_contribution
  per_hypothesis_delta
  unresolved_delta | null
  caused_by_event_ids
```

Only this delta enters the Belief Solver.

## 8. Corrected Core Pipeline

```text
Belief State_t
  -> Probe Designer sees belief and uncertainty
  -> Probe Selector chooses bounded Probe Set
  -> Probe Executor receives a blind execution brief
  -> Active returns and Passive arrivals become Signals
  -> Signal Collection Boundary closes
  -> Provenance Normalization
  -> LLM-assisted blind Evidence Assessment
  -> Deterministic quality caps and schema validation
  -> Evidence Root Resolution
  -> Root Contribution Reconciliation
  -> Evidence Contribution Deltas
  -> Belief Solver
  -> Frame Adequacy and Hypothesis Evolution
  -> Belief State_t+1
  -> Answer Projection or next Probe cycle
```

No step may send a model response directly to the solver. No controller may
construct a contribution delta.

## 9. Module Interfaces

### 9.1 Probe Designer

The Probe Designer may see posterior, uncertainty, Evidence Memory summaries,
and current Change-My-Mind Conditions. Its job is to decide what should be
tested, so belief-aware selection is legitimate.

### 9.2 Probe Selector

The selector remains deterministic and budget-aware. After the first integrated
cycle, when at least one valid candidate exists, one selected slot must satisfy
all of these conditions:

- `purpose == hypothesis_falsification`;
- the current top hypothesis is an explicit target;
- a non-empty weaken condition exists for that hypothesis;
- the expected observation can in principle lower its support.

Merely mentioning the top hypothesis does not qualify.

### 9.3 Probe Executor

Replace the full-state execution interface with a blind brief:

```text
ProbeExecutionBrief
  run_id
  cycle_id
  problem
  task_context
  probe
  hypothesis_descriptions
  available_capability
```

The brief excludes:

- prior and posterior values;
- current winner and ranking;
- posterior summaries;
- Evidence Memory credit or root contributions;
- Answer Projection;
- any instruction to preserve the current answer.

Hypothesis descriptions retain ids, statements, scope, predictions, and
falsifiers because the executor must understand what the Probe is testing.

### 9.4 Evidence Assessor

Natural-language Signals use an LLM-assisted structured assessment. The request
contains:

- raw Signal content;
- epistemic origin and non-directional provenance facts;
- current hypothesis statements and scopes;
- the Probe inquiry goal and observed return;
- permitted evidence types and likelihood bands.

It excludes:

- posterior and ranking;
- prior Evidence Root contribution values;
- remaining directional credit;
- final answer or correctness labels;
- Probe support/weaken templates that disclose the desired judgment direction.

The LLM returns semantic likelihood bands and interpretation. It cannot assign
posterior, numerical update weight, root identity, independence status, or
contribution mode.

Machine-readable tool facts such as exit status, test counts, and numeric output
remain deterministic source fields. The LLM may assess their relevance to a
hypothesis but may not rewrite the underlying result.

### 9.5 Evidence Root Reconciler

This is the corrected deep module at the trust seam. Its interface consumes the
prior Evidence Memory, normalized Signal, and validated Evidence Event, and
returns:

```text
ReconciliationResult
  evidence_event
  contribution_delta
  next_evidence_memory
```

It owns root resolution, same-root batch aggregation, replacement,
deterministic likelihood conversion, no-change detection, and atomic memory
transition validation.

### 9.6 Belief Solver

The Belief Solver consumes `EvidenceContributionDelta`, not raw
`EvidenceEvent` weight.

For an exclusive frame:

```text
score_new(h) = log(p_old(h)) + delta_log_likelihood(h)
posterior_new = softmax(score_new)
```

For independent hypotheses, the delta is applied in log-odds space. Existing
coverage and unresolved-mass rules remain unchanged.

The solver performs no model calls and makes no provenance decisions.

## 10. Root Reconciliation Semantics

For root `r`, let `C_old(r, h)` be the stored contribution to hypothesis `h` and
`C_new(r, h)` be the contribution derived from the latest accepted assessment.

```text
Delta(r, h) = C_new(r, h) - C_old(r, h)
```

Reconciliation is cycle-batched and order-independent. The gate first groups
all accepted Events in the closed Signal Inbox by contribution root. If one
cycle contains multiple Events from the same root, their candidate
contributions are combined with an arithmetic mean. Each candidate vector is
already quality-weighted as defined in section 7.4. The vectors are never
summed. Repeating the same conclusion therefore leaves the root position
unchanged, low-quality Events retain proportionally low influence, and
contradictory same-root conclusions pull the root position toward neutral
rather than allowing whichever Event happened to be processed last to win.

One root produces at most one `EvidenceContributionDelta` per cycle.

Cases:

| Situation | Mode | Solver effect |
| --- | --- | --- |
| New independent root | `new_root` | Add full new contribution |
| Same root, identical assessment | `no_change` | No posterior movement |
| Same root, stronger same direction | `revise_root` | Add only strength difference |
| Same root, weaker same direction | `revise_root` | Remove excess prior strength |
| Same root changes B to C | `revise_root` | Reverse old B contribution and apply C contribution |
| Root assessment invalidated | `retract_root` | Remove its stored contribution |

This preserves the meaning of continuous revision without treating cycles as
independent samples.

## 11. Frame Evolution Semantics

The correction must not let old discovery information automatically confirm a
newly created hypothesis.

- surviving hypothesis ids retain their existing root contributions;
- a new hypothesis begins with zero contribution from all pre-existing roots;
- retired hypotheses retain historical contribution records but do not
  participate in the active solver;
- a later Signal from an existing root may be reassessed against the expanded
  frame and revise that root's contribution vector;
- discovery evidence remains ineligible for immediate confirmation of the
  hypothesis it introduced.

This is sufficient for the MVP without rebuilding all past Events whenever a
frame changes.

## 12. Epistemic Progress and Stopping

Each integrated cycle produces an `EpistemicProgress` summary:

```text
new_root_count
revised_root_count
retracted_root_count
no_change_count
max_absolute_contribution_delta
falsification_probe_executed
```

The No-New-Information Rule is:

```text
if every accepted event is no_change
and no new root entered
and no hypothesis/frame revision occurred:
    posterior must remain unchanged
    autonomous run stops with epistemic_stagnation
```

Existing hard budgets remain. A confidence threshold cannot convert a
zero-information cycle into additional confidence.

## 13. State and Ledger Changes

New native runs use Evidence Memory version 3 with root contributions. The full
ledger remains append-only.

Ledger additions:

- canonical contribution-root id on each Evidence Event;
- one `evidence_contribution_delta` record per reconciliation;
- root revision number and mode;
- cycle-level `epistemic_progress` record;
- explicit blind-context metadata for model execution and evidence assessment.

Historical v0.2 HLE artifacts remain immutable and readable as historical
experiments. An in-progress v0.2 Belief State is not silently resumed under v3
semantics because its accumulated directional credit cannot be converted into a
truthful root contribution. The MVP starts a new run and records the semantic
version explicitly.

## 14. Required Conformance Tests

### 14.1 Root behavior

1. One model root weakly supporting B moves B once.
2. Ten identical or paraphrased B assessments from that root do not move B
   again.
3. A stronger B assessment moves B only by the difference from the old root
   contribution.
4. A later C assessment from the same root removes the old B position and
   applies the C position.
5. A genuinely independent tool or source root can add a second contribution.
6. A derived summary of an existing source cannot create another contribution.

### 14.2 Blind interfaces

1. Probe execution payload contains no posterior, ranking, or answer
   projection.
2. Evidence assessment payload contains no posterior, winner, directional
   credit, or correctness label.
3. Planner and projector may still receive the belief information their roles
   require.

### 14.3 Falsification

1. A generic Probe targeting all hypotheses does not satisfy the falsification
   requirement.
2. A selected falsification Probe must include a concrete weaken condition for
   the current top hypothesis.
3. The ledger records whether that Probe actually executed.

### 14.4 No-new-information behavior

1. A no-change cycle produces byte-equivalent hypothesis posteriors.
2. It records zero contribution delta.
3. Autonomous mode stops with `epistemic_stagnation` rather than spending the
   remaining cycle budget.

### 14.5 Small frozen benchmark check

Before another expensive HLE run, select exactly 30 sample ids by deterministic
hash order from the previously completed frozen 77-case paired set. Gold labels
remain hidden from runtime components. Compare cycle 1 with the actual cycle 4
state when reached, or with the terminal stopped state when epistemic
stagnation ends the run earlier. The primary acceptance criteria are process
invariants, not an accuracy claim:

- stable-wrong cases without a new root do not gain confidence;
- same-root repetition produces zero net contribution;
- first-cycle decisions can reverse when the root assessment reverses;
- every posterior movement can be attributed to a root delta;
- final accuracy and calibration are reported, but no superiority claim is
  required for architectural acceptance.

## 15. Explicit Non-Goals

This correction does not add:

- a complete proposition or knowledge graph;
- symbolic theorem proving;
- a multi-model debate protocol;
- a general search platform;
- a production tool ecosystem;
- distributed execution or durable orchestration;
- partial-correlation Bayesian networks;
- broad benchmark expansion;
- provider-specific reasoning heuristics.

It also does not tune likelihood bands to improve HLE accuracy. First the
experiment must measure the intended method.

## 16. Acceptance Definition

The P0 correction is architecturally complete when all of the following are
true:

1. The only belief-changing path remains
   `Belief State -> Probe -> Signal -> Evidence -> Update`.
2. Model reasoning remains a typed Signal and never bypasses Evidence.
3. Same-root assessments revise one current contribution rather than
   accumulating Events as independent support.
4. Executor and Evidence Assessor are blind to the current answer ranking.
5. The planner distinguishes falsification from mere targeting.
6. No-new-information cycles cannot increase confidence.
7. Ledger traces identify the exact root delta behind every posterior change.
8. A frozen replay demonstrates these invariants before HLE is resumed.

Meeting these criteria establishes implementation fidelity. It does not, by
itself, establish that BayesProbe outperforms a direct model.
