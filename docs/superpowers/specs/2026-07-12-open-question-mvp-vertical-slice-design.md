# Open-Question MVP Vertical Slice Design

Date: 2026-07-12
Status: Design approved; written specification awaiting user review

## 1. Decision

BayesProbe will complete one contract-preserving open-question vertical slice.
The existing epistemic kernel remains authoritative for Signal admission,
Evidence integration, belief revision, frame adequacy, and bounded control. A
model may propose semantic content, but it may not bypass those decisions.

The MVP must run two kinds of open tasks end to end:

1. an explanation or design question whose answer synthesizes several
   independently testable claims; and
2. an exact-answer question for which every initial answer candidate is wrong,
   requiring the hypothesis frame to expand before a final answer is possible.

This phase stops when those paths work through the existing autonomous runner
and WebUI. It does not complete the full architecture roadmap.

## 2. Why the Current Runtime Is Not Yet an Open-Question Agent

The current runtime already provides model-backed task admission, task
framing, probe execution, Evidence integration, open-frame state, and belief
revision. Four remaining behaviors prevent a meaningful open-question run:

1. initialization emits generic per-hypothesis `source_tracing` probes instead
   of task-specific discriminative probes;
2. the autonomous runner omits Core-produced probe candidates from the next
   candidate pool;
3. deterministic anomaly evolution creates placeholder hypothesis text rather
   than substantive new answer candidates or explanatory claims; and
4. answer projection restates the highest-ranked hypothesis instead of
   satisfying the TaskFrame's Answer Contract.

The MVP corrects these four behaviors. It does not redesign the working middle
of the kernel.

## 3. Hypothesis Semantics by Task Type

BayesProbe must not force every question into a finite list of mutually
exclusive complete answers.

### 3.1 Closed Selection

For multiple-choice and explicit yes/no tasks, a hypothesis is an answer
candidate. The frame is normally `exclusive + exhaustive`, and the Answer
Projection selects one candidate.

### 3.2 Open Exact Answer

For a number or short answer without supplied choices, initial hypotheses are
provisional answer candidates. The frame is `exclusive + open`, so unresolved
alternative mass explicitly represents the possibility that every named
candidate is wrong.

If admitted Evidence disconfirms the named candidates or supports the
unresolved alternative, the deterministic Frame Adequacy Policy decides
whether expansion is required. A model may then materialize new candidates,
but the discovery signal cannot immediately confirm the candidate it caused.
A later probe must discriminate among the expanded candidates.

### 3.3 Open Synthesis

For explanation and design tasks, a hypothesis is a revisable, truth-apt claim
that contributes to an answer. Examples include a causal mechanism, a
confounder, a boundary condition, a risk, or an experimental assumption.
These hypotheses may be independently credible and may all appear in the final
answer. The Answer Projection synthesizes supported claims according to the
Answer Contract instead of selecting a single H id.

The model may generate an initial finite working frame, but that frame is a
bounded representation of current uncertainty, not a claim that the open
question has only those possible answers.

## 4. Runtime Architecture

```text
Question + Task Context + Initial Signals
  -> Task Admission
  -> Task Framing + Answer Contract
  -> Initial BeliefState and FrameState
  -> Model Probe Design
  -> Deterministic Capability Filter and Probe Selection
  -> Model-Reasoning Probe Execution
  -> ExternalSignal collection boundary
  -> Evidence Integration Gate
  -> Belief and Frame Adequacy update
  -> Authorized Semantic Hypothesis Expansion, when required
  -> Next-cycle Candidate Pool
  -> Answer-Contract Projection or another bounded cycle
```

The autonomous runner owns cycle budgets and stopping. `BayesProbeCore` remains
the only component that converts Signals into Evidence, changes belief, or
authorizes frame expansion. Projection remains downstream and cannot mutate
belief.

## 5. Components

### 5.1 ProbeDesigner

Add a small `ProbeDesigner` interface with model-backed and recorded adapters.
It receives the TaskFrame, current BeliefState, current FrameState, uncertainty
summary, and available capability descriptors. It returns semantic probe
proposals containing:

- purpose;
- target hypotheses;
- inquiry goal;
- expected observation;
- support and weakening conditions;
- frame-expansion condition;
- required capability.

The model does not assign ids, priorities, costs, or expected-information-gain
numbers. The server validates hypothesis references, assigns stable ids and
bounded numeric features, removes semantic duplicates, and produces existing
`ProbeCandidate` objects.

For open tasks, initialization requests one frame-level discriminative probe
or a small set of complementary probes. It no longer creates one generic
`source_tracing` probe for every hypothesis.

### 5.2 Minimal Capability Boundary

The WebUI MVP exposes one executable capability:

```text
model_reasoning
```

Search, browsing, retrieval, code execution, and expert consultation are not
implemented in this slice. A proposal requiring an unavailable capability is
visible as unavailable and cannot be selected or silently impersonated by
model reasoning.

This is a fixed capability set, not a general plugin registry.

### 5.3 Semantic HypothesisExpander

Add a model-backed and recorded `HypothesisExpander`. It is invoked only when
the Core's `FrameAdequacyDecision.should_expand` is true. Its input includes:

- the TaskFrame and current FrameState;
- active and retired hypotheses;
- the Core's expansion reason and triggering Evidence ids;
- compact, sanitized summaries of the triggering Evidence; and
- the remaining revision budget.

For exact-answer tasks it proposes one to three new candidates with typed
`answer_value` fields. For synthesis tasks it proposes one to three substantive
claims with scope, falsifiers, and predictions. The server assigns ids and
priors, enforces the frame revision limit, and rejects duplicates or candidates
that merely paraphrase an existing hypothesis.

New hypotheses begin as unconfirmed. The Candidate Pool receives a mandatory
follow-up discriminator for the next cycle. Placeholder statements such as
`Spawned anomaly hypothesis` are not valid model-backed expansion output.

### 5.4 Candidate Pool Wiring

The autonomous next-cycle pool merges, in deterministic order:

1. mandatory follow-up probes for newly expanded hypotheses;
2. `CycleResult.probe_candidates` from Evidence integration and evolution;
3. fresh model-designed probes for the updated uncertainty;
4. projection change-my-mind probes; and
5. unselected prior candidates.

Candidates are deduplicated by id and normalized semantic identity. The
existing ProbePlanner remains responsible for selecting at most the configured
number of probes.

Synchronized-runner parity is explicitly deferred.

### 5.5 AnswerProjector

Add a model-backed and recorded `AnswerProjector`. It receives only:

- the immutable TaskFrame and Answer Contract;
- current hypotheses and belief values;
- FrameState and remaining unresolved mass;
- admitted Evidence summaries and provenance labels; and
- the run stop reason.

For `selection`, it emits a contract-typed answer value from an active
hypothesis. For `synthesis`, it composes the supported independent claims into
the required sections. It must preserve material disagreement, uncertainty,
and limitations rather than treating the top claim as the whole answer.

The server validates output shape and permits one structured repair. The
projector cannot add Evidence, alter posterior values, create hypotheses, or
claim that model reasoning came from an external source.

If the contract cannot be satisfied, projection returns an explicit
abstention/partial result rather than the generic `Current best hypothesis is
H1` fallback.

The existing `ProjectionMode` values (`selection`, `synthesis`, and
`abstention`) identify these outcomes explicitly; the UI must not infer them
from answer text.

## 6. Provider Calls and Error Handling

The OpenAI-compatible gateway gains three structured task families:

```text
design_probes / repair_probe_design
expand_hypotheses / repair_hypothesis_expansion
project_answer / repair_answer_projection
```

Each call uses the existing request-scoped provider configuration and timeout.
API keys never enter prompts, fixtures, ledgers, progress events, or errors.
Invalid structured output receives one repair attempt; a second failure ends
the run with the failing stage named.

A provider failure does not fall back to deterministic generic probes,
placeholder hypotheses, or top-hypothesis answer text. The WebUI preserves the
last valid TaskFrame and BeliefState and shows the failed stage.

## 7. Stop and Projection Rules

All runs remain bounded by `max_cycles` and `max_probes_per_cycle`.

An exact-answer selection is final only when:

- at least one Evidence-integration cycle completed;
- Frame Adequacy is not `inadequate` or `expanding`;
- the selected hypothesis has a value matching the Answer Contract; and
- unresolved alternative mass does not outrank the selected named candidate.

Otherwise the run continues while budget and executable probes remain, then
returns abstention with the current candidates and uncertainty.

A synthesis answer is final only after at least one integrated cycle and when
no authorized frame expansion is pending. At the cycle limit it may return an
explicitly marked partial synthesis if the contract can be satisfied without
hiding unresolved uncertainty.

`no_probes` is an honest stop. It never manufactures a final answer.

## 8. WebUI Behavior

The existing provider form remains unchanged. During a real open-question run,
the progress stream adds visible stages for:

- probe design;
- capability filtering;
- frame-adequacy decision;
- hypothesis expansion; and
- answer projection.

The Belief State view distinguishes:

- answer candidates for selection tasks;
- contributing claims for synthesis tasks; and
- unresolved alternative mass for open exact-answer frames.

The answer area displays the contract-facing answer first. Hypothesis ids and
belief details remain inspectable as the reasoning trace, not as a substitute
for the answer.

## 9. MVP Acceptance Tests

### 9.1 Open Explanation/Design Fixture

Prompt: a team claims that making the model larger always improves an agent's
real-task performance; design a way to test that claim.

The recorded run must demonstrate:

- multiple distinct claims, including the causal claim and plausible
  confounders or boundary conditions;
- a discriminative experimental probe rather than mirrored support/refute
  hypotheses;
- Evidence integration and belief updates; and
- a final experimental protocol satisfying the required Answer Contract
  sections, not a restatement of one H id.

### 9.2 Open Exact-Answer Expansion Fixture

The recorded frame begins with answer candidates `1`, `2`, and `3`. Admitted
Evidence indicates that all three are wrong or that an unresolved alternative
is better supported.

The run must demonstrate:

- an open frame with unresolved alternative mass;
- a Core-authorized expansion;
- substantive new candidates such as `4` and `5` with typed values;
- a later discriminative probe that does not reuse discovery Evidence as
  confirmation; and
- sufficient recorded follow-up Evidence to select a final numeric answer from
  the expanded frame.

### 9.3 WebUI Acceptance

Using a user-supplied OpenAI-compatible base URL, model, and API key, the user
can submit an unseeded open question and observe framing, probe design, Signal,
Evidence judgment, belief update, optional expansion, and final projection as
they occur.

Provider-specific live results are a manual smoke test. Secret-free recorded
fixtures provide the repeatable automated contract.

## 10. Test Boundary

Implementation is complete for this MVP when:

- focused unit tests cover each new adapter and validation boundary;
- both recorded vertical slices pass end to end;
- existing Python and WebUI tests pass without semantic regressions;
- a manual real-provider WebUI run completes for the explanation/design case;
  and
- the answer shown to the user is meaningful without reading internal H ids.

## 11. Explicit Non-Goals

This slice does not implement:

- web search, retrieval, browser, repository, shell, or coding tools;
- a general Capability Registry or plugin system;
- synchronized multi-agent parity;
- SWE-bench, Terminal-Bench, RE-Bench, or HLE experiment execution;
- a private benchmark;
- calibration claims for posterior values;
- exhaustive adversarial hardening, migration edge cases, or unrelated schema
  security work;
- a redesign of Evidence Memory, the Evidence Gate, or the belief solver.

These remain future work after the open-question MVP is demonstrated.

## 12. Scope Guardrail

The implementation plan may touch only the contracts and runtime paths needed
for Probe Design, authorized semantic expansion, candidate-pool feedback,
task-aware projection, progress display, and their tests. Any finding outside
those paths is recorded for later and does not block the MVP unless it directly
breaks one of the two acceptance fixtures.
