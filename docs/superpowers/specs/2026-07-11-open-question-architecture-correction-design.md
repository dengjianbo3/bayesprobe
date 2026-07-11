# Open-Question Architecture Correction Design

Date: 2026-07-11
Status: Approved and implemented (Milestones 1-2)

## 1. Decision

BayesProbe will not patch the WebUI with an isolated hypothesis-generation
call. The correction introduces an explicit task-framing stage and repairs the
runtime feedback paths required for open questions.

The current implementation remains useful as a categorical belief-revision
kernel and as a multiple-choice evaluation path. It is not accepted as a
general open-question agent until the gates in this design pass.

No formal HLE run or live-key experiment begins while this correction is under
implementation. HLE remains a separate multiple-choice capability experiment;
it cannot establish open-question or full-paradigm validity.

## 2. Audit Findings Driving the Correction

The correction is based on reproduced runtime behavior, not only code review:

1. `initialization_completed` occurs before any model request. Every unseeded,
   non-MCQ question is mapped to a fixed support/refute pair.
2. `judge_evidence` receives hypothesis ids but no hypothesis semantics.
3. `CycleResult.probe_candidates` are written by the core but omitted from both
   autonomous and synchronized next-cycle candidate pools.
4. All active hypotheses are normalized as one categorical distribution even
   when open-task hypotheses may coexist.
5. The model-backed probe adapter produces internal model reasoning, not an
   external observation, and the same model then judges that reasoning.
6. Duplicate detection is cycle-local; repeated model content can accumulate
   belief weight across cycles.
7. hypothesis evolution uses only current-cycle events, produces placeholder
   hypothesis text, and cannot accumulate independent counterevidence across
   cycles.
8. answer projection selects and restates the top hypothesis rather than
   satisfying the user's task.
9. existing tests intentionally encode the fixed binary initializer and focus
   provider-backed WebUI acceptance on MCQ behavior.

These findings show an imbalanced architecture: the middle belief-revision
path is substantially implemented, while task framing, cross-cycle epistemic
memory, feedback control, and task-facing projection are shallow or absent.

## 3. Goals

The corrected vertical slice must:

- support unseeded open questions through model-backed task framing before a
  Belief State is exposed;
- preserve deterministic framing for explicit answer choices and explicit
  caller-supplied hypothesis seeds;
- represent whether hypotheses are exclusive or independently credible;
- generate task-specific, falsifiable hypotheses and discriminative probes;
- provide full hypothesis semantics to evidence judgment;
- route evidence/evolution probe candidates into later cycles;
- preserve provenance and correlation across cycle boundaries;
- distinguish internal model reasoning from external observations;
- generate an Answer Projection that satisfies the original task contract;
- fail explicitly instead of silently replacing a failed open-question frame
  with generic H1/H2 templates;
- remain compatible with Autonomous and Synchronized BayesProbe through the
  same core semantics.

## 4. Non-Goals

This correction does not:

- claim calibrated Bayesian probabilities;
- add unrestricted browser automation or arbitrary host code execution;
- make model reasoning an independent external source;
- treat HLE-MCQ accuracy as proof of open-question capability;
- implement every graph-shaped belief relation in the first release;
- make model-generated text evidence without passing the Evidence Integration
  Gate;
- change the rule that received human/agent projections re-enter as signals.

## 5. Corrected Runtime

```text
Question + Task Context + Initial Signals
  -> Task Framing
  -> TaskFrame + HypothesisFrame + AnswerContract
  -> Belief State 0 materialization
  -> Probe Design
  -> Probe Candidate Pool
  -> Probe Selection
  -> Capability-Aware Probe Execution
  -> Signal Inbox
  -> Signal Collection Boundary
  -> Evidence Integration Gate + Evidence Memory
  -> Relation-Aware Belief Solver
  -> History-Aware Hypothesis Evolution
  -> Belief State t+1 + New Probe Candidates
  -> Task-Aware Answer Projection or Next Cycle
```

The run regime still owns timing, boundary closure, budgets, and continuation.
The core still owns evidence admission, belief revision, and hypothesis
evolution. Task framing happens before cycle 1; task-aware projection happens
after belief revision and cannot mutate the Belief State.

## 6. Task Framing Module

### 6.1 Interface

The new deep module exposes one interface:

```python
class TaskFramer(Protocol):
    def frame(self, input: TaskFramingInput) -> TaskFrame: ...
```

`TaskFramingInput` contains:

- `run_id`;
- `question`;
- optional non-evidentiary `task_context`;
- explicit hypothesis seeds when supplied;
- an explicit structured `answer_choices` field when supplied;
- model invocation metadata.

It does not contain passive evidence text. Initial evidence enters later as an
`ExternalSignal`.

The versioned HTTP and Python request contracts expose `answer_choices`
directly. A compatibility parser may extract conventional English or Chinese
choice blocks from legacy question text, but parser failure never converts an
open question into a binary claim.

### 6.2 TaskFrame

```text
TaskFrame
  task_kind
  normalized_question
  answer_contract
  hypothesis_frame
  framing_method
  framing_trace
```

Supported first-release task kinds are:

- `multiple_choice`;
- `claim_verification`;
- `explanation`;
- `diagnosis`;
- `design`;
- `decision`.

Unsupported framing fails before Belief State creation. It is never coerced to
`binary_claim`.

### 6.3 Adapters

Three adapters make the seam real:

- `ExplicitTaskFramer`: deterministic mapping for answer choices and explicit
  hypothesis seeds;
- `ModelTaskFramer`: provider-backed framing for unseeded open questions;
- `RecordedTaskFramer`: frozen provider-shaped fixtures for offline tests.

The WebUI uses `ModelTaskFramer` whenever an OpenAI-compatible provider is
selected and no choices/seeds exist. Deterministic mode requires choices,
seeds, or a recorded frame for open questions.

### 6.4 Structured Model Output

`frame_open_question` returns:

```json
{
  "task_kind": "claim_verification",
  "answer_contract": {
    "objective": "design a discriminating validation protocol",
    "required_sections": [
      "hypotheses",
      "experimental_design",
      "controls",
      "metrics",
      "decision_rule",
      "limitations"
    ],
    "decision_form": "experimental_protocol",
    "permits_synthesis": true
  },
  "hypothesis_relation": "independent",
  "hypotheses": [
    {
      "statement": "Holding agent scaffolding, inference budget, and task conditions fixed, increasing model scale improves real-task performance.",
      "type": "causal_claim",
      "scope": "The declared task distribution and controlled resource envelope.",
      "falsifiers": ["The preregistered size coefficient is non-positive or practically negligible."],
      "predictions": ["Performance improves monotonically across controlled model sizes."]
    },
    {
      "statement": "Apparent gains attributed to model size are materially explained by uncontrolled scaffolding, inference compute, or task-selection differences.",
      "type": "confounding_explanation",
      "scope": "Published or observed comparisons that do not hold the resource envelope fixed.",
      "falsifiers": ["The size effect remains after matched controls and ablation of the suspected confounders."],
      "predictions": ["The apparent size effect shrinks under matched-budget and matched-scaffold evaluation."]
    }
  ],
  "coverage_statement": "These hypotheses test the target causal effect and a major confounding explanation; they are not claimed to exhaust every failure mode.",
  "coverage_limitation": "Task-specific interactions and unmodeled deployment conditions may require additional hypotheses."
}
```

The server assigns stable ids and initial priors. The model never assigns
prior or posterior values. Exclusive frames use a uniform prior over active
hypotheses unless an explicit admissible prior policy is configured.
Independent frames default each active hypothesis to credence `0.5`; they are
not divided by hypothesis count. An open frame contains 2-6 distinct
hypotheses. Validation requires non-empty semantics, unique normalized
statements, at least one falsifier and prediction per hypothesis, and a
relation compatible with the selected solver. One structured repair is
permitted. A second failure ends the run before `initialization_completed`.

## 7. Question, Context, and Signal Semantics

The current single `Context` field overloads two meanings and must be split:

- `Task Context`: constraints, audience, scope, definitions, desired output,
  and other non-evidentiary framing information;
- `Initial Signals`: source-bearing claims, documents, observations, expert
  feedback, logs, or other raw information that must pass through the Evidence
  Integration Gate.

The compatibility request field `context` is temporarily interpreted as one
initial passive signal and is visibly labelled as such. It is not also supplied
to model probe execution as independent background. A later removal requires a
versioned request contract.

## 8. Hypothesis Frame and Belief Semantics

### 8.1 HypothesisFrame

`HypothesisFrame` records:

- `relation`;
- active hypothesis ids;
- explicit rival sets;
- coverage statement;
- unresolved alternative mass or coverage limitation;
- framing trace.

`BeliefState` stores the complete `HypothesisFrame` identity and relation plus
a compact Evidence Memory snapshot. Relation semantics are therefore part of
the state transition contract, not UI-only metadata or a value inferred by the
solver.

The first release supports two relations:

- `exclusive_exhaustive`: exactly one active rival is treated as correct and
  active posterior mass sums to one;
- `independent`: multiple hypotheses may be credible at once and each carries
  an independent credence.

### 8.2 Relation-Aware Solver

The existing softmax update remains the implementation for
`exclusive_exhaustive` frames.

For `independent` frames, each hypothesis updates in log-odds space without
cross-normalization:

```text
logit(p_h,t+1)
  = logit(p_h,t)
  + sum(log(LR_h,event) * admitted_event_weight)
  - newly_applicable_complexity_penalty
  - newly_applicable_ad_hoc_penalty
```

An admitted event is applied once by ledger identity. Static penalties are not
subtracted again on every cycle; the transition records which penalties have
already been incorporated.

New states without relation metadata are invalid. A versioned deserialization
migration assigns legacy serialized states `exclusive_exhaustive` before they
reach the solver; the solver itself never infers a relation from hypothesis
count, ids, or posterior values.

All-to-all rivals are valid only for an exclusive frame. Independent frames
carry explicit rival/conflict links.

### 8.3 Display Semantics

The API and WebUI label exclusive values as posterior mass and may show a
top-versus-runner-up gap. Independent values are labelled hypothesis credence;
they need not sum to one, and the UI must not imply that their ranking alone
selects the final answer. Coverage limitations remain visible in both modes.

## 9. Probe Design and Selection

The current `ProbePlanner` performs selection, not design. The correction
separates two modules:

```python
class ProbeDesigner(Protocol):
    def propose(self, context: ProbeDesignContext) -> list[ProbeCandidate]: ...

class ProbeSelector:
    def select(self, candidates, belief_state, budget) -> ProbeSet: ...
```

`ProbeDesignContext` contains the full TaskFrame, current Belief State,
Evidence Memory summary, unresolved uncertainty, change-my-mind conditions,
and available capability descriptors.

Initial open-task probes are generated after framing. At least one first-cycle
probe must discriminate two or more relevant rivals; one-hypothesis generic
`source_tracing` cannot be the only initial probe. The existing deterministic
answer-choice discriminator remains for MCQ.

The selector remains deterministic and budget-aware. Model output proposes
candidate semantics; it does not decide posterior values or bypass the
selector.

## 10. Capability-Aware Probe Execution

Each Probe Design declares a required capability, such as:

- `model_reasoning`;
- `python_computation`;
- `search`;
- `document_retrieval`;
- `external_agent_request`;
- `human_request`.

The run receives an explicit capability registry. A probe requiring an
unavailable capability is rejected or converted into a visible request for an
external signal; it is not silently executed by the language model.

`ModelBackedProbeToolGateway` is renamed conceptually to model reasoning. Its
signals remain admissible but carry `epistemic_origin=model_reasoning`, low
independence, and a provider/session correlation group. They must never claim
search, retrieval, source verification, or experimental observation.

## 11. Signal Provenance and Evidence Memory

### 11.1 Signal Provenance

Each signal records:

- epistemic origin;
- source identity;
- provider/model or tool identity when applicable;
- parent signal ids;
- derivation root;
- correlation group;
- canonical content fingerprint;
- citations or artifact references when present.

A model summary derived from an initial passive signal shares that signal's
derivation root and cannot be counted as independent confirmation.

### 11.2 Cross-Cycle Evidence Memory

Belief State carries a compact Evidence Memory summary sufficient to detect:

- exact repeated content;
- repeated source/content pairs;
- different text derived from the same root signal;
- repeated reasoning from the same provider/session;
- previously accepted counterevidence for lifecycle decisions.

Duplicate and correlation checks operate across the run, not only within one
cycle. The ledger remains the full audit record; the Belief State stores only
the compact state needed for deterministic next-cycle behavior.

## 12. Evidence Judgment Correction

Every `judge_evidence` request includes:

- the raw signal and complete provenance summary;
- the full semantic descriptor of each assessed hypothesis;
- relation type and relevant rival ids;
- probe support/weaken conditions;
- prior accepted evidence fingerprints needed for novelty judgment;
- allowed evidence and likelihood enums.

For an exclusive frame, discriminative evidence is judged against all relevant
rivals, not only the single hypothesis named by a probe. For an independent
frame, targets are explicit and untargeted claims remain unchanged.

The judge may lower quality dimensions. Model reasoning and Python-derived
signals remain capped by deterministic source baselines. No model override can
increase independence or verifiability above source policy.

## 13. Hypothesis Evolution Correction

Evolution becomes a two-part deep module:

1. deterministic trigger policy decides whether spawn, reframe, merge, split,
   retire, or reactivate is eligible;
2. a semantic evolution adapter materializes meaningful hypothesis content and
   corresponding discriminative probe candidates.

The engine receives current-cycle events plus Evidence Memory. Retirement and
correlation decisions can therefore accumulate evidence across cycles.

Every new or reframed hypothesis must include substantive statement, scope,
falsifiers, predictions, relation links, rationale explaining frame failure,
and at least one next probe. Placeholder text such as `Spawned anomaly
hypothesis for E1` is invalid.

`CycleResult.probe_candidates` is mandatory runner input for the next cycle.
Autonomous and Synchronized runners merge candidates in this order:

1. evolution/anomaly/source-verification candidates;
2. task-aware change-my-mind candidates;
3. remaining unselected candidates.

Stable candidate ids prevent duplicates across these sources.

## 14. Task-Aware Projection

Projection remains downstream from belief and cannot alter it.

MCQ uses deterministic label projection. Open tasks use a
`ProjectionGenerator` with the interface:

```python
class ProjectionGenerator(Protocol):
    def project(self, input: ProjectionInput) -> AnswerProjection: ...
```

`ProjectionInput` contains the AnswerContract, full ranked/independent belief
summary, accepted evidence summaries and ids, unresolved uncertainty,
falsifiers, and stop reason.

Both `AnswerProjection` and the Synchronized-mode `BeliefStateProjection`
evolve from a forced single-winner contract to include:

```text
projection_mode: selection | synthesis | abstention
current_best_hypothesis: hypothesis id | null
basis_hypotheses: ordered list of hypothesis ids
```

`selection` requires one current best id and remains the MCQ-compatible path.
`synthesis` may combine several independently credible hypotheses and leaves
`current_best_hypothesis` null when no single hypothesis is the answer.
`abstention` records why the AnswerContract cannot yet be met. Existing
single-winner consumers receive a versioned compatibility projection only for
exclusive frames; synthesis is never disguised as selection.

`project_open_answer` must:

- answer the original question directly;
- satisfy every required AnswerContract section;
- distinguish accepted external evidence from model reasoning;
- state material uncertainty and missing capabilities;
- identify the belief-backed current best hypothesis or synthesis;
- include a change-my-mind condition;
- cite only sources present in signal provenance;
- never invent a new Evidence Event or change posterior values.

One structured repair is allowed. Projection failure is explicit and the WebUI
shows the preserved Belief State with `answer_unavailable`; it does not present
a generic top-hypothesis string as a successful answer.

## 15. Progress and Failure Semantics

The autonomous progress stream adds:

```text
run_started
task_framing_started
task_framing_completed
initialization_completed
probe_design_started
probe_set_planned
...
answer_projection_started
run_completed | run_failed
```

The WebUI does not render a Belief State before
`initialization_completed`. Provider failures during framing produce
`run_failed` with the same secret-safe diagnostics used elsewhere.

No silent fallbacks are permitted for:

- open-task framing;
- relation selection;
- semantic hypothesis evolution;
- open-answer projection.

## 16. Compatibility Rules

- explicit MCQ choices keep deterministic ids and uniform priors;
- new request versions carry answer choices structurally; legacy text parsing
  is compatibility behavior, not the task-typing authority;
- explicit hypothesis seeds remain supported and now require or inherit an
  explicit relation from a versioned input default;
- HLE Text-MCQ continues to use its frozen explicit-choice path;
- old deterministic open-question fixtures migrate to explicit recorded
  TaskFrames instead of relying on generic H1/H2;
- existing ModelGateway adapters gain new structured tasks without changing
  `complete_structured(request) -> dict`;
- provider keys remain request-scoped and never enter TaskFrame, ledger, or
  shareable telemetry;
- Context-as-passive-signal compatibility is retained until the versioned WebUI
  request fields are deployed.

## 17. Testing Strategy

### 17.1 Contract Tests

Tests must prove:

- no open Belief State appears before a framing model call;
- an explicit MCQ or seed frame does not require a framing call;
- invalid or duplicate hypotheses receive one repair and then fail closed;
- no failed frame silently becomes support/refute H1/H2;
- relation-aware solvers preserve categorical and independent invariants;
- evidence judgment receives complete hypothesis semantics;
- discriminative probes assess relevant rivals;
- core-generated probe candidates are selectable in the next autonomous and
  synchronized cycles;
- cross-cycle repeated/correlated signals are downweighted;
- independent counterevidence can accumulate across cycles for evolution;
- open projection satisfies its AnswerContract and cannot mutate belief;
- unavailable capabilities are visible instead of impersonated by a model;
- all secret-redaction and artifact rules remain intact.

### 17.2 Open-Question Fixtures

The offline suite contains self-authored recorded fixtures for:

- claim verification;
- causal explanation;
- incident diagnosis;
- experiment/system design;
- decision/recommendation;
- one Chinese claim-verification question matching the reported failure mode.

The Chinese fixture's frame must distinguish at least these causal states:

- model scale has an independent positive effect under controlled conditions;
- apparent gains disappear after controlling data, training, inference budget,
  and scaffolding;
- effects are conditional or non-monotonic across task/tool regimes.

Its AnswerContract requires controlled experimental design, confound controls,
task metrics, statistical decision rule, and limitations. Tests assert these
semantics without requiring a single exact prose answer.

### 17.3 Regression Tests

The complete Python, WebUI stream, provider adapter, Docker isolation, MCQ, and
HLE synthetic workflow suites must remain green. Existing tests that assert the
generic binary fallback are replaced, not layered with contradictory behavior.

## 18. Experimental Gates

### Gate 0: Offline Semantic Gate

- all contract and regression tests pass;
- recorded open fixtures pass semantic assertions;
- no generic unseeded binary fallback remains on provider-backed paths.

### Gate 1: One-Time-Key Live Smoke

Use the supplied one-time key only in process memory. Run five self-authored
open questions, including the reported Chinese question, with at most two
cycles each. Persist secret-free request/task telemetry and inspect:

- TaskFrame quality;
- hypothesis coverage and distinctness;
- probe discrimination;
- signal origin/provenance;
- evidence-target correctness;
- candidate feedback;
- AnswerContract completion;
- provider errors, tokens, and latency.

### Gate 2: Open-Question Pilot

Compare Direct versus BayesProbe on a frozen self-authored open set. Use a
blind rubric or independent evaluator for answer utility and deterministic
structural metrics for internal behavior. This is the first experiment that
can support an open-question capability claim.

### Gate 3: Formal HLE-MCQ

Only after Gates 0-2 pass should the separate HLE-MCQ pilot run. Its claim
remains limited to multiple-choice capability. It does not substitute for the
open-question pilot.

## 19. Documentation Corrections

When implementation begins, `docs/ARCHITECTURE.md` must be corrected before
any new progress estimate is published:

- distinguish categorical MCQ maturity from open-question maturity;
- downgrade autonomous question and hypothesis evolution status until their
  gates pass;
- state that model reasoning is not external verification;
- state the exact implemented evolution operations;
- remove the obsolete global completion estimate;
- document TaskFrame, HypothesisFrame, relation-aware solving, Evidence Memory,
  and task-aware projection.

## 20. Delivery Sequence

The correction is delivered as six independently reviewed milestones:

1. TaskFrame, explicit/model/recorded framers, and progress events;
2. HypothesisFrame relations and relation-aware belief solving;
3. evidence request semantics and cross-cycle provenance memory;
4. ProbeDesigner, capability declarations, and candidate feedback repair;
5. history-aware semantic evolution and task-aware projection;
6. WebUI field correction, live smoke, open-question pilot tooling, and
   architecture-status correction.

No milestone may claim completion from unit tests alone. Each milestone ends
with an end-to-end recorded open-question check through the public runner.

## 21. Definition of Done

The architecture correction is complete only when:

1. the reported Chinese question shows no Belief State before model framing;
2. its initial hypotheses are task-specific, distinct, and auditable;
3. its first probe is meaningfully discriminative rather than generic H1
   source tracing;
4. evidence judgment sees full hypothesis semantics;
5. repeated reasoning cannot accumulate as independent evidence;
6. evolved candidates reach the next cycle;
7. the final answer contains an actionable validation design matching the
   AnswerContract;
8. explicit MCQ and seed behavior remains correct;
9. the offline suite, WebUI stream suite, and Docker isolation suite pass;
10. the one-time-key smoke passes without writing or exposing the key;
11. architecture documentation and progress estimates reflect the verified
    capability rather than the intended capability.
