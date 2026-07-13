# BayesProbe

BayesProbe is the project context for a new agent paradigm centered on signal-grounded belief revision over evolving hypotheses. This glossary fixes the vocabulary used by the documents, implementation, and experiments.

## Language

**BayesProbe Agent Paradigm**:
A complete agent methodology with its own lifecycle, control flow, state objects, action semantics, and evaluation criteria. It is not a wrapper around ReAct/ReWOO and should not be described as a mixture of older paradigms.
_Avoid_: BayesProbe-over-ReAct, BayesProbe-over-ReWOO, ReAct with posterior fields, ReWOO with evidence events

**Paradigm Comparison**:
The relationship between BayesProbe and ReAct/ReWOO/ToT/GoT is comparison, not inheritance or composition. These paradigms can be discussed as neighboring answers to different assumptions about what an agent's core object should be.
_Avoid_: ReAct as a BayesProbe component, ReWOO as a BayesProbe execution substrate

**Epistemic Humility**:
BayesProbe's first principle: an agent does not directly possess an answer, but maintains a revisable belief state under uncertainty. Answers are compressed presentations of the current belief state, not the primitive object of the paradigm.
_Avoid_: direct answer as primary state, confidence without revisability

**BayesProbe-Batch**:
A BayesProbe run mode that designs and executes a batch of probes within BayesProbe's own control flow. The name describes batching inside BayesProbe, not dependence on ReWOO.
_Avoid_: BayesProbe-WOO

**BayesProbe-Iterative**:
A BayesProbe run mode that selects the next probe after each belief-state update. The name describes iterative BayesProbe control, not dependence on ReAct.
_Avoid_: BayesProbe-Act

**BayesProbe-Graph**:
A BayesProbe run mode that maintains long-lived graph structure among hypotheses, probes, signals, evidence events, updates, and evolutions.
_Avoid_: Graph-of-Thoughts wrapper

**Synchronized BayesProbe**:
A BayesProbe run regime whose cycle boundaries are shaped by an external collaboration protocol, such as multi-agent rounds, human review turns, or fixed one-round/N-round exchange windows.
_Avoid_: separate paradigm, debate wrapper

**Autonomous BayesProbe**:
A BayesProbe run regime whose cycle boundaries are shaped by internal stopping conditions, such as budget, convergence, confidence, uncertainty reduction, or self-evaluation. It is used when the agent explores independently without waiting for external participants.
_Avoid_: separate paradigm, unconstrained infinite loop

**Dual-Regime BayesProbe**:
BayesProbe support for both Synchronized and Autonomous run regimes as first-class modes. The same belief-revision core serves externally coordinated collaboration and independent agent exploration.
_Avoid_: autonomous-first-only core, synchronized as afterthought

**BayesProbe Core**:
The shared belief-revision machinery used by all BayesProbe run regimes. It owns Belief State, Probe Set Design, Signal Inbox, Evidence Event construction, likelihood judgment, Belief Update, Hypothesis Evolution, and Answer Projection.
_Avoid_: separate cores per regime, controller-owned belief update

**Run Regime Controller**:
The component that governs cycle boundaries and continuation rules for a run regime while delegating belief-revision transitions to BayesProbe Core. Controllers decide when to collect, wait, close a boundary, continue, or emit; they do not bypass the Evidence Integration Gate.
_Avoid_: controller as reasoning core, controller-specific evidence rules

**Controller/Core Boundary**:
The invariant that Run Regime Controllers may govern timing, waiting, boundary closure, continuation, and output emission, but may not define evidence rules, likelihood judgment, posterior updates, or Hypothesis Evolution. Those belong to BayesProbe Core.
_Avoid_: regime-specific belief update, controller bypass of Evidence Integration Gate

**Synchronized Controller**:
A Run Regime Controller driven by external collaboration protocol, such as human review turns, multi-agent rounds, or fixed one-round/N-round exchange windows.
_Avoid_: synchronized-only core, debate logic as belief update

**Fixed-Round Synchronization**:
The MVP synchronization style for Synchronized BayesProbe. A run receives passive signals for a named round or fixed N-round window, closes the Signal Collection Boundary, processes the signals through BayesProbe Core, and emits a Belief State Projection for the next exchange.
_Avoid_: real-time interruption, arbitrary mid-update insertion, direct internal-state sharing

**Passive-Only Cycle**:
A valid Synchronized BayesProbe cycle with an empty Probe Set and one or more Passive External Signals. It still closes a Signal Collection Boundary and processes the Signal Inbox through the Evidence Integration Gate before emitting a Belief State Projection.
_Avoid_: forced active probing every round, passive signal bypass

**Cycle Signal Shape**:
The signal composition of a BayesProbe cycle. Valid shapes are active-only, passive-only, and active-plus-passive; all must close a Signal Collection Boundary and pass through the same Evidence Integration Gate.
_Avoid_: special evidence rules by signal shape, passive-signal shortcut

**Autonomous Controller**:
A Run Regime Controller driven by internal continuation and stop conditions, such as budget, convergence, uncertainty reduction, confidence stability, or answer-utility threshold.
_Avoid_: autonomous-only core, uncontrolled loop

**Autonomous Stop Conditions**:
The combined hard limits and epistemic criteria that end an Autonomous BayesProbe run. Hard limits include max cycles, cost, time, and tool calls; epistemic criteria include stable top hypothesis, low entropy reduction, no high-value probe candidates, out-of-scope change-my-mind conditions, or sufficient answer utility.
_Avoid_: stopping by intuition, endless self-analysis

**Belief State Projection**:
A collaboration-facing compression of an agent's current Belief State, used in human review or multi-agent exchange. It exposes the current best hypothesis, confidence, main evidence events, uncertainties, questions for others, and change-my-mind conditions without requiring the full internal state to be shared.
_Avoid_: full belief-state dump, answer-only message

**Change-My-Mind Condition**:
An explicit condition describing what kind of signal, source challenge, counterevidence, or boundary case would materially change the current belief state. It is required in all BayesProbe outputs and has two layers: a human-readable condition for review/collaboration, and structured probe candidates for execution or autonomous continuation.
_Avoid_: unchallengeable conclusion, vague uncertainty

**Projection-as-Signal Rule**:
A Belief State Projection received from another agent or human enters BayesProbe as a Passive External Signal, not as an Evidence Event. It must pass through the same evidence construction, quality assessment, likelihood judgment, and update process as any other signal.
_Avoid_: authority shortcut, projection as evidence

**Projection Decomposition Rule**:
When an external Belief State Projection contains both a conclusion and cited evidence, BayesProbe separates the sender's judgment from the claimed underlying source. The judgment remains a Passive External Signal, while the claimed source becomes a separate signal or a candidate for direct verification.
_Avoid_: treating another agent's posterior as evidence, merging cited sources with sender belief

**Probe Design**:
A hypothesis-conditioned inquiry plan that specifies what kind of information, test, comparison, question, or observation would help support, weaken, distinguish, or reframe target hypotheses. Probe Design is not the external information itself and is not yet evidence.
_Avoid_: tool call as probe, evidence as probe, generic action

**Probe Set**:
A bounded set of Probe Designs selected for one BayesProbe cycle. A Probe Set expresses the cycle's active control signal and must share the same signal collection boundary.
_Avoid_: unbounded search plan, ReWOO evidence slots

**Probe Candidate Pool**:
The set of candidate Probe Designs available for future cycles, including candidates derived from Change-My-Mind Conditions, unresolved uncertainties, anomalies, or passive signals. Candidates do not execute automatically; the next Probe Set Design selects from them under budget, expected value, and current belief-state constraints.
_Avoid_: automatic next probe set, unprioritized todo list

**Active Control Signal**:
The BayesProbe agent's cycle-local expression of agency, usually represented by one or more Probe Designs. It controls what active external information the agent seeks, while passive external information can still enter the same cycle through the Signal Inbox.
_Avoid_: treating Probe Design as the only information source, passive-signal exclusion

**Signal Intake**:
The point where raw external information enters BayesProbe, whether solicited by a Probe Design or provided independently by a user, benchmark, log, tool, expert, document, or experiment.
_Avoid_: treating signal intake as belief update

**Active External Signal**:
Raw external information returned from a BayesProbe-initiated Probe Design, such as a search result, tool call result, skill output, document retrieval, experiment, or simulation.
_Avoid_: treating active signals as evidence before evaluation

**Passive External Signal**:
Raw external information that arrives without a BayesProbe-initiated Probe Design, such as human expert feedback, another agent's message, system logs, user corrections, benchmark evidence streams, or environmental events.
_Avoid_: ignoring passive signals, forcing every passive signal into a fake probe

**Signal Inbox**:
The cycle-local holding area for Active and Passive External Signals before they are admitted into Evidence Event construction. It preserves passive human or agent inputs that may arrive before the current agent's active probe returns.
_Avoid_: letting passive signals bypass evidence judgment, treating the inbox as belief state

**Evidence Integration Gate**:
The cycle boundary where BayesProbe takes the accumulated Signal Inbox and decides which signals become Evidence Events, which are discarded, and how they enter likelihood judgment. Active and passive signals are integrated at this same gate.
_Avoid_: updating belief immediately on signal arrival, separate evidence rules for active and passive signals

**Signal Collection Boundary**:
The closure point for a cycle's Signal Inbox. Passive signals received before the boundary are integrated with the active probe return in the current cycle; passive signals received after the boundary wait for the next cycle.
_Avoid_: mid-update signal injection, non-reproducible belief updates

**External Signal**:
Raw information received from outside the current belief state before it has been judged as evidence. Active and Passive External Signals are equally eligible for Evidence Event construction, and neither affects synthesis until BayesProbe converts it into an Evidence Event or explicitly discards it.
_Avoid_: Observation as answer input, raw evidence

**Evidence Event**:
A structured, quality-assessed interpretation derived from an External Signal that can participate in likelihood judgment and belief revision.
_Avoid_: Evidence slot, tool result, observation

**Anomaly Signal**:
An Evidence Event whose likelihood is low under all active hypotheses. It should first trigger Hypothesis Evolution rather than more probing inside the old hypothesis frame.
_Avoid_: forcing anomaly into existing hypotheses, probing stale hypotheses first

**Hypothesis Evolution Trigger**:
A condition that requires BayesProbe to consider changing the hypothesis space through spawn, split, reframe, merge, reject, or retire operations before designing the next Probe Set.
_Avoid_: probability-only update, probe-before-evolution on anomaly

**New Hypothesis Entry Rule**:
A spawned, split, or reframed hypothesis enters the Belief State with a small but viable prior, explicit rationale, and required audit fields. It must explain what active hypotheses failed to explain, carry scope/rivals/falsifier/complexity penalty, and generate at least one candidate probe that could weaken it.
_Avoid_: instant high-posterior anomaly explanation, decorative low-prior hypothesis

**Hypothesis Lifecycle**:
The status progression of a hypothesis through active, weakened, reframed/split, retired, and archived states. Retired hypotheses are not deleted; they remain in the ledger as historical explanations or possible future rivals if new evidence reactivates them.
_Avoid_: deleting failed hypotheses, untracked hypothesis replacement

**Belief State**:
The current set of hypotheses, posterior beliefs, scopes, rival relations, uncertainty, and update history maintained by BayesProbe.
_Avoid_: Plan state, scratchpad, reasoning trace

**Task Admission**:
The pre-framing decision that a task is epistemically admissible, needs reframing, or is outside BayesProbe's current scope. It prevents non-epistemic prompts from creating a decorative Belief State.
_Avoid_: universal task routing, framing every prompt, silent rejection

**Epistemic Applicability Gate**:
The rule that admits only tasks with a definable Answer Contract, revisable claims, and at least one possible discriminating probe.
_Avoid_: universal language-task support, hypothesis generation as proof of applicability

**Task Frame**:
The explicit interpretation of a user's task that fixes the task kind, Answer Contract, and Hypothesis Frame before a Belief State is created.
_Avoid_: prompt classification, implicit binary fallback, question text copied into H1/H2

**Task Context**:
Non-evidentiary constraints and definitions that shape the Task Frame, such as audience, scope, available resources, and required output form.
_Avoid_: evidence context, passive signal, source claim

**Answer Contract**:
The task-facing requirements an Answer Projection must satisfy, including its objective, required sections, decision form, and permitted synthesis.
_Avoid_: output prompt, answer template, top-hypothesis restatement

**Answer Relationship**:
The declared relationship between belief claims and the task-facing answer: `selection` chooses an answer-candidate hypothesis, while `synthesis` constructs an answer from one or more supported claims.
_Avoid_: hypothesis always equals answer, top claim always wins

**Hypothesis Frame**:
The task-scoped hypothesis set together with its competition and coverage semantics, rival links, coverage limitations, and framing provenance.
_Avoid_: untyped list of candidates, answer choices assumed for every task

**Hypothesis Competition**:
The rule describing whether named hypotheses compete exclusively or carry independently revisable credences.
_Avoid_: competition implies exhaustiveness, unconditional softmax, automatic all-to-all rivalry

**Hypothesis Coverage**:
The rule describing whether the named hypothesis space is exhaustive or remains open to missing alternatives.
_Avoid_: model candidate count as exhaustive proof, coverage inferred from competition

**Unresolved Alternative**:
The latent reserve for possibilities outside a named exclusive-open hypothesis set. It is not an ordinary hypothesis and cannot be selected as an answer.
_Avoid_: H-other answer choice, renormalizing unresolved mass into named candidates

**Frame Adequacy**:
The current assessment of whether a Hypothesis Frame explains enough of the admitted signal history to support its Answer Contract or requires expansion.
_Avoid_: highest posterior proves adequate, schema validity as epistemic adequacy

**Exact-Answer Task**:
An admitted task requiring a typed scalar or short-text answer without an explicit closed candidate list.
_Avoid_: implicit multiple choice, exhaustive model guesses

**Signal Provenance**:
The source, derivation root, correlation group, parent signals, and content identity needed to judge whether a signal is external, derived, repeated, or independent.
_Avoid_: source label only, cycle-local duplicate flag

**Model Reasoning Signal**:
A Signal produced by model inference without external retrieval, tool observation, or source verification. It remains inside the one Signal-to-Evidence path and may revise one run-scoped Model Reasoning Root, but repeated model calls from that root do not accumulate as independent support.
_Avoid_: search result, verified evidence, tool result

**Evidence Root**:
The canonical unit of potentially independent information behind one or more Signals and Evidence Events. An Evidence Root owns one current likelihood contribution to Belief State; later assessments from the same root revise that contribution instead of adding another independent contribution.
_Avoid_: one root per cycle, one root per paraphrase, correlation cap as repeated evidence allowance

**Evidence Root Revision**:
The replacement, weakening, reversal, or retraction of an Evidence Root's current likelihood contribution. Belief Update applies only the difference between the root's new and previous contributions.
_Avoid_: appending a fresh Bayes factor for every same-root Event, freezing the first assessment forever

**No-New-Information Rule**:
If a cycle adds no independent Evidence Root, produces no nonzero Evidence Root Revision, and performs no hypothesis-frame revision, it cannot increase confidence. Autonomous mode should stop for epistemic stagnation or seek a materially different capability.
_Avoid_: confidence gain from repetition, max-cycle self-confirmation

**Probe Designer**:
The BayesProbe module that proposes hypothesis-conditioned inquiries from the Task Frame, current Belief State, Evidence Memory, uncertainty, and available capabilities.
_Avoid_: probe selector, generic source-tracing template

**Probe Selector**:
The budget-aware module that chooses a bounded Probe Set from an existing Probe Candidate Pool without inventing evidence or updating belief.
_Avoid_: probe designer, tool router

**Capability Registry**:
The run-scoped declaration of available probe capabilities and their provenance, cost, latency, and quality policies.
_Avoid_: model impersonating tools, hidden capability fallback

**Evidence Memory**:
The compact cross-cycle record of accepted evidence identity, provenance, Evidence Roots, current root contributions, and lifecycle relevance needed for deterministic future belief revision.
_Avoid_: raw chain of thought, ledger replacement, cycle-local seen set

**Discovery Evidence**:
A signal used to justify creating or expanding a hypothesis, which cannot immediately be reused as independent confirmation of that same hypothesis.
_Avoid_: candidate proposal as proof, self-confirming hypothesis generation

**Intervention**:
A world-changing action such as applying a code patch, distinct from a Probe that seeks information. Intervention results may return as External Signals only after execution.
_Avoid_: code mutation as probe, action result as immediate evidence

**Projection Mode**:
The declared relationship between a task-facing answer and the current Belief State: `selection` chooses one exclusive hypothesis, `synthesis` combines multiple supported hypotheses, and `abstention` reports that the Answer Contract cannot yet be met.
_Avoid_: always pick H1, synthesis disguised as winner selection

**Answer Projection**:
The user-facing compression of BayesProbe's current Belief State for a specific task or decision. An answer is not BayesProbe's primitive target state.
_Avoid_: final answer as belief state, answer-first agent target

**Answer Utility**:
The practical value of BayesProbe's Answer Projection on a concrete task, benchmark, or decision. BayesProbe must still produce useful answers; belief-state quality is the internal operating philosophy, not a substitute for external task value.
_Avoid_: belief-state-only evaluation, dismissing final answer quality

**Dual-Objective Evaluation**:
BayesProbe is evaluated by both external answer utility and internal belief-revision quality. Either objective alone misrepresents the paradigm.
_Avoid_: answer-only evaluation, belief-state-only evaluation

**Cycle Shape Evaluation Priority**:
The first benchmark plan should cover active-only and passive-only cycle shapes as core cases, with active-plus-passive covered by smaller smoke tests until the mixed collaboration setting is better specified.
_Avoid_: active-only benchmark suite, premature large-scale mixed-cycle benchmark
