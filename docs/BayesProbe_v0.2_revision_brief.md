# BayesProbe v0.2 Revision Brief

Date: 2026-07-07
Status: Draft revision brief
Source baseline: `BayesProbe_docpack/*_v0.1.docx`

## Purpose

This brief captures the conceptual corrections agreed after the v0.1 docpack review and grilling session. It should guide the v0.2 rewrite of the paradigm, engineering, and benchmark documents before implementation starts.

The central correction is that BayesProbe should be presented as a complete agent paradigm with its own philosophical foundation, control flow, state objects, and evaluation criteria. It should not be described as a wrapper around, upgrade layer over, or engineering adaptation of ReAct/ReWOO.

## Core Positioning

### Replace Wrapper Language

v0.1 repeatedly frames BayesProbe as close to ReWOO or as a ReWOO-style transformation. v0.2 should remove that as the primary framing.

Use:

> BayesProbe is a complete agent paradigm for signal-grounded belief revision over evolving hypotheses.

Avoid:

> BayesProbe-WOO, BayesProbe-Act, ReWOO execution substrate, ReAct-style execution, ReWOO-style agent conversion.

ReAct, ReWOO, ToT, GoT, Reflexion, and related methods should appear as neighboring paradigms for comparison, not as BayesProbe components.

### Define the Philosophical Difference

v0.2 should explicitly state the first principle:

> Epistemic humility: an agent does not directly possess an answer; it maintains a revisable belief state under uncertainty.

The contrast should be:

```text
ReAct:    knowing through action and observation
ReWOO:    knowing through planned evidence collection
ToT/GoT:  knowing through structured thought search
BayesProbe: knowing through signal-grounded belief revision
```

BayesProbe's primitive target state is the Belief State. The answer is an Answer Projection: a user-facing compression of the current Belief State.

## Revised Core Cycle

v0.1's short cycle `H -> P -> S -> E -> LR -> BΔ -> H'` is useful but too probe-centric. v0.2 should distinguish active control from signal intake.

Recommended core cycle:

```text
Belief State_t
→ Probe Set Design
→ Signal Inbox
→ Signal Collection Boundary
→ Evidence Integration Gate
→ Evidence Event Construction
→ Likelihood Judgment
→ Belief Update
→ Hypothesis Evolution
→ Belief State_t+1
→ Answer Projection / Next Cycle
```

`Probe Set Design` expresses the agent's active control signal, but it is not the only way information enters the cycle.

## Probe, Signal, Evidence

v0.2 should cleanly separate three concepts:

```text
Probe Design = based on hypotheses, what should be checked?
External Signal = what external information arrived?
Evidence Event = what does that signal mean after assessment?
```

Probe Design is a hypothesis-conditioned inquiry plan. It can request search, tool calls, skill outputs, document retrieval, simulations, user questions, or other tests. It is not raw external information and not evidence.

External Signal is the raw intake point. It may be active or passive.

Evidence Event is derived from an External Signal after quality assessment and interpretation.

## Active and Passive Signals

v0.2 must support both signal types as first-class:

```text
Active External Signal:
  Returned from a BayesProbe-initiated Probe Design.
  Examples: search results, tool outputs, skill results, retrieval results, simulations.

Passive External Signal:
  Arrives without being initiated by the current Probe Set.
  Examples: human expert feedback, another agent's message, user correction, system logs, benchmark evidence stream.
```

Active and passive signals are equally eligible for Evidence Event construction. Neither may update belief directly.

Introduce:

```text
Signal Inbox:
  cycle-local holding area for active and passive signals

Signal Collection Boundary:
  closure point for the current cycle's inbox

Evidence Integration Gate:
  unified gate where signals become Evidence Events or are discarded
```

Passive signals received before the boundary are integrated in the current cycle. Passive signals received after the boundary wait for the next cycle.

## Cycle Shapes

v0.2 should define three legal cycle shapes:

```text
active-only:
  Probe Set != []
  Passive External Signals = []

passive-only:
  Probe Set = []
  Passive External Signals != []

active-plus-passive:
  Probe Set != []
  Passive External Signals != []
```

All shapes must close a Signal Collection Boundary and pass through the same Evidence Integration Gate.

## Dual Run Regimes

v0.2 should support two first-class run regimes, not one primary mode with a future extension.

### Synchronized BayesProbe

Used for human-in-the-loop and multi-agent collaboration.

Cycle boundaries are defined by an external protocol:

```text
round_id
participant_id
incoming passive signals
fixed one-round or N-round window
close boundary
emit Belief State Projection
```

First implementation should support fixed-round synchronization, not arbitrary real-time interruption.

Passive-only cycles must be valid. A synchronized round may receive other agents' or human signals, perform belief revision, and emit a projection without initiating an active probe.

### Autonomous BayesProbe

Used for independent single-agent research, sub-agent task execution, and benchmark runs where no external participant must be awaited.

Cycle boundaries are defined by internal continuation and stop conditions:

```text
hard limits:
  max_cycles, max_cost, max_time, max_tool_calls

epistemic criteria:
  stable top hypothesis for N cycles
  posterior entropy reduction below threshold
  no high-value probe candidates remain
  change-my-mind condition is outside scope/budget
  answer utility threshold is met
```

## Core and Controller Split

v0.2 engineering docs should introduce:

```text
BayesProbe Core:
  owns belief revision machinery

Run Regime Controller:
  owns cycle boundary and continuation policy
```

Controllers may decide:

```text
when to collect
when to wait
when to close boundary
when to continue
when to emit output
```

Controllers may not define:

```text
what counts as evidence
how reliability is assessed
how likelihood is judged
how posterior is updated
how hypotheses evolve
```

Those rules belong to BayesProbe Core and must be shared by Synchronized and Autonomous modes.

## Collaboration Interface

v0.2 should define Belief State Projection as the exchange object for human and multi-agent collaboration.

Required fields:

```text
current_best_hypothesis
posterior_or_confidence_interval
main_evidence_events
main_uncertainties
questions_for_others
change_my_mind_condition
requested_signal_type
```

Do not share full internal belief state by default.

### Projection-as-Signal Rule

A Belief State Projection received from another agent or human enters BayesProbe as a Passive External Signal, not as Evidence Event.

It must pass through:

```text
Evidence Event construction
quality assessment
likelihood judgment
belief update
```

### Projection Decomposition Rule

If an external projection contains both a conclusion and cited evidence, separate them:

```text
Signal 1: sender believes H2 is likely
Signal 2: sender claims source A supports H2
Potential probe: verify source A directly
```

Do not treat another agent's posterior as evidence.

## Change-My-Mind Condition

v0.2 should make Change-My-Mind Condition mandatory for all BayesProbe outputs.

It should have two layers:

```text
human_readable_condition:
  clear statement of what would materially change the current belief

structured_probe_candidates:
  candidate probes that could test the condition in a later cycle
```

Structured candidates should enter a Probe Candidate Pool, not the next Probe Set directly. The next Probe Set Designer selects under budget, value, and current belief-state constraints.

## Hypothesis Evolution

v0.2 should emphasize that BayesProbe is not probability-only updating. It must maintain an open hypothesis space.

If an Evidence Event has low likelihood under all active hypotheses:

```text
Anomaly Signal
→ Hypothesis Evolution Trigger
→ spawn / split / reframe / merge / reject / retire
→ updated Belief State
→ new Probe Candidate Pool
```

Do not keep probing inside stale hypotheses before considering evolution.

### New Hypothesis Entry Rule

Spawned, split, or reframed hypotheses should enter with:

```text
small but viable prior
explicit rationale
scope
rivals
falsifier
complexity penalty
why existing hypotheses failed
at least one candidate probe that could weaken the new hypothesis
```

Do not grant instant high posterior merely because the new hypothesis explains an anomaly.

### Hypothesis Lifecycle

Do not delete failed hypotheses. Use lifecycle states:

```text
active
weakened
reframed / split
retired
archived
```

Retired hypotheses remain in the ledger and can function as historical rivals or be reactivated by later evidence.

## Evaluation Revision

v0.2 should use dual-objective evaluation:

```text
External Answer Utility:
  final accuracy, task success, decision usefulness, cost-normalized performance

Internal Belief-Revision Quality:
  hypothesis coverage
  rival quality
  falsifier quality
  evidence event correctness
  likelihood direction
  posterior update direction
  calibration
  counterevidence response
  uncertainty expression
  change-my-mind quality
```

The project should avoid both extremes:

```text
answer-only evaluation -> BayesProbe becomes a benchmark trick
belief-state-only evaluation -> BayesProbe loses practical agent value
```

### Cycle Shape Evaluation Priority

First benchmark plan should cover:

```text
active-only: core autonomous / tool / benchmark workflow
passive-only: synchronized / human / multi-agent input workflow
active-plus-passive: smaller smoke tests first
```

Do not build an active-only benchmark suite if synchronized use is required by downstream projects.

## Required v0.2 Document Changes

### 01 Paradigm Document

Revise:

- Replace "upgrade layer over existing paradigms" with "complete agent paradigm."
- Present ReAct/ReWOO/ToT/GoT as paradigm comparison only.
- Add epistemic humility as first principle.
- Replace probe-centric cycle with signal-inbox/evidence-gate cycle.
- Add active/passive external signals.
- Add dual run regimes.
- Add Answer Projection and Belief State Projection.
- Add Projection-as-Signal Rule for multi-agent exchange.

Remove or rewrite:

- Any language implying ReAct/ReWOO are internal components.
- "BayesProbe-WOO" and "BayesProbe-Act" as primary names.
- "ReWOO as execution layer" framing.

### 02 Engineering Practice Document

Revise:

- Rename architecture from BayesProbe-WOO to BayesProbe Core + Run Regime Controllers.
- Add Synchronized Controller and Autonomous Controller.
- Add Signal Inbox, Signal Collection Boundary, and Evidence Integration Gate.
- Add active-only, passive-only, and active-plus-passive cycle shapes.
- Add Belief State Projection schema.
- Add Projection-as-Signal and Projection Decomposition handling.
- Add Change-My-Mind Condition schema with structured probe candidates.
- Add Probe Candidate Pool.
- Add Hypothesis Lifecycle.

MVP must include both:

```text
Autonomous BayesProbe
Synchronized BayesProbe with fixed-round synchronization
```

### 03 Experiment and Benchmark Document

Revise:

- Keep final answer accuracy as Layer 1, but explicitly frame it as Answer Utility.
- Expand Layer 2/3 around internal belief-revision quality.
- Add passive-only benchmark scenarios.
- Add synchronized fixed-round tests.
- Add tests where another agent or human projection is treated as Passive External Signal.
- Add tests for Projection Decomposition Rule.
- Add active-plus-passive smoke tests.
- Add metrics for Change-My-Mind Condition quality and Probe Candidate Pool usefulness.

Rename or clarify:

- `BayesProbe-WOO` -> `BayesProbe-Batch` only if describing run rhythm, or avoid if not needed.
- `BayesProbe-Act` -> `BayesProbe-Iterative` only if describing run rhythm, or avoid if not needed.
- Main systems should be `BayesProbe-Autonomous` and `BayesProbe-Synchronized` where the regime matters.

## Open Questions for v0.2

1. What exact JSON schema should represent Belief State Projection?
2. What minimal fields are required for Passive External Signal in fixed-round synchronization?
3. Should Probe Set Design be LLM-only in MVP, or should it use deterministic ranking over Probe Candidate Pool?
4. What thresholds should define "small but viable prior" for new hypotheses?
5. Which passive-only benchmark should be built first: human-feedback simulation, multi-agent projection exchange, or benchmark-provided evidence stream?

## Immediate Next Step

Create v0.2 outlines for the three docpack documents before editing DOCX files. The outlines should be approved before rewriting content.
