# BayesProbe v0.2 Engineering Design

Date: 2026-07-07
Status: Draft detailed engineering design
Inputs:
- `CONTEXT.md`
- `docs/BayesProbe_v0.2_revision_brief.md`
- `docs/BayesProbe_02_engineering_v0.2_outline.md`

## 1. Design Intent

BayesProbe v0.2 is a complete agent paradigm implementation for signal-grounded belief revision over evolving hypotheses. The engineering design must not encode BayesProbe as a ReAct/ReWOO wrapper. ReAct, ReWOO, ToT, GoT, and related systems are comparison baselines, not internal modules.

The first version must support two first-class run regimes:

- **Autonomous BayesProbe**: independent exploration with internal stop conditions.
- **Synchronized BayesProbe**: fixed-round collaboration with humans or other agents.

Both regimes share one **BayesProbe Core**. Controllers may decide timing and cycle boundaries, but they may not define evidence rules, likelihood judgment, posterior updates, or Hypothesis Evolution.

## 2. Architecture Summary

```text
Client / Runner
  |
  v
Run Regime Controller
  |-- Autonomous Controller
  |-- Synchronized Controller
  |
  v
BayesProbe Core
  |-- Frame Builder
  |-- Hypothesis Manager
  |-- Probe Set Designer
  |-- Signal Inbox Manager
  |-- Evidence Integration Gate
  |-- Evidence Event Builder
  |-- Signal Quality Assessor
  |-- Likelihood Judge
  |-- Belief Solver
  |-- Hypothesis Evolver
  |-- Projection Generator
  |
  v
Persistence / Ledger
  |
  v
Evaluation Harness
```

The important seam is between **Run Regime Controller** and **BayesProbe Core**. The controller drives cycles. The core owns belief revision.

## 3. Deep Module Strategy

### 3.1 Main Modules

The implementation should be organized around deep modules with small interfaces:

| Module | Interface role | Hidden implementation |
|---|---|---|
| `BayesProbeCore` | Advance a cycle from belief state + inbox to updated belief state + projections | Evidence construction, quality scoring, likelihood judging, belief updates, hypothesis evolution |
| `RunRegimeController` | Govern cycle timing and continuation | Autonomous loop, synchronized fixed rounds, boundary closure |
| `ProbeSetDesigner` | Select a bounded Probe Set | Ranking candidates, budget tradeoffs, target-hypothesis coverage |
| `SignalInbox` | Collect active/passive signals until boundary closure | Signal validation, cycle assignment, late signal deferral |
| `LedgerStore` | Persist and retrieve run records | JSONL/SQLite/Postgres adapters |
| `ModelGateway` | Run LLM judgments through typed prompts | Prompting, retries, schema validation |
| `ToolGateway` | Execute active probe methods | Search, retrieval, skill execution, simulations |
| `Evaluator` | Score runs and samples | Metrics, rubrics, benchmark-specific gold comparison |

### 3.2 External Seams

Use real seams only where behavior varies:

- `LedgerStore`: in-memory, JSONL, SQLite/Postgres.
- `ModelGateway`: model provider, local fake, recorded fixture.
- `ToolGateway`: web/search/document/tool adapters, fake benchmark adapters.
- `Clock`: deterministic tests for synchronized deadlines.

Do not create seams for every internal sub-step unless there are at least two adapters or a strong testability need.

## 4. Core Runtime Concepts

### 4.1 Cycle

A cycle is the atomic runtime unit for signal collection and belief revision.

```text
BeliefState_t
→ ProbeSetDesign
→ SignalInbox
→ SignalCollectionBoundary
→ EvidenceIntegrationGate
→ EvidenceEvents
→ BeliefUpdates
→ HypothesisEvolution
→ BeliefState_t+1
→ Projection
```

Valid cycle signal shapes:

- `active_only`
- `passive_only`
- `active_plus_passive`

All three shapes must use the same Evidence Integration Gate.

### 4.2 Active and Passive Signals

Active External Signals are returned by current-cycle Probe Designs.

Passive External Signals arrive without a current-cycle Probe Design:

- human feedback
- external agent Belief State Projection
- user correction
- system log
- benchmark signal stream
- environmental event

Neither signal kind may update belief directly.

### 4.3 Signal Inbox and Boundary

The Signal Inbox is cycle-local. It holds active and passive signals before evidence construction.

The Signal Collection Boundary closes the inbox:

- Signals received before closure enter the current cycle.
- Signals received after closure are assigned to the next cycle.
- Synchronized cycles may close by round deadline or explicit controller command.
- Autonomous cycles close after active probes complete or a controller-defined collection timeout.

### 4.4 Evidence Integration Gate

The Evidence Integration Gate is the core admission point:

```text
Closed Signal Inbox
→ classify signal
→ construct Evidence Event or discard/defer
→ assess quality
→ judge likelihood
→ pass to Belief Solver
```

The gate must treat active and passive signals with the same evidence rules.

## 5. Public Interfaces

These are conceptual interfaces. Exact language syntax can be chosen during implementation.

### 5.1 BayesProbeCore

Small external interface:

```ts
interface BayesProbeCore {
  initializeRun(input: InitializeRunInput): RunSnapshot
  designProbeSet(input: DesignProbeSetInput): ProbeSet
  integrateCycle(input: IntegrateCycleInput): CycleResult
  generateProjection(input: ProjectionInput): ProjectionResult
}
```

The controller should not call `EvidenceEventBuilder`, `BeliefSolver`, or `HypothesisEvolver` directly. Those are internal to `integrateCycle`.

### 5.2 RunRegimeController

```ts
interface RunRegimeController {
  start(input: ControllerStartInput): RunSnapshot
  step(input: ControllerStepInput): ControllerStepResult
  close(input: ControllerCloseInput): ProjectionResult
}
```

Controller responsibilities:

- open cycle
- accept passive signals
- request Probe Set
- execute active probes through `ToolGateway`
- close Signal Collection Boundary
- call `core.integrateCycle`
- decide continue/stop/emit

Controller non-responsibilities:

- evidence classification
- reliability scoring
- likelihood judgment
- posterior update
- hypothesis evolution

### 5.3 LedgerStore

```ts
interface LedgerStore {
  createRun(run: RunRecord): void
  appendCycle(cycle: CycleRecord): void
  appendSignals(signals: ExternalSignal[]): void
  appendEvidence(events: EvidenceEvent[]): void
  appendUpdates(updates: BeliefUpdate[]): void
  appendEvolutions(evolutions: HypothesisEvolution[]): void
  saveProjection(projection: AnswerProjection | BeliefStateProjection): void
  loadRun(runId: string): RunSnapshot
}
```

MVP can use JSONL or SQLite. The interface should not expose storage-specific query details to core logic.

### 5.4 ModelGateway

```ts
interface ModelGateway {
  completeStructured<T>(request: StructuredModelRequest<T>): T
}
```

Use it for:

- frame building
- hypothesis generation
- evidence event construction
- quality assessment
- likelihood judgment
- projection generation

For tests, support deterministic fixture adapters.

### 5.5 ToolGateway

```ts
interface ToolGateway {
  executeProbe(probe: ProbeDesign, context: ToolExecutionContext): Promise<ActiveExternalSignal[]>
}
```

Supported MVP methods:

- `document_retrieval`
- `web_search` if available
- `tool_result`
- `skill_output`
- `simulation`
- `ask_user` as a deferred passive signal request

## 6. Data Model

### 6.1 RunRecord

```json
{
  "run_id": "run_001",
  "regime": "autonomous",
  "status": "running",
  "problem": "...",
  "current_cycle_id": "cycle_001",
  "budget": {
    "max_cycles": 5,
    "max_tool_calls": 20,
    "max_tokens": 50000,
    "max_cost": null
  },
  "created_at": "2026-07-07T00:00:00Z",
  "updated_at": "2026-07-07T00:00:00Z",
  "metadata": {}
}
```

### 6.2 CycleRecord

```json
{
  "cycle_id": "cycle_001",
  "run_id": "run_001",
  "round_id": null,
  "cycle_index": 1,
  "signal_shape": "active_plus_passive",
  "boundary_status": "closed",
  "started_at": "...",
  "boundary_closed_at": "...",
  "completed_at": null,
  "controller_metadata": {}
}
```

### 6.3 Hypothesis

```json
{
  "id": "H1",
  "statement": "...",
  "type": "causal_explanation",
  "scope": "...",
  "prior": 0.33,
  "posterior": 0.45,
  "status": "active",
  "rivals": ["H2", "H3"],
  "falsifiers": ["..."],
  "predictions": ["..."],
  "complexity_penalty": 0.1,
  "ad_hoc_penalty": 0.0,
  "created_by": "initial",
  "why_existing_hypotheses_failed": null
}
```

### 6.4 BeliefState

```json
{
  "belief_state_id": "bs_001",
  "run_id": "run_001",
  "cycle_id": "cycle_001",
  "hypotheses": [],
  "posterior_summary": {
    "top_hypothesis": "H1",
    "entropy": 1.02,
    "confidence_band": "moderate"
  },
  "uncertainty_summary": "...",
  "ledger_refs": {
    "evidence_events": [],
    "belief_updates": [],
    "hypothesis_evolutions": []
  }
}
```

### 6.5 ProbeDesign

```json
{
  "id": "P1",
  "cycle_id": "cycle_001",
  "target_hypotheses": ["H1", "H2"],
  "probe_type": "discriminative_test",
  "inquiry_goal": "Distinguish whether the observed error supports H1 or H2.",
  "method": "document_retrieval",
  "support_condition": {"H1": "..."},
  "weaken_condition": {"H2": "..."},
  "reframe_condition": null,
  "expected_information_gain": 0.7,
  "decision_relevance": 0.9,
  "cost_estimate": 0.3,
  "priority": 0.63,
  "status": "selected"
}
```

### 6.6 ProbeSet

```json
{
  "probe_set_id": "ps_001",
  "cycle_id": "cycle_001",
  "probes": [],
  "boundary_id": "boundary_001",
  "selection_reason": "Passive-only synchronized cycle; no active probe requested.",
  "budget_allocated": {
    "tool_calls": 0,
    "tokens": 0
  },
  "may_be_empty": true
}
```

### 6.7 ProbeCandidate

```json
{
  "candidate_id": "pc_001",
  "source": "change_my_mind",
  "candidate_probe": {},
  "priority_features": {
    "expected_information_gain": 0.8,
    "cost_estimate": 0.4,
    "decision_relevance": 0.9,
    "attacks_top_hypothesis": true
  },
  "selected_in_cycle": null
}
```

### 6.8 ExternalSignal

```json
{
  "id": "S1",
  "signal_kind": "passive",
  "source_type": "external_agent_projection",
  "source": "agent_critic_01",
  "raw_content": "...",
  "generated_by_probe": null,
  "received_at": "...",
  "cycle_id": "cycle_001",
  "inbox_status": "accepted",
  "initial_target_hypotheses": ["H2"]
}
```

### 6.9 EvidenceEvent

```json
{
  "id": "E1",
  "derived_from_signal": "S1",
  "target_hypotheses": ["H1", "H2"],
  "evidence_type": "counterevidence",
  "content": "...",
  "reliability": 0.75,
  "independence": 0.6,
  "relevance": 0.9,
  "novelty": 0.8,
  "specificity": 0.7,
  "verifiability": 0.8,
  "likelihoods": {
    "H1": "moderately_disconfirming",
    "H2": "weakly_confirming"
  },
  "interpretation": "...",
  "discard_reason": null
}
```

### 6.10 BeliefUpdate

```json
{
  "update_id": "U1",
  "cycle_id": "cycle_001",
  "evidence_id": "E1",
  "hypothesis_id": "H1",
  "prior": 0.45,
  "posterior": 0.32,
  "direction": "weakened",
  "reason": "...",
  "sensitivity": {
    "depends_on_reliability": true,
    "depends_on_independence": true
  }
}
```

### 6.11 HypothesisEvolution

```json
{
  "evolution_id": "HE1",
  "cycle_id": "cycle_001",
  "operation": "spawn",
  "from_hypothesis": null,
  "to_hypothesis": "H4",
  "triggered_by": ["E3"],
  "reason": "E3 has low likelihood under all active hypotheses.",
  "audit_fields": {
    "why_existing_hypotheses_failed": "...",
    "new_hypothesis_prior": 0.12,
    "required_next_probe": "P_candidate_4"
  }
}
```

### 6.12 Projection Records

Answer Projection:

```json
{
  "answer": "...",
  "current_best_hypothesis": "H2",
  "posterior_summary": "...",
  "main_uncertainty": "...",
  "weakest_assumption": "...",
  "main_evidence_events": ["E1", "E2"],
  "change_my_mind_condition": {
    "human_readable_condition": "...",
    "structured_probe_candidates": []
  },
  "answer_utility_notes": "Meets task threshold under current budget."
}
```

Belief State Projection:

```json
{
  "current_best_hypothesis": "H2",
  "posterior_or_confidence_interval": "moderate-high",
  "main_evidence_events": ["E1", "E2"],
  "main_uncertainties": ["..."],
  "questions_for_others": ["..."],
  "change_my_mind_condition": {
    "human_readable_condition": "...",
    "structured_probe_candidates": []
  },
  "requested_signal_type": "source_independence_challenge",
  "cited_sources": [],
  "projection_metadata": {
    "run_id": "run_001",
    "cycle_id": "cycle_002"
  }
}
```

## 7. Core Pipeline Design

### 7.1 Initialize Run

Input:

- problem
- context
- initial passive signals optional
- regime config

Output:

- initial RunRecord
- initial Belief State
- initial Probe Candidate Pool

Steps:

1. Build problem frame.
2. Generate initial rival hypotheses.
3. Assign conservative priors.
4. Validate each high-impact hypothesis has scope, rivals, falsifiers, and predictions.
5. Create initial Belief State.
6. Generate first Change-My-Mind Condition and Probe Candidate Pool.

### 7.2 Design Probe Set

Input:

- current Belief State
- Probe Candidate Pool
- cycle constraints
- budget
- regime hints

Output:

- bounded Probe Set

Rules:

- Probe Set may be empty only when cycle has passive signals or controller explicitly requests passive-only.
- Every selected probe must bind target hypotheses.
- At least one probe should attack the top hypothesis when budget allows.
- Do not select candidates automatically from Change-My-Mind Condition without ranking.
- Freeze Probe Set within the cycle for MVP.

Ranking features:

```text
priority =
  expected_information_gain
  * decision_relevance
  * attacks_top_hypothesis_bonus
  * unresolved_uncertainty_bonus
  / cost_estimate
```

The exact formula can be heuristic in MVP, but the selected Probe Set must record selection reasons.

### 7.3 Collect Signals

Active path:

1. Controller sends Probe Set to ToolGateway.
2. ToolGateway returns Active External Signals.
3. Signals are appended to Signal Inbox.

Passive path:

1. Controller accepts passive input from human, agent, log, benchmark, or user.
2. Signals are appended to Signal Inbox.
3. Late signals after boundary closure are assigned to next cycle.

### 7.4 Close Boundary

Boundary can close when:

- all active probes returned
- fixed round deadline reached
- manual `close_boundary`
- autonomous timeout reached

After closure:

- no new signal may enter current inbox
- controller calls `core.integrateCycle`

### 7.5 Integrate Cycle

Input:

- previous Belief State
- closed Signal Inbox
- Probe Set

Output:

- Evidence Events
- discarded/deferred signals
- Belief Updates
- Hypothesis Evolutions
- new Belief State
- Probe Candidate Pool updates

Internal steps:

1. Classify signals.
2. Apply Projection-as-Signal Rule.
3. Apply Projection Decomposition Rule.
4. Build Evidence Events or discard/defer.
5. Assess signal quality.
6. Judge likelihood under active hypotheses.
7. Solve belief updates.
8. Detect anomaly and counterevidence.
9. Trigger Hypothesis Evolution where needed.
10. Update Hypothesis Lifecycle.
11. Generate new Probe Candidate Pool.

### 7.6 Generate Projection

For autonomous run:

- generate Answer Projection
- include final answer and practical utility notes

For synchronized run:

- generate Belief State Projection
- include requested signal type and questions for others

Both must include:

- current best hypothesis
- main uncertainty
- main evidence events
- Change-My-Mind Condition

## 8. Run Regime Designs

### 8.1 Autonomous Controller

Autonomous Controller is for independent exploration.

Pseudo-flow:

```text
start run
initialize Belief State

while true:
  open cycle
  design Probe Set
  execute active probes
  collect active signals
  close boundary
  integrate cycle through BayesProbe Core

  if autonomous stop condition reached:
    emit Answer Projection
    break
```

Stop conditions:

- hard max cycles
- hard max tokens/cost/tool calls
- top hypothesis stable for N cycles
- posterior entropy reduction below threshold
- no high-value probe candidates remain
- Change-My-Mind Condition outside budget/scope
- answer utility threshold met

Autonomous controller may still accept passive signals if they are provided by caller or environment during a cycle, but it does not wait indefinitely for them.

### 8.2 Synchronized Controller

Synchronized Controller is for fixed-round collaboration.

Pseudo-flow:

```text
open round cycle
accept passive signals until round boundary
optionally design Probe Set
optionally execute active probes
close boundary
integrate cycle through BayesProbe Core
emit Belief State Projection
```

MVP synchronization:

- fixed `round_id`
- participant id
- one-round or N-round window
- manual or deadline-based boundary closure
- passive-only cycles allowed
- no real-time interruption of belief update
- no direct internal-state sharing

Synchronized controller should support three use cases:

1. Passive-only review round.
2. Active-only self-update round inside collaboration window.
3. Active-plus-passive mixed round.

### 8.3 Shared Controller Invariants

- Controllers cannot create Evidence Events directly.
- Controllers cannot update posterior.
- Controllers cannot retire/spawn/reframe hypotheses.
- Controllers cannot bypass Signal Inbox or Evidence Integration Gate.
- Controllers must persist cycle records.

## 9. Projection Handling

### 9.1 Incoming Projection

Incoming Belief State Projection from another agent or human:

```text
incoming projection
→ Passive External Signal
→ Signal Inbox
→ Evidence Integration Gate
```

It is never accepted directly as evidence.

### 9.2 Decomposition

If projection contains a conclusion and cited evidence:

```text
Sender judgment:
  Passive External Signal, source_type = external_agent_projection

Cited source claim:
  Passive External Signal or Probe Candidate for direct verification
```

Example:

```text
Agent X: "H2 is likely because Table 3 in Paper A supports it."
```

Decompose:

- `S1`: Agent X believes H2 is likely.
- `S2`: Agent X claims Paper A Table 3 supports H2.
- `PC1`: retrieve Paper A Table 3 directly.

### 9.3 Outgoing Projection

Belief State Projection should avoid flooding another agent with full state.

It must expose:

- current best hypothesis
- confidence/posterior summary
- main evidence events
- main uncertainties
- questions for others
- Change-My-Mind Condition
- requested signal type

## 10. Belief Update Algorithm

### 10.1 LR Bands

MVP uses qualitative bands mapped to conservative numeric LR values:

| Band | LR |
|---|---:|
| strongly_disconfirming | 0.1 |
| moderately_disconfirming | 0.3 |
| weakly_disconfirming | 0.7 |
| neutral | 1.0 |
| weakly_confirming | 1.5 |
| moderately_confirming | 3.0 |
| strongly_confirming | 10.0 |

### 10.2 Weighted Update

Single-hypothesis log-odds:

```text
weighted_log_lr =
  log(LR)
  * reliability
  * independence
  * relevance
  * novelty

logit_posterior = logit_prior + weighted_log_lr
posterior = sigmoid(logit_posterior)
```

Multi-hypothesis softmax:

```text
score_i =
  log(prior_i)
  + weight * log(P(E | H_i))
  - complexity_penalty_i
  - ad_hoc_penalty_i

posterior_i = softmax(score_i over active hypotheses)
```

### 10.3 Guardrails

- Do not output unjustified precise probabilities.
- Strong update requires high reliability and high relevance.
- Low independence signals are downweighted or merged.
- Repeated projections from agents using the same cited source must not be counted as independent evidence.
- Passive signals have no special privilege or penalty solely because they are passive.

## 11. Hypothesis Evolution

### 11.1 Evolution Triggers

Triggers:

- Evidence Event has low likelihood under all active hypotheses.
- strong counterevidence against top hypothesis.
- repeated ad hoc narrowing.
- high posterior uncertainty persists after high-value probes.
- passive signal introduces a new explanatory frame.

### 11.2 New Hypothesis Entry

New hypothesis enters with:

- small but viable prior
- rationale
- scope
- rivals
- falsifier
- predictions
- complexity penalty
- why existing hypotheses failed
- at least one candidate probe that could weaken it

### 11.3 Lifecycle

```text
active
→ weakened
→ reframed / split
→ retired
→ archived
```

Retired hypotheses are not deleted. They remain in the ledger and can be reactivated.

## 12. Persistence Plan

### 12.1 MVP Storage

Start with JSONL or SQLite:

- JSONL is easiest for experiments and audit.
- SQLite is better for local querying.
- Postgres + JSONB can come later.

Recommended MVP:

```text
JSONL ledger for append-only audit
SQLite index for local query and benchmark scoring
```

If implementation speed matters, start with JSONL only and provide loader utilities.

### 12.2 Append-Only Ledger

Every cycle writes:

- cycle record
- probe set
- external signals
- evidence events
- belief updates
- hypothesis evolutions
- projections
- metrics snapshot

Never mutate historical ledger entries. Store corrections as new records.

### 12.3 Reproducibility

Each run should record:

- model name and parameters
- prompt version
- tool adapter versions
- signal order
- random seed if used
- scoring version
- schema version

## 13. Error Handling

### 13.1 Schema Failures

If model output fails schema:

1. retry with schema repair prompt
2. if still invalid, mark `schema_violation`
3. either discard signal or defer to manual review depending on module

### 13.2 Tool Failures

Tool failure becomes Active External Signal only if the failure itself is informative.

Otherwise:

- record failed probe execution
- no Evidence Event
- optionally create retry candidate

### 13.3 Late Passive Signal

If passive signal arrives after Signal Collection Boundary:

- assign to next cycle
- record `deferred_due_to_boundary`

### 13.4 Empty Inbox

Empty inbox is legal only if controller is stopping or emitting projection without update.

Otherwise:

- autonomous: design another Probe Set or stop if no high-value candidates remain
- synchronized: wait until fixed boundary or emit no-update projection if protocol requires

### 13.5 Contradictory Evidence

Contradictory Evidence Events should not be collapsed prematurely.

Core should:

- preserve both events
- update likelihoods separately
- surface uncertainty in projection
- generate candidate probes for source quality or boundary conditions

## 14. Testing Strategy

### 14.1 Unit Tests

Test modules through public interfaces:

- `BayesProbeCore.integrateCycle`
- `ProbeSetDesigner.design`
- `SignalInbox.closeBoundary`
- `ProjectionGenerator.generate`
- `AutonomousController.step`
- `SynchronizedController.step`

### 14.2 Golden Tests

Use deterministic fixture inputs:

- active-only FEVER-like claim
- passive-only external projection
- active-plus-passive mixed cycle
- anomaly-triggered new hypothesis
- repeated source downweighting

Expected outputs:

- Evidence Event classification
- update direction
- hypothesis lifecycle change
- projection fields

### 14.3 Property Checks

Invariants:

- no signal directly updates belief
- no projection directly becomes evidence
- controller does not produce Evidence Events
- every important update has prior/posterior/direction/reason
- retired hypotheses remain queryable
- every output has Change-My-Mind Condition

### 14.4 Integration Tests

Autonomous:

- run to stop condition
- verify Answer Projection

Synchronized:

- fixed round with passive-only inputs
- verify Belief State Projection

Mixed:

- active probe result plus passive agent projection
- verify shared Evidence Integration Gate

### 14.5 Evaluation Harness Tests

- final accuracy calculation
- update direction accuracy
- neutral signal drift
- projection decomposition accuracy
- duplicate evidence detection

## 15. MVP Implementation Roadmap

### Phase 0: Repository Setup

- choose implementation language
- create package structure
- add schema validation library
- add test runner
- add JSONL ledger

Recommended stack if not otherwise constrained:

- Python
- Pydantic for schemas
- pytest
- JSONL ledger
- optional SQLite index

### Phase 1: Core Data Schemas

Deliver:

- RunRecord
- CycleRecord
- BeliefState
- Hypothesis
- ProbeDesign
- ProbeSet
- ExternalSignal
- EvidenceEvent
- BeliefUpdate
- HypothesisEvolution
- AnswerProjection
- BeliefStateProjection

Acceptance:

- schemas validate sample fixtures
- JSON serialization round-trips

### Phase 2: Core Single-Cycle Integration

Deliver:

- Signal Inbox
- Evidence Integration Gate
- Evidence Event Builder
- Quality Assessor
- Likelihood Judge
- Belief Solver

Acceptance:

- active-only fixture updates posterior correctly
- passive-only fixture also uses same gate
- no signal bypasses evidence construction

### Phase 3: Controllers

Deliver:

- Autonomous Controller
- Synchronized Controller
- fixed-round synchronization
- boundary closure logic

Acceptance:

- autonomous active-only run completes
- synchronized passive-only run completes
- active-plus-passive mixed cycle completes

### Phase 4: Projection Protocol

Deliver:

- Belief State Projection
- Projection-as-Signal Rule
- Projection Decomposition Rule
- Change-My-Mind Condition schema
- Probe Candidate Pool generation

Acceptance:

- external projection never becomes Evidence Event directly
- cited source becomes independent signal or verification candidate

### Phase 5: Hypothesis Evolution

Deliver:

- anomaly detection
- spawn/split/reframe/retire
- New Hypothesis Entry Rule
- Hypothesis Lifecycle ledger

Acceptance:

- anomaly fixture spawns viable new hypothesis
- retired hypothesis remains in ledger

### Phase 6: Benchmark Harness

Deliver:

- active-only fixtures
- passive-only fixtures
- active-plus-passive smoke fixtures
- metrics calculation

Acceptance:

- outputs answer utility metrics
- outputs belief-revision quality metrics
- outputs cycle-shape metrics

## 16. Suggested Package Layout

If implemented in Python:

```text
bayesprobe/
  __init__.py
  core/
    core.py
    frame_builder.py
    hypothesis_manager.py
    probe_set_designer.py
    signal_inbox.py
    evidence_gate.py
    evidence_builder.py
    quality_assessor.py
    likelihood_judge.py
    belief_solver.py
    hypothesis_evolver.py
    projection_generator.py
  controllers/
    autonomous.py
    synchronized.py
  schemas/
    run.py
    cycle.py
    belief.py
    hypothesis.py
    probe.py
    signal.py
    evidence.py
    update.py
    projection.py
  adapters/
    model_gateway.py
    tool_gateway.py
    ledger_store.py
    clock.py
  ledger/
    jsonl_store.py
    sqlite_index.py
  eval/
    metrics.py
    harness.py
    fixtures.py
  tests/
```

Do not create this package until implementation starts. This layout is the proposed target.

## 17. Minimal API Sketch

### 17.1 Autonomous Run

```python
core = BayesProbeCore(model_gateway, ledger_store)
controller = AutonomousController(core, tool_gateway, ledger_store)

result = controller.run(
    problem=problem,
    context=context,
    config=AutonomousConfig(max_cycles=5, max_tool_calls=20),
)

print(result.answer_projection)
```

### 17.2 Synchronized Round

```python
core = BayesProbeCore(model_gateway, ledger_store)
controller = SynchronizedController(core, tool_gateway, ledger_store)

round_result = controller.process_round(
    run_id=run_id,
    round_id="round_03",
    passive_signals=[external_agent_projection],
    probe_policy="passive_only",
)

print(round_result.belief_state_projection)
```

### 17.3 Mixed Round

```python
round_result = controller.process_round(
    run_id=run_id,
    round_id="round_04",
    passive_signals=[human_feedback],
    probe_policy="allow_active",
)
```

## 18. First Engineering Risks

### Risk 1: Controller leaks into reasoning

Mitigation:

- keep `integrateCycle` as the only belief update entry point
- tests assert controller never creates Evidence Events or Belief Updates

### Risk 2: Passive signals bypass evidence judgment

Mitigation:

- all passive signals enter Signal Inbox
- no projection field is allowed to map directly to posterior

### Risk 3: LLM produces decorative posterior updates

Mitigation:

- structured LR bands
- required update direction and reason
- benchmark update-direction metrics

### Risk 4: Probe Set becomes unbounded search

Mitigation:

- bounded Probe Set
- frozen within cycle
- cost and expected value fields required

### Risk 5: Synchronized mode gets under-tested

Mitigation:

- passive-only fixture in Phase 2
- synchronized controller in Phase 3
- passive-only benchmark in Phase 6

## 19. Open Design Decisions

1. Implementation language: Python recommended, but not locked.
2. MVP persistence: JSONL-only vs JSONL + SQLite.
3. Model provider abstraction: one gateway with typed prompts vs separate gateways per judgment type.
4. Probe Set ranking: LLM-only vs deterministic ranker over LLM-generated candidates.
5. Initial benchmark fixture: FEVER/PubMedQA active-only vs external-agent projection passive-only.

## 20. Recommended Next Step

Before coding, resolve these two implementation choices:

1. Choose Python/Pydantic or TypeScript/Zod for schemas.
2. Choose JSONL-only or JSONL + SQLite for the first ledger.

After that, create an implementation plan with milestones Phase 0 through Phase 6.
