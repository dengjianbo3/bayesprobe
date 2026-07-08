# Question-to-Belief Initialization Design

## Goal

Build the first initialization layer that turns a user problem, optional context, and optional benchmark-provided hypothesis seeds into a valid BayesProbe run starting point: a `RunRecord`, an initial `BeliefState`, and an initial pool of `ProbeCandidate`s.

This is the missing bridge between "the user asks a question" and "the autonomous self-loop has a belief state to revise."

## Design Position

This feature belongs before probe execution and before benchmark harness work. The autonomous runner can already advance an existing belief state, but the system still lacks a first-class way to create that belief state from a question.

The initializer must preserve the BayesProbe philosophy:

- A question does not become an answer directly.
- A question becomes a bounded problem frame and a set of rival hypotheses.
- The initial hypotheses are provisional, conservative, and designed to be challenged by later signals.
- No raw signal becomes evidence during initialization.
- No posterior update happens during initialization.

## Non-Goals

This slice will not implement:

- LLM-backed hypothesis generation.
- Tool execution.
- Probe execution.
- Evidence interpretation.
- Belief updates.
- Synchronized round orchestration.
- Benchmark scoring.
- Natural-language parsing beyond deterministic MVP heuristics.

## New Module

Create:

```text
bayesprobe/initialization.py
tests/test_initialization.py
```

The module should expose a small public API:

```python
@dataclass(frozen=True)
class HypothesisSeed:
    id: str | None
    statement: str
    scope: str | None = None
    prior: float | None = None
    falsifiers: list[str] = field(default_factory=list)
    predictions: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class InitializeRunInput:
    run_id: str
    problem: str
    context: str = ""
    regime: RunRegime = RunRegime.AUTONOMOUS
    hypothesis_seeds: list[HypothesisSeed] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InitializationResult:
    run: RunRecord
    belief_state: BeliefState
    probe_candidates: list[ProbeCandidate]
```

And:

```python
class BayesProbeInitializer:
    def __init__(self, ledger: JsonlLedgerStore | None = None) -> None:
        ...

    def initialize(self, input: InitializeRunInput) -> InitializationResult:
        ...
```

## Behavior

### Input Validation

The initializer should reject:

- Empty or whitespace-only `run_id`.
- Empty or whitespace-only `problem`.
- Fewer than two effective hypotheses after seed/default generation.
- Invalid seed priors outside `[0, 1]`.

### Default Hypotheses

When no `hypothesis_seeds` are provided, the initializer creates two conservative rival hypotheses:

- `H1`: the claim/problem direction is supported or true enough to proceed.
- `H2`: the claim/problem direction is refuted, false, or materially misleading.

Both hypotheses should:

- Use `created_by="initial"`.
- Start with equal priors/posteriors, default `0.5` for two hypotheses.
- List each other as rivals.
- Include at least one falsifier.
- Include at least one prediction.
- Use the problem text as part of their statement/scope.

### Seeded Hypotheses

When seeds are provided:

- Preserve provided seed statements.
- Generate missing IDs as `H1`, `H2`, etc.
- Generate missing scopes from the problem.
- Generate missing falsifiers and predictions deterministically.
- Normalize missing priors to equal priors.
- Preserve explicit priors when provided, as long as they are valid.
- Set posterior equal to prior at initialization.
- Assign rivals to all other hypotheses.

This keeps benchmark fixtures stable: if a benchmark provides hypothesis candidates, BayesProbe should not rewrite them before evidence arrives.

### RunRecord

The initializer creates a `RunRecord` with:

- `run_id` from input.
- `regime` from input.
- `problem` from input.
- `status=RunStatus.RUNNING`.
- `current_cycle_id="cycle_0"`.
- `metadata` copied from input and enriched with deterministic initialization fields.

### BeliefState

The initializer creates a `BeliefState` with:

- `belief_state_id=f"{run_id}_bs_0"`.
- `run_id` from input.
- `cycle_id="cycle_0"`.
- `cycle_index=0`.
- The initialized hypotheses.
- `posterior_summary` containing at least:
  - `initialization_method`
  - `hypothesis_count`
  - `top_hypothesis`
- `uncertainty_summary` derived from the problem and rival hypothesis setup.

### Initial Probe Candidate Pool

The initializer creates one `ProbeCandidate` per initial hypothesis.

Each candidate should:

- Use `source="manual"` for MVP deterministic initialization.
- Target one hypothesis.
- Ask for information that could support or weaken that hypothesis.
- Include `support_condition` and `weaken_condition`.
- Use `cycle_id="cycle_0"`.

These candidates are not executed by this slice. They are a handoff to the later probe planner/executor slice.

### Ledger Behavior

If a `JsonlLedgerStore` is provided, append records in this order:

1. `run`
2. `belief_state`
3. one `probe_candidate` record per generated candidate

The initializer should not append `evidence_event`, `belief_update`, or `answer_projection` records.

## Integration With Existing Runtime

The output `belief_state` must be immediately usable by:

```python
AutonomousLoopRunner.run(
    run_id=result.run.run_id,
    initial_belief_state=result.belief_state,
    signal_provider=provider,
)
```

This verifies the initializer is not a parallel abstraction. It feeds the existing BayesProbe autonomous lifecycle.

## Testing Plan

Add behavior-first tests:

1. `test_initializer_creates_default_rival_hypotheses_from_problem`
   - No seeds.
   - Creates `RunRecord`, `BeliefState`, two rival hypotheses, and probe candidates.

2. `test_initializer_preserves_seeded_hypotheses`
   - Seeds contain benchmark-like statements.
   - Statements are preserved.
   - Missing IDs/scopes/falsifiers/predictions are filled.
   - Rivals are assigned.

3. `test_initializer_rejects_invalid_input`
   - Empty `run_id`.
   - Empty `problem`.
   - One seed only.
   - Invalid seed prior.

4. `test_initializer_writes_ledger_records_without_evidence_or_answers`
   - Uses `JsonlLedgerStore`.
   - Asserts record order and absence of evidence/update/projection records.

5. `test_initialized_belief_state_can_run_autonomous_loop`
   - Initializes a run.
   - Passes the belief state into `AutonomousLoopRunner`.
   - Uses a deterministic signal provider.
   - Confirms at least one cycle executes and returns an answer projection.

## Acceptance Criteria

- `bayesprobe/initialization.py` exposes `HypothesisSeed`, `InitializeRunInput`, `InitializationResult`, and `BayesProbeInitializer`.
- Initialization from a plain problem creates a valid initial BayesProbe state.
- Initialization from benchmark-provided seeds preserves benchmark hypotheses.
- The initializer does not update beliefs or emit answers.
- Ledger records are deterministic and do not include evidence/update/projection records.
- Existing tests still pass.
- New initialization tests pass.

## Known Follow-Ups

After this slice, the next likely layers are:

1. Probe planner that ranks the initial `ProbeCandidate` pool into a bounded `ProbeSet`.
2. Tool/probe executor adapter that turns selected probes into active `ExternalSignal`s.
3. Benchmark harness that loads `question_or_claim`, hypotheses, and cycle signal streams.
4. LLM-backed frame builder behind the same initializer interface.

## Self-Review

- Placeholder scan: No placeholder fields or deferred requirements remain.
- Internal consistency: Initialization creates state only; it does not perform evidence integration or posterior updates.
- Scope check: This is one focused implementation slice.
- Ambiguity check: Seed handling, default handling, ledger order, and runner integration are explicit.
