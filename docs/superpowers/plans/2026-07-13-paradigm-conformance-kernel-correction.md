# Paradigm-Conformance Kernel Correction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the BayesProbe MVP so posterior change is caused by reconciled information-root deltas rather than repeated model assertions, while preserving the atomic `Belief State -> Probe -> Signal -> Evidence -> Belief Update` paradigm.

**Architecture:** Native runs use Evidence Memory v3. Every assessed `EvidenceEvent` belongs to one deterministic Evidence Root; that root owns one replaceable current log-likelihood contribution, and only the difference between the previous and current contribution enters the solver. Probe execution and Evidence assessment receive blind briefs without posterior or winner information, while runners expose and stop on epistemic stagnation.

**Tech Stack:** Python 3.11+, Pydantic 2.7+, pytest 8, synchronous `ModelGateway`, JSONL ledger, vanilla JavaScript and Node test runner.

## Global Constraints

- Preserve the atomic pipeline: `Belief State -> Probe -> Signal -> Evidence -> Belief Update -> Belief State`.
- `EvidenceEvent` is an assessed semantic record, not direct authorization for additive posterior credit.
- One Evidence Root owns one current contribution; a later same-root assessment revises, retracts, or leaves that contribution unchanged.
- Only `EvidenceContributionDelta = current_root_contribution - previous_root_contribution` enters the native solver.
- Events from one root in one cycle are combined by arithmetic mean, never sum, and their input order cannot affect output.
- Model reasoning from one provider/model/run session shares one root, regardless of cycle or probe count.
- Independently rooted tool results, observations, sources, human inputs, and agent messages may accumulate.
- Probe execution and Evidence assessment requests must contain no prior, posterior, current winner, credit balance, or preferred-answer hint.
- After the first integrated cycle, a valid top-hypothesis falsification probe is reserved when one exists.
- A no-new-information cycle cannot increase confidence and autonomous mode stops with `epistemic_stagnation`.
- LLMs may assess Evidence semantics; deterministic code owns provenance, root identity, contribution reconciliation, and update arithmetic.
- Historical memory v1/v2 artifacts remain readable. Native writes use memory v3; explicit legacy migration keeps the historical update path.
- Do not add a proposition graph, general tool ecosystem, multi-model debate, benchmark-specific likelihood tuning, or a new runtime dependency.
- Provider credentials remain request-scoped and never enter prompts, traces, ledger records, fixtures, artifacts, or errors.
- Each task follows red-green-refactor, runs its focused tests, and leaves `git diff --check` clean. The full offline suite is the final integration gate.

---

## File Map

### New deep module

- Create `bayesprobe/evidence_roots.py`: canonical root resolution, per-event candidate vectors, same-cycle grouping, root replacement, contribution deltas, and epistemic progress.

### Existing modules with focused edits

- Modify `bayesprobe/schemas.py`: root contribution, delta, progress, memory-v3, and root-bound Evidence Event contracts.
- Modify `bayesprobe/evidence_memory.py`: v3 identity/history commits without correlation-credit accumulation; retain v1/v2 behavior.
- Modify `bayesprobe/initialization.py`: initialize native runs with memory v3.
- Modify `bayesprobe/migrations.py`: keep explicit legacy migrations on memory v1.
- Modify `bayesprobe/lifecycle.py`: fail closed when a native run does not carry memory v3.
- Modify `bayesprobe/evidence.py`: blind Evidence request, root assignment, batch reconciliation, and contribution output.
- Modify `bayesprobe/belief.py`: consume contribution deltas and adapt legacy events before solving.
- Modify `bayesprobe/core.py`: atomically commit deltas/progress, ledger them, and pass only deltas to the solver.
- Modify `bayesprobe/probe_executor.py`: replace the full-state execution context with a blind execution brief.
- Modify `bayesprobe/probe_planner.py`: distinguish actual falsification from generic top-hypothesis targeting.
- Modify `bayesprobe/question_runner.py`: expose contribution progress and stop on epistemic stagnation.
- Modify `bayesprobe/synchronized_runner.py`: use the blind brief and expose contribution progress without adding autonomous stopping.
- Modify `bayesprobe/webui.py` and `bayesprobe/webui_static/app.js`: show root deltas and epistemic progress in cycle traces.
- Modify `bayesprobe/evaluation/arms.py`: record root and falsification process metrics for later HLE analysis.
- Modify `bayesprobe/__init__.py`: export the approved public contracts.
- Modify `docs/ARCHITECTURE.md` and `CONTEXT.md`: record the implemented native-v3 semantics and remaining experiment checkpoint.

### Focused tests

- Create `tests/test_evidence_roots.py`.
- Create `tests/test_paradigm_conformance.py`.
- Modify `tests/test_schemas.py`, `tests/test_initialization.py`, `tests/test_migrations.py`, and `tests/test_evidence_memory.py`.
- Modify `tests/test_evidence.py`, `tests/test_belief.py`, `tests/test_frame_policy.py`, and `tests/test_core_cycles.py`.
- Modify `tests/test_probe_executor.py`, `tests/test_probe_planner.py`, `tests/test_question_runner.py`, and `tests/test_synchronized_runner.py`.
- Modify `tests/test_webui.py`, `tests/test_webui_stream.js`, and `tests/evaluation/test_bayesprobe_arm.py`.

---

### Task 0: Freeze the Approved Baseline and Create an Isolated Worktree

**Files:**
- Verify only; no source edit.

**Interfaces:**
- Consumes: `main` containing the approved design and this implementation plan.
- Produces: branch `codex/paradigm-conformance-kernel` at `.worktrees/paradigm-conformance-kernel`.

- [ ] **Step 1: Verify the plan branch and baseline are clean**

Run:

```bash
cd /Users/dengjianbo/Documents/BayesProbe
git status --short
git diff --check
python3 -m pytest -q -p no:cacheprovider
node --test tests/test_webui_stream.js
```

Expected: Git commands produce no output; Python and Node suites pass without failures. Record the exact baseline counts in the execution log rather than hard-coding an old count.

- [ ] **Step 2: Create the isolated implementation worktree**

Run:

```bash
cd /Users/dengjianbo/Documents/BayesProbe
git worktree add .worktrees/paradigm-conformance-kernel -b codex/paradigm-conformance-kernel main
git -C .worktrees/paradigm-conformance-kernel status --short --branch
```

Expected: the new branch is shown with no modified files.

---

### Task 1: Add Memory-v3 and Contribution Contracts

**Files:**
- Modify: `bayesprobe/schemas.py`
- Modify: `bayesprobe/__init__.py`
- Modify: `tests/test_schemas.py`

**Interfaces:**
- Produces: `EvidenceContributionMode`, `EvidenceRootContribution`, `EvidenceContributionDelta`, and `EpistemicProgress`.
- Extends: `EvidenceMemorySnapshot.root_contributions: dict[str, EvidenceRootContribution]`.
- Extends: `EvidenceEvent.contribution_root_id: str | None`.
- Compatibility: memory v1/v2 reject root contributions; memory v3 rejects correlation credit; historical rootless v0.2 Events still require `effective_update_weight`.

- [ ] **Step 1: Write failing schema tests**

Add these cases to `tests/test_schemas.py`:

```python
def test_v3_memory_owns_root_contributions_without_correlation_credit():
    contribution = EvidenceRootContribution(
        contribution_root_id="eroot:model-run",
        revision=1,
        assessment_event_ids=["E1", "E2"],
        epistemic_origin=EpistemicOrigin.MODEL_REASONING,
        per_hypothesis_log_likelihood={"H1": 0.25, "H2": -0.25},
        active=True,
    )
    memory = EvidenceMemorySnapshot(
        memory_version=3,
        root_contributions={contribution.contribution_root_id: contribution},
    )

    assert memory.root_contributions[contribution.contribution_root_id].revision == 1
    assert memory.correlation_credit == {}


def test_v3_memory_rejects_correlation_credit():
    with pytest.raises(ValueError, match="memory v3 does not use correlation credit"):
        EvidenceMemorySnapshot(
            memory_version=3,
            correlation_credit={"group|H1|confirming": 0.2},
        )


def test_v2_memory_rejects_root_contributions():
    contribution = EvidenceRootContribution(
        contribution_root_id="eroot:model-run",
        revision=1,
        assessment_event_ids=["E1"],
        epistemic_origin=EpistemicOrigin.MODEL_REASONING,
        per_hypothesis_log_likelihood={"H1": 0.1},
    )
    with pytest.raises(ValueError, match="root contributions require memory version 3"):
        EvidenceMemorySnapshot(
            memory_version=2,
            root_contributions={contribution.contribution_root_id: contribution},
        )


def test_root_bound_native_event_cannot_carry_legacy_effective_weight():
    with pytest.raises(ValueError, match="root-bound evidence uses contribution reconciliation"):
        make_native_evidence_event(
            contribution_root_id="eroot:model-run",
            effective_update_weight=0.5,
        )
```

Change the unsupported-version parameterization from `[0, 3, 999]` to `[0, 4, 999]`.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
python3 -m pytest tests/test_schemas.py -q -p no:cacheprovider
```

Expected: collection or assertions fail because the v3 contracts do not exist.

- [ ] **Step 3: Implement strict contribution models and version coherence**

Add the following contracts before `EvidenceMemorySnapshot` in `bayesprobe/schemas.py`:

```python
class EvidenceContributionMode(StrEnum):
    NEW_ROOT = "new_root"
    REVISE_ROOT = "revise_root"
    RETRACT_ROOT = "retract_root"
    NO_CHANGE = "no_change"


class EvidenceRootContribution(StrictTaskModel):
    contribution_root_id: str
    revision: int = Field(ge=1)
    assessment_event_ids: list[str]
    epistemic_origin: EpistemicOrigin
    per_hypothesis_log_likelihood: dict[str, float] = Field(default_factory=dict)
    unresolved_log_likelihood: float | None = None
    active: bool = True


class EvidenceContributionDelta(StrictTaskModel):
    contribution_root_id: str
    mode: EvidenceContributionMode
    previous_contribution: EvidenceRootContribution | None = None
    current_contribution: EvidenceRootContribution
    per_hypothesis_delta: dict[str, float] = Field(default_factory=dict)
    unresolved_delta: float | None = None
    caused_by_event_ids: list[str]


class EpistemicProgress(StrictTaskModel):
    new_root_count: int = Field(default=0, ge=0)
    revised_root_count: int = Field(default=0, ge=0)
    retracted_root_count: int = Field(default=0, ge=0)
    no_change_count: int = Field(default=0, ge=0)
    max_absolute_contribution_delta: float = Field(default=0.0, ge=0.0)
    falsification_probe_executed: bool = False
```

Use validators already present in `schemas.py` to enforce non-empty canonical ids, unique event ids, finite numeric maps, a root key matching `contribution_root_id`, and a delta whose previous/current roots match its own root. Update `EvidenceMemorySnapshot` coherence exactly as follows:

```python
if self.memory_version == 3 and self.correlation_credit:
    raise ValueError("memory v3 does not use correlation credit")
if self.memory_version in {1, 2} and self.root_contributions:
    raise ValueError("root contributions require memory version 3")
if self.memory_version not in {2, 3} and self.event_signal_identity_digests:
    raise ValueError("event signal identity bindings require memory version 2 or 3")
```

For native v0.2 Events, enforce one of two mutually exclusive contracts: historical rootless Event plus `effective_update_weight`, or root-bound Event plus `effective_update_weight=None`.

- [ ] **Step 4: Export the new contracts and verify GREEN**

Run:

```bash
python3 -m pytest tests/test_schemas.py -q -p no:cacheprovider
git diff --check
```

Expected: all schema tests pass and the diff check is silent.

- [ ] **Step 5: Commit the contract layer**

```bash
git add bayesprobe/schemas.py bayesprobe/__init__.py tests/test_schemas.py
git commit -m "feat: add evidence root contribution contracts"
```

---

### Task 2: Implement Deterministic Evidence Root Reconciliation

**Files:**
- Create: `bayesprobe/evidence_roots.py`
- Create: `tests/test_evidence_roots.py`

**Interfaces:**
- Produces: `resolve_contribution_root_id(signal: ExternalSignal) -> str`.
- Produces: `EvidenceRootReconciler.reconcile_cycle(snapshot, evidence_events, falsification_probe_executed) -> RootReconciliationResult`.
- Produces: `RootReconciliationResult(evidence_events, contribution_deltas, evidence_memory, epistemic_progress)`.
- Consumes: root-bound v0.2 `EvidenceEvent` records and memory v3 only.

- [ ] **Step 1: Write red tests for the invariants that exposed the HLE failure**

Create `tests/test_evidence_roots.py` with fixtures for root-bound Events and these assertions:

```python
def test_same_root_repetition_replaces_instead_of_accumulating():
    reconciler = EvidenceRootReconciler()
    first = reconciler.reconcile_cycle(
        snapshot=EvidenceMemorySnapshot(memory_version=3),
        evidence_events=[event("E1", root="eroot:model", h1=LikelihoodBand.MODERATELY_CONFIRMING)],
        falsification_probe_executed=False,
    )
    second = reconciler.reconcile_cycle(
        snapshot=first.evidence_memory,
        evidence_events=[event("E2", root="eroot:model", h1=LikelihoodBand.MODERATELY_CONFIRMING)],
        falsification_probe_executed=False,
    )

    assert first.contribution_deltas[0].mode == EvidenceContributionMode.NEW_ROOT
    assert second.contribution_deltas[0].mode == EvidenceContributionMode.NO_CHANGE
    assert second.contribution_deltas[0].per_hypothesis_delta == {"H1": 0.0, "H2": 0.0}


def test_same_cycle_same_root_events_are_meaned_and_order_independent():
    events = [
        event("E1", root="eroot:model", h1=LikelihoodBand.STRONGLY_CONFIRMING),
        event("E2", root="eroot:model", h1=LikelihoodBand.WEAKLY_DISCONFIRMING),
    ]
    forward = EvidenceRootReconciler().reconcile_cycle(
        snapshot=EvidenceMemorySnapshot(memory_version=3),
        evidence_events=events,
        falsification_probe_executed=False,
    )
    reverse = EvidenceRootReconciler().reconcile_cycle(
        snapshot=EvidenceMemorySnapshot(memory_version=3),
        evidence_events=list(reversed(events)),
        falsification_probe_executed=False,
    )

    assert forward.contribution_deltas == reverse.contribution_deltas
    assert len(forward.contribution_deltas) == 1


def test_same_root_counterassessment_can_reverse_prior_contribution():
    first = reconcile_one("E1", LikelihoodBand.STRONGLY_CONFIRMING)
    second = EvidenceRootReconciler().reconcile_cycle(
        snapshot=first.evidence_memory,
        evidence_events=[event("E2", root="eroot:model", h1=LikelihoodBand.STRONGLY_DISCONFIRMING)],
        falsification_probe_executed=True,
    )

    delta = second.contribution_deltas[0]
    assert delta.mode == EvidenceContributionMode.REVISE_ROOT
    assert delta.per_hypothesis_delta["H1"] < 0
    assert second.epistemic_progress.falsification_probe_executed is True


def test_independent_roots_create_two_independent_contributions():
    result = EvidenceRootReconciler().reconcile_cycle(
        snapshot=EvidenceMemorySnapshot(memory_version=3),
        evidence_events=[
            event("E1", root="eroot:source-a", h1=LikelihoodBand.MODERATELY_CONFIRMING),
            event("E2", root="eroot:source-b", h1=LikelihoodBand.MODERATELY_CONFIRMING),
        ],
        falsification_probe_executed=False,
    )

    assert [item.mode for item in result.contribution_deltas] == [
        EvidenceContributionMode.NEW_ROOT,
        EvidenceContributionMode.NEW_ROOT,
    ]
    assert result.epistemic_progress.new_root_count == 2
```

Also test all-neutral reassessment retracts a previously active root, a same-root assessment can move support from B to C while removing the old B position, discarded Events contribute zero, a derived summary inherits its source root, a model signal from the same provider/session resolves to the same root across cycles, and deterministic tool roots with different canonical inputs remain distinct.

- [ ] **Step 2: Run the new tests and verify RED**

Run:

```bash
python3 -m pytest tests/test_evidence_roots.py -q -p no:cacheprovider
```

Expected: import fails because `bayesprobe.evidence_roots` does not exist.

- [ ] **Step 3: Implement the root resolver and candidate-vector arithmetic**

Use a canonical SHA-256 id. Model reasoning, retrieved sources, human input, agent messages, and derived summaries use canonical `correlation_group`; tool results and direct external observations use `derivation_root_id`:

```python
_CORRELATION_ROOTED_ORIGINS = frozenset(
    {
        EpistemicOrigin.MODEL_REASONING,
        EpistemicOrigin.RETRIEVED_SOURCE,
        EpistemicOrigin.HUMAN_INPUT,
        EpistemicOrigin.AGENT_MESSAGE,
        EpistemicOrigin.DERIVED_SUMMARY,
    }
)


def resolve_contribution_root_id(signal: ExternalSignal) -> str:
    provenance = signal.provenance
    if provenance is None:
        raise ValueError("contribution root resolution requires normalized provenance")
    basis = (
        provenance.correlation_group
        if provenance.epistemic_origin in _CORRELATION_ROOTED_ORIGINS
        else provenance.derivation_root_id
    )
    encoded = json.dumps(
        {"basis": basis, "origin": provenance.epistemic_origin.value},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    return f"evidence-root:sha256:{digest}"
```

For each accepted Event, compute:

```python
quality = event.reliability * event.independence * event.relevance * event.novelty
candidate[hypothesis_id] = quality * math.log(LIKELIHOOD_RATIO_BY_BAND[band])
```

Define `LIKELIHOOD_RATIO_BY_BAND` once in this module with the exact ratios currently used by `belief.py`. Sort Event ids and root ids before aggregation. Arithmetic-mean every hypothesis coordinate and unresolved coordinate within a root.

- [ ] **Step 4: Implement replacement and delta modes**

For each root, create revision `1` when absent or `previous.revision + 1` when present. Compare complete vectors with absolute tolerance `1e-12`:

```python
if previous is None:
    mode = EvidenceContributionMode.NEW_ROOT
elif vectors_equal(previous, current):
    mode = EvidenceContributionMode.NO_CHANGE
elif previous.active and not current.active:
    mode = EvidenceContributionMode.RETRACT_ROOT
else:
    mode = EvidenceContributionMode.REVISE_ROOT
```

Calculate every delta over the union of old and new hypothesis ids, store the latest contribution in a copied memory snapshot, and derive `EpistemicProgress` from the resulting modes and maximum absolute delta. Never mutate the input snapshot.

- [ ] **Step 5: Verify GREEN and commit**

Run:

```bash
python3 -m pytest tests/test_evidence_roots.py tests/test_schemas.py -q -p no:cacheprovider
git diff --check
git add bayesprobe/evidence_roots.py tests/test_evidence_roots.py
git commit -m "feat: reconcile evidence by information root"
```

Expected: focused tests pass and one root produces at most one delta per cycle.

---

### Task 3: Integrate Blind Evidence Assessment with Native Memory v3

**Files:**
- Modify: `bayesprobe/evidence_memory.py`
- Modify: `bayesprobe/evidence.py`
- Modify: `bayesprobe/initialization.py`
- Modify: `bayesprobe/migrations.py`
- Modify: `bayesprobe/lifecycle.py`
- Modify: `tests/test_evidence_memory.py`
- Modify: `tests/test_evidence.py`
- Modify: `tests/test_initialization.py`
- Modify: `tests/test_migrations.py`

**Interfaces:**
- Extends: `EvidenceIntegrationResult.contribution_deltas` and `.epistemic_progress`.
- Produces: native memory-v3 integration that commits identity/history, then reconciles roots once per closed cycle.
- Preserves: existing v1/v2 correlation-credit path only for explicit legacy migration.

- [ ] **Step 1: Write failing tests for native-v3 activation and Evidence blindness**

Add tests proving:

```python
def test_native_initializer_uses_memory_v3():
    result = initialize_native_fixture()
    assert result.belief_state.evidence_memory.memory_version == 3
    assert result.belief_state.evidence_memory.root_contributions == {}


def test_legacy_migration_keeps_memory_v1():
    migrated = migrate_belief_state_v0_1(make_legacy_state())
    assert migrated.evidence_memory.memory_version == 1


def test_native_lifecycle_rejects_historical_memory_semantics():
    native = make_native_state().model_copy(
        update={"evidence_memory": EvidenceMemorySnapshot(memory_version=2)}
    )
    with pytest.raises(ValueError, match="native v0.2 requires evidence memory version 3"):
        resolve_belief_lifecycle(native)


def test_evidence_judge_request_is_blind_to_belief_and_credit():
    gateway = RecordingEvidenceGateway()
    integrate_native_signal(gateway=gateway)
    request = gateway.requests[-1]
    forbidden = {
        "prior",
        "posterior",
        "current_best_hypothesis",
        "correlation_credit",
        "remaining_credit",
        "support_condition",
        "weaken_condition",
        "reframe_condition",
    }
    assert forbidden.isdisjoint(recursive_keys(request.input))
    assert request.metadata["belief_context_policy"] == "blind_no_scores_v1"
```

Add a two-cycle integration test where two different model outputs from one run produce one stored root and the second cycle returns `revise_root` or `no_change`, never `new_root`.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
python3 -m pytest tests/test_evidence_memory.py tests/test_evidence.py tests/test_initialization.py tests/test_migrations.py -q -p no:cacheprovider
```

Expected: native initialization still uses historical memory and requests still expose posterior/credit fields.

- [ ] **Step 3: Separate identity commit from legacy credit commit**

In `EvidenceMemoryManager`, accept versions `{1, 2, 3}`. Add:

```python
def commit_identity(
    self,
    snapshot: EvidenceMemorySnapshot,
    *,
    signal: ExternalSignal,
    event: EvidenceEvent,
) -> EvidenceMemorySnapshot:
    """Commit canonical signal/event identity without assigning update credit."""
```

Move accepted/discarded event history, signal digest binding, discovery ids, and counterevidence indexing from `commit` into `commit_identity`. Keep `commit` as the v1/v2 wrapper that calls `commit_identity` and then applies correlation credit. For v3, preserve `memory_version=3`, copy `root_contributions`, and require empty `correlation_credit`.

- [ ] **Step 4: Add native batch reconciliation to the Evidence Gate**

Extend the result dataclass with default-safe fields:

```python
@dataclass(frozen=True)
class EvidenceIntegrationResult:
    evidence_events: list[EvidenceEvent]
    probe_candidates: list[ProbeCandidate]
    evidence_memory: EvidenceMemorySnapshot | None = None
    normalized_signals: list[ExternalSignal] | None = None
    contribution_deltas: list[EvidenceContributionDelta] = field(default_factory=list)
    epistemic_progress: EpistemicProgress = field(default_factory=EpistemicProgress)
```

In the memory-v3 path, retain provenance normalization, duplicate detection, and model judgment. Replace `_apply_memory_decision` credit assignment with:

```python
root_id = resolve_contribution_root_id(signal)
event = event.model_copy(
    update={
        "contribution_root_id": root_id,
        "effective_update_weight": None,
    }
)
working_memory = self._memory_manager.commit_identity(
    working_memory,
    signal=signal,
    event=event,
)
```

After every Signal in the closed boundary has been assessed, invoke `EvidenceRootReconciler.reconcile_cycle` once. Compute `falsification_probe_executed` only when an accepted active Signal names `generated_by_probe` for a `ProbePurpose.HYPOTHESIS_FALSIFICATION` probe in the frozen Probe Set. Return the reconciler's memory, deltas, and progress atomically.

- [ ] **Step 5: Build the Evidence request from an explicit allowlist**

The `judge_evidence` input may contain problem/task context, hypothesis id/statement/scope/predictions/falsifiers, raw Signal, normalized provenance, and probe id/purpose/inquiry goal/expected observation. Remove posterior values, winner summaries, correlation statuses/credit, and designer-authored support/weaken/reframe conditions. Set request metadata `belief_context_policy` to `blind_no_scores_v1`; `ModelInvocationTrace` carries this explicit policy into the Evidence Event ledger trace.

- [ ] **Step 6: Activate memory v3 for native initialization and fail closed**

Change native initialization to `EvidenceMemorySnapshot(memory_version=3)`. Leave migration as `EvidenceMemorySnapshot(memory_version=1)`. In `resolve_belief_lifecycle`, require native states to carry memory v3 and legacy-migrated states to carry memory v1 or v2.

- [ ] **Step 7: Verify focused behavior and commit**

Run:

```bash
python3 -m pytest tests/test_evidence_memory.py tests/test_evidence.py tests/test_initialization.py tests/test_migrations.py -q -p no:cacheprovider
git diff --check
git add bayesprobe/evidence_memory.py bayesprobe/evidence.py bayesprobe/initialization.py bayesprobe/migrations.py bayesprobe/lifecycle.py tests/test_evidence_memory.py tests/test_evidence.py tests/test_initialization.py tests/test_migrations.py
git commit -m "feat: integrate blind root-reconciled evidence"
```

Expected: focused tests pass; no native Evidence request contains a forbidden belief/credit key.

---

### Task 4: Make the Solver and Core Consume Only Contribution Deltas

**Files:**
- Modify: `bayesprobe/belief.py`
- Modify: `bayesprobe/core.py`
- Modify: `tests/test_belief.py`
- Modify: `tests/test_frame_policy.py`
- Modify: `tests/test_core_cycles.py`

**Interfaces:**
- Changes: `CoverageAwareBeliefSolver.solve(belief_state, contribution_deltas, *, run_id, cycle_id)`.
- Produces: `legacy_event_contribution_deltas(events) -> list[EvidenceContributionDelta]` for explicit legacy migration only.
- Extends: `CycleResult.contribution_deltas` and `.epistemic_progress`.
- Ledger kinds: `evidence_contribution_delta` and `epistemic_progress`.

- [ ] **Step 1: Write solver red tests for replacement arithmetic**

Add to `tests/test_belief.py`:

```python
def test_no_change_delta_cannot_move_posterior():
    state = exclusive_state(h1=0.6, h2=0.4)
    result = CoverageAwareBeliefSolver().solve(
        state,
        [no_change_delta(root="eroot:model", hypotheses={"H1": 0.0, "H2": 0.0})],
        run_id=state.run_id,
        cycle_id="cycle_2",
    )

    assert result.hypotheses_by_id()["H1"].posterior == 0.6
    assert result.hypotheses_by_id()["H2"].posterior == 0.4
    assert result.belief_updates == []


def test_revision_applies_only_new_minus_previous_contribution():
    state = exclusive_state(h1=0.7, h2=0.3)
    result = CoverageAwareBeliefSolver().solve(
        state,
        [revision_delta(root="eroot:model", h1=-0.8, h2=0.8)],
        run_id=state.run_id,
        cycle_id="cycle_2",
    )

    assert result.hypotheses_by_id()["H1"].posterior < 0.7
    assert result.hypotheses_by_id()["H2"].posterior > 0.3
```

Add an independent-frame test, an unresolved-mass delta test, and an order-invariance test for deltas from distinct roots.

- [ ] **Step 2: Run solver tests and verify RED**

Run:

```bash
python3 -m pytest tests/test_belief.py tests/test_frame_policy.py -q -p no:cacheprovider
```

Expected: solver rejects contribution deltas because it still expects Events.

- [ ] **Step 3: Replace Event multiplication with direct log-delta application**

For exclusive frames, calculate each root update from the current log scores:

```python
log_scores = {
    hypothesis.id: math.log(max(hypothesis.posterior, _MIN_PROBABILITY))
    for hypothesis in active_hypotheses
}
for delta in sorted(contribution_deltas, key=lambda item: item.contribution_root_id):
    for hypothesis_id in log_scores:
        log_scores[hypothesis_id] += delta.per_hypothesis_delta.get(hypothesis_id, 0.0)
```

Apply `unresolved_delta` to the unresolved log score when the frame is exclusive-open, then normalize once. For independent frames, add each hypothesis delta to its logit. Skip all-zero deltas and emit no `BeliefUpdate` for them. Use the root id as `BeliefUpdate.evidence_id` and store `caused_by_event_ids` plus mode in `sensitivity`.

Move the existing likelihood-ratio table to `evidence_roots.py`. The legacy adapter creates one synthetic root delta per historical accepted Event using its stored `effective_update_weight`; native code never invokes this adapter.

- [ ] **Step 4: Route native core integration through deltas and ledger them**

Extend `CycleResult` and `_append_ledger_records`. In `integrate_cycle`:

```python
contribution_deltas = (
    integration.contribution_deltas
    if lifecycle == BeliefLifecycle.NATIVE_V02
    else legacy_event_contribution_deltas(evidence_events)
)
solve_result = self._belief_solver.solve(
    authoritative_belief_state,
    contribution_deltas,
    run_id=authoritative_cycle.run_id,
    cycle_id=authoritative_cycle.cycle_id,
)
```

Fail closed when a native integration result contains accepted root-bound Events but omits their contribution deltas. Keep raw Events available to frame adequacy and hypothesis evolution; they do not enter posterior arithmetic.

- [ ] **Step 5: Add the core regression that directly reproduces self-reinforcement**

In `tests/test_core_cycles.py`, integrate the same model root in two cycles. Assert cycle one moves posterior, cycle two emits `no_change`, cycle-two posterior equals cycle-one posterior exactly, and the ledger has two Evidence Events but one active root contribution.

- [ ] **Step 6: Verify focused behavior and commit**

Run:

```bash
python3 -m pytest tests/test_belief.py tests/test_frame_policy.py tests/test_core_cycles.py -q -p no:cacheprovider
git diff --check
git add bayesprobe/belief.py bayesprobe/core.py tests/test_belief.py tests/test_frame_policy.py tests/test_core_cycles.py
git commit -m "fix: update beliefs from root contribution deltas"
```

Expected: repeated same-root reasoning cannot move posterior after its current contribution is already represented.

---

### Task 5: Blind Probe Execution and Reserve Real Falsification

**Files:**
- Modify: `bayesprobe/probe_executor.py`
- Modify: `bayesprobe/probe_planner.py`
- Modify: `bayesprobe/question_runner.py`
- Modify: `bayesprobe/synchronized_runner.py`
- Modify: `bayesprobe/__init__.py`
- Modify: `tests/test_probe_executor.py`
- Modify: `tests/test_probe_planner.py`

**Interfaces:**
- Replaces: `ProbeExecutionContext` with `ProbeExecutionBrief`.
- Produces: `build_probe_execution_brief(*, run_id: str, cycle_id: str, belief_state: BeliefState, problem: str, task_context: str = "", metadata: Mapping[str, Any] | None = None) -> ProbeExecutionBrief`.
- Produces: `_is_top_falsification(candidate, top_hypothesis_id) -> bool`.

- [ ] **Step 1: Write failing execution-blindness and planner tests**

Add:

```python
def test_model_probe_execution_receives_no_belief_scores():
    gateway = RecordingGateway()
    brief = build_probe_execution_brief(
        run_id="run-1",
        cycle_id="cycle-2",
        belief_state=make_native_belief_state(h1=0.91, h2=0.09),
        problem="Which claim survives testing?",
        task_context="Use only the supplied conditions.",
    )
    ModelBackedProbeToolGateway(gateway).execute_probe(
        probe=make_probe("P1", ["H1"]),
        context=brief,
    )

    assert not hasattr(brief, "belief_state")
    assert {"prior", "posterior", "current_best_hypothesis"}.isdisjoint(
        recursive_keys(gateway.requests[-1].input)
    )
    assert (
        gateway.requests[-1].metadata["belief_context_policy"]
        == "blind_no_scores_v1"
    )


def test_top_targeting_probe_is_not_mistaken_for_falsification():
    targeting = candidate(
        "C-target",
        target="H1",
        purpose=ProbePurpose.HYPOTHESIS_DISCRIMINATION,
        weaken_condition={},
        score=100.0,
    )
    falsifier = candidate(
        "C-falsify",
        target="H1",
        purpose=ProbePurpose.HYPOTHESIS_FALSIFICATION,
        weaken_condition={"H1": "Observation X would contradict H1."},
        score=1.0,
    )
    result = ProbePlanner().design_probe_set(
        run_id="run-1",
        cycle_id="cycle-2",
        belief_state=make_state(cycle_index=1, h1=0.8, h2=0.2),
        candidates=[targeting, falsifier],
        config=ProbePlanningConfig(max_probes=1),
    )

    assert [probe.id for probe in result.probe_set.probes] == ["P-C-falsify"]
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
python3 -m pytest tests/test_probe_executor.py tests/test_probe_planner.py -q -p no:cacheprovider
```

Expected: the current execution context exposes `belief_state`, and generic targeting satisfies the planner guard.

- [ ] **Step 3: Implement the blind execution brief**

Define immutable hypothesis views with id, statement, scope, predictions, and falsifiers. `ProbeExecutionBrief` contains run id, cycle id, problem, task context, task frame, provider schema version, hypothesis views, and secret-free metadata. `build_probe_execution_brief` is the only function allowed to read a BeliefState before execution; downstream gateways receive only the brief.

Build the model request from the brief and omit every probability field. Set request metadata `belief_context_policy` to `blind_no_scores_v1` so provider invocation traces audit the blind interface. Update deterministic, model-backed, Python-augmented, autonomous, and synchronized execution call sites.

- [ ] **Step 4: Require actual falsification semantics in the planner**

Implement:

```python
def _is_top_falsification(
    candidate: ProbeCandidate,
    top_hypothesis_id: str,
) -> bool:
    probe = candidate.candidate_probe
    return (
        probe.purpose == ProbePurpose.HYPOTHESIS_FALSIFICATION
        and top_hypothesis_id in probe.target_hypotheses
        and bool(probe.weaken_condition.get(top_hypothesis_id, "").strip())
    )
```

Only after `belief_state.cycle_index > 0`, reserve the highest-ranked valid falsifier when one exists. Apply `attack_top_hypothesis_bonus` only to actual falsifiers, not every candidate that mentions the top hypothesis.

- [ ] **Step 5: Verify focused behavior and commit**

Run:

```bash
python3 -m pytest tests/test_probe_executor.py tests/test_probe_planner.py -q -p no:cacheprovider
git diff --check
git add bayesprobe/probe_executor.py bayesprobe/probe_planner.py bayesprobe/question_runner.py bayesprobe/synchronized_runner.py bayesprobe/__init__.py tests/test_probe_executor.py tests/test_probe_planner.py
git commit -m "fix: blind probe execution and reserve falsifiers"
```

Expected: model execution payloads reveal no current ranking, and top-hypothesis targeting alone does not satisfy falsification.

---

### Task 6: Add Epistemic Stagnation to Both Control Regimes

**Files:**
- Modify: `bayesprobe/question_runner.py`
- Modify: `bayesprobe/synchronized_runner.py`
- Modify: `tests/test_question_runner.py`
- Modify: `tests/test_synchronized_runner.py`

**Interfaces:**
- Adds: `AutonomousQuestionStopReason.EPISTEMIC_STAGNATION`.
- Extends: `AutonomousQuestionCycleResult` and `SynchronizedRoundResult` with contribution deltas and epistemic progress.
- Autonomous behavior: stop after a cycle with no new/revised/retracted root, no nonzero delta, and no frame/hypothesis revision.
- Synchronized behavior: report stagnation-relevant progress but wait for the external round controller.

- [ ] **Step 1: Write failing autonomous and synchronized tests**

Add:

```python
def test_autonomous_runner_stops_when_same_root_adds_no_information():
    runner = repeated_same_root_runner(max_cycles=10)
    result = runner.run_question(make_input())

    assert result.stop_reason == AutonomousQuestionStopReason.EPISTEMIC_STAGNATION
    assert len(result.cycle_results) == 2
    assert result.cycle_results[-1].epistemic_progress.no_change_count == 1
    assert (
        result.cycle_results[-1].belief_state.hypotheses_by_id()["H1"].posterior
        == result.cycle_results[-2].belief_state.hypotheses_by_id()["H1"].posterior
    )


def test_synchronized_round_exposes_stagnation_without_ending_external_session():
    first, second = run_two_synchronized_same_root_rounds()
    assert second.epistemic_progress.no_change_count == 1
    assert second.contribution_deltas[0].mode == EvidenceContributionMode.NO_CHANGE
    assert second.belief_state.run_id == first.belief_state.run_id
```

Also test that a genuinely new independent root prevents stagnation and that hypothesis evolution or frame-version change counts as progress even when the numeric root delta is zero.

- [ ] **Step 2: Run focused tests and verify RED**

Run:

```bash
python3 -m pytest tests/test_question_runner.py tests/test_synchronized_runner.py -q -p no:cacheprovider
```

Expected: the stop reason and result fields do not exist.

- [ ] **Step 3: Propagate contribution progress through both result types**

Copy `core_result.contribution_deltas` and `core_result.epistemic_progress` into each autonomous cycle and synchronized round result. Do not recompute these values in a controller.

- [ ] **Step 4: Implement the autonomous stagnation predicate**

Use the core-owned progress plus actual frame/evolution changes:

```python
def _is_epistemically_stagnant(
    *,
    previous: BeliefState,
    current: BeliefState,
    cycle_result: CycleResult,
) -> bool:
    progress = cycle_result.epistemic_progress
    has_root_change = (
        progress.new_root_count
        + progress.revised_root_count
        + progress.retracted_root_count
    ) > 0
    frame_changed = previous.frame_state != current.frame_state
    return (
        not has_root_change
        and progress.max_absolute_contribution_delta == 0.0
        and not cycle_result.hypothesis_evolutions
        and not frame_changed
    )
```

Evaluate stagnation before confidence and posterior-stability stops so the trace reports why the loop could no longer learn. `max_cycles` remains the hard safety bound if reached before an integrated cycle can be classified.

- [ ] **Step 5: Verify focused behavior and commit**

Run:

```bash
python3 -m pytest tests/test_question_runner.py tests/test_synchronized_runner.py -q -p no:cacheprovider
git diff --check
git add bayesprobe/question_runner.py bayesprobe/synchronized_runner.py tests/test_question_runner.py tests/test_synchronized_runner.py
git commit -m "feat: stop autonomous runs on epistemic stagnation"
```

Expected: same-root loops terminate after demonstrating no new information; synchronized rounds remain externally controlled.

---

### Task 7: Expose Root Semantics in WebUI and Evaluation Metrics

**Files:**
- Modify: `bayesprobe/webui.py`
- Modify: `bayesprobe/webui_static/app.js`
- Modify: `bayesprobe/evaluation/arms.py`
- Modify: `tests/test_webui.py`
- Modify: `tests/test_webui_stream.js`
- Modify: `tests/evaluation/test_bayesprobe_arm.py`

**Interfaces:**
- Web cycle JSON adds `contribution_deltas` and `epistemic_progress`.
- Evaluation `process_metrics` adds root-mode counts, maximum root delta, falsification count, and stagnation stop status.

- [ ] **Step 1: Write failing serialization and metric tests**

Add assertions:

```python
cycle = serialize_autonomous_cycle_result(make_cycle_result())
assert set(cycle["epistemic_progress"]) == {
    "new_root_count",
    "revised_root_count",
    "retracted_root_count",
    "no_change_count",
    "max_absolute_contribution_delta",
    "falsification_probe_executed",
}
assert cycle["contribution_deltas"][0]["contribution_root_id"].startswith(
    "evidence-root:sha256:"
)
```

For the evaluation arm, assert `new_evidence_roots`, `revised_evidence_roots`, `retracted_evidence_roots`, `unchanged_evidence_roots`, `falsification_cycles`, `max_absolute_contribution_delta`, and `epistemic_stagnation` are present.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
python3 -m pytest tests/test_webui.py tests/evaluation/test_bayesprobe_arm.py -q -p no:cacheprovider
node --test tests/test_webui_stream.js
```

Expected: new JSON fields and rendered trace sections are absent.

- [ ] **Step 3: Serialize and render the root trace**

Add to `serialize_autonomous_cycle_result`:

```python
"contribution_deltas": _dump_domain(cycle.contribution_deltas),
"epistemic_progress": _dump_domain(cycle.epistemic_progress),
```

In `renderCycle`, place `Evidence root deltas` after `Evidence` and `Epistemic progress` before `Belief updates`. Use the existing `block` renderer; do not redesign the page.

- [ ] **Step 4: Extend HLE process metrics without changing scoring**

Aggregate progress directly from cycle results. Accuracy, gold handling, answer extraction, provider policy, and selection policy remain unchanged. The metrics must permit later comparison of cycle-one versus final answer, root novelty, revisions, falsification, and stagnation without reading model prose.

- [ ] **Step 5: Verify and commit**

Run:

```bash
python3 -m pytest tests/test_webui.py tests/evaluation/test_bayesprobe_arm.py -q -p no:cacheprovider
node --test tests/test_webui_stream.js
git diff --check
git add bayesprobe/webui.py bayesprobe/webui_static/app.js bayesprobe/evaluation/arms.py tests/test_webui.py tests/test_webui_stream.js tests/evaluation/test_bayesprobe_arm.py
git commit -m "feat: expose epistemic root progress"
```

Expected: WebUI and evaluation artifacts expose the same core-owned root semantics.

---

### Task 8: Prove Paradigm Conformance End to End and Update Architecture State

**Files:**
- Create: `tests/test_paradigm_conformance.py`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `CONTEXT.md`

**Interfaces:**
- Produces: executable regression proof for the five-stage BayesProbe loop.
- Documents: native memory-v3 status and the remaining frozen 30-case experiment checkpoint.

- [ ] **Step 1: Write the end-to-end conformance test before documentation**

Create a deterministic/scripted gateway fixture and cover these complete runs:

```python
def test_same_model_root_cannot_self_reinforce_across_ten_requested_cycles():
    result = run_scripted_same_root_question(max_cycles=10)
    posteriors = [
        cycle.belief_state.hypotheses_by_id()["H1"].posterior
        for cycle in result.cycle_results
    ]

    assert len(posteriors) == 2
    assert posteriors[1] == posteriors[0]
    assert result.stop_reason == AutonomousQuestionStopReason.EPISTEMIC_STAGNATION


def test_independent_tool_root_can_change_model_reasoning_conclusion():
    result = run_scripted_model_then_tool_counterevidence()
    assert result.cycle_results[0].contribution_deltas[0].mode == EvidenceContributionMode.NEW_ROOT
    assert result.cycle_results[1].epistemic_progress.new_root_count == 1
    assert result.final_belief_state.hypotheses_by_id()["H1"].posterior < 0.5


def test_same_root_revision_can_reverse_without_double_counting():
    result = run_scripted_same_root_reversal()
    first, second = result.cycle_results
    assert second.contribution_deltas[0].mode == EvidenceContributionMode.REVISE_ROOT
    assert second.belief_state.hypotheses_by_id()["H1"].posterior < 0.5
    assert len(second.belief_state.evidence_memory.root_contributions) == 1
```

The fixture must also assert the Signal Collection Boundary closes before Evidence integration, each accepted Event has a root, the solver receives no raw Event, and the final trace contains a falsification probe when one was available after cycle one.

- [ ] **Step 2: Run the conformance test and fix only integration defects**

Run:

```bash
python3 -m pytest tests/test_paradigm_conformance.py -q -p no:cacheprovider
```

Expected: all paradigm invariants pass. Do not tune likelihood ratios or prompts in response to answer correctness.

- [ ] **Step 3: Update architecture and project context**

Document:

- native memory v3 is implemented;
- Evidence Events remain audit records;
- Evidence Root Contribution is the unit of update ownership;
- same-root repeats revise rather than accumulate;
- executor and assessor blindness are enforced by request tests;
- autonomous stagnation is implemented, synchronized mode reports progress;
- HLE accuracy claims remain unchanged until the frozen 30-case process check is run.

- [ ] **Step 4: Run the complete offline verification gate**

Run:

```bash
python3 -m pytest -q -p no:cacheprovider
node --test tests/test_webui_stream.js
git diff --check
```

Expected: all offline tests pass, Node reports zero failures, and the diff check is silent. If a historical test expects additive same-root credit in a native run, update that test to explicit legacy migration or replace its expectation with root reconciliation; do not weaken the new invariant.

- [ ] **Step 5: Commit the conformance proof and documentation**

```bash
git add tests/test_paradigm_conformance.py docs/ARCHITECTURE.md CONTEXT.md
git commit -m "test: prove paradigm-conformant belief revision"
```

---

## Frozen 30-Case Experiment Checkpoint

This checkpoint follows implementation; it is not a reason to change kernel semantics during Tasks 1-8.

1. Use the previously completed restricted HLE v0.1 paired set as the source population.
2. Include only sample ids with terminal results for both `direct_flash` and `bayesprobe_python`; this recreates the frozen 77-case population without reading gold labels.
3. Sort candidates by `sha256("paradigm-conformance-v3:" + sample_id)`, then select the first exactly 30 ids.
4. Freeze those ids before loading `gold_store.json`.
5. Run corrected BayesProbe for exactly four cycles maximum with the existing provider/model/prompt controls; do not alter likelihood bands after viewing answers. Report the actual cycle-four state when reached and the terminal stopped state as the cycle-four-equivalent state when epistemic stagnation ends the run earlier.
6. Report cycle-one and final accuracy separately, answer-change matrix, correct-to-wrong and wrong-to-correct counts, new/revised/retracted/no-change root counts, falsification-cycle rate, stagnation rate, and same-root posterior-drift violations.
7. The methodological pass condition is zero same-root posterior-drift violations, zero confidence increases on no-change cycles, order-invariant same-cycle reconciliation, and falsification visibility. Accuracy superiority is measured but is not a kernel conformance requirement.

Do not start a 100-case rerun until the 30-case process report satisfies all methodological pass conditions.

---

## Self-Review

- **Spec coverage:** Tasks 1-4 implement root ownership and delta-only solving; Tasks 3 and 5 enforce assessor/executor blindness; Task 5 implements real falsification reservation; Task 6 implements stagnation in autonomous and visibility in synchronized mode; Task 7 makes the mechanism observable; Task 8 proves the atomic loop and preserves the experiment boundary.
- **Compatibility:** Memory v1/v2 and rootless v0.2 Events remain historical-readable. Native initialization and lifecycle move to memory v3 only. Legacy correlation credit is isolated from native runs.
- **Type consistency:** `assessment_event_ids` and `caused_by_event_ids` are plural throughout. The gate returns `EvidenceContributionDelta`; the solver consumes the same type; runners and WebUI expose it unchanged.
- **Scope:** No new provider, tool ecosystem, proposition graph, benchmark-specific tuning, or runtime dependency is introduced.
- **Failure interpretation:** A wrong semantic judgment remains visible as an Evidence problem; repeated accumulation is prevented by deterministic root reconciliation; stopping and falsification defects remain controller/planner problems rather than being hidden inside prompts.
