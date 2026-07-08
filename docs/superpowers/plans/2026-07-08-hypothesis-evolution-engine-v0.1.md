# Hypothesis Evolution Engine v0.1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract and deepen BayesProbe hypothesis evolution so anomaly, counterevidence, and stale hypotheses can produce auditable spawn, reframe, and retire operations.

**Architecture:** Create `bayesprobe/hypothesis_evolution.py` as a deep module with one external interface: `HypothesisEvolutionEngine.evolve(...)`. `BayesProbeCore` will call that interface after belief solving and will no longer contain concrete evolution rules or spawned hypothesis materialization logic.

**Tech Stack:** Python 3.11+, dataclasses, existing Pydantic schemas, pytest.

## Global Constraints

- `BayesProbeCore` must not contain concrete spawn/reframe/retire rules after this slice.
- Existing anomaly-spawn behavior and IDs must remain compatible with current tests.
- No merge, split, reject, or reactivate implementation in this slice.
- No LLM model gateway.
- No probabilistic structural learning.
- No changes to Evidence Integration Gate.
- No changes to Belief Solver likelihood math.
- No changes to benchmark scoring.
- Every spawn or reframe must create a follow-up `ProbeCandidate`.
- Retirement must require independent counterevidence and must ignore low-independence duplicate evidence.
- Do not attempt git commits because this workspace is not currently a git repository.

---

## File Structure

- Create `tests/test_hypothesis_evolution.py`: direct tests across the module interface.
- Create `bayesprobe/hypothesis_evolution.py`: config, result, engine, and internal rules.
- Modify `bayesprobe/core.py`: delegate evolution to the new engine and merge evolution probe candidates.
- Modify `tests/test_core_cycles.py`: add a core-level assertion that evolution probe candidates flow through result/ledger refs.

### Task 1: Hypothesis Evolution Module Tests

**Files:**
- Create: `tests/test_hypothesis_evolution.py`

**Interfaces:**
- Consumes planned:
  - `HypothesisEvolutionEngine`
  - `HypothesisEvolutionConfig`
  - `HypothesisEvolutionResult`
- Produces failing tests that define spawn, reframe, retire, and duplicate-protection behavior.

- [x] **Step 1: Write failing tests**

Create tests covering:

```python
def test_anomaly_spawns_hypothesis_and_probe_candidate():
    result = HypothesisEvolutionEngine().evolve(...)
    assert result.evolutions[0].operation == EvolutionOperation.SPAWN
    assert result.evolutions[0].to_hypothesis == "H_run_1_cycle_1_E1_spawned"
    assert result.hypotheses_by_id()["H_run_1_cycle_1_E1_spawned"].created_by == "spawned"
    assert result.probe_candidates[0].source == "anomaly"
```

Also add:

- `test_low_independence_duplicate_counterevidence_does_not_retire_hypothesis`
- `test_independent_counterevidence_retires_stale_hypothesis`
- `test_counterevidence_reframes_scoped_top_hypothesis`

- [x] **Step 2: Verify RED**

Run:

```bash
python3 -m pytest tests/test_hypothesis_evolution.py -q
```

Expected: failure because `bayesprobe.hypothesis_evolution` does not exist yet.

### Task 2: Hypothesis Evolution Engine Implementation

**Files:**
- Create: `bayesprobe/hypothesis_evolution.py`
- Test: `tests/test_hypothesis_evolution.py`

**Interfaces:**
- Produces:
  - `HypothesisEvolutionConfig`
  - `HypothesisEvolutionResult`
  - `HypothesisEvolutionEngine.evolve(...)`

- [x] **Step 1: Add dataclasses and public interface**

Implement:

```python
@dataclass(frozen=True)
class HypothesisEvolutionConfig:
    spawn_prior: float = 0.12
    reframe_drop_threshold: float = 0.08
    reframe_min_previous_posterior: float = 0.6
    retire_posterior_threshold: float = 0.2
    retire_min_independent_counterevents: int = 2
    independent_event_threshold: float = 0.5


@dataclass(frozen=True)
class HypothesisEvolutionResult:
    hypotheses: list[Hypothesis]
    evolutions: list[HypothesisEvolution]
    probe_candidates: list[ProbeCandidate] = field(default_factory=list)

    def hypotheses_by_id(self) -> dict[str, Hypothesis]:
        return {hypothesis.id: hypothesis for hypothesis in self.hypotheses}
```

- [x] **Step 2: Implement spawn rule**

Trigger on anomaly events and preserve existing ID behavior:

```python
spawned_hypothesis_id = f"H_{event.id}_spawned"
```

Create a spawned `Hypothesis`, `HypothesisEvolution`, and anomaly follow-up `ProbeCandidate`.

- [x] **Step 3: Implement retire rule**

Retire a hypothesis only when:

- updated posterior is below `retire_posterior_threshold`
- at least `retire_min_independent_counterevents` independent counterevidence events target it
- counted events have `independence >= independent_event_threshold`

Materialize retirement by copying the existing hypothesis with `status=HypothesisStatus.RETIRED`.

- [x] **Step 4: Implement reframe rule**

Reframe a hypothesis when:

- it receives a weakening update from counterevidence
- posterior drop is at least `reframe_drop_threshold`
- previous posterior is at least `reframe_min_previous_posterior`
- previous hypothesis has non-empty scope

Create a reframed hypothesis with ID:

```python
H_<hypothesis_id>_<cycle_id>_reframed
```

Create a `REFRAME` evolution and scope-disambiguation `ProbeCandidate`.

- [x] **Step 5: Run focused module tests**

Run:

```bash
python3 -m pytest tests/test_hypothesis_evolution.py -q
```

Expected: all direct evolution tests pass.

### Task 3: Core Integration

**Files:**
- Modify: `bayesprobe/core.py`
- Modify: `tests/test_core_cycles.py`

**Interfaces:**
- Consumes:
  - `HypothesisEvolutionEngine`
  - `HypothesisEvolutionResult`
- Produces:
  - core-level delegation to new engine
  - result/ledger refs containing both evidence-gate and evolution probe candidates

- [x] **Step 1: Add core integration test**

Update the existing anomaly test or add a new assertion:

```python
assert result.probe_candidates
assert result.probe_candidates[0].source == "anomaly"
assert result.belief_state.ledger_refs["probe_candidates"] == [
    result.probe_candidates[0].candidate_id
]
```

- [x] **Step 2: Verify RED for core integration**

Run:

```bash
python3 -m pytest tests/test_core_cycles.py::test_anomaly_triggers_hypothesis_evolution_before_next_probe -q
```

Expected: failure because current core anomaly evolution creates no evolution probe candidate.

- [x] **Step 3: Replace old core policy with engine delegation**

Modify `BayesProbeCore`:

- remove internal concrete `HypothesisEvolutionPolicy`
- import `HypothesisEvolutionEngine`
- `_create_hypothesis_evolution_policy` returns `HypothesisEvolutionEngine`
- call `evolve(...)`
- use `evolution_result.hypotheses`
- merge `integration.probe_candidates + evolution_result.probe_candidates`

- [x] **Step 4: Run focused core/evolution tests**

Run:

```bash
python3 -m pytest tests/test_hypothesis_evolution.py tests/test_core_cycles.py tests/test_autonomous_runner.py tests/test_inbox_and_ledger.py -q
```

Expected: all focused tests pass.

### Task 4: Regression Verification

**Files:**
- Test: all pytest files

**Interfaces:**
- Confirms the new evolution engine does not alter unrelated BayesProbe behavior.

- [x] **Step 1: Run full test suite**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
```

Expected: all tests pass with no failures.

- [x] **Step 2: Remove generated caches**

Run:

```bash
find . -type d \( -name __pycache__ -o -name .pytest_cache \) -prune -exec rm -rf {} +
```

Expected: no generated cache directories remain.

## Self-Review

- Spec coverage: The plan covers the new deep module, spawn/reframe/retire rules, duplicate-protected retirement, evolution probe candidates, core integration, focused tests, and full regression verification.
- Placeholder scan: No unspecified implementation placeholders remain.
- Type consistency: Public names and signatures match the design spec.
