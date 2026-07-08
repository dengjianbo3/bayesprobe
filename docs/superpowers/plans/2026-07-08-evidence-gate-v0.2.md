# Evidence Gate v0.2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the deterministic Evidence Integration Gate with projection decomposition, source-quality assessment, duplicate downweighting, and core-generated verification probe candidates.

**Architecture:** Extract evidence integration from `bayesprobe/core.py` into `bayesprobe/evidence.py`. `BayesProbeCore.integrate_cycle(...)` keeps the same call signature, normalizes either legacy `list[EvidenceEvent]` gate output or new `EvidenceIntegrationResult`, then writes generated probe candidates into `CycleResult`, ledger refs, and the ledger.

**Tech Stack:** Python 3.11+, dataclasses, existing Pydantic schemas, existing JSONL ledger, pytest.

## Global Constraints

- `BayesProbeCore.integrate_cycle(...)` arguments must not change.
- Controllers and runners must not create evidence, update posteriors, or evolve hypotheses.
- External projections remain passive `ExternalSignal`s and are not direct evidence.
- `SOURCE_CLAIM` events are neutral in v0.2 and generate verification `ProbeCandidate`s.
- Existing imports of `EvidenceIntegrationGate` from `bayesprobe.core` must continue to work.
- Existing custom gates returning `list[EvidenceEvent]` must continue to work.
- No LLM calls, real citation parser, cross-run source memory, new external dependency, or runner candidate-pool changes in this slice.
- Do not attempt git commits because this workspace is not currently a git repository.

---

## File Structure

- Create `bayesprobe/evidence.py`: evidence gate, result type, quality assessor, projection decomposer.
- Modify `bayesprobe/core.py`: import/re-export gate, normalize gate output, carry generated probe candidates into result, ledger refs, and ledger writes.
- Modify `tests/test_core_cycles.py`: add tests for projection decomposition, source quality, duplicate downweighting, generated candidates, ledger refs, and legacy gate compatibility.

### Task 1: Evidence Gate v0.2 Tests

**Files:**
- Modify: `tests/test_core_cycles.py`

**Interfaces:**
- Consumes: `BayesProbeCore`, `EvidenceIntegrationGate`, schemas
- Produces: failing tests for v0.2 gate behavior before implementation

- [x] **Step 1: Write failing tests**

Add tests covering:

- external projection with cited source decomposes into `SENDER_JUDGMENT` and `SOURCE_CLAIM`.
- sender judgment weakly supports only the endorsed hypothesis.
- source claim has neutral likelihoods and generates a verification `ProbeCandidate`.
- direct benchmark `REFUTES` still produces counterevidence and updates H1/H2.
- low-reliability signal caps reliability and verifiability.
- duplicate signals downweight later event independence and novelty.
- generated probe candidates are written to `CycleResult.probe_candidates`, `BeliefState.ledger_refs["probe_candidates"]`, and ledger records.
- legacy gate subclasses returning plain `list[EvidenceEvent]` continue to work.

- [x] **Step 2: Verify RED**

Run:

```bash
python3 -m pytest tests/test_core_cycles.py -q
```

Expected result: new tests fail because `CycleResult.probe_candidates` and v0.2 decomposition behavior do not exist yet.

### Task 2: Evidence Module Extraction And Gate Behavior

**Files:**
- Create: `bayesprobe/evidence.py`
- Modify: `bayesprobe/core.py`

**Interfaces:**
- Consumes: `CycleRecord`, `BeliefState`, `ProbeSet`, `ExternalSignal`
- Produces:
  - `EvidenceIntegrationResult`
  - `EvidenceIntegrationGate`
  - `SignalQualityAssessor`
  - `ProjectionDecomposer`

- [x] **Step 1: Add `bayesprobe/evidence.py`**

Implement:

- `EvidenceIntegrationResult(evidence_events, probe_candidates)`
- `SignalQualityAssessor.assess(...)`
- `ProjectionDecomposer.should_decompose(...)`
- `EvidenceIntegrationGate.integrate(...)`

- [x] **Step 2: Preserve direct signal semantics**

Keep current deterministic direct signal behavior for:

- `REFUTES` / `CONTRADICTS`
- `SUPPORTS`
- `ANOMALY`
- target-hypothesis resolution from signal targets and generated probe ids

- [x] **Step 3: Implement projection decomposition**

For `source_type="external_agent_projection"` with source cues, emit:

- one `SENDER_JUDGMENT` event
- one neutral `SOURCE_CLAIM` event
- one source-tracing `ProbeCandidate`

- [x] **Step 4: Implement quality rules**

Apply deterministic quality defaults for:

- direct signals
- external agent projections
- source claims
- low-reliability cues
- duplicate signals within cycle

### Task 3: Core Integration Of Gate Result

**Files:**
- Modify: `bayesprobe/core.py`

**Interfaces:**
- Consumes: `EvidenceIntegrationResult | list[EvidenceEvent]`
- Produces: `CycleResult.probe_candidates`, ledger refs, ledger writes

- [x] **Step 1: Import and re-export evidence gate**

`from bayesprobe.evidence import EvidenceIntegrationGate, EvidenceIntegrationResult`

Existing imports from `bayesprobe.core` must keep working.

- [x] **Step 2: Add `CycleResult.probe_candidates`**

Add a default empty list field so existing construction remains compatible.

- [x] **Step 3: Normalize gate output**

If gate returns `EvidenceIntegrationResult`, use its events and candidates.

If gate returns a plain list, treat it as events with no candidates.

- [x] **Step 4: Add ledger refs and ledger records**

Add generated candidate ids to `belief_state.ledger_refs["probe_candidates"]`.

Append `probe_candidate` records in `_append_ledger_records(...)`.

- [x] **Step 5: Verify GREEN for core tests**

Run:

```bash
python3 -m pytest tests/test_core_cycles.py -q
```

Expected result: core tests pass.

### Task 4: Regression Verification

**Files:**
- Test: all pytest files

**Interfaces:**
- Confirms: evidence gate v0.2 does not break autonomous, synchronized, or benchmark paths.

- [x] **Step 1: Run focused integration tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_core_cycles.py tests/test_controllers.py tests/test_question_runner.py tests/test_synchronized_runner.py tests/test_benchmark_harness.py -q -p no:cacheprovider
```

Expected result: all focused tests pass.

- [x] **Step 2: Run full test suite**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
```

Expected result: all tests pass with no failures.

- [x] **Step 3: Remove generated caches**

Run:

```bash
find . -type d \( -name __pycache__ -o -name .pytest_cache \) -prune -exec rm -rf {} +
```

Expected result: no generated cache directories remain.

## Self-Review

- Spec coverage: The plan covers extraction, projection decomposition, quality scoring, duplicate downweighting, generated probe candidates, ledger integration, and legacy gate compatibility.
- Placeholder scan: No unspecified implementation placeholders remain.
- Type consistency: Public class names and method signatures match the approved design spec.
