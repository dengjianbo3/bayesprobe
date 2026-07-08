# Question-to-Belief Initialization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a deterministic initialization layer that turns a problem plus optional hypothesis seeds into a valid BayesProbe `RunRecord`, initial `BeliefState`, and initial `ProbeCandidate` pool.

**Architecture:** Add one focused module, `bayesprobe/initialization.py`, that constructs initial state without performing evidence integration, posterior updates, probe execution, or answer generation. Tests verify that the initialized belief state feeds the existing `AutonomousLoopRunner`, keeping initialization inside the BayesProbe lifecycle rather than creating a parallel abstraction.

**Tech Stack:** Python 3.11+, dataclasses, existing Pydantic schemas in `bayesprobe.schemas`, existing `JsonlLedgerStore`, pytest.

## Global Constraints

- A question becomes a bounded problem frame and rival hypotheses; it does not become an answer directly.
- No raw signal becomes evidence during initialization.
- No posterior update happens during initialization.
- Do not implement LLM-backed hypothesis generation, tool execution, probe execution, synchronized orchestration, benchmark scoring, or natural-language parsing beyond deterministic MVP heuristics.
- Preserve benchmark-provided hypothesis statements before evidence arrives.
- Do not attempt git commits because this workspace is not currently a git repository.

---

## File Structure

- Create `tests/test_initialization.py`: behavior tests for default hypotheses, seeded hypotheses, validation, ledger records, and runner integration.
- Create `bayesprobe/initialization.py`: initializer dataclasses and deterministic construction logic.

### Task 1: Initialization Tests

**Files:**
- Create: `tests/test_initialization.py`

**Interfaces:**
- Consumes: `BayesProbeInitializer`, `InitializeRunInput`, `HypothesisSeed`, `AutonomousLoopRunner`, `AutonomousLoopConfig`, `ExternalSignal`
- Produces: failing tests for initialization behavior before production code exists

- [x] **Step 1: Write the failing tests**

Create `tests/test_initialization.py` with these behaviors:

```python
from pathlib import Path

import pytest

from bayesprobe.core import BayesProbeCore
from bayesprobe.initialization import BayesProbeInitializer, HypothesisSeed, InitializeRunInput
from bayesprobe.ledger import JsonlLedgerStore
from bayesprobe.runners import AutonomousLoopConfig, AutonomousLoopRunner
from bayesprobe.schemas import ExternalSignal, RunRegime, RunStatus, SignalKind


class OneBatchSignalProvider:
    def __init__(self):
        self.calls = 0

    def collect_signals(self, *, run_id, cycle_index, belief_state, previous_answer):
        self.calls += 1
        if self.calls > 1:
            return []
        return [
            ExternalSignal(
                id="S_init_support",
                cycle_id="pending",
                signal_kind=SignalKind.ACTIVE,
                source_type="benchmark_stream",
                source="fixture",
                raw_content="SUPPORTS: The initialized claim direction is supported.",
            )
        ]
```

Required tests:

- `test_initializer_creates_default_rival_hypotheses_from_problem`
- `test_initializer_preserves_seeded_hypotheses`
- `test_initializer_rejects_invalid_input`
- `test_initializer_writes_ledger_records_without_evidence_or_answers`
- `test_initialized_belief_state_can_run_autonomous_loop`

- [x] **Step 2: Verify RED**

Run:

```bash
python3 -m pytest tests/test_initialization.py -q
```

Expected result: failure because `bayesprobe.initialization` does not exist yet.

### Task 2: Initialization Module

**Files:**
- Create: `bayesprobe/initialization.py`

**Interfaces:**
- Consumes: `JsonlLedgerStore`, `BeliefState`, `Hypothesis`, `ProbeCandidate`, `ProbeDesign`, `RunRecord`, `RunRegime`, `RunStatus`
- Produces:
  - `HypothesisSeed`
  - `InitializeRunInput`
  - `InitializationResult`
  - `BayesProbeInitializer.initialize(input: InitializeRunInput) -> InitializationResult`

- [x] **Step 1: Add dataclasses and initializer shell**

Implement:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from bayesprobe.ledger import JsonlLedgerStore
from bayesprobe.schemas import (
    BeliefState,
    Hypothesis,
    ProbeCandidate,
    ProbeDesign,
    RunRecord,
    RunRegime,
    RunStatus,
)
```

And:

```python
@dataclass(frozen=True)
class HypothesisSeed:
    statement: str
    id: str | None = None
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

- [x] **Step 2: Implement validation and hypothesis construction**

Implement deterministic helpers:

- `_clean_required(value: str, field_name: str) -> str`
- `_validate_seed(seed: HypothesisSeed) -> None`
- `_default_seeds(problem: str) -> list[HypothesisSeed]`
- `_build_hypotheses(input: InitializeRunInput, problem: str) -> list[Hypothesis]`

Rules:

- Empty `run_id` or `problem` raises `ValueError`.
- One effective seed raises `ValueError`.
- Seed prior outside `[0, 1]` raises `ValueError`.
- Missing seed priors become `1 / seed_count`.
- Explicit seed priors are preserved.
- `posterior == prior` at initialization.
- Each hypothesis rivals every other hypothesis.

- [x] **Step 3: Implement run, belief state, probe candidates, and ledger writes**

`BayesProbeInitializer.initialize(...)` should:

1. Validate and clean input.
2. Build hypotheses.
3. Create `RunRecord(current_cycle_id="cycle_0", status=RunStatus.RUNNING)`.
4. Create `BeliefState(belief_state_id=f"{run_id}_bs_0", cycle_id="cycle_0", cycle_index=0)`.
5. Create one `ProbeCandidate` per hypothesis.
6. Append ledger records in order: `run`, `belief_state`, then each `probe_candidate`.
7. Return `InitializationResult`.

- [x] **Step 4: Verify GREEN**

Run:

```bash
python3 -m pytest tests/test_initialization.py -q
```

Expected result: all initialization tests pass.

### Task 3: Regression Verification

**Files:**
- Test: all pytest files

**Interfaces:**
- Confirms: initialization integrates with existing core, controllers, runner, schemas, and ledger.

- [x] **Step 1: Run full test suite**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
```

Expected result: all tests pass with no failures.

- [x] **Step 2: Remove generated caches**

Run:

```bash
find . -type d \( -name __pycache__ -o -name .pytest_cache \) -prune -exec rm -rf {} +
```

Expected result: no generated cache directories remain.

## Self-Review

- Spec coverage: The plan covers default initialization, seeded initialization, validation, ledger behavior, and autonomous runner integration.
- Placeholder scan: No unspecified implementation placeholders remain.
- Type consistency: Public class names and method signatures match the approved design spec.
