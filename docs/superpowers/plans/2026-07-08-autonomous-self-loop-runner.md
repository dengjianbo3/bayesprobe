# Autonomous Self-Loop Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first bounded autonomous self-loop runner on top of the existing BayesProbe one-cycle autonomous controller.

**Architecture:** The runner is a thin orchestration layer. It asks a signal provider for active signals, delegates one cycle to `AutonomousController`, carries forward the returned `BeliefState`, and exits with an explicit stop reason. It must not duplicate evidence interpretation, posterior update, ledger writing, or hypothesis evolution logic already owned by `BayesProbeCore`.

**Tech Stack:** Python 3.11+, Pydantic models already in `bayesprobe.schemas`, dataclasses, `enum.StrEnum`, pytest.

## Global Constraints

- Keep BayesProbe belief-state-centered: runner controls loop timing only.
- Do not implement natural-language question parsing, real tool execution, LLM evidence interpretation, synchronized meeting orchestration, or benchmark scoring in this slice.
- Use TDD: add failing tests before production runner code.
- Preserve existing controller/core contracts and ledger behavior.
- Do not attempt git commits because this workspace is not currently a git repository.

---

## File Structure

- Create `bayesprobe/runners.py`: public autonomous loop API, config validation, stop reasons, run result, and loop orchestration.
- Create `tests/test_autonomous_runner.py`: behavior tests for max-cycle stop, no-signal stop, state carry-forward, confidence stop, anomaly-spawned hypotheses, ledger records, and invalid config.

### Task 1: Autonomous Runner Tests

**Files:**
- Create: `tests/test_autonomous_runner.py`

**Interfaces:**
- Consumes: `BayesProbeCore`, `JsonlLedgerStore`, `BeliefState`, `ExternalSignal`, `Hypothesis`, `SignalKind`
- Produces: failing expectations for `AutonomousLoopRunner`, `AutonomousLoopConfig`, `AutonomousStopReason`, and `AutonomousSignalProvider`

- [x] **Step 1: Write the failing test file**

Create `tests/test_autonomous_runner.py` with deterministic fixtures:

```python
from pathlib import Path

import pytest

from bayesprobe.core import BayesProbeCore
from bayesprobe.ledger import JsonlLedgerStore
from bayesprobe.runners import AutonomousLoopConfig, AutonomousLoopRunner, AutonomousStopReason
from bayesprobe.schemas import BeliefState, ExternalSignal, Hypothesis, SignalKind


class SequenceSignalProvider:
    def __init__(self, batches: list[list[ExternalSignal]]):
        self._batches = list(batches)
        self.calls = []

    def collect_signals(self, *, run_id, cycle_index, belief_state, previous_answer):
        self.calls.append(
            {
                "run_id": run_id,
                "cycle_index": cycle_index,
                "belief_state": belief_state,
                "previous_answer": previous_answer,
            }
        )
        if self._batches:
            return self._batches.pop(0)
        return []
```

Include tests for:

- `test_runner_stops_after_max_cycles`
- `test_runner_stops_before_cycle_when_no_signals`
- `test_runner_feeds_updated_belief_state_into_next_cycle`
- `test_runner_stops_when_confidence_threshold_reached`
- `test_runner_materializes_anomaly_spawned_hypothesis_across_cycles`
- `test_runner_writes_ledger_records_for_each_executed_cycle`
- `test_invalid_runner_config_is_rejected`

- [x] **Step 2: Run the new test file and verify RED**

Run:

```bash
python3 -m pytest tests/test_autonomous_runner.py -q
```

Expected result: failure because `bayesprobe.runners` does not exist yet.

### Task 2: Runner API And Loop Implementation

**Files:**
- Create: `bayesprobe/runners.py`

**Interfaces:**
- Consumes: `AutonomousController`, `ControllerResult`, `BayesProbeCore`, `AnswerProjection`, `BeliefState`, `ExternalSignal`
- Produces:
  - `AutonomousSignalProvider.collect_signals(...) -> list[ExternalSignal]`
  - `AutonomousLoopConfig`
  - `AutonomousStopReason`
  - `AutonomousRunResult`
  - `AutonomousLoopRunner.run(...) -> AutonomousRunResult`

- [x] **Step 1: Implement the public API**

Create `bayesprobe/runners.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from bayesprobe.controllers import AutonomousController, ControllerResult
from bayesprobe.core import BayesProbeCore
from bayesprobe.schemas import AnswerProjection, BeliefState, ExternalSignal, Hypothesis
```

Add:

```python
class AutonomousSignalProvider(Protocol):
    def collect_signals(
        self,
        *,
        run_id: str,
        cycle_index: int,
        belief_state: BeliefState,
        previous_answer: AnswerProjection | None,
    ) -> list[ExternalSignal]:
        ...
```

Add validated `AutonomousLoopConfig`, `AutonomousStopReason`, `AutonomousRunResult`, and `AutonomousLoopRunner`.

- [x] **Step 2: Implement stop logic**

The runner loop should:

1. Start with `initial_belief_state`.
2. Ask the provider for signals using `cycle_index=current_state.cycle_index + 1`.
3. If there are no signals and `stop_on_no_signals=True`, return without creating a cycle.
4. Run one cycle through a reused `AutonomousController`.
5. Append the cycle result and update current belief state.
6. Stop on confidence threshold or posterior stability when configured.
7. Stop with `MAX_CYCLES` after `max_cycles` executed cycles.

Use helper functions:

```python
def _top_hypothesis(belief_state: BeliefState) -> Hypothesis:
    return max(belief_state.hypotheses, key=lambda hypothesis: (hypothesis.posterior, hypothesis.id))


def _posterior_delta_is_stable(
    previous: BeliefState,
    current: BeliefState,
    threshold: float,
) -> bool:
    previous_by_id = previous.hypotheses_by_id()
    current_by_id = current.hypotheses_by_id()
    continuing_ids = set(previous_by_id).intersection(current_by_id)
    if not continuing_ids:
        return False
    return all(
        abs(current_by_id[hypothesis_id].posterior - previous_by_id[hypothesis_id].posterior) <= threshold
        for hypothesis_id in continuing_ids
    )
```

- [x] **Step 3: Run focused tests and fix failures**

Run:

```bash
python3 -m pytest tests/test_autonomous_runner.py -q
```

Expected result: all runner tests pass.

### Task 3: Regression Verification

**Files:**
- Test: all existing pytest files

**Interfaces:**
- Confirms: runner did not change core/controller behavior.

- [x] **Step 1: Run full suite**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
```

Expected result: all tests pass with no failures.

- [x] **Step 2: Remove generated caches if created**

Run:

```bash
find . -type d \( -name __pycache__ -o -name .pytest_cache \) -prune -exec rm -rf {} +
```

Expected result: no generated cache directories remain in the working tree.

## Self-Review

- Spec coverage: The plan covers the public runner API, config validation, stop reasons, loop behavior, no-signal ledger cleanliness, anomaly carry-forward, and ledger verification.
- Placeholder scan: No unspecified implementation placeholders remain.
- Type consistency: The method and dataclass names match the approved design spec.
