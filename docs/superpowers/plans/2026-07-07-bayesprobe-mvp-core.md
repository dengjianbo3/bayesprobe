# BayesProbe MVP Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first testable BayesProbe MVP core with schemas, JSONL ledger, signal inbox, evidence integration, belief update, hypothesis evolution, and minimal Autonomous/Synchronized controllers.

**Architecture:** Implement a Python/Pydantic tracer bullet that keeps `BayesProbeCore` as the only belief-revision entry point. Controllers govern cycle timing and signal collection but never create Evidence Events, update posterior, or evolve hypotheses directly.

**Tech Stack:** Python 3.11+, Pydantic 2.x, pytest, JSONL append-only ledger, no network dependency, no LLM dependency in MVP tests.

## Global Constraints

- BayesProbe is a complete agent paradigm, not a ReAct/ReWOO wrapper.
- Synchronized and Autonomous run regimes are both first-class in MVP.
- All active/passive signals must enter `SignalInbox` and pass through `EvidenceIntegrationGate`.
- Controllers must not define evidence rules, likelihood judgment, posterior updates, or Hypothesis Evolution.
- Incoming Belief State Projections are Passive External Signals, not Evidence Events.
- Every Answer Projection and Belief State Projection must include a Change-My-Mind Condition.
- Probe Sets are frozen within a cycle and may be empty for passive-only cycles.
- Use append-only ledger records; do not mutate historical entries.
- Current workspace is not a git repository. Commit steps should run `git rev-parse --is-inside-work-tree` first and skip commit when it prints an error.

---

## File Structure

Create this minimal MVP package:

```text
bayesprobe/
  __init__.py
  schemas.py
  ledger.py
  inbox.py
  belief.py
  core.py
  controllers.py
tests/
  test_schemas.py
  test_inbox_and_ledger.py
  test_core_cycles.py
  test_controllers.py
pyproject.toml
```

This is intentionally smaller than the final target layout. The first slice optimizes for a working, auditable tracer bullet; later phases can split `schemas.py` into a `schemas/` package without changing domain names.

---

### Task 1: Project Scaffold And Domain Schemas

**Files:**
- Create: `pyproject.toml`
- Create: `bayesprobe/__init__.py`
- Create: `bayesprobe/schemas.py`
- Create: `tests/test_schemas.py`

**Interfaces:**
- Produces: Pydantic models and enums imported by every later task.
- Consumes: No prior code.

- [ ] **Step 1: Write the failing schema tests**

Create `tests/test_schemas.py`:

```python
from bayesprobe.schemas import (
    BeliefState,
    ChangeMyMindCondition,
    CycleRecord,
    CycleSignalShape,
    ExternalSignal,
    Hypothesis,
    HypothesisStatus,
    ProbeCandidate,
    ProbeDesign,
    ProbeSet,
    RunRecord,
    RunRegime,
    SignalKind,
)


def test_minimal_run_cycle_and_belief_state_round_trip():
    run = RunRecord(run_id="run_1", regime=RunRegime.AUTONOMOUS, problem="Decide X")
    cycle = CycleRecord(
        cycle_id="cycle_1",
        run_id=run.run_id,
        cycle_index=1,
        signal_shape=CycleSignalShape.ACTIVE_ONLY,
    )
    hypothesis = Hypothesis(
        id="H1",
        statement="X is true",
        scope="sample scope",
        prior=0.5,
        posterior=0.5,
        rivals=["H2"],
        falsifiers=["A strong counterexample would weaken H1."],
        predictions=["Evidence A is likely if H1 is true."],
    )
    belief_state = BeliefState(
        belief_state_id="bs_1",
        run_id=run.run_id,
        cycle_id=cycle.cycle_id,
        hypotheses=[hypothesis],
    )

    loaded = BeliefState.model_validate_json(belief_state.model_dump_json())

    assert loaded.hypotheses[0].id == "H1"
    assert loaded.hypotheses[0].status == HypothesisStatus.ACTIVE


def test_probe_set_can_be_empty_for_passive_only_cycle():
    probe_set = ProbeSet(
        probe_set_id="ps_1",
        cycle_id="cycle_1",
        probes=[],
        selection_reason="Passive-only synchronized cycle.",
        may_be_empty=True,
    )

    assert probe_set.probes == []
    assert probe_set.may_be_empty is True


def test_external_signal_kinds_and_change_my_mind_candidates():
    candidate = ProbeCandidate(
        candidate_id="pc_1",
        source="change_my_mind",
        candidate_probe=ProbeDesign(
            id="P1",
            cycle_id="cycle_2",
            target_hypotheses=["H1"],
            inquiry_goal="Check if source A is independent.",
            method="source_tracing",
            support_condition={"H1": "Source A is independent."},
            weaken_condition={"H1": "Source A shares origin with source B."},
        ),
    )
    condition = ChangeMyMindCondition(
        human_readable_condition="I would lower H1 if source A is not independent.",
        structured_probe_candidates=[candidate],
    )
    signal = ExternalSignal(
        id="S1",
        cycle_id="cycle_1",
        signal_kind=SignalKind.PASSIVE,
        source_type="external_agent_projection",
        source="agent_a",
        raw_content="Agent A believes H1 because source A supports it.",
    )

    assert condition.structured_probe_candidates[0].candidate_probe.method == "source_tracing"
    assert signal.signal_kind == SignalKind.PASSIVE
```

- [ ] **Step 2: Run the schema tests and verify they fail**

Run:

```bash
python -m pytest tests/test_schemas.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'bayesprobe'`.

- [ ] **Step 3: Create project metadata**

Create `pyproject.toml`:

```toml
[project]
name = "bayesprobe"
version = "0.1.0"
description = "BayesProbe MVP core for signal-grounded belief revision"
requires-python = ">=3.11"
dependencies = [
  "pydantic>=2.7,<3",
]

[project.optional-dependencies]
dev = [
  "pytest>=8,<9",
]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

- [ ] **Step 4: Create package init**

Create `bayesprobe/__init__.py`:

```python
"""BayesProbe MVP core package."""

__all__ = [
    "schemas",
]
```

- [ ] **Step 5: Create domain schemas**

Create `bayesprobe/schemas.py`:

```python
from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RunRegime(StrEnum):
    AUTONOMOUS = "autonomous"
    SYNCHRONIZED = "synchronized"


class RunStatus(StrEnum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class CycleSignalShape(StrEnum):
    ACTIVE_ONLY = "active_only"
    PASSIVE_ONLY = "passive_only"
    ACTIVE_PLUS_PASSIVE = "active_plus_passive"


class BoundaryStatus(StrEnum):
    OPEN = "open"
    CLOSED = "closed"
    INTEGRATED = "integrated"


class HypothesisStatus(StrEnum):
    ACTIVE = "active"
    WEAKENED = "weakened"
    REFRAMED = "reframed"
    SPLIT = "split"
    RETIRED = "retired"
    ARCHIVED = "archived"


class SignalKind(StrEnum):
    ACTIVE = "active"
    PASSIVE = "passive"


class SignalInboxStatus(StrEnum):
    ACCEPTED = "accepted"
    DEFERRED = "deferred"


class EvidenceType(StrEnum):
    SUPPORTING = "supporting"
    COUNTEREVIDENCE = "counterevidence"
    BOUNDARY_CONDITION = "boundary_condition"
    ANOMALY = "anomaly"
    NEUTRAL = "neutral"
    SOURCE_CLAIM = "source_claim"
    SENDER_JUDGMENT = "sender_judgment"


class LikelihoodBand(StrEnum):
    STRONGLY_DISCONFIRMING = "strongly_disconfirming"
    MODERATELY_DISCONFIRMING = "moderately_disconfirming"
    WEAKLY_DISCONFIRMING = "weakly_disconfirming"
    NEUTRAL = "neutral"
    WEAKLY_CONFIRMING = "weakly_confirming"
    MODERATELY_CONFIRMING = "moderately_confirming"
    STRONGLY_CONFIRMING = "strongly_confirming"


class UpdateDirection(StrEnum):
    STRENGTHENED = "strengthened"
    WEAKENED = "weakened"
    NEUTRAL = "neutral"


class EvolutionOperation(StrEnum):
    SPAWN = "spawn"
    SPLIT = "split"
    MERGE = "merge"
    REFRAME = "reframe"
    REJECT = "reject"
    RETIRE = "retire"
    REACTIVATE = "reactivate"


class RunBudget(BaseModel):
    max_cycles: int = 5
    max_tool_calls: int = 20
    max_tokens: int | None = None
    max_cost: float | None = None


class RunRecord(BaseModel):
    run_id: str
    regime: RunRegime
    problem: str
    status: RunStatus = RunStatus.RUNNING
    current_cycle_id: str | None = None
    budget: RunBudget = Field(default_factory=RunBudget)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CycleRecord(BaseModel):
    cycle_id: str
    run_id: str
    cycle_index: int
    signal_shape: CycleSignalShape
    round_id: str | None = None
    boundary_status: BoundaryStatus = BoundaryStatus.OPEN
    started_at: datetime = Field(default_factory=utc_now)
    boundary_closed_at: datetime | None = None
    completed_at: datetime | None = None
    controller_metadata: dict[str, Any] = Field(default_factory=dict)


class Hypothesis(BaseModel):
    id: str
    statement: str
    scope: str
    prior: float
    posterior: float
    type: str = "claim"
    status: HypothesisStatus = HypothesisStatus.ACTIVE
    rivals: list[str] = Field(default_factory=list)
    falsifiers: list[str] = Field(default_factory=list)
    predictions: list[str] = Field(default_factory=list)
    complexity_penalty: float = 0.0
    ad_hoc_penalty: float = 0.0
    created_by: Literal["initial", "spawned", "split", "reframed"] = "initial"
    why_existing_hypotheses_failed: str | None = None

    @field_validator("prior", "posterior", "complexity_penalty", "ad_hoc_penalty")
    @classmethod
    def probability_like(cls, value: float) -> float:
        if value < 0 or value > 1:
            raise ValueError("value must be between 0 and 1")
        return value


class BeliefState(BaseModel):
    belief_state_id: str
    run_id: str
    cycle_id: str
    hypotheses: list[Hypothesis]
    posterior_summary: dict[str, Any] = Field(default_factory=dict)
    uncertainty_summary: str = ""
    ledger_refs: dict[str, list[str]] = Field(default_factory=dict)


class ProbeDesign(BaseModel):
    id: str
    cycle_id: str
    target_hypotheses: list[str]
    inquiry_goal: str
    method: str
    probe_type: str = "discriminative_test"
    support_condition: dict[str, str] = Field(default_factory=dict)
    weaken_condition: dict[str, str] = Field(default_factory=dict)
    reframe_condition: dict[str, str] | None = None
    expected_information_gain: float = 0.5
    decision_relevance: float = 0.5
    cost_estimate: float = 0.5
    priority: float = 0.5
    status: str = "candidate"

    @field_validator("expected_information_gain", "decision_relevance", "cost_estimate", "priority")
    @classmethod
    def score_between_zero_and_one(cls, value: float) -> float:
        if value < 0 or value > 1:
            raise ValueError("score must be between 0 and 1")
        return value


class ProbeSet(BaseModel):
    probe_set_id: str
    cycle_id: str
    probes: list[ProbeDesign] = Field(default_factory=list)
    boundary_id: str | None = None
    selection_reason: str
    budget_allocated: dict[str, int | float] = Field(default_factory=dict)
    may_be_empty: bool = False


class ProbeCandidate(BaseModel):
    candidate_id: str
    source: Literal["change_my_mind", "uncertainty", "anomaly", "passive_signal", "manual"]
    candidate_probe: ProbeDesign
    priority_features: dict[str, Any] = Field(default_factory=dict)
    selected_in_cycle: str | None = None


class ChangeMyMindCondition(BaseModel):
    human_readable_condition: str
    structured_probe_candidates: list[ProbeCandidate] = Field(default_factory=list)


class ExternalSignal(BaseModel):
    id: str
    cycle_id: str
    signal_kind: SignalKind
    source_type: str
    source: str
    raw_content: str
    generated_by_probe: str | None = None
    received_at: datetime = Field(default_factory=utc_now)
    inbox_status: SignalInboxStatus = SignalInboxStatus.ACCEPTED
    initial_target_hypotheses: list[str] = Field(default_factory=list)


class EvidenceEvent(BaseModel):
    id: str
    derived_from_signal: str
    target_hypotheses: list[str]
    evidence_type: EvidenceType
    content: str
    reliability: float = 0.5
    independence: float = 0.5
    relevance: float = 0.5
    novelty: float = 0.5
    specificity: float = 0.5
    verifiability: float = 0.5
    likelihoods: dict[str, LikelihoodBand] = Field(default_factory=dict)
    interpretation: str = ""
    discard_reason: str | None = None

    @field_validator("reliability", "independence", "relevance", "novelty", "specificity", "verifiability")
    @classmethod
    def score_between_zero_and_one(cls, value: float) -> float:
        if value < 0 or value > 1:
            raise ValueError("quality score must be between 0 and 1")
        return value


class BeliefUpdate(BaseModel):
    update_id: str
    cycle_id: str
    evidence_id: str
    hypothesis_id: str
    prior: float
    posterior: float
    direction: UpdateDirection
    reason: str
    sensitivity: dict[str, Any] = Field(default_factory=dict)


class HypothesisEvolution(BaseModel):
    evolution_id: str
    cycle_id: str
    operation: EvolutionOperation
    from_hypothesis: str | None = None
    to_hypothesis: str | None = None
    triggered_by: list[str] = Field(default_factory=list)
    reason: str
    audit_fields: dict[str, Any] = Field(default_factory=dict)


class AnswerProjection(BaseModel):
    answer: str
    current_best_hypothesis: str
    posterior_summary: str
    main_uncertainty: str
    weakest_assumption: str
    main_evidence_events: list[str]
    change_my_mind_condition: ChangeMyMindCondition
    answer_utility_notes: str = ""


class BeliefStateProjection(BaseModel):
    current_best_hypothesis: str
    posterior_or_confidence_interval: str
    main_evidence_events: list[str]
    main_uncertainties: list[str]
    questions_for_others: list[str]
    change_my_mind_condition: ChangeMyMindCondition
    requested_signal_type: str
    cited_sources: list[str] = Field(default_factory=list)
    projection_metadata: dict[str, Any] = Field(default_factory=dict)
```

- [ ] **Step 6: Run schema tests and verify they pass**

Run:

```bash
python -m pytest tests/test_schemas.py -v
```

Expected: all tests PASS.

- [ ] **Step 7: Checkpoint**

Run:

```bash
git rev-parse --is-inside-work-tree
```

Expected: if this prints `true`, run:

```bash
git add pyproject.toml bayesprobe/__init__.py bayesprobe/schemas.py tests/test_schemas.py
git commit -m "feat: add BayesProbe MVP schemas"
```

If it prints `fatal: not a git repository`, skip commit and record this checkpoint in your final task note.

---

### Task 2: Append-Only Ledger And Signal Inbox

**Files:**
- Create: `bayesprobe/ledger.py`
- Create: `bayesprobe/inbox.py`
- Create: `tests/test_inbox_and_ledger.py`

**Interfaces:**
- Consumes: schema models from `bayesprobe.schemas`.
- Produces:
  - `JsonlLedgerStore.append(record_type: str, record: BaseModel) -> None`
  - `JsonlLedgerStore.read_all(record_type: str | None = None) -> list[dict]`
  - `SignalInbox.add(signal: ExternalSignal) -> ExternalSignal`
  - `SignalInbox.close() -> list[ExternalSignal]`

- [ ] **Step 1: Write failing tests**

Create `tests/test_inbox_and_ledger.py`:

```python
from pathlib import Path

from bayesprobe.inbox import SignalInbox
from bayesprobe.ledger import JsonlLedgerStore
from bayesprobe.schemas import ExternalSignal, SignalInboxStatus, SignalKind


def test_jsonl_ledger_appends_and_reads_records(tmp_path: Path):
    ledger = JsonlLedgerStore(tmp_path / "ledger.jsonl")
    signal = ExternalSignal(
        id="S1",
        cycle_id="cycle_1",
        signal_kind=SignalKind.PASSIVE,
        source_type="user_feedback",
        source="user",
        raw_content="This claim seems too broad.",
    )

    ledger.append("external_signal", signal)
    records = ledger.read_all("external_signal")

    assert len(records) == 1
    assert records[0]["record_type"] == "external_signal"
    assert records[0]["payload"]["id"] == "S1"


def test_signal_inbox_defers_late_signals_after_close():
    inbox = SignalInbox(cycle_id="cycle_1")
    first = ExternalSignal(
        id="S1",
        cycle_id="cycle_1",
        signal_kind=SignalKind.PASSIVE,
        source_type="external_agent_projection",
        source="agent_a",
        raw_content="Agent A reports uncertainty about H1.",
    )
    late = ExternalSignal(
        id="S2",
        cycle_id="cycle_1",
        signal_kind=SignalKind.PASSIVE,
        source_type="system_log",
        source="log",
        raw_content="Late log signal.",
    )

    accepted = inbox.add(first)
    closed_signals = inbox.close()
    deferred = inbox.add(late)

    assert accepted.inbox_status == SignalInboxStatus.ACCEPTED
    assert [signal.id for signal in closed_signals] == ["S1"]
    assert deferred.inbox_status == SignalInboxStatus.DEFERRED
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
python -m pytest tests/test_inbox_and_ledger.py -v
```

Expected: FAIL with `ModuleNotFoundError` for `bayesprobe.inbox` or `bayesprobe.ledger`.

- [ ] **Step 3: Implement append-only JSONL ledger**

Create `bayesprobe/ledger.py`:

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from bayesprobe.schemas import utc_now


class JsonlLedgerStore:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record_type: str, record: BaseModel | dict[str, Any]) -> None:
        payload = record.model_dump(mode="json") if isinstance(record, BaseModel) else record
        envelope = {
            "record_type": record_type,
            "recorded_at": utc_now().isoformat(),
            "payload": payload,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(envelope, ensure_ascii=False, sort_keys=True) + "\n")

    def read_all(self, record_type: str | None = None) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                envelope = json.loads(line)
                if record_type is None or envelope["record_type"] == record_type:
                    records.append(envelope)
        return records
```

- [ ] **Step 4: Implement SignalInbox**

Create `bayesprobe/inbox.py`:

```python
from __future__ import annotations

from bayesprobe.schemas import ExternalSignal, SignalInboxStatus


class SignalInbox:
    def __init__(self, cycle_id: str):
        self.cycle_id = cycle_id
        self._signals: list[ExternalSignal] = []
        self._closed = False

    @property
    def is_closed(self) -> bool:
        return self._closed

    def add(self, signal: ExternalSignal) -> ExternalSignal:
        if self._closed:
            return signal.model_copy(update={"inbox_status": SignalInboxStatus.DEFERRED})
        accepted = signal.model_copy(
            update={
                "cycle_id": self.cycle_id,
                "inbox_status": SignalInboxStatus.ACCEPTED,
            }
        )
        self._signals.append(accepted)
        return accepted

    def close(self) -> list[ExternalSignal]:
        self._closed = True
        return list(self._signals)
```

- [ ] **Step 5: Run tests and verify they pass**

Run:

```bash
python -m pytest tests/test_inbox_and_ledger.py -v
```

Expected: all tests PASS.

- [ ] **Step 6: Run schema tests to catch regressions**

Run:

```bash
python -m pytest tests/test_schemas.py tests/test_inbox_and_ledger.py -v
```

Expected: all tests PASS.

- [ ] **Step 7: Checkpoint**

Run:

```bash
git rev-parse --is-inside-work-tree
```

Expected: if this prints `true`, run:

```bash
git add bayesprobe/ledger.py bayesprobe/inbox.py tests/test_inbox_and_ledger.py
git commit -m "feat: add ledger and signal inbox"
```

If it prints `fatal: not a git repository`, skip commit and record this checkpoint in your final task note.

---

### Task 3: Deterministic Belief Solver And Evidence Integration

**Files:**
- Create: `bayesprobe/belief.py`
- Create: `bayesprobe/core.py`
- Create: `tests/test_core_cycles.py`

**Interfaces:**
- Consumes: `BeliefState`, `ExternalSignal`, `EvidenceEvent`, `ProbeSet`.
- Produces:
  - `likelihood_band_to_lr(band: LikelihoodBand) -> float`
  - `solve_updates(cycle_id: str, belief_state: BeliefState, events: list[EvidenceEvent]) -> tuple[list[Hypothesis], list[BeliefUpdate]]`
  - `BayesProbeCore.integrate_cycle(cycle: CycleRecord, belief_state: BeliefState, probe_set: ProbeSet, signals: list[ExternalSignal]) -> CycleResult`

- [ ] **Step 1: Write failing core cycle tests**

Create `tests/test_core_cycles.py`:

```python
from bayesprobe.core import BayesProbeCore
from bayesprobe.schemas import (
    BeliefState,
    CycleRecord,
    CycleSignalShape,
    EvidenceType,
    ExternalSignal,
    Hypothesis,
    LikelihoodBand,
    ProbeSet,
    SignalKind,
)


def make_belief_state(cycle_id: str = "cycle_1") -> BeliefState:
    return BeliefState(
        belief_state_id="bs_1",
        run_id="run_1",
        cycle_id=cycle_id,
        hypotheses=[
            Hypothesis(
                id="H1",
                statement="The claim is supported.",
                scope="claim verification",
                prior=0.5,
                posterior=0.5,
                rivals=["H2"],
                falsifiers=["A refuting sentence weakens H1."],
                predictions=["Supporting evidence is likely."],
            ),
            Hypothesis(
                id="H2",
                statement="The claim is refuted.",
                scope="claim verification",
                prior=0.5,
                posterior=0.5,
                rivals=["H1"],
                falsifiers=["A supporting sentence weakens H2."],
                predictions=["Refuting evidence is likely."],
            ),
        ],
    )


def test_active_only_signal_updates_belief_through_evidence_gate():
    core = BayesProbeCore()
    cycle = CycleRecord(
        cycle_id="cycle_1",
        run_id="run_1",
        cycle_index=1,
        signal_shape=CycleSignalShape.ACTIVE_ONLY,
    )
    signal = ExternalSignal(
        id="S1",
        cycle_id="cycle_1",
        signal_kind=SignalKind.ACTIVE,
        source_type="benchmark_stream",
        source="fixture",
        raw_content="REFUTES: The cited sentence contradicts the claim.",
    )
    result = core.integrate_cycle(
        cycle=cycle,
        belief_state=make_belief_state(),
        probe_set=ProbeSet(
            probe_set_id="ps_1",
            cycle_id="cycle_1",
            probes=[],
            selection_reason="Fixture active-only cycle.",
            may_be_empty=True,
        ),
        signals=[signal],
    )

    h1 = result.belief_state.hypotheses_by_id()["H1"]
    h2 = result.belief_state.hypotheses_by_id()["H2"]

    assert result.evidence_events[0].evidence_type == EvidenceType.COUNTEREVIDENCE
    assert result.evidence_events[0].likelihoods["H1"] == LikelihoodBand.MODERATELY_DISCONFIRMING
    assert h1.posterior < 0.5
    assert h2.posterior > 0.5
    assert result.belief_updates[0].evidence_id == "E1"


def test_passive_projection_is_signal_not_direct_evidence():
    core = BayesProbeCore()
    cycle = CycleRecord(
        cycle_id="cycle_2",
        run_id="run_1",
        cycle_index=2,
        signal_shape=CycleSignalShape.PASSIVE_ONLY,
    )
    signal = ExternalSignal(
        id="S2",
        cycle_id="cycle_2",
        signal_kind=SignalKind.PASSIVE,
        source_type="external_agent_projection",
        source="agent_a",
        raw_content="Agent A believes H2 because Source A refutes the claim.",
    )

    result = core.integrate_cycle(
        cycle=cycle,
        belief_state=make_belief_state(cycle_id="cycle_2"),
        probe_set=ProbeSet(
            probe_set_id="ps_2",
            cycle_id="cycle_2",
            probes=[],
            selection_reason="Passive-only synchronized round.",
            may_be_empty=True,
        ),
        signals=[signal],
    )

    assert result.evidence_events[0].derived_from_signal == "S2"
    assert result.evidence_events[0].evidence_type == EvidenceType.SENDER_JUDGMENT
    assert result.belief_updates
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
python -m pytest tests/test_core_cycles.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'bayesprobe.core'`.

- [ ] **Step 3: Add hypothesis lookup helper to schemas**

Modify `bayesprobe/schemas.py` by adding this method inside `BeliefState`:

```python
    def hypotheses_by_id(self) -> dict[str, Hypothesis]:
        return {hypothesis.id: hypothesis for hypothesis in self.hypotheses}
```

The resulting `BeliefState` class should include the new method after the `ledger_refs` field.

- [ ] **Step 4: Implement belief solver**

Create `bayesprobe/belief.py`:

```python
from __future__ import annotations

import math

from bayesprobe.schemas import (
    BeliefState,
    BeliefUpdate,
    EvidenceEvent,
    Hypothesis,
    LikelihoodBand,
    UpdateDirection,
)


LR_BY_BAND: dict[LikelihoodBand, float] = {
    LikelihoodBand.STRONGLY_DISCONFIRMING: 0.1,
    LikelihoodBand.MODERATELY_DISCONFIRMING: 0.3,
    LikelihoodBand.WEAKLY_DISCONFIRMING: 0.7,
    LikelihoodBand.NEUTRAL: 1.0,
    LikelihoodBand.WEAKLY_CONFIRMING: 1.5,
    LikelihoodBand.MODERATELY_CONFIRMING: 3.0,
    LikelihoodBand.STRONGLY_CONFIRMING: 10.0,
}


def likelihood_band_to_lr(band: LikelihoodBand) -> float:
    return LR_BY_BAND[band]


def _logit(probability: float) -> float:
    clipped = min(max(probability, 0.001), 0.999)
    return math.log(clipped / (1 - clipped))


def _sigmoid(value: float) -> float:
    return 1 / (1 + math.exp(-value))


def _direction(prior: float, posterior: float) -> UpdateDirection:
    if posterior > prior + 0.01:
        return UpdateDirection.STRENGTHENED
    if posterior < prior - 0.01:
        return UpdateDirection.WEAKENED
    return UpdateDirection.NEUTRAL


def solve_updates(
    cycle_id: str,
    belief_state: BeliefState,
    events: list[EvidenceEvent],
) -> tuple[list[Hypothesis], list[BeliefUpdate]]:
    hypotheses = belief_state.hypotheses_by_id()
    current_posteriors = {hypothesis.id: hypothesis.posterior for hypothesis in belief_state.hypotheses}
    updates: list[BeliefUpdate] = []

    for event_index, event in enumerate(events, start=1):
        for hypothesis_id, band in event.likelihoods.items():
            if hypothesis_id not in hypotheses:
                continue
            prior = current_posteriors[hypothesis_id]
            weight = event.reliability * event.independence * event.relevance * event.novelty
            weighted_log_lr = math.log(likelihood_band_to_lr(band)) * weight
            posterior = _sigmoid(_logit(prior) + weighted_log_lr)
            current_posteriors[hypothesis_id] = posterior
            updates.append(
                BeliefUpdate(
                    update_id=f"U{event_index}_{hypothesis_id}",
                    cycle_id=cycle_id,
                    evidence_id=event.id,
                    hypothesis_id=hypothesis_id,
                    prior=round(prior, 4),
                    posterior=round(posterior, 4),
                    direction=_direction(prior, posterior),
                    reason=f"{event.evidence_type.value} is {band.value} for {hypothesis_id}.",
                    sensitivity={
                        "weight": round(weight, 4),
                        "likelihood_band": band.value,
                    },
                )
            )

    updated_hypotheses = [
        hypothesis.model_copy(update={"posterior": round(current_posteriors[hypothesis.id], 4)})
        for hypothesis in belief_state.hypotheses
    ]
    return updated_hypotheses, updates
```

- [ ] **Step 5: Implement deterministic core integration**

Create `bayesprobe/core.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

from bayesprobe.belief import solve_updates
from bayesprobe.schemas import (
    BeliefState,
    CycleRecord,
    EvidenceEvent,
    EvidenceType,
    ExternalSignal,
    HypothesisEvolution,
    LikelihoodBand,
    ProbeSet,
)


@dataclass(frozen=True)
class CycleResult:
    cycle: CycleRecord
    belief_state: BeliefState
    evidence_events: list[EvidenceEvent]
    belief_updates: list
    hypothesis_evolutions: list[HypothesisEvolution]


class BayesProbeCore:
    def integrate_cycle(
        self,
        cycle: CycleRecord,
        belief_state: BeliefState,
        probe_set: ProbeSet,
        signals: list[ExternalSignal],
    ) -> CycleResult:
        evidence_events = [
            self._build_evidence_event(index=index, signal=signal, belief_state=belief_state)
            for index, signal in enumerate(signals, start=1)
        ]
        updated_hypotheses, belief_updates = solve_updates(
            cycle_id=cycle.cycle_id,
            belief_state=belief_state,
            events=evidence_events,
        )
        evolutions = self._detect_anomalies(cycle.cycle_id, evidence_events)
        updated_state = belief_state.model_copy(
            update={
                "cycle_id": cycle.cycle_id,
                "hypotheses": updated_hypotheses,
                "ledger_refs": {
                    "evidence_events": [event.id for event in evidence_events],
                    "belief_updates": [update.update_id for update in belief_updates],
                    "hypothesis_evolutions": [evolution.evolution_id for evolution in evolutions],
                },
            }
        )
        return CycleResult(
            cycle=cycle,
            belief_state=updated_state,
            evidence_events=evidence_events,
            belief_updates=belief_updates,
            hypothesis_evolutions=evolutions,
        )

    def _build_evidence_event(
        self,
        index: int,
        signal: ExternalSignal,
        belief_state: BeliefState,
    ) -> EvidenceEvent:
        content_upper = signal.raw_content.upper()
        hypothesis_ids = [hypothesis.id for hypothesis in belief_state.hypotheses]
        likelihoods = {hypothesis_id: LikelihoodBand.NEUTRAL for hypothesis_id in hypothesis_ids}
        evidence_type = EvidenceType.NEUTRAL

        if "AGENT" in content_upper and signal.source_type == "external_agent_projection":
            evidence_type = EvidenceType.SENDER_JUDGMENT
            if "H2" in content_upper and "H2" in likelihoods:
                likelihoods["H2"] = LikelihoodBand.WEAKLY_CONFIRMING
            if "H1" in content_upper and "H1" in likelihoods:
                likelihoods["H1"] = LikelihoodBand.WEAKLY_CONFIRMING
        elif "REFUTES" in content_upper or "CONTRADICTS" in content_upper:
            evidence_type = EvidenceType.COUNTEREVIDENCE
            if "H1" in likelihoods:
                likelihoods["H1"] = LikelihoodBand.MODERATELY_DISCONFIRMING
            if "H2" in likelihoods:
                likelihoods["H2"] = LikelihoodBand.MODERATELY_CONFIRMING
        elif "SUPPORTS" in content_upper:
            evidence_type = EvidenceType.SUPPORTING
            if "H1" in likelihoods:
                likelihoods["H1"] = LikelihoodBand.MODERATELY_CONFIRMING
            if "H2" in likelihoods:
                likelihoods["H2"] = LikelihoodBand.MODERATELY_DISCONFIRMING
        elif "ANOMALY" in content_upper:
            evidence_type = EvidenceType.ANOMALY
            likelihoods = {
                hypothesis_id: LikelihoodBand.MODERATELY_DISCONFIRMING
                for hypothesis_id in hypothesis_ids
            }

        return EvidenceEvent(
            id=f"E{index}",
            derived_from_signal=signal.id,
            target_hypotheses=hypothesis_ids,
            evidence_type=evidence_type,
            content=signal.raw_content,
            reliability=0.8,
            independence=0.8,
            relevance=0.9,
            novelty=0.8,
            specificity=0.7,
            verifiability=0.7,
            likelihoods=likelihoods,
            interpretation=f"Deterministic MVP interpretation for {signal.source_type}.",
        )

    def _detect_anomalies(
        self,
        cycle_id: str,
        events: list[EvidenceEvent],
    ) -> list[HypothesisEvolution]:
        evolutions: list[HypothesisEvolution] = []
        for event in events:
            if event.evidence_type == EvidenceType.ANOMALY:
                evolutions.append(
                    HypothesisEvolution(
                        evolution_id=f"HE_{event.id}",
                        cycle_id=cycle_id,
                        operation="spawn",
                        from_hypothesis=None,
                        to_hypothesis="H_new",
                        triggered_by=[event.id],
                        reason="Anomaly has low likelihood under all active hypotheses.",
                        audit_fields={
                            "new_hypothesis_prior": 0.12,
                            "required_next_probe": "probe anomaly boundary condition",
                        },
                    )
                )
        return evolutions
```

- [ ] **Step 6: Run core tests and verify they pass**

Run:

```bash
python -m pytest tests/test_core_cycles.py -v
```

Expected: all tests PASS.

- [ ] **Step 7: Run all tests**

Run:

```bash
python -m pytest -v
```

Expected: all tests PASS.

- [ ] **Step 8: Checkpoint**

Run:

```bash
git rev-parse --is-inside-work-tree
```

Expected: if this prints `true`, run:

```bash
git add bayesprobe/belief.py bayesprobe/core.py bayesprobe/schemas.py tests/test_core_cycles.py
git commit -m "feat: add deterministic core integration"
```

If it prints `fatal: not a git repository`, skip commit and record this checkpoint in your final task note.

---

### Task 4: Minimal Autonomous And Synchronized Controllers

**Files:**
- Create: `bayesprobe/controllers.py`
- Create: `tests/test_controllers.py`

**Interfaces:**
- Consumes: `BayesProbeCore`, `SignalInbox`, schema models.
- Produces:
  - `AutonomousController.run_once(run_id: str, belief_state: BeliefState, active_signals: list[ExternalSignal]) -> ControllerResult`
  - `SynchronizedController.process_round(run_id: str, round_id: str, belief_state: BeliefState, passive_signals: list[ExternalSignal]) -> ControllerResult`

- [ ] **Step 1: Write failing controller tests**

Create `tests/test_controllers.py`:

```python
from bayesprobe.controllers import AutonomousController, SynchronizedController
from bayesprobe.core import BayesProbeCore
from bayesprobe.schemas import (
    BeliefState,
    ExternalSignal,
    Hypothesis,
    SignalKind,
)


def make_belief_state() -> BeliefState:
    return BeliefState(
        belief_state_id="bs_1",
        run_id="run_1",
        cycle_id="cycle_0",
        hypotheses=[
            Hypothesis(
                id="H1",
                statement="The claim is supported.",
                scope="claim verification",
                prior=0.5,
                posterior=0.5,
                rivals=["H2"],
                falsifiers=["Refuting evidence weakens H1."],
                predictions=["Supporting signal is likely."],
            ),
            Hypothesis(
                id="H2",
                statement="The claim is refuted.",
                scope="claim verification",
                prior=0.5,
                posterior=0.5,
                rivals=["H1"],
                falsifiers=["Supporting evidence weakens H2."],
                predictions=["Refuting signal is likely."],
            ),
        ],
    )


def test_autonomous_active_only_run_once_emits_answer_projection():
    controller = AutonomousController(core=BayesProbeCore())
    signal = ExternalSignal(
        id="S1",
        cycle_id="pending",
        signal_kind=SignalKind.ACTIVE,
        source_type="benchmark_stream",
        source="fixture",
        raw_content="REFUTES: The claim is contradicted.",
    )

    result = controller.run_once(
        run_id="run_1",
        belief_state=make_belief_state(),
        active_signals=[signal],
    )

    assert result.cycle.signal_shape == "active_only"
    assert result.answer_projection is not None
    assert result.answer_projection.change_my_mind_condition.human_readable_condition
    assert result.belief_state.hypotheses_by_id()["H2"].posterior > 0.5


def test_synchronized_passive_only_round_emits_belief_state_projection():
    controller = SynchronizedController(core=BayesProbeCore())
    signal = ExternalSignal(
        id="S2",
        cycle_id="pending",
        signal_kind=SignalKind.PASSIVE,
        source_type="external_agent_projection",
        source="agent_a",
        raw_content="Agent A believes H2 because source A refutes the claim.",
    )

    result = controller.process_round(
        run_id="run_1",
        round_id="round_1",
        belief_state=make_belief_state(),
        passive_signals=[signal],
    )

    assert result.cycle.signal_shape == "passive_only"
    assert result.belief_state_projection is not None
    assert result.belief_state_projection.requested_signal_type == "counterevidence_or_source_challenge"
    assert result.evidence_events[0].derived_from_signal == "S2"
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
python -m pytest tests/test_controllers.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'bayesprobe.controllers'`.

- [ ] **Step 3: Implement controllers**

Create `bayesprobe/controllers.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

from bayesprobe.core import BayesProbeCore, CycleResult
from bayesprobe.inbox import SignalInbox
from bayesprobe.schemas import (
    AnswerProjection,
    BeliefState,
    BeliefStateProjection,
    ChangeMyMindCondition,
    CycleRecord,
    CycleSignalShape,
    ExternalSignal,
    ProbeCandidate,
    ProbeDesign,
    ProbeSet,
    SignalKind,
)


@dataclass(frozen=True)
class ControllerResult:
    cycle: CycleRecord
    belief_state: BeliefState
    evidence_events: list
    belief_updates: list
    hypothesis_evolutions: list
    answer_projection: AnswerProjection | None = None
    belief_state_projection: BeliefStateProjection | None = None


def _next_cycle_id(belief_state: BeliefState) -> str:
    current = belief_state.cycle_id
    if current.startswith("cycle_"):
        try:
            return f"cycle_{int(current.split('_', 1)[1]) + 1}"
        except ValueError:
            return "cycle_1"
    return "cycle_1"


def _change_my_mind_condition(cycle_id: str, top_hypothesis: str) -> ChangeMyMindCondition:
    candidate = ProbeCandidate(
        candidate_id=f"pc_{cycle_id}_{top_hypothesis}",
        source="change_my_mind",
        candidate_probe=ProbeDesign(
            id=f"P_{cycle_id}_{top_hypothesis}",
            cycle_id=cycle_id,
            target_hypotheses=[top_hypothesis],
            inquiry_goal=f"Find counterevidence or source-quality challenge for {top_hypothesis}.",
            method="source_tracing",
            support_condition={top_hypothesis: "Independent supporting source is found."},
            weaken_condition={top_hypothesis: "Evidence source is duplicated, unreliable, or refuted."},
        ),
    )
    return ChangeMyMindCondition(
        human_readable_condition=(
            f"I would materially lower {top_hypothesis} if a reliable independent signal "
            "refutes it or shows its main evidence is not independent."
        ),
        structured_probe_candidates=[candidate],
    )


def _top_hypothesis_id(belief_state: BeliefState) -> str:
    return max(belief_state.hypotheses, key=lambda hypothesis: hypothesis.posterior).id


class AutonomousController:
    def __init__(self, core: BayesProbeCore):
        self.core = core

    def run_once(
        self,
        run_id: str,
        belief_state: BeliefState,
        active_signals: list[ExternalSignal],
    ) -> ControllerResult:
        cycle_id = _next_cycle_id(belief_state)
        cycle = CycleRecord(
            cycle_id=cycle_id,
            run_id=run_id,
            cycle_index=1,
            signal_shape=CycleSignalShape.ACTIVE_ONLY,
        )
        inbox = SignalInbox(cycle_id=cycle_id)
        accepted = [inbox.add(signal.model_copy(update={"signal_kind": SignalKind.ACTIVE})) for signal in active_signals]
        closed_signals = inbox.close()
        probe_set = ProbeSet(
            probe_set_id=f"ps_{cycle_id}",
            cycle_id=cycle_id,
            probes=[],
            selection_reason="MVP active-only run uses provided active fixture signals.",
            may_be_empty=True,
        )
        core_result = self.core.integrate_cycle(cycle, belief_state, probe_set, closed_signals)
        projection = self._answer_projection(core_result)
        return ControllerResult(
            cycle=cycle,
            belief_state=core_result.belief_state,
            evidence_events=core_result.evidence_events,
            belief_updates=core_result.belief_updates,
            hypothesis_evolutions=core_result.hypothesis_evolutions,
            answer_projection=projection,
        )

    def _answer_projection(self, result: CycleResult) -> AnswerProjection:
        top = _top_hypothesis_id(result.belief_state)
        return AnswerProjection(
            answer=f"Current best hypothesis is {top}.",
            current_best_hypothesis=top,
            posterior_summary=str(result.belief_state.posterior_summary or {}),
            main_uncertainty="MVP uncertainty summary is based on remaining rival posterior mass.",
            weakest_assumption="The deterministic MVP evidence builder may under-model source quality.",
            main_evidence_events=[event.id for event in result.evidence_events],
            change_my_mind_condition=_change_my_mind_condition(result.cycle.cycle_id, top),
            answer_utility_notes="MVP answer projection is suitable for fixture evaluation.",
        )


class SynchronizedController:
    def __init__(self, core: BayesProbeCore):
        self.core = core

    def process_round(
        self,
        run_id: str,
        round_id: str,
        belief_state: BeliefState,
        passive_signals: list[ExternalSignal],
    ) -> ControllerResult:
        cycle_id = _next_cycle_id(belief_state)
        cycle = CycleRecord(
            cycle_id=cycle_id,
            run_id=run_id,
            round_id=round_id,
            cycle_index=1,
            signal_shape=CycleSignalShape.PASSIVE_ONLY,
        )
        inbox = SignalInbox(cycle_id=cycle_id)
        accepted = [inbox.add(signal.model_copy(update={"signal_kind": SignalKind.PASSIVE})) for signal in passive_signals]
        closed_signals = inbox.close()
        probe_set = ProbeSet(
            probe_set_id=f"ps_{cycle_id}",
            cycle_id=cycle_id,
            probes=[],
            selection_reason="Passive-only synchronized cycle.",
            may_be_empty=True,
        )
        core_result = self.core.integrate_cycle(cycle, belief_state, probe_set, closed_signals)
        projection = self._belief_state_projection(core_result)
        return ControllerResult(
            cycle=cycle,
            belief_state=core_result.belief_state,
            evidence_events=core_result.evidence_events,
            belief_updates=core_result.belief_updates,
            hypothesis_evolutions=core_result.hypothesis_evolutions,
            belief_state_projection=projection,
        )

    def _belief_state_projection(self, result: CycleResult) -> BeliefStateProjection:
        top = _top_hypothesis_id(result.belief_state)
        return BeliefStateProjection(
            current_best_hypothesis=top,
            posterior_or_confidence_interval="mvp_fixture_confidence",
            main_evidence_events=[event.id for event in result.evidence_events],
            main_uncertainties=["Source independence still needs verification."],
            questions_for_others=["Can another participant verify whether the cited source is independent?"],
            change_my_mind_condition=_change_my_mind_condition(result.cycle.cycle_id, top),
            requested_signal_type="counterevidence_or_source_challenge",
            cited_sources=[],
            projection_metadata={"cycle_id": result.cycle.cycle_id},
        )
```

- [ ] **Step 4: Run controller tests**

Run:

```bash
python -m pytest tests/test_controllers.py -v
```

Expected: all tests PASS.

- [ ] **Step 5: Run all tests**

Run:

```bash
python -m pytest -v
```

Expected: all tests PASS.

- [ ] **Step 6: Checkpoint**

Run:

```bash
git rev-parse --is-inside-work-tree
```

Expected: if this prints `true`, run:

```bash
git add bayesprobe/controllers.py tests/test_controllers.py
git commit -m "feat: add minimal BayesProbe controllers"
```

If it prints `fatal: not a git repository`, skip commit and record this checkpoint in your final task note.

---

### Task 5: MVP Invariants And Regression Coverage

**Files:**
- Modify: `tests/test_core_cycles.py`
- Modify: `tests/test_controllers.py`

**Interfaces:**
- Consumes: all MVP modules.
- Produces: regression coverage for core BayesProbe guardrails.

- [ ] **Step 1: Add invariant tests to `tests/test_core_cycles.py`**

Append these tests:

```python
def test_anomaly_triggers_hypothesis_evolution_before_next_probe():
    core = BayesProbeCore()
    cycle = CycleRecord(
        cycle_id="cycle_3",
        run_id="run_1",
        cycle_index=3,
        signal_shape=CycleSignalShape.PASSIVE_ONLY,
    )
    signal = ExternalSignal(
        id="S3",
        cycle_id="cycle_3",
        signal_kind=SignalKind.PASSIVE,
        source_type="system_log",
        source="log",
        raw_content="ANOMALY: This signal is poorly explained by current hypotheses.",
    )

    result = core.integrate_cycle(
        cycle=cycle,
        belief_state=make_belief_state(cycle_id="cycle_3"),
        probe_set=ProbeSet(
            probe_set_id="ps_3",
            cycle_id="cycle_3",
            probes=[],
            selection_reason="Passive-only anomaly fixture.",
            may_be_empty=True,
        ),
        signals=[signal],
    )

    assert result.evidence_events[0].evidence_type == EvidenceType.ANOMALY
    assert result.hypothesis_evolutions
    assert result.hypothesis_evolutions[0].operation == "spawn"


def test_active_and_passive_shapes_use_same_evidence_gate():
    core = BayesProbeCore()
    active_cycle = CycleRecord(
        cycle_id="cycle_4",
        run_id="run_1",
        cycle_index=4,
        signal_shape=CycleSignalShape.ACTIVE_ONLY,
    )
    passive_cycle = CycleRecord(
        cycle_id="cycle_5",
        run_id="run_1",
        cycle_index=5,
        signal_shape=CycleSignalShape.PASSIVE_ONLY,
    )
    active_signal = ExternalSignal(
        id="S4",
        cycle_id="cycle_4",
        signal_kind=SignalKind.ACTIVE,
        source_type="benchmark_stream",
        source="fixture",
        raw_content="SUPPORTS: A sentence supports the claim.",
    )
    passive_signal = ExternalSignal(
        id="S5",
        cycle_id="cycle_5",
        signal_kind=SignalKind.PASSIVE,
        source_type="benchmark_stream",
        source="fixture",
        raw_content="SUPPORTS: A sentence supports the claim.",
    )

    active_result = core.integrate_cycle(
        cycle=active_cycle,
        belief_state=make_belief_state(cycle_id="cycle_4"),
        probe_set=ProbeSet(
            probe_set_id="ps_4",
            cycle_id="cycle_4",
            probes=[],
            selection_reason="Active fixture.",
            may_be_empty=True,
        ),
        signals=[active_signal],
    )
    passive_result = core.integrate_cycle(
        cycle=passive_cycle,
        belief_state=make_belief_state(cycle_id="cycle_5"),
        probe_set=ProbeSet(
            probe_set_id="ps_5",
            cycle_id="cycle_5",
            probes=[],
            selection_reason="Passive fixture.",
            may_be_empty=True,
        ),
        signals=[passive_signal],
    )

    assert active_result.evidence_events[0].evidence_type == passive_result.evidence_events[0].evidence_type
    assert active_result.evidence_events[0].likelihoods == passive_result.evidence_events[0].likelihoods
```

- [ ] **Step 2: Add projection invariant tests to `tests/test_controllers.py`**

Append this test:

```python
def test_every_controller_output_has_change_my_mind_condition():
    autonomous = AutonomousController(core=BayesProbeCore())
    synchronized = SynchronizedController(core=BayesProbeCore())
    active_signal = ExternalSignal(
        id="S6",
        cycle_id="pending",
        signal_kind=SignalKind.ACTIVE,
        source_type="benchmark_stream",
        source="fixture",
        raw_content="SUPPORTS: The claim is supported.",
    )
    passive_signal = ExternalSignal(
        id="S7",
        cycle_id="pending",
        signal_kind=SignalKind.PASSIVE,
        source_type="user_feedback",
        source="user",
        raw_content="The claim may be too broad.",
    )

    autonomous_result = autonomous.run_once(
        run_id="run_1",
        belief_state=make_belief_state(),
        active_signals=[active_signal],
    )
    synchronized_result = synchronized.process_round(
        run_id="run_1",
        round_id="round_2",
        belief_state=make_belief_state(),
        passive_signals=[passive_signal],
    )

    assert autonomous_result.answer_projection is not None
    assert autonomous_result.answer_projection.change_my_mind_condition.structured_probe_candidates
    assert synchronized_result.belief_state_projection is not None
    assert synchronized_result.belief_state_projection.change_my_mind_condition.structured_probe_candidates
```

- [ ] **Step 3: Run all tests**

Run:

```bash
python -m pytest -v
```

Expected: all tests PASS.

- [ ] **Step 4: Checkpoint**

Run:

```bash
git rev-parse --is-inside-work-tree
```

Expected: if this prints `true`, run:

```bash
git add tests/test_core_cycles.py tests/test_controllers.py
git commit -m "test: cover BayesProbe MVP invariants"
```

If it prints `fatal: not a git repository`, skip commit and record this checkpoint in your final task note.

---

## Self-Review Checklist

- [ ] Task 1 implements schema coverage for runs, cycles, hypotheses, probe sets, signals, evidence, updates, evolutions, and projections.
- [ ] Task 2 implements append-only JSONL ledger and Signal Inbox boundary behavior.
- [ ] Task 3 implements the shared Evidence Integration Gate through `BayesProbeCore.integrate_cycle`.
- [ ] Task 4 implements both Autonomous and Synchronized controller paths.
- [ ] Task 5 adds guardrail tests for anomaly evolution, shared active/passive evidence rules, and required Change-My-Mind Conditions.
- [ ] No task allows a controller to create Evidence Events or update posterior directly.
- [ ] No task treats external Belief State Projection as Evidence Event directly.
- [ ] Passive-only and active-only cycle shapes are both tested.
