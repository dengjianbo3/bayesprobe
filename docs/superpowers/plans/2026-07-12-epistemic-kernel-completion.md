# Epistemic Kernel Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the BayesProbe v0.2 epistemic kernel so admitted open tasks can preserve unresolved alternatives, revise their hypothesis frame, accumulate provenance-aware evidence across cycles, design capability-aware probes, and project selection, synthesis, or abstention without regressing MCQ or synchronized behavior.

**Architecture:** Preserve `BayesProbeCore` as the atomic belief-revision module and the autonomous/synchronized runners as regime controllers. Add deep modules at the admission, frame-policy, evidence-memory, probe-design, capability, candidate-pool, semantic-evolution, and projection seams; callers pass compact domain objects while each module owns validation and deterministic policy. Keep provider behavior behind the existing `ModelGateway` seam and make every model proposal subordinate to server-owned ids, policy, belief updates, and audit records.

**Tech Stack:** Python 3.11+, Pydantic 2.7+, pytest 8, synchronous `ModelGateway`, stdlib JSON/HTTP, vanilla JavaScript and Node test runner, Docker for the existing Python isolation suite.

## Global Constraints

- New native writes use schema `v0.2`; v0.1 states enter only through explicit migration.
- Task Admission precedes Task Framing. `needs_reframing` and `out_of_scope` create no Run, Cycle, TaskFrame, or BeliefState.
- A hypothesis is a revisable claim relevant to an answer and is not automatically an answer candidate.
- `HypothesisCompetition` and `HypothesisCoverage` are independent properties.
- Exclusive-open defaults are initial unresolved mass `0.50`, named mass `0.50 / count`, and minimum unresolved reserve `0.05`.
- Exact-answer frames contain one to six initial named candidates; zero valid candidates fails before BeliefState creation.
- Expansion defaults are maximum `3` frame revisions, maximum `8` active hypotheses, and one structured repair.
- Exact-answer selection requires frame adequacy `adequate`, top posterior at least `0.60`, top margin at least `0.15`, unresolved mass at most `0.20`, a conforming typed value, and no unexplained high-quality anomaly.
- Discovery evidence may justify candidate creation but cannot immediately confirm that candidate.
- Exact repeats produce no BeliefUpdate. Same-root restatements have zero independence. Correlation credit is bounded across cycles.
- Initial open-task probes include a multi-hypothesis discriminator or frame-coverage probe.
- Unavailable capabilities remain visible and model reasoning never impersonates search, retrieval, test execution, or external verification.
- Core-produced probe candidates are mandatory input to the next autonomous and synchronized cycle.
- Passive signals remain raw signals until their Signal Collection Boundary closes and they pass through the Evidence Integration Gate.
- Projection cannot create Evidence Events, alter posterior values, or invent citations.
- Provider credentials remain request-scoped and never enter config JSON, prompts, traces, ledger records, fixtures, artifacts, or error text.
- Do not add real search, document retrieval, repository mutation, coding Intervention, SWE-bench, Terminal-Bench, RE-Bench, or a formal HLE run in this plan.
- Do not add a runtime dependency beyond the current `pydantic` dependency.
- Every task follows red-green-refactor, runs focused tests, runs the full offline suite before commit, and leaves `git diff --check` clean.

---

## File Map

### New deep modules

- Create `bayesprobe/migrations.py`: v0.1 TaskFrame and BeliefState migration into explicit v0.2 competition, coverage, FrameState, and EvidenceMemorySnapshot.
- Create `bayesprobe/task_admission.py`: TaskAdmitter interface plus explicit, model, recorded, and routing adapters with one repair.
- Create `bayesprobe/kernel_config.py`: immutable run-policy values for open coverage, frame adequacy, correlation credit, expansion, capabilities, and projection.
- Create `bayesprobe/frame_policy.py`: coverage-aware belief solving, unresolved-mass updates, and deterministic frame-adequacy transitions.
- Create `bayesprobe/evidence_memory.py`: provenance normalization, content identity, cross-cycle duplicate/correlation classification, credit accounting, and memory snapshots.
- Create `bayesprobe/capabilities.py`: CapabilityRegistry and capability availability decisions.
- Create `bayesprobe/probe_design.py`: ProbeDesigner interface plus deterministic and model adapters.
- Create `bayesprobe/candidate_pool.py`: one merge, semantic de-duplication, and carry-forward policy shared by both runners.
- Create `bayesprobe/hypothesis_expansion.py`: structured expansion proposals, repair, server-owned allocation, and discovery-evidence rules.
- Create `bayesprobe/projection_generator.py`: Answer Contract-aware selection, synthesis, abstention, repair, and citation validation.

### Existing modules with focused integration edits

- Modify `bayesprobe/schemas.py`: v0.2 enums and strict domain records; retain the legacy enum only for migration input.
- Modify `bayesprobe/task_framing.py`: consume an admitted decision and produce v0.2 frames; re-export legacy migration wrappers.
- Modify `bayesprobe/initialization.py`: initialize FrameState and EvidenceMemorySnapshot only after admission/framing.
- Modify `bayesprobe/model_gateway.py`: v0.2 evidence judgment fields and deterministic fixtures for new structured tasks.
- Modify `bayesprobe/openai_gateway.py`: strict schemas/instructions for admission, framing, probe design, evidence, expansion, and projection.
- Modify `bayesprobe/recorded_gateway.py`: structured request identity matching for repeated tasks without storing restricted question text.
- Modify `bayesprobe/evidence.py`: full semantic judgment input, provenance/memory enforcement, unresolved likelihood, and frame fit.
- Modify `bayesprobe/belief.py`: delegate v0.2 solving and summaries to `frame_policy`; preserve v0.1 migration wrappers.
- Modify `bayesprobe/core.py`: atomically commit evidence memory, frame state, unresolved updates, semantic evolution, and candidates.
- Modify `bayesprobe/probe_planner.py`: expose `ProbeSelector`, reject unavailable capabilities, and retain `ProbePlanner` as a compatibility alias.
- Modify `bayesprobe/probe_executor.py`: require a registry-authorized executor and stamp signal provenance.
- Modify `bayesprobe/hypothesis_evolution.py`: deterministic trigger plus semantic adapter; remove generated placeholder text.
- Modify `bayesprobe/projections.py`: compatibility wrappers around ProjectionGenerator.
- Modify `bayesprobe/question_runner.py`: tagged admission results, complete progress lifecycle, shared candidate pool, and contract-aware stopping.
- Modify `bayesprobe/synchronized_runner.py`: shared candidate pool, passive boundary rules, capability visibility, and task-aware projection.
- Modify `bayesprobe/config.py`, `bayesprobe/experiment_runner.py`, and `bayesprobe/experiment_artifacts.py`: policy parsing and secret-free snapshots.
- Modify `bayesprobe/__init__.py`: export the approved public v0.2 interfaces.
- Modify `bayesprobe/webui.py` and `bayesprobe/webui_static/{index.html,app.js,styles.css}`: request config, tagged admission, live state, frame history, capabilities, and projection modes.
- Modify `docs/ARCHITECTURE.md` and `CONTEXT.md`: implemented-state table and final terminology.

### New focused tests and fixtures

- Create `tests/test_migrations.py`.
- Create `tests/test_task_admission.py`.
- Create `tests/test_frame_policy.py`.
- Create `tests/test_evidence_memory.py`.
- Create `tests/test_capabilities.py`.
- Create `tests/test_probe_design.py`.
- Create `tests/test_candidate_pool.py`.
- Create `tests/test_hypothesis_expansion.py`.
- Create `tests/test_projection_generator.py`.
- Create `tests/test_epistemic_kernel_vertical_slices.py`.
- Create `tests/fixtures/epistemic_kernel/exact_answer_missing_candidate_v0.2.json`.
- Create `tests/fixtures/epistemic_kernel/design_synthesis_v0.2.json`.
- Create `tests/fixtures/epistemic_kernel/admission_reframing_v0.2.json`.
- Create `tests/fixtures/epistemic_kernel/out_of_scope_generation_v0.2.json`.
- Create `tests/fixtures/epistemic_kernel/unavailable_capability_v0.2.json`.
- Modify existing schema, framing, initialization, core, runner, provider, SDK/config, artifact, WebUI, benchmark-regression, and stream tests at the matching seam.

---

### Task 0: Freeze the Approved Correction Baseline and Create the Execution Worktree

**Files:**
- Verify only; no source edit.

**Interfaces:**
- Consumes: approved branch `codex/open-question-correction` containing this plan and design commit.
- Produces: pushed frozen `main` and isolated branch `codex/epistemic-kernel-completion` in `.worktrees/epistemic-kernel-completion`.

- [ ] **Step 1: Verify the correction branch is clean**

Run:

```bash
git -C /Users/dengjianbo/Documents/BayesProbe/.worktrees/open-question-correction status --short
git -C /Users/dengjianbo/Documents/BayesProbe/.worktrees/open-question-correction diff --check
```

Expected: both commands produce no output.

- [ ] **Step 2: Re-run the frozen offline baseline**

Run:

```bash
cd /Users/dengjianbo/Documents/BayesProbe/.worktrees/open-question-correction
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
node --test tests/test_webui_stream.js
```

Expected: Python remains at the approved baseline of `797 passed, 10 skipped`; Node reports `14` passing tests and zero failures.

- [ ] **Step 3: Fast-forward local main and push the frozen baseline**

Run from the primary worktree after confirming it is clean:

```bash
cd /Users/dengjianbo/Documents/BayesProbe
git status --short
git merge --ff-only codex/open-question-correction
git push origin main
```

Expected: the status command is empty, the merge is a fast-forward, and `origin/main` points at the plan commit.

- [ ] **Step 4: Create the isolated implementation branch and worktree**

Run:

```bash
cd /Users/dengjianbo/Documents/BayesProbe
git worktree add .worktrees/epistemic-kernel-completion -b codex/epistemic-kernel-completion main
git -C .worktrees/epistemic-kernel-completion status --short
```

Expected: the branch is created and the final status is empty.

---

### Task 1: Add v0.2 Domain Contracts and Explicit Migration

**Files:**
- Create: `bayesprobe/migrations.py`
- Modify: `bayesprobe/schemas.py`
- Modify: `bayesprobe/task_framing.py`
- Modify: `bayesprobe/__init__.py`
- Create: `tests/test_migrations.py`
- Modify: `tests/test_schemas.py`
- Modify: `tests/test_task_framing.py`

**Interfaces:**
- Produces: `TaskAdmissionStatus`, `AnswerRelationship`, `AnswerValueType`, `HypothesisCompetition`, `HypothesisCoverage`, `FrameAdequacyStatus`, `FrameFit`, `EpistemicOrigin`, `ProbePurpose`, `CapabilityKind`, and `ProjectionMode`.
- Produces: `AnswerContractOutline`, `TaskAdmissionDecision`, `FrameState`, `SignalProvenance`, `EvidenceMemorySnapshot`, `FrameMassUpdate`, `CapabilityDescriptor`, and `CapabilityDecision`.
- Produces: `migrate_task_frame_v0_1(payload) -> TaskFrame` and `migrate_belief_state_v0_1(payload) -> BeliefState`.
- Preserves: `HypothesisRelation` as deprecated migration vocabulary; no v0.2 native constructor writes it.

- [ ] **Step 1: Write failing v0.2 schema tests**

Add tests proving competition and coverage are orthogonal, unresolved mass is legal only for exclusive-open frames, exact-answer frames accept one candidate, and a v0.2 BeliefState requires frame and memory state:

```python
def test_exact_answer_frame_is_exclusive_open_with_unresolved_mass():
    frame = make_v02_task_frame(
        task_kind=TaskKind.EXACT_ANSWER,
        competition=HypothesisCompetition.EXCLUSIVE,
        coverage=HypothesisCoverage.OPEN,
        priors=[0.25, 0.25],
        unresolved=0.50,
    )

    assert frame.answer_relationship == AnswerRelationship.SELECTION
    assert frame.hypothesis_frame.coverage == HypothesisCoverage.OPEN
    assert frame.hypothesis_frame.unresolved_alternative_mass == 0.50


def test_independent_frame_rejects_shared_unresolved_mass():
    with pytest.raises(ValueError, match="independent frames do not use shared unresolved mass"):
        make_v02_task_frame(
            competition=HypothesisCompetition.INDEPENDENT,
            coverage=HypothesisCoverage.OPEN,
            priors=[0.5, 0.5],
            unresolved=0.2,
        )


def test_independent_exhaustive_frame_is_valid_without_shared_mass():
    frame = make_v02_task_frame(
        competition=HypothesisCompetition.INDEPENDENT,
        coverage=HypothesisCoverage.EXHAUSTIVE,
        priors=[0.5, 0.5],
        unresolved=None,
    )
    assert frame.hypothesis_frame.coverage == HypothesisCoverage.EXHAUSTIVE


def test_v02_belief_state_requires_frame_state_and_evidence_memory():
    with pytest.raises(ValueError, match="v0.2 belief state requires frame_state"):
        BeliefState(
            schema_version="v0.2",
            belief_state_id="bs",
            run_id="run",
            cycle_id="cycle_0",
            hypotheses=[],
            task_frame=make_v02_task_frame(),
        )
```

- [ ] **Step 2: Run schema tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_schemas.py -q -p no:cacheprovider
```

Expected: collection or assertions fail because the v0.2 types and fields do not exist.

- [ ] **Step 3: Implement strict v0.2 records**

Add the enums and records to `bayesprobe/schemas.py`. Keep exact values and strict validation:

```python
class TaskAdmissionStatus(StrEnum):
    ADMITTED = "admitted"
    NEEDS_REFRAMING = "needs_reframing"
    OUT_OF_SCOPE = "out_of_scope"


class AnswerRelationship(StrEnum):
    SELECTION = "selection"
    SYNTHESIS = "synthesis"


class AnswerValueType(StrEnum):
    CHOICE_LABEL = "choice_label"
    INTEGER = "integer"
    NUMBER = "number"
    SHORT_TEXT = "short_text"
    STRUCTURED_TEXT = "structured_text"


class HypothesisCompetition(StrEnum):
    EXCLUSIVE = "exclusive"
    INDEPENDENT = "independent"


class HypothesisCoverage(StrEnum):
    EXHAUSTIVE = "exhaustive"
    OPEN = "open"


class FrameAdequacyStatus(StrEnum):
    PROVISIONAL = "provisional"
    ADEQUATE = "adequate"
    CHALLENGED = "challenged"
    INADEQUATE = "inadequate"
    EXPANDING = "expanding"


class FrameFit(StrEnum):
    EXPLAINED_BY_NAMED = "explained_by_named"
    UNDERDETERMINED = "underdetermined"
    SUPPORTS_UNRESOLVED = "supports_unresolved"


class ProjectionMode(StrEnum):
    SELECTION = "selection"
    SYNTHESIS = "synthesis"
    ABSTENTION = "abstention"


class EpistemicOrigin(StrEnum):
    EXTERNAL_OBSERVATION = "external_observation"
    RETRIEVED_SOURCE = "retrieved_source"
    TOOL_RESULT = "tool_result"
    MODEL_REASONING = "model_reasoning"
    HUMAN_INPUT = "human_input"
    AGENT_MESSAGE = "agent_message"
    DERIVED_SUMMARY = "derived_summary"


class ProbePurpose(StrEnum):
    HYPOTHESIS_DISCRIMINATION = "hypothesis_discrimination"
    HYPOTHESIS_FALSIFICATION = "hypothesis_falsification"
    FRAME_COVERAGE = "frame_coverage"
    SOURCE_VERIFICATION = "source_verification"
    ANOMALY_CLARIFICATION = "anomaly_clarification"
    ANSWER_CONTRACT_GAP = "answer_contract_gap"


class CapabilityKind(StrEnum):
    MODEL_REASONING = "model_reasoning"
    PYTHON_COMPUTATION = "python_computation"
    SEARCH = "search"
    DOCUMENT_RETRIEVAL = "document_retrieval"
    REPOSITORY_READ = "repository_read"
    TEST_EXECUTION = "test_execution"
    EXTERNAL_AGENT_REQUEST = "external_agent_request"
    HUMAN_REQUEST = "human_request"
```

Add strict records with these fields:

```python
class AnswerContractOutline(StrictTaskModel):
    objective: str
    answer_value_type: AnswerValueType
    decision_form: str
    permits_synthesis: bool
    required_sections: list[str]


class TaskAdmissionDecision(StrictTaskModel):
    attempt_id: str
    status: TaskAdmissionStatus
    epistemic_basis: list[str]
    proposed_task_kind: TaskKind | None = None
    answer_contract_outline: AnswerContractOutline | None = None
    clarification_questions: list[str] = Field(default_factory=list)
    reason: str
    model_trace: dict[str, Any] = Field(default_factory=dict)


class FrameState(StrictTaskModel):
    frame_id: str
    frame_version: int = 1
    parent_frame_version: int | None = None
    competition: HypothesisCompetition
    coverage: HypothesisCoverage
    active_hypothesis_ids: list[str]
    unresolved_alternative_mass: float | None = None
    adequacy_status: FrameAdequacyStatus
    revision_reason: str | None = None
    trigger_event_ids: list[str] = Field(default_factory=list)
    revision_count: int = 0


class EvidenceMemorySnapshot(StrictTaskModel):
    memory_version: int = 1
    accepted_evidence_ids: list[str] = Field(default_factory=list)
    content_fingerprints: dict[str, str] = Field(default_factory=dict)
    source_content_fingerprints: dict[str, str] = Field(default_factory=dict)
    derivation_roots: dict[str, str] = Field(default_factory=dict)
    correlation_credit: dict[str, float] = Field(default_factory=dict)
    discovery_evidence_ids: list[str] = Field(default_factory=list)
    counterevidence_ids_by_hypothesis: dict[str, list[str]] = Field(default_factory=dict)
    discard_and_schema_history: list[str] = Field(default_factory=list)


class SignalProvenance(StrictTaskModel):
    epistemic_origin: EpistemicOrigin
    source_identity: str
    provider_model_or_tool_identity: str | None = None
    session_id: str | None = None
    parent_signal_ids: list[str] = Field(default_factory=list)
    derivation_root_id: str
    correlation_group: str
    canonical_content_fingerprint: str
    citations: list[str] = Field(default_factory=list)
    artifact_refs: list[str] = Field(default_factory=list)
    environment_state_id: str | None = None


class FrameMassUpdate(StrictTaskModel):
    update_id: str
    cycle_id: str
    evidence_id: str
    prior: float
    posterior: float
    direction: UpdateDirection
    reason: str


class CapabilityDescriptor(StrictTaskModel):
    kind: CapabilityKind
    available: bool
    cost_class: str = "bounded"
    latency_class: str = "interactive"
    epistemic_origin: EpistemicOrigin = EpistemicOrigin.MODEL_REASONING
    quality_caps: dict[str, float] = Field(default_factory=dict)
    executor_adapter_id: str


class CapabilityDecision(StrictTaskModel):
    kind: CapabilityKind
    available: bool
    descriptor: CapabilityDescriptor | None
    reason: str
```

`TaskAdmissionDecision` validation requires `proposed_task_kind` and `answer_contract_outline` for `admitted`, at least one clarification question for `needs_reframing`, and no fabricated TaskKind/contract for `out_of_scope`. All semantic text lists are non-empty and de-duplicated.

Add `EXACT_ANSWER = "exact_answer"` to `TaskKind`. Update `AnswerContract` with `answer_value_type`, `answer_format`, required sections, decision form, and synthesis permission. Add `answer_value: str | int | float | None` to both framed and runtime hypotheses. Add optional migration fields `ExternalSignal.provenance: SignalProvenance | None`, `EvidenceEvent.unresolved_likelihood: LikelihoodBand | None`, `EvidenceEvent.frame_fit: FrameFit`, `EvidenceEvent.unexplained_observation: str | None`, `EvidenceEvent.correlation_status: str`, and `EvidenceEvent.effective_update_weight: float | None`; Task 4 makes them mandatory on native v0.2 evidence writes. Update `HypothesisFrame`, `TaskFrame`, and `BeliefState` with the exact v0.2 fields from the design. A `HypothesisFrame` validator must enforce:

```python
if self.competition == HypothesisCompetition.EXCLUSIVE:
    unresolved = self.unresolved_alternative_mass or 0.0
    if not math.isclose(sum(h.initial_prior for h in self.hypotheses) + unresolved, 1.0, abs_tol=1e-6):
        raise ValueError("exclusive named and unresolved mass must sum to one")
elif self.unresolved_alternative_mass is not None:
    raise ValueError("independent frames do not use shared unresolved mass")
```

Provide a migration-window bridge so Task 1 remains independently green:

- `TaskFrame.schema_version` defaults to `v0.1` only for existing constructors; every Task 2 native path passes `v0.2` explicitly.
- v0.1 Answer Contracts receive compatibility defaults for value type/format; v0.2 validation requires explicit values.
- v0.1 admission id, answer relationship, FrameState, EvidenceMemory, provenance, and new evidence fields may be absent; v0.2 validation requires them at the owning lifecycle point.
- a `mode="before"` HypothesisFrame validator consumes legacy `relation` input and maps it through `_relation_mapping` without serializing `relation` in v0.2 output.
- a deprecated read-only `relation` property returns `INDEPENDENT` for independent frames and `EXCLUSIVE_EXHAUSTIVE` only for exclusive-exhaustive frames; it raises for exclusive-open frames so old code cannot silently erase unresolved mass.

Task 2 converts framing/initialization callers to the native fields, Task 3 converts solving/core callers, and Tasks 5-9 remove the remaining internal reads. The compatibility bridge remains only for explicit migration/public input at branch completion.

- [ ] **Step 4: Write failing migration tests**

```python
def test_migrates_legacy_exclusive_frame_to_exclusive_exhaustive():
    migrated = migrate_task_frame_v0_1(legacy_mcq_frame_payload())
    assert migrated.schema_version == "v0.2"
    assert migrated.hypothesis_frame.competition == HypothesisCompetition.EXCLUSIVE
    assert migrated.hypothesis_frame.coverage == HypothesisCoverage.EXHAUSTIVE
    assert migrated.hypothesis_frame.unresolved_alternative_mass == 0.0


def test_migrates_legacy_independent_frame_to_independent_open():
    migrated = migrate_task_frame_v0_1(legacy_independent_frame_payload())
    assert migrated.hypothesis_frame.competition == HypothesisCompetition.INDEPENDENT
    assert migrated.hypothesis_frame.coverage == HypothesisCoverage.OPEN
    assert migrated.hypothesis_frame.unresolved_alternative_mass is None
```

- [ ] **Step 5: Implement explicit migration**

In `bayesprobe/migrations.py`, validate old payloads through private strict v0.1 records, map only the two documented relations, synthesize a migration admission id, and initialize FrameState/Memory without inferring coverage from ids or posterior values:

```python
def _relation_mapping(relation: HypothesisRelation) -> tuple[HypothesisCompetition, HypothesisCoverage]:
    if relation == HypothesisRelation.EXCLUSIVE_EXHAUSTIVE:
        return HypothesisCompetition.EXCLUSIVE, HypothesisCoverage.EXHAUSTIVE
    if relation == HypothesisRelation.INDEPENDENT:
        return HypothesisCompetition.INDEPENDENT, HypothesisCoverage.OPEN
    raise ValueError(f"unsupported legacy relation: {relation}")
```

Keep `task_framing.migrate_legacy_belief_state` as a thin deprecated wrapper around the new module so existing imports remain valid during this branch.

- [ ] **Step 6: Run focused and full tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_schemas.py tests/test_migrations.py tests/test_task_framing.py -q -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
git diff --check
```

Expected: all tests pass; legacy callers still migrate; diff check is clean.

- [ ] **Step 7: Commit the domain foundation**

```bash
git add bayesprobe/schemas.py bayesprobe/migrations.py bayesprobe/task_framing.py bayesprobe/__init__.py tests/test_schemas.py tests/test_migrations.py tests/test_task_framing.py
git commit -m "feat: add epistemic kernel v0.2 contracts"
```

---

### Task 2: Add Task Admission and v0.2 Task Framing

**Files:**
- Create: `bayesprobe/task_admission.py`
- Modify: `bayesprobe/task_framing.py`
- Modify: `bayesprobe/initialization.py`
- Modify: `bayesprobe/model_gateway.py`
- Modify: `bayesprobe/openai_gateway.py`
- Modify: `bayesprobe/question_runner.py`
- Modify: `bayesprobe/__init__.py`
- Create: `tests/fixtures/open_questions/model_scale_validation_v0.2.json`
- Create: `tests/test_task_admission.py`
- Modify: `tests/test_recorded_model_gateway.py`
- Modify: `tests/test_task_framing.py`
- Modify: `tests/test_initialization.py`
- Modify: `tests/test_openai_gateway.py`
- Modify: `tests/test_question_runner.py`

**Interfaces:**
- Produces: `TaskAdmissionInput`, `TaskAdmitter.assess(input) -> TaskAdmissionDecision`, `ExplicitTaskAdmitter`, `ModelTaskAdmitter`, `RecordedTaskAdmitter`, and `RoutingTaskAdmitter`.
- Changes: `TaskFramingInput` requires an admitted `TaskAdmissionDecision`; `TaskFrame.admission_decision_id` records it.
- Produces: tagged `NeedsReframingResult` and `OutOfScopeResult`, both with no BeliefState.

- [ ] **Step 1: Write failing admission tests**

```python
def test_explicit_mcq_is_admitted_without_model_call():
    gateway = ScriptedModelGateway(responses={})
    admitter = RoutingTaskAdmitter(
        explicit_admitter=ExplicitTaskAdmitter(),
        open_admitter=ModelTaskAdmitter(gateway),
    )
    decision = admitter.assess(mcq_admission_input())
    assert decision.status == TaskAdmissionStatus.ADMITTED
    assert decision.proposed_task_kind == TaskKind.MULTIPLE_CHOICE
    assert gateway.requests == []


@pytest.mark.parametrize("status", ["needs_reframing", "out_of_scope"])
def test_non_admitted_result_creates_no_belief_state(status):
    runner, ledger = runner_with_admission_response(status)
    result = runner.run_question(open_initialize_input())
    assert result.result_type == status
    assert not hasattr(result, "initial_belief_state")
    assert len(ledger.records("task_admission")) == 1
    assert ledger.records("belief_state") == []
```

- [ ] **Step 2: Run admission tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_task_admission.py tests/test_question_runner.py -q -p no:cacheprovider
```

Expected: tests fail because the TaskAdmitter interface and tagged results do not exist.

- [ ] **Step 3: Implement the admission deep module**

Create `bayesprobe/task_admission.py` with this public seam:

```python
@dataclass(frozen=True)
class TaskAdmissionInput:
    attempt_id: str
    question: str
    task_context: str = ""
    answer_choices: list[AnswerChoice] = field(default_factory=list)
    hypothesis_seeds: list[HypothesisSeed] = field(default_factory=list)
    requested_output_shape: str | None = None
    available_capabilities: list[CapabilityDescriptor] = field(default_factory=list)
    model_metadata: dict[str, Any] = field(default_factory=dict)


class TaskAdmitter(Protocol):
    def assess(self, input: TaskAdmissionInput) -> TaskAdmissionDecision: ...
```

`ExplicitTaskAdmitter` admits explicit choices and explicit hypothesis seeds. `ModelTaskAdmitter` sends `assess_task_admission`, validates the exact status enum and required fields, permits one `repair_task_admission` request, and raises `TaskAdmissionError` after the second invalid payload. `RoutingTaskAdmitter` chooses the explicit adapter only when the caller supplied a real frame.

- [ ] **Step 4: Add provider schemas and instructions**

Extend both OpenAI transport adapters with strict `TaskAdmissionDecision` output. The request must contain question, Task Context, requested output shape, and sanitized capability descriptors, but not passive signals or API credentials. Require the model to return:

```json
{
  "status": "admitted",
  "epistemic_basis": ["The requested answer can be tested against discriminating claims."],
  "proposed_task_kind": "exact_answer",
  "answer_contract_outline": {
    "objective": "Return the supported integer value.",
    "answer_value_type": "integer",
    "decision_form": "single_value",
    "permits_synthesis": false,
    "required_sections": ["answer", "basis", "uncertainty"]
  },
  "clarification_questions": [],
  "reason": "The task has a verifiable scalar answer."
}
```

Add schema-matrix tests for Responses and Chat Completions and verify provider errors remain secret-safe.

- [ ] **Step 5: Write failing v0.2 framing tests**

```python
def test_exact_answer_framing_preserves_open_coverage():
    frame = ModelTaskFramer(scripted_exact_answer_gateway()).frame(
        exact_answer_framing_input(admitted_decision())
    )
    assert frame.task_kind == TaskKind.EXACT_ANSWER
    assert frame.answer_relationship == AnswerRelationship.SELECTION
    assert frame.hypothesis_frame.competition == HypothesisCompetition.EXCLUSIVE
    assert frame.hypothesis_frame.coverage == HypothesisCoverage.OPEN
    assert frame.hypothesis_frame.unresolved_alternative_mass == 0.50


def test_zero_candidate_exact_frame_fails_after_one_repair():
    framer = ModelTaskFramer(gateway_with_two_zero_candidate_frames())
    with pytest.raises(TaskFramingError, match="invalid after 1 repair attempt"):
        framer.frame(exact_answer_framing_input(admitted_decision()))
```

- [ ] **Step 6: Upgrade framing and initialization**

Change model framing output to `schema_version=v0.2` with `answer_relationship`, typed Answer Contract, competition, coverage, and hypotheses. Server code assigns ids and exact initial mass:

```python
def _exclusive_open_priors(count: int, *, unresolved: float = 0.50) -> list[float]:
    if not 1 <= count <= 6:
        raise TaskFramingError("exclusive-open framing requires one to six candidates")
    return [round((1.0 - unresolved) / count, 12) for _ in range(count)]
```

`BayesProbeInitializer.initialize(input, admission_decision=...)` must reject non-admitted decisions, copy the admission id into TaskFrame, create `FrameState` and an empty `EvidenceMemorySnapshot`, then write admission, frame, run, state, and initial probe records in order. For direct legacy callers that omit the decision, the initializer invokes its configured TaskAdmitter exactly once; its default `ExplicitTaskAdmitter` admits only real choices/seeds and fails closed for unseeded open input. A runner-supplied decision is never reassessed. No implicit support/refute fallback returns.

Create `model_scale_validation_v0.2.json` with recorded `assess_task_admission` and v0.2 `frame_open_question` responses, plus the still-compatible probe/evidence responses needed by the current vertical slice. Keep the v0.1 file unchanged as migration coverage and switch the public-runner recorded test to v0.2.

- [ ] **Step 7: Add tagged early results to the public runner**

Before invoking the initializer, `AutonomousQuestionRunner.run_question` calls TaskAdmitter. Return these concrete dataclasses for non-admitted outcomes:

```python
@dataclass(frozen=True)
class NeedsReframingResult:
    admission: TaskAdmissionDecision
    result_type: Literal["needs_reframing"] = "needs_reframing"


@dataclass(frozen=True)
class OutOfScopeResult:
    admission: TaskAdmissionDecision
    result_type: Literal["out_of_scope"] = "out_of_scope"
```

Do not emit task-framing or initialization progress for those results.

- [ ] **Step 8: Run focused and full tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_task_admission.py tests/test_task_framing.py tests/test_initialization.py tests/test_openai_gateway.py tests/test_recorded_model_gateway.py tests/test_question_runner.py -q -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
git diff --check
```

Expected: all explicit MCQ regressions pass, open tasks are admitted before framing, and non-admitted outcomes contain no state.

- [ ] **Step 9: Commit admission and framing**

```bash
git add bayesprobe/task_admission.py bayesprobe/task_framing.py bayesprobe/initialization.py bayesprobe/model_gateway.py bayesprobe/openai_gateway.py bayesprobe/question_runner.py bayesprobe/__init__.py tests/fixtures/open_questions/model_scale_validation_v0.2.json tests/test_task_admission.py tests/test_recorded_model_gateway.py tests/test_task_framing.py tests/test_initialization.py tests/test_openai_gateway.py tests/test_question_runner.py
git commit -m "feat: admit tasks before epistemic framing"
```

---

### Task 3: Add the Coverage-Aware Solver and Frame-Adequacy Policy

**Files:**
- Create: `bayesprobe/kernel_config.py`
- Create: `bayesprobe/frame_policy.py`
- Modify: `bayesprobe/belief.py`
- Modify: `bayesprobe/core.py`
- Modify: `bayesprobe/initialization.py`
- Modify: `bayesprobe/schemas.py`
- Create: `tests/test_frame_policy.py`
- Modify: `tests/test_belief.py`
- Modify: `tests/test_core_cycles.py`
- Modify: `tests/test_initialization.py`

**Interfaces:**
- Produces: `OpenCoveragePolicy`, `FrameAdequacyPolicyConfig`, `BeliefSolveResult`, `FrameAdequacyDecision`, `CoverageAwareBeliefSolver.solve(...)`, and `FrameAdequacyPolicy.assess(...)`.
- Changes: `CycleResult` includes `frame_mass_updates`; `BeliefState.frame_state` is updated atomically with named hypotheses.
- Preserves: independent credences never cross-normalize; exhaustive MCQ mass remains exactly one over named active choices.

- [ ] **Step 1: Write failing exclusive-open solver tests**

```python
def test_exclusive_open_solver_updates_named_and_unresolved_as_one_distribution():
    state = exact_state(named={"H1": 0.25, "H2": 0.25}, unresolved=0.50)
    event = evidence_event(
        likelihoods={
            "H1": LikelihoodBand.MODERATELY_DISCONFIRMING,
            "H2": LikelihoodBand.MODERATELY_DISCONFIRMING,
        },
        unresolved_likelihood=LikelihoodBand.MODERATELY_CONFIRMING,
        frame_fit=FrameFit.SUPPORTS_UNRESOLVED,
        effective_update_weight=1.0,
    )

    result = CoverageAwareBeliefSolver().solve(state, [event], run_id="run", cycle_id="cycle_1")
    total = sum(h.posterior for h in result.hypotheses) + result.frame_state.unresolved_alternative_mass
    assert total == pytest.approx(1.0)
    assert result.frame_state.unresolved_alternative_mass > 0.50
    assert all(h.posterior < 0.25 for h in result.hypotheses)


def test_all_named_candidates_can_lose_without_forced_winner():
    result = solve_all_named_disconfirmed()
    assert result.frame_state.unresolved_alternative_mass > max(
        h.posterior for h in result.hypotheses
    )
    assert result.frame_state.adequacy_status == FrameAdequacyStatus.CHALLENGED
```

- [ ] **Step 2: Run solver tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_frame_policy.py tests/test_belief.py -q -p no:cacheprovider
```

Expected: imports fail because `frame_policy` and the coverage-aware result do not exist.

- [ ] **Step 3: Implement immutable policy config**

Create `bayesprobe/kernel_config.py` and validate every field:

```python
@dataclass(frozen=True)
class OpenCoveragePolicy:
    initial_unresolved_mass: float = 0.50
    minimum_unresolved_reserve: float = 0.05


@dataclass(frozen=True)
class FrameAdequacyPolicyConfig:
    high_verifiability_threshold: float = 0.75
    moderate_verifiability_threshold: float = 0.50
    required_distinct_moderate_roots: int = 2


@dataclass(frozen=True)
class ExpansionPolicy:
    max_frame_revisions: int = 3
    max_active_hypotheses: int = 8
    max_repair_attempts: int = 1


@dataclass(frozen=True)
class ProjectionPolicy:
    exact_top_threshold: float = 0.60
    exact_margin_threshold: float = 0.15
    exact_max_unresolved_mass: float = 0.20
    max_repair_attempts: int = 1
```

Reject booleans as numbers, non-finite values, invalid probability ranges, a reserve greater than initial unresolved mass, and non-positive integer limits.

- [ ] **Step 4: Implement coverage-aware solving**

`CoverageAwareBeliefSolver.solve` computes log scores for named active hypotheses and a private unresolved slot for exclusive-open frames, normalizes all participants together, rounds once, and writes unresolved movement to `FrameMassUpdate`, never to `BeliefUpdate`:

```python
@dataclass(frozen=True)
class BeliefSolveResult:
    hypotheses: list[Hypothesis]
    frame_state: FrameState
    belief_updates: list[BeliefUpdate]
    frame_mass_updates: list[FrameMassUpdate]


class CoverageAwareBeliefSolver:
    def solve(
        self,
        belief_state: BeliefState,
        events: list[EvidenceEvent],
        *,
        run_id: str,
        cycle_id: str,
    ) -> BeliefSolveResult: ...
```

Use `event.effective_update_weight` instead of recomputing quality in the solver. For exclusive-open, include `event.unresolved_likelihood`; for exclusive-exhaustive, omit the private slot; for independent, apply per-hypothesis log-odds exactly as today. Retirement in an exclusive-open frame returns the retired posterior to unresolved mass before normalization.

- [ ] **Step 5: Write and implement deterministic adequacy transitions**

Add tests for provisional, challenged, inadequate, expanding, and adequate transitions. Implement this interface:

```python
@dataclass(frozen=True)
class FrameAdequacyDecision:
    frame_state: FrameState
    should_expand: bool
    trigger_event_ids: list[str]
    reason: str


class FrameAdequacyPolicy:
    def assess(
        self,
        *,
        previous: FrameState,
        events: list[EvidenceEvent],
        hypotheses: list[Hypothesis],
    ) -> FrameAdequacyDecision: ...
```

Any accepted `supports_unresolved` event may set `should_expand=True`. One strongly confirming unresolved event with non-model origin and verifiability at least `0.75`, or two moderately-or-strongly confirming events from distinct derivation roots with verifiability at least `0.50`, marks the frame inadequate. Model reasoning alone may challenge and request expansion but does not establish externally verified adequacy.

- [ ] **Step 6: Integrate solver and frame policy atomically in core**

Replace direct `solve_updates` calls with the deep solver, apply adequacy after solving, and include these records in one ledger commit order:

```text
external_signal
evidence_event
belief_update
frame_mass_update
frame_adequacy_decision
hypothesis_evolution
probe_candidate
belief_state
```

Keep `belief.solve_updates` as a deprecated v0.1 wrapper that migrates before delegation. Until Task 4 upgrades native evidence construction, a migrated v0.1 event with `effective_update_weight=None` uses the existing quality product and a neutral unresolved likelihood; an explicit zero always remains zero. Add `unresolved_alternative_mass`, named active mass, and frame adequacy to posterior summaries. Initialization marks exhaustive MCQ frames `adequate` and every open frame `provisional`.

- [ ] **Step 7: Run focused and full tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_frame_policy.py tests/test_belief.py tests/test_core_cycles.py tests/test_initialization.py -q -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
git diff --check
```

Expected: open distributions preserve named plus unresolved mass, independent tests remain non-normalized, MCQ tests remain categorical, and all tests pass.

- [ ] **Step 8: Commit coverage-aware belief revision**

```bash
git add bayesprobe/kernel_config.py bayesprobe/frame_policy.py bayesprobe/belief.py bayesprobe/core.py bayesprobe/initialization.py bayesprobe/schemas.py tests/test_frame_policy.py tests/test_belief.py tests/test_core_cycles.py tests/test_initialization.py
git commit -m "feat: preserve unresolved hypothesis mass"
```

---

### Task 4: Add Signal Provenance and Cross-Cycle Evidence Memory

**Files:**
- Create: `bayesprobe/evidence_memory.py`
- Modify: `bayesprobe/kernel_config.py`
- Modify: `bayesprobe/schemas.py`
- Modify: `bayesprobe/model_gateway.py`
- Modify: `bayesprobe/openai_gateway.py`
- Modify: `bayesprobe/evidence.py`
- Modify: `bayesprobe/core.py`
- Modify: `bayesprobe/belief.py`
- Modify: `bayesprobe/probe_executor.py`
- Modify: `tests/fixtures/open_questions/model_scale_validation_v0.2.json`
- Create: `tests/test_evidence_memory.py`
- Modify: `tests/test_model_gateway.py`
- Modify: `tests/test_openai_gateway.py`
- Modify: `tests/test_core_cycles.py`
- Modify: `tests/test_probe_executor.py`

**Interfaces:**
- Produces: `SignalProvenanceNormalizer.normalize(signal, *, run_id) -> ExternalSignal`.
- Produces: `EvidenceMemoryManager.classify(...)` and `EvidenceMemoryManager.commit(...)` behind the Evidence Integration Gate.
- Changes: `EvidenceIntegrationResult` includes `evidence_memory`; native v0.2 Evidence Events populate and strictly validate the provenance identity, frame fit, unresolved likelihood, and effective update-weight fields declared in Task 1.

- [ ] **Step 1: Write failing provenance and duplicate tests**

```python
def test_exact_cross_cycle_repeat_produces_no_update_or_provider_call():
    gateway = CountingGateway(valid_judgment())
    state = state_with_memory_for(signal("first", root="root-1", content="same fact"))
    result = gate(gateway).integrate(
        cycle=cycle("cycle_2"),
        belief_state=state,
        probe_set=empty_probe_set("cycle_2"),
        signals=[signal("repeat", root="root-1", content="same fact")],
    )
    assert result.evidence_events[0].discard_reason == "duplicate_exact"
    assert result.evidence_events[0].effective_update_weight == 0.0
    assert gateway.requests == []


def test_same_root_restatement_has_zero_independence():
    result = integrate_restatement("same fact in different words", root="root-1")
    event = result.evidence_events[0]
    assert event.correlation_status == "correlated_restatement"
    assert event.independence == 0.0
    assert event.effective_update_weight == 0.0
```

- [ ] **Step 2: Run memory tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_evidence_memory.py tests/test_core_cycles.py -q -p no:cacheprovider
```

Expected: tests fail because provenance and cross-cycle memory do not exist.

- [ ] **Step 3: Normalize the structured provenance introduced in Task 1**

Use `SignalProvenance` and the exact origins already defined in Task 1:

```python
class EpistemicOrigin(StrEnum):
    EXTERNAL_OBSERVATION = "external_observation"
    RETRIEVED_SOURCE = "retrieved_source"
    TOOL_RESULT = "tool_result"
    MODEL_REASONING = "model_reasoning"
    HUMAN_INPUT = "human_input"
    AGENT_MESSAGE = "agent_message"
    DERIVED_SUMMARY = "derived_summary"
```

The normalizer fills source identity, provider/model/tool identity, session id, parent ids, derivation root, correlation group, canonical content fingerprint, citations, artifact refs, and optional environment state. Use SHA-256 over Unicode-normalized, whitespace-collapsed content plus source identity; never hash an API key or provider request headers.

Model reasoning from the same provider, model, and run session shares one correlation group. A derived summary retains its parent's derivation root. Deterministic recomputation over the same input may raise verifiability but remains the same factual root. Independent accumulation requires both a distinct source identity and a distinct derivation root with no parent/derived relationship.

- [ ] **Step 4: Implement Evidence Memory classification and credit accounting**

Add this deterministic policy to `bayesprobe/kernel_config.py`:

```python
@dataclass(frozen=True)
class CorrelationCreditPolicy:
    max_cumulative_effective_weight_per_direction: float = 1.0

    def __post_init__(self) -> None:
        value = self.max_cumulative_effective_weight_per_direction
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("correlation credit cap must be finite and positive")
```

The manager keys named credit by `correlation_group | hypothesis_id | confirming-or-disconfirming`. Unresolved credit uses the internal subject `frame:<frame_version>:unresolved`, never an `H_other` id, so repeated calls cannot bypass the cap through latent mass. It returns:

```python
@dataclass(frozen=True)
class EvidenceMemoryDecision:
    correlation_status: Literal["novel", "duplicate_exact", "correlated_restatement", "correlated_novel"]
    effective_update_weight: float
    discard_reason: str | None
    remaining_credit: dict[str, float]
```

Exact source/root/content repeats get weight zero. Same-root restatements get independence zero. Other signals consume at most remaining group credit after origin quality caps. A saturated event remains ledger-visible with weight zero and reason `correlation_credit_saturated`.

- [ ] **Step 5: Upgrade Evidence Judgment v0.2**

Change `EvidenceJudgment` and parsing to require:

```python
@dataclass(frozen=True)
class EvidenceJudgment:
    evidence_type: EvidenceType
    likelihoods: dict[str, LikelihoodBand]
    unresolved_likelihood: LikelihoodBand | None
    frame_fit: FrameFit
    unexplained_observation: str | None
    interpretation: str
    quality_overrides: dict[str, float] = field(default_factory=dict)
```

For exclusive-open frames, unresolved likelihood is required. For all other frames it must be null. Enforce these cross-field rules:

```text
supports_unresolved -> unresolved likelihood is confirming
explained_by_named -> unresolved likelihood is not confirming
underdetermined -> unresolved likelihood is neutral
```

Add a test proving a model quality override may reduce but cannot increase reliability, independence, novelty, or verifiability beyond the deterministic cap for its epistemic origin.

The `judge_evidence` request includes full hypothesis statement, type, scope, posterior, predictions, falsifiers, competition, coverage, frame version, rivals, probe purpose and conditions, provenance summary, and prior memory classification. It must never send only hypothesis ids.

- [ ] **Step 6: Update provider schemas and deterministic behavior**

Add v0.2 JSON schemas to both OpenAI transports and repair requests. `DeterministicModelGateway` maps `ANOMALY` to `supports_unresolved` only for exclusive-open frames and otherwise returns `underdetermined` with null unresolved likelihood. Add tests that invalid field combinations fail before the solver. Upgrade the v0.2 model-scale fixture's evidence response with frame fit, unexplained observation, and the correct null unresolved likelihood for its independent-open frame.

- [ ] **Step 7: Persist memory atomically in core**

Pass the prior snapshot into the gate, commit accepted/discarded identity and credit once, attach the new snapshot to the next BeliefState, and ensure replayed Evidence Event ids do not append a second ledger record. Discovery ids and counterevidence ids remain compact refs rather than copied raw text.

- [ ] **Step 8: Run focused and full tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_evidence_memory.py tests/test_model_gateway.py tests/test_openai_gateway.py tests/test_core_cycles.py tests/test_probe_executor.py -q -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
git diff --check
```

Expected: exact repeats and correlated restatements are neutral across cycles; full semantics reach judgment; no existing evidence id is duplicated.

- [ ] **Step 9: Commit provenance-aware memory**

```bash
git add bayesprobe/evidence_memory.py bayesprobe/kernel_config.py bayesprobe/schemas.py bayesprobe/model_gateway.py bayesprobe/openai_gateway.py bayesprobe/evidence.py bayesprobe/core.py bayesprobe/belief.py bayesprobe/probe_executor.py tests/fixtures/open_questions/model_scale_validation_v0.2.json tests/test_evidence_memory.py tests/test_model_gateway.py tests/test_openai_gateway.py tests/test_core_cycles.py tests/test_probe_executor.py
git commit -m "feat: remember evidence provenance across cycles"
```

---

### Task 5: Add Capability-Aware Probe Design, Selection, and Candidate Pool Management

**Files:**
- Create: `bayesprobe/capabilities.py`
- Create: `bayesprobe/probe_design.py`
- Create: `bayesprobe/candidate_pool.py`
- Modify: `bayesprobe/schemas.py`
- Modify: `bayesprobe/probe_planner.py`
- Modify: `bayesprobe/probe_executor.py`
- Modify: `bayesprobe/initialization.py`
- Modify: `bayesprobe/model_gateway.py`
- Modify: `bayesprobe/openai_gateway.py`
- Modify: `tests/fixtures/open_questions/model_scale_validation_v0.2.json`
- Create: `tests/test_capabilities.py`
- Create: `tests/test_probe_design.py`
- Create: `tests/test_candidate_pool.py`
- Modify: `tests/test_probe_planner.py`
- Modify: `tests/test_probe_executor.py`
- Modify: `tests/test_initialization.py`
- Modify: `tests/test_openai_gateway.py`

**Interfaces:**
- Consumes: `CapabilityDescriptor` and `CapabilityDecision` from Task 1.
- Produces: `CapabilityRegistry.resolve(kind) -> CapabilityDecision`.
- Produces: `ProbeDesignContext`, `ProbeDesigner.propose(context)`, `DeterministicProbeDesigner`, and `ModelProbeDesigner`.
- Produces: `ProbeSelector.select(...)`; retains `ProbePlanner.design_probe_set(...)` as a compatibility wrapper.
- Produces: `CandidatePoolManager.build_next_pool(...)` shared by both runners.

- [ ] **Step 1: Write failing capability tests**

```python
def test_unavailable_capability_is_rejected_without_executor_fallback():
    registry = CapabilityRegistry([
        CapabilityDescriptor(kind=CapabilityKind.MODEL_REASONING, available=True, executor_adapter_id="model")
    ])
    candidate = candidate_requiring(CapabilityKind.SEARCH)
    result = ProbeSelector(registry=registry).select(selection_input([candidate]))
    assert result.probe_set.probes == []
    assert result.rejected_candidates[0].reason == "capability_unavailable:search"


def test_model_reasoning_cannot_execute_search_probe():
    with pytest.raises(CapabilityUnavailableError, match="search"):
        model_executor().execute_probe_set(
            probe_set=probe_set_requiring(CapabilityKind.SEARCH),
            context=execution_context(model_only_registry()),
        )
```

- [ ] **Step 2: Run capability tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_capabilities.py tests/test_probe_executor.py -q -p no:cacheprovider
```

Expected: imports fail because capability records and enforcement do not exist.

- [ ] **Step 3: Implement CapabilityRegistry**

Use this small public interface:

```python
class CapabilityRegistry:
    def __init__(self, descriptors: Iterable[CapabilityDescriptor]) -> None: ...

    def resolve(self, kind: CapabilityKind) -> CapabilityDecision: ...

    def snapshot(self) -> list[CapabilityDescriptor]: ...
```

`CapabilityDescriptor` records kind, availability, cost class, latency class, epistemic origin, quality caps, and executor adapter id. Duplicate kinds fail. Every selected probe must resolve to one available descriptor whose executor id matches the executor handling it.

- [ ] **Step 4: Write failing ProbeDesigner tests**

```python
def test_initial_open_design_contains_discriminator_or_coverage_probe():
    proposals = DeterministicProbeDesigner().propose(open_probe_context())
    assert any(
        proposal.candidate_probe.purpose in {
            ProbePurpose.HYPOTHESIS_DISCRIMINATION,
            ProbePurpose.FRAME_COVERAGE,
        }
        and len(proposal.candidate_probe.target_hypotheses) >= 2
        for proposal in proposals
    )


def test_model_probe_design_cannot_assign_posterior_or_priority():
    with pytest.raises(ModelGatewayValidationError, match="unsupported field"):
        ModelProbeDesigner(gateway_with_probe_payload({"posterior": 0.9})).propose(open_probe_context())
```

- [ ] **Step 5: Implement typed ProbeDesigner adapters**

Use the purposes defined in Task 1 exactly as specified:

```python
class ProbePurpose(StrEnum):
    HYPOTHESIS_DISCRIMINATION = "hypothesis_discrimination"
    HYPOTHESIS_FALSIFICATION = "hypothesis_falsification"
    FRAME_COVERAGE = "frame_coverage"
    SOURCE_VERIFICATION = "source_verification"
    ANOMALY_CLARIFICATION = "anomaly_clarification"
    ANSWER_CONTRACT_GAP = "answer_contract_gap"
```

`ProbeDesignContext` contains TaskFrame, FrameState, BeliefState, memory summary, unresolved uncertainty, contract gaps, change-my-mind conditions, and capability descriptors. The model returns purpose, target ids, inquiry goal, expected observation, support/weaken/reframe conditions, and required capability only. Server policy assigns ids and numeric score features.

- [ ] **Step 6: Convert planner into a selector**

Rename the deep implementation to `ProbeSelector.select`, keep the old class/method as a wrapper, and reject unknown targets, unavailable capabilities, repeated semantic fingerprints, and mismatched frame versions before scoring. Preserve deterministic expected-value ordering and top-hypothesis attack policy where applicable.

- [ ] **Step 7: Write and implement CandidatePoolManager**

Test merge order and de-duplication:

```python
def test_next_pool_preserves_core_candidates_before_designer_and_remaining():
    pool = CandidatePoolManager().build_next_pool(
        frame_version=2,
        core_candidates=[candidate("core")],
        designed_candidates=[candidate("designed")],
        change_my_mind_candidates=[candidate("change")],
        remaining_candidates=[candidate("remaining")],
        selected_candidate_ids=set(),
    )
    assert [item.candidate_id for item in pool] == ["core", "designed", "change", "remaining"]
```

The semantic fingerprint is SHA-256 over purpose, sorted targets, required capability, normalized inquiry goal, and frame version. Temporary ids never determine equality.

- [ ] **Step 8: Replace generic initialization probes**

Remove `_probe_candidate(... method="source_tracing")` from initialization. After BeliefState creation, invoke ProbeDesigner with the registry snapshot. Explicit MCQ keeps one answer-choice discriminator; open tasks get a discriminator or frame-coverage probe. No generic one-hypothesis source-tracing set satisfies initialization.

- [ ] **Step 9: Add provider support and run tests**

Add `design_probes` schema/instructions and one repair through the existing strict model path. Add a matching v0.2 recorded model-scale probe-design response so the public recorded slice remains green. Then run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_capabilities.py tests/test_probe_design.py tests/test_candidate_pool.py tests/test_probe_planner.py tests/test_probe_executor.py tests/test_initialization.py tests/test_openai_gateway.py -q -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
git diff --check
```

Expected: unavailable capabilities are visible, initial open probes are discriminative, and all offline tests pass.

- [ ] **Step 10: Commit the probe pipeline**

```bash
git add bayesprobe/capabilities.py bayesprobe/probe_design.py bayesprobe/candidate_pool.py bayesprobe/schemas.py bayesprobe/probe_planner.py bayesprobe/probe_executor.py bayesprobe/initialization.py bayesprobe/model_gateway.py bayesprobe/openai_gateway.py tests/fixtures/open_questions/model_scale_validation_v0.2.json tests/test_capabilities.py tests/test_probe_design.py tests/test_candidate_pool.py tests/test_probe_planner.py tests/test_probe_executor.py tests/test_initialization.py tests/test_openai_gateway.py
git commit -m "feat: design capability-aware probes"
```

---

### Task 6: Add History-Aware Semantic Evolution and Bounded Frame Expansion

**Files:**
- Create: `bayesprobe/hypothesis_expansion.py`
- Modify: `bayesprobe/hypothesis_evolution.py`
- Modify: `bayesprobe/schemas.py`
- Modify: `bayesprobe/core.py`
- Modify: `bayesprobe/model_gateway.py`
- Modify: `bayesprobe/openai_gateway.py`
- Modify: `tests/fixtures/open_questions/model_scale_validation_v0.2.json`
- Create: `tests/test_hypothesis_expansion.py`
- Modify: `tests/test_hypothesis_evolution.py`
- Modify: `tests/test_core_cycles.py`
- Modify: `tests/test_openai_gateway.py`

**Interfaces:**
- Produces: `HypothesisExpansionRequest`, `HypothesisCandidateProposal`, `HypothesisExpansionProposal`, `HypothesisExpansionAdapter.expand(request)`, and `ModelHypothesisExpansionAdapter`.
- Produces: `HypothesisEvolutionTriggerPolicy` and `SemanticEvolutionAdapter`; `HypothesisEvolutionEngine` coordinates but does not invent statements.
- Changes: `HypothesisEvolutionResult` contains `hypotheses`, `evolutions`, `probe_candidates`, next `frame_state`, `discovery_evidence_ids`, and `failure_reason: str | None`.

- [ ] **Step 1: Write failing missing-candidate recovery tests**

```python
def test_expansion_adds_missing_exact_answer_without_reusing_discovery_evidence():
    state = exact_state(
        named_values=[1, 2, 3],
        unresolved=0.70,
        adequacy=FrameAdequacyStatus.INADEQUATE,
    )
    result = core_with_expansion_proposal(value=4).integrate_cycle(
        cycle=cycle("cycle_2"),
        belief_state=state,
        probe_set=empty_probe_set("cycle_2"),
        signals=[supports_unresolved_signal("E_discovery")],
    )
    created = next(h for h in result.belief_state.hypotheses if h.answer_value == 4)
    assert created.created_by == "spawned"
    assert created.posterior <= 0.35
    assert "E_discovery" in result.belief_state.evidence_memory.discovery_evidence_ids
    assert not any(
        update.hypothesis_id == created.id and update.evidence_id == "E_discovery"
        for update in result.belief_updates
    )


def test_expansion_failure_preserves_state_and_creates_no_placeholder():
    result = engine_with_invalid_proposal_twice().evolve_from(inadequate_state())
    assert result.hypotheses == inadequate_state().hypotheses
    assert result.frame_state == inadequate_state().frame_state
    assert all("Spawned anomaly hypothesis" not in h.statement for h in result.hypotheses)
    assert result.failure_reason == "semantic_evolution_unavailable"


def test_expansion_proposal_is_not_an_evidence_event():
    result = core_with_expansion_proposal(value=4).integrate_cycle_from(inadequate_state())
    proposal_ref = result.hypothesis_evolutions[0].audit_fields["proposal_record_id"]
    assert all(event.id != proposal_ref for event in result.evidence_events)
    assert all(event.derived_from_signal != proposal_ref for event in result.evidence_events)
    assert all(update.evidence_id != proposal_ref for update in result.belief_updates)
```

- [ ] **Step 2: Run evolution tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_hypothesis_expansion.py tests/test_hypothesis_evolution.py -q -p no:cacheprovider
```

Expected: tests fail because semantic proposal adapters and frame expansion do not exist.

- [ ] **Step 3: Implement strict expansion proposals**

Create this seam in `bayesprobe/hypothesis_expansion.py`:

```python
@dataclass(frozen=True)
class HypothesisExpansionRequest:
    task_frame: TaskFrame
    frame_state: FrameState
    hypotheses: list[Hypothesis]
    evidence_memory: EvidenceMemorySnapshot
    unexplained_signals: list[dict[str, Any]]
    answer_value_type: AnswerValueType
    capabilities: list[CapabilityDescriptor]


class HypothesisExpansionAdapter(Protocol):
    def expand(self, request: HypothesisExpansionRequest) -> HypothesisExpansionProposal: ...
```

The model proposal permits only candidate statement or typed value, scope, falsifiers, predictions, current-frame failure rationale, discovery signal ids, and required next probe descriptions. Reject ids, priors, posteriors, status, frame version, unresolved mass, and numeric priority fields. After validation, the server assigns a `proposal_record_id` for ledger/audit references; that id is never a signal or Evidence Event id.

- [ ] **Step 4: Implement server-owned allocation**

Validate semantic distinctness against active and historical hypotheses, Answer Contract type conformance, at least one falsifier, prediction, and next probe, then allocate:

```python
transferable = max(
    0.0,
    min(
        current_unresolved / 2.0,
        current_unresolved - coverage_policy.minimum_unresolved_reserve,
    ),
)
per_candidate = transferable / len(valid_candidates)
```

Assign stable ids from run id, next frame version, and ordinal. Retired exclusive-open candidates return their mass to unresolved before allocation. Reject expansion when revision count is `3`, active count would exceed `8`, or no reserve can be transferred.

- [ ] **Step 5: Split trigger policy from semantic materialization**

Refactor `HypothesisEvolutionEngine` so deterministic history-aware policy authorizes spawn, expand, split, reframe, merge, retire, or reactivate based on FrameState and EvidenceMemory. The semantic adapter materializes substantive content only after authorization. Remove all generated statements of the forms `Spawned anomaly hypothesis...` and `Reframed scope of...`.

For every created or reframed hypothesis, require one returned probe candidate whose purpose is falsification, discrimination, coverage, or anomaly clarification. Trigger logic may use counterevidence ids accumulated in memory rather than only current-cycle events.

- [ ] **Step 6: Enforce discovery-evidence separation**

When a proposal cites discovery ids, add them to memory and candidate audit fields but do not generate BeliefUpdates for the new hypothesis from those events. A confirming event is eligible only when it comes from a later probe or a different derivation root. Add an explicit test where the same event is replayed and remains neutral.

- [ ] **Step 7: Add structured provider tasks and repair**

Add `expand_hypothesis_frame`, `repair_hypothesis_expansion`, and semantic evolution schemas to both OpenAI transports. One invalid proposal permits one repair; another invalid proposal returns a state-preserving failure. Prompt text must state that proposals do not assign belief.

- [ ] **Step 8: Integrate frame revisions in core**

After adequacy assessment, invoke expansion only when authorized. Commit the evolution, new hypotheses, discovery refs, required candidates, and new FrameState atomically. A successful expansion increments frame version/revision count, records parent version, and returns to `provisional`; failure retains the previous frame version.

- [ ] **Step 9: Run focused and full tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_hypothesis_expansion.py tests/test_hypothesis_evolution.py tests/test_core_cycles.py tests/test_openai_gateway.py -q -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
git diff --check
```

Expected: a wrong initial exact-answer set can expand, discovery evidence cannot self-confirm, placeholder content is absent, and full regression passes.

- [ ] **Step 10: Commit semantic evolution**

```bash
git add bayesprobe/hypothesis_expansion.py bayesprobe/hypothesis_evolution.py bayesprobe/schemas.py bayesprobe/core.py bayesprobe/model_gateway.py bayesprobe/openai_gateway.py tests/test_hypothesis_expansion.py tests/test_hypothesis_evolution.py tests/test_core_cycles.py tests/test_openai_gateway.py
git commit -m "feat: expand inadequate hypothesis frames"
```

---

### Task 7: Add Answer Contract-Aware Projection

**Files:**
- Create: `bayesprobe/projection_generator.py`
- Modify: `bayesprobe/projections.py`
- Modify: `bayesprobe/schemas.py`
- Modify: `bayesprobe/model_gateway.py`
- Modify: `bayesprobe/openai_gateway.py`
- Create: `tests/test_projection_generator.py`
- Modify: `tests/test_openai_gateway.py`
- Modify: `tests/test_question_runner.py`
- Modify: `tests/test_synchronized_runner.py`

**Interfaces:**
- Produces: `ProjectionInput`, `ProjectionGenerator.project(input) -> AnswerProjection`, `DeterministicProjectionGenerator`, `ModelProjectionGenerator`, and `ProjectionUnavailableError`.
- Changes: `AnswerProjection` v0.2 has selection, synthesis, or abstention mode and typed contract completion fields.
- Preserves: `build_answer_projection` and `build_belief_state_projection` as compatibility wrappers.

- [ ] **Step 1: Write failing selection and abstention tests**

```python
def test_exact_answer_abstains_when_unresolved_mass_is_too_high():
    projection = DeterministicProjectionGenerator().project(
        projection_input(
            exact_state(top=0.55, runner_up=0.10, unresolved=0.35, adequacy="challenged")
        )
    )
    assert projection.projection_mode == ProjectionMode.ABSTENTION
    assert projection.typed_answer_value is None
    assert "frame_adequacy" in projection.unmet_contract_sections


def test_mcq_still_returns_required_label_at_low_confidence():
    projection = DeterministicProjectionGenerator().project(low_confidence_mcq_input())
    assert projection.projection_mode == ProjectionMode.SELECTION
    assert projection.typed_answer_value == "C"
    assert projection.material_uncertainties
```

- [ ] **Step 2: Run projection tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_projection_generator.py -q -p no:cacheprovider
```

Expected: tests fail because v0.2 projection modes and generator do not exist.

- [ ] **Step 3: Implement the projection records and deterministic policy**

Use the exact interface:

```python
@dataclass(frozen=True)
class ProjectionInput:
    admission: TaskAdmissionDecision
    task_frame: TaskFrame
    frame_state: FrameState
    belief_state: BeliefState
    accepted_evidence: list[EvidenceEvent]
    evidence_memory: EvidenceMemorySnapshot
    missing_capabilities: list[CapabilityKind]
    stop_reason: str


class ProjectionGenerator(Protocol):
    def project(self, input: ProjectionInput) -> AnswerProjection: ...
```

`AnswerProjection` contains mode, answer, typed value, current best hypothesis, basis hypotheses, completed and unmet contract sections, evidence ids, citations, material uncertainties, missing capabilities, and change-my-mind condition. For exact selection apply all fixed thresholds from Global Constraints. Validate integer/number/label/short-text output types before selection.

- [ ] **Step 4: Implement model-backed synthesis and repair**

Use deterministic policy to choose mode and permitted basis hypotheses before any model call. `ModelProjectionGenerator` calls `project_open_answer` only for synthesis or natural-language rendering of an already-authorized selection/abstention. The model cannot change mode, ids, typed value, posterior, or citations.

Validate every required section, every cited source against Signal Provenance, and every basis id against BeliefState. Permit one `repair_answer_projection`; a second failure raises `ProjectionUnavailableError`, preserves state, and yields the public stop reason `answer_unavailable`.

- [ ] **Step 5: Prove projection purity**

```python
def test_projection_does_not_mutate_belief_or_create_evidence():
    input = projection_input_for_design()
    before = input.belief_state.model_copy(deep=True)
    projection = model_projection_generator().project(input)
    assert input.belief_state == before
    assert not hasattr(projection, "likelihoods")
    assert set(projection.source_citations) <= provenance_citations(before)
```

Use deep copies in tests and compare serialized BeliefState before/after.

- [ ] **Step 6: Add provider schemas and compatibility wrappers**

Add projection/repair tasks to OpenAI Responses and Chat Completions. Replace generic `Current best hypothesis is...` wrappers with calls into the generator. Preserve MCQ text expected by benchmark scoring and keep synchronized projection fields available in `projection_metadata` during the migration window. Add a contract-complete `project_open_answer` response to the v0.2 model-scale fixture.

- [ ] **Step 7: Run focused and full tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_projection_generator.py tests/test_openai_gateway.py tests/test_question_runner.py tests/test_synchronized_runner.py -q -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
git diff --check
```

Expected: exact questions can abstain, synthesis satisfies contracts, MCQ labels remain stable, and no projection mutates state.

- [ ] **Step 8: Commit task-aware projection**

```bash
git add bayesprobe/projection_generator.py bayesprobe/projections.py bayesprobe/schemas.py bayesprobe/model_gateway.py bayesprobe/openai_gateway.py tests/fixtures/open_questions/model_scale_validation_v0.2.json tests/test_projection_generator.py tests/test_openai_gateway.py tests/test_question_runner.py tests/test_synchronized_runner.py
git commit -m "feat: project answers from task contracts"
```

---

### Task 8: Integrate the Complete Autonomous Lifecycle

**Files:**
- Modify: `bayesprobe/question_runner.py`
- Modify: `bayesprobe/initialization.py`
- Modify: `bayesprobe/core.py`
- Modify: `bayesprobe/schemas.py`
- Modify: `bayesprobe/runners.py`
- Modify: `tests/test_question_runner.py`
- Modify: `tests/test_autonomous_runner.py`
- Modify: `tests/test_core_cycles.py`

**Interfaces:**
- Changes: `AutonomousQuestionRunner.run_question` returns a tagged admission/run result and uses all deep modules in the approved order.
- Produces: complete progress events and stop reasons from the design.
- Guarantees: no BeliefState progress before initialization and no omission of core-produced candidates.

- [ ] **Step 1: Write failing lifecycle-order tests**

```python
def test_progress_never_exposes_belief_before_admission_and_framing():
    observed = []
    runner = complete_runner(progress_observer=observed.append)
    runner.run_question(exact_answer_input())
    kinds = [event.kind for event in observed]
    assert kinds[:5] == [
        AutonomousQuestionProgressKind.RUN_STARTED,
        AutonomousQuestionProgressKind.TASK_ADMISSION_STARTED,
        AutonomousQuestionProgressKind.TASK_ADMISSION_COMPLETED,
        AutonomousQuestionProgressKind.TASK_FRAMING_STARTED,
        AutonomousQuestionProgressKind.TASK_FRAMING_COMPLETED,
    ]
    assert all(event.belief_state is None for event in observed[:5])


def test_core_candidate_is_available_to_next_autonomous_cycle():
    result = runner_with_cycle_one_core_candidate("core-follow-up", max_cycles=2).run_question(open_input())
    assert "core-follow-up" in {
        c.candidate_id for c in result.cycle_results[1].planning_result.selected_candidates
    }
```

- [ ] **Step 2: Run lifecycle tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_question_runner.py tests/test_autonomous_runner.py -q -p no:cacheprovider
```

Expected: progress and carry-forward assertions fail against the existing runner.

- [ ] **Step 3: Wire the autonomous runtime in the approved order**

The runner sequence is exactly:

```text
admit -> frame -> initialize -> seed passive inbox -> design -> build pool
-> select -> execute -> close boundary -> core integrate/evolve
-> project -> decide stop -> build next pool
```

At each next-pool build, pass `core_result.probe_candidates`, fresh ProbeDesigner candidates, projection change-my-mind candidates, and unselected remaining candidates to `CandidatePoolManager`. Never reconstruct the pool in the runner.

- [ ] **Step 4: Add the complete progress vocabulary**

Add and emit this exact vocabulary:

```python
class AutonomousQuestionProgressKind(StrEnum):
    RUN_STARTED = "run_started"
    TASK_ADMISSION_STARTED = "task_admission_started"
    TASK_ADMISSION_COMPLETED = "task_admission_completed"
    TASK_FRAMING_STARTED = "task_framing_started"
    TASK_FRAMING_COMPLETED = "task_framing_completed"
    INITIALIZATION_COMPLETED = "initialization_completed"
    CYCLE_STARTED = "cycle_started"
    PROBE_DESIGN_STARTED = "probe_design_started"
    PROBE_DESIGN_COMPLETED = "probe_design_completed"
    PROBE_SET_PLANNED = "probe_set_planned"
    PROBE_EXECUTION_STARTED = "probe_execution_started"
    CAPABILITY_UNAVAILABLE = "capability_unavailable"
    EXTERNAL_SIGNAL_REQUIRED = "external_signal_required"
    SIGNALS_COLLECTED = "signals_collected"
    EVIDENCE_INTEGRATION_STARTED = "evidence_integration_started"
    HYPOTHESIS_EXPANSION_STARTED = "hypothesis_expansion_started"
    HYPOTHESIS_EXPANSION_COMPLETED = "hypothesis_expansion_completed"
    CYCLE_INTEGRATED = "cycle_integrated"
    ANSWER_PROJECTION_STARTED = "answer_projection_started"
    ANSWER_PROJECTION_COMPLETED = "answer_projection_completed"
    ANSWER_PROJECTION_FAILED = "answer_projection_failed"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"
```

Snapshot mutable domain records before notifying observers. Observer exceptions remain isolated from execution.

- [ ] **Step 5: Implement contract-aware stopping**

Add these reasons:

```python
class AutonomousQuestionStopReason(StrEnum):
    ANSWER_CONTRACT_SATISFIED = "answer_contract_satisfied"
    MAX_CYCLES = "max_cycles"
    NO_VALUABLE_PROBES = "no_valuable_probes"
    EXTERNAL_SIGNAL_REQUIRED = "external_signal_required"
    FRAME_REVISION_BUDGET_EXHAUSTED = "frame_revision_budget_exhausted"
    ANSWER_UNAVAILABLE = "answer_unavailable"
```

An open task stopped by budget still invokes projection and normally abstains. Explicit MCQ continues to return its required label. Confidence stopping is disabled for independent frames and cannot bypass exact-answer adequacy/coverage thresholds.

- [ ] **Step 6: Preserve pre-cycle passive signals**

Keep compatibility `context` as an Initial Passive External Signal. Buffer it after initialization, execute selected active probes, combine active and passive signals, then close one boundary. It must not enter Task Admission or Task Framing input.

- [ ] **Step 7: Add failure-path tests**

Cover unavailable material capability, expansion repair failure, projection repair failure, revision budget exhaustion, no valuable probes, provider failure, and passive-only first cycle. Each failure must preserve the latest valid BeliefState and use secret-safe diagnostics.

- [ ] **Step 8: Run focused and full tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_question_runner.py tests/test_autonomous_runner.py tests/test_core_cycles.py -q -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
git diff --check
```

Expected: lifecycle order, candidate carry-forward, passive buffering, stopping, and failure semantics pass.

- [ ] **Step 9: Commit autonomous integration**

```bash
git add bayesprobe/question_runner.py bayesprobe/initialization.py bayesprobe/core.py bayesprobe/schemas.py bayesprobe/runners.py tests/test_question_runner.py tests/test_autonomous_runner.py tests/test_core_cycles.py
git commit -m "feat: run the complete autonomous epistemic loop"
```

---

### Task 9: Integrate the Complete Synchronized Lifecycle

**Files:**
- Modify: `bayesprobe/synchronized_runner.py`
- Modify: `bayesprobe/projections.py`
- Modify: `tests/test_synchronized_runner.py`
- Modify: `tests/test_candidate_pool.py`
- Modify: `tests/test_core_cycles.py`

**Interfaces:**
- Changes: synchronized rounds use the same CandidatePoolManager, CapabilityRegistry, ProbeDesigner, ProbeSelector, core, and ProjectionGenerator as autonomous mode.
- Preserves: fixed rounds, passive-only, active-only, and mixed signal shapes.
- Guarantees: a passive-only round remains a normal BayesProbe cycle with an empty ProbeSet.

- [ ] **Step 1: Write failing synchronized carry-forward and passive tests**

```python
def test_passive_only_round_closes_boundary_and_projects_normally():
    result = synchronized_runner().run_rounds(passive_only_run_input())
    round_result = result.round_results[0]
    assert round_result.probe_set.probes == []
    assert round_result.cycle.boundary_status == BoundaryStatus.INTEGRATED
    assert round_result.signals[0].signal_kind == SignalKind.PASSIVE
    assert round_result.belief_state_projection is not None


def test_core_candidate_reaches_next_synchronized_round():
    result = synchronized_runner_with_core_candidate("sync-follow-up").run_rounds(two_round_input())
    assert "sync-follow-up" in {
        c.candidate_id for c in result.round_results[1].selected_probe_candidates
    }
```

- [ ] **Step 2: Run synchronized tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_synchronized_runner.py tests/test_candidate_pool.py -q -p no:cacheprovider
```

Expected: the core-candidate test fails because `_next_candidate_pool` currently omits core results.

- [ ] **Step 3: Replace local pool logic with CandidatePoolManager**

Delete `_next_candidate_pool` from `synchronized_runner.py`. After each integrated round, pass core, designed, projection, and remaining candidates through the shared manager in the same order as autonomous mode. Persist selected and rejected capability decisions in each round result.

- [ ] **Step 4: Preserve synchronized boundary semantics**

For passive-only rounds, construct an empty allowed ProbeSet and process all supplied passive signals. For mixed rounds, wait until active execution returns, then combine signals and integrate once. The provided round list is the external barrier; no runner path may skip a non-empty passive round because no active probe exists.

- [ ] **Step 5: Use task-aware projection without forcing a terminal answer**

Emit a Belief State Projection for collaboration every round. Its metadata includes current Answer Projection mode, frame adequacy, unresolved mass, questions for others, requested capability/signal type, and cited sources. A collaborative projection may request external input even when the task-facing answer abstains.

- [ ] **Step 6: Run focused and full tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_synchronized_runner.py tests/test_candidate_pool.py tests/test_core_cycles.py -q -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
git diff --check
```

Expected: passive-only, active-only, mixed, fixed-round, candidate carry-forward, and projection tests all pass.

- [ ] **Step 7: Commit synchronized integration**

```bash
git add bayesprobe/synchronized_runner.py bayesprobe/projections.py tests/test_synchronized_runner.py tests/test_candidate_pool.py tests/test_core_cycles.py
git commit -m "feat: align synchronized epistemic rounds"
```

---

### Task 10: Complete Provider, Recorded Fixture, Config, SDK, and Artifact Surfaces

**Files:**
- Modify: `bayesprobe/kernel_config.py`
- Modify: `bayesprobe/config.py`
- Modify: `bayesprobe/experiment_runner.py`
- Modify: `bayesprobe/experiment_artifacts.py`
- Modify: `bayesprobe/recorded_gateway.py`
- Modify: `bayesprobe/openai_gateway.py`
- Modify: `bayesprobe/__init__.py`
- Modify: `tests/test_recorded_model_gateway.py`
- Modify: `tests/test_public_api_and_config.py`
- Modify: `tests/test_experiment_runner.py`
- Modify: `tests/test_experiment_artifacts.py`
- Modify: `tests/test_openai_gateway.py`

**Interfaces:**
- Produces: `EpistemicKernelConfig` as one public immutable policy object.
- Changes: JSON config parses admission, coverage, adequacy, correlation, capability, expansion, and projection policies without accepting raw credentials.
- Changes: recorded fixtures match structured task identity and request metadata subsets.
- Exports: all approved v0.2 interfaces from package root.

- [ ] **Step 1: Write failing config and export tests**

```python
def test_public_package_exports_epistemic_kernel_interfaces():
    for name in {
        "TaskAdmitter",
        "TaskAdmissionDecision",
        "FrameState",
        "EvidenceMemorySnapshot",
        "ProbeDesigner",
        "ProbeSelector",
        "CandidatePoolManager",
        "CapabilityRegistry",
        "HypothesisExpansionAdapter",
        "ProjectionGenerator",
        "EpistemicKernelConfig",
    }:
        assert hasattr(bayesprobe, name)


def test_config_rejects_nested_raw_api_key():
    payload = valid_config_mapping()
    payload["epistemic_kernel"] = {"projection": {"api_key": "forbidden"}}
    with pytest.raises(ValueError, match="api_key"):
        experiment_config_from_mapping(payload)
```

- [ ] **Step 2: Run public-surface tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_public_api_and_config.py tests/test_recorded_model_gateway.py tests/test_experiment_artifacts.py -q -p no:cacheprovider
```

Expected: exports and policy config parsing fail because they do not yet exist.

- [ ] **Step 3: Compose EpistemicKernelConfig**

```python
@dataclass(frozen=True)
class EpistemicKernelConfig:
    open_coverage: OpenCoveragePolicy = field(default_factory=OpenCoveragePolicy)
    frame_adequacy: FrameAdequacyPolicyConfig = field(default_factory=FrameAdequacyPolicyConfig)
    correlation_credit: CorrelationCreditPolicy = field(default_factory=CorrelationCreditPolicy)
    expansion: ExpansionPolicy = field(default_factory=ExpansionPolicy)
    projection: ProjectionPolicy = field(default_factory=ProjectionPolicy)
```

Parse nested JSON objects with unknown-field rejection. Snapshot every value in artifact config and manifest. Never serialize capability executor objects; serialize only secret-free descriptors and adapter ids.

- [ ] **Step 4: Improve recorded request matching**

Extend fixture matches with optional `prompt_id`, `schema_version`, and a metadata subset:

```json
{
  "match": {
    "task": "execute_probe",
    "prompt_id": "probe_execution",
    "metadata": {"probe_id": "P_cycle_1_frame_coverage"}
  },
  "response": {"raw_content": "A recorded self-authored observation."}
}
```

Matching is exact for scalar values and subset-based for metadata. Reject raw question text, provider headers, key-like names, and secret-like values recursively. Preserve existing task-only and signal-id fixtures.

- [ ] **Step 5: Audit every OpenAI structured task**

The schema matrix must include admission, admission repair, v0.2 framing, framing repair, probe design, probe execution, evidence judgment, evidence repair, hypothesis expansion, expansion repair, open projection, and projection repair for both Responses and Chat Completions. Add one test per task proving request schema version, required keys, parser behavior, and secret-safe provider errors.

- [ ] **Step 6: Wire policy config through public composition roots**

Pass one `EpistemicKernelConfig` from config/WebUI/SDK into initializer, core, runners, selector, expansion, and projection. Modules receive only their policy slice internally. Do not add parallel keyword defaults at every caller.

- [ ] **Step 7: Update artifact provenance**

Artifact manifests record schema version, policy snapshot, capability snapshot, frame revision summary, projection mode counts, prompt/schema versions, and model invocation identity. Evidence Memory raw content and request-scoped credentials remain absent. Add recursive leak tests over JSON and JSONL outputs.

- [ ] **Step 8: Run focused and full tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_recorded_model_gateway.py tests/test_public_api_and_config.py tests/test_experiment_runner.py tests/test_experiment_artifacts.py tests/test_openai_gateway.py -q -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
git diff --check
```

Expected: package exports, policy config, recorded matching, provider matrix, artifact snapshots, and leak scans pass.

- [ ] **Step 9: Commit external programmatic surfaces**

```bash
git add bayesprobe/kernel_config.py bayesprobe/config.py bayesprobe/experiment_runner.py bayesprobe/experiment_artifacts.py bayesprobe/recorded_gateway.py bayesprobe/openai_gateway.py bayesprobe/__init__.py tests/test_recorded_model_gateway.py tests/test_public_api_and_config.py tests/test_experiment_runner.py tests/test_experiment_artifacts.py tests/test_openai_gateway.py
git commit -m "feat: expose epistemic kernel configuration"
```

---

### Task 11: Make the Complete Kernel Observable and Testable in the WebUI

**Files:**
- Modify: `bayesprobe/webui.py`
- Modify: `bayesprobe/webui_static/index.html`
- Modify: `bayesprobe/webui_static/app.js`
- Modify: `bayesprobe/webui_static/styles.css`
- Modify: `tests/test_webui.py`
- Modify: `tests/test_webui_stream.js`

**Interfaces:**
- Changes: autonomous request accepts secret-safe kernel policy overrides and capability declarations.
- Changes: streamed events expose admission, framing, frame state, probes, signals, evidence memory status, evolution, and projection without revealing provider credentials.
- Preserves: API key remains in page memory for repeated runs and is never returned by the server.

- [ ] **Step 1: Write failing request and serialization tests**

```python
def test_webui_serializes_needs_reframing_without_belief_state():
    payload = serialize_autonomous_run_result(needs_reframing_result())
    assert payload["result_type"] == "needs_reframing"
    assert payload["admission"]["clarification_questions"]
    assert "final_belief_state" not in payload


def test_webui_serializes_frame_and_memory_without_secret_material():
    payload = serialize_autonomous_run_result(completed_v02_result())
    assert payload["final_belief_state"]["frame_state"]["coverage"] == "open"
    assert payload["final_answer"]["projection_mode"] in {"selection", "synthesis", "abstention"}
    assert "api_key" not in json.dumps(payload).lower()
```

- [ ] **Step 2: Run WebUI Python tests and verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_webui.py -q -p no:cacheprovider
```

Expected: tagged admission and v0.2 state serialization assertions fail.

- [ ] **Step 3: Parse and compose request-scoped runtime safely**

Extend `_parse_autonomous_request` with nested `epistemic_kernel` and `capabilities` objects, strict unknown-field rejection, numeric validation, and recursive secret detection. `_prepare_autonomous_run` creates one registry/config and injects the same modules into initializer, core, runner, and executor. Client input may request only server-registered executor adapters; checking a capability cannot make an unavailable adapter available. Do not make a second provider request during preflight.

- [ ] **Step 4: Add complete streaming events**

Serialize all progress events in order and include only the domain record relevant to that event. `task_admission_completed` may terminate successfully with `needs_reframing` or `out_of_scope`. `initialization_completed` remains the first event allowed to carry BeliefState. A failed model adapter emits one sanitized `run_failed` terminal event.

- [ ] **Step 5: Write failing browser-state tests**

Add Node tests proving:

```javascript
test("API key remains in page memory after a completed run", async () => {
  document.querySelector("#api-key").value = "page-memory-only";
  await consumeRunStream(completedStream());
  assert.equal(document.querySelector("#api-key").value, "page-memory-only");
});

test("belief state is rejected before initialization", () => {
  assert.throws(
    () => handleProgressEvent(eventWithBelief("task_framing_completed")),
    /before initialization/
  );
});
```

Also cover needs-reframing rendering, unresolved mass, frame history, capability unavailable, correlated/discarded evidence, expansion, synthesis, abstention, and stream termination.

- [ ] **Step 6: Update the operational UI**

Keep the existing dense workbench layout. Add compact controls for maximum frame revisions and active hypotheses, plus a capability checklist using checkboxes. Render:

```text
Admission: status, reason, clarification questions
Task Frame: kind, answer relationship, Answer Contract
Belief State: competition, coverage, adequacy, unresolved mass, named claims
Current Cycle: probe purpose/capability, signal origin, evidence correlation status
Frame History: version, trigger ids, expansion/evolution operation
Answer Projection: mode, typed value, completed/unmet sections, uncertainties
```

Use existing panel bands; do not add nested cards, decorative gradients, or explanatory marketing copy. Only implemented executor capabilities are selectable; unimplemented search, retrieval, repository mutation, and test execution remain visibly disabled. Ensure long Chinese and English content wraps without overlapping controls.

- [ ] **Step 7: Run WebUI test suites**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_webui.py -q -p no:cacheprovider
node --test tests/test_webui_stream.js
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
git diff --check
```

Expected: Python WebUI tests and Node stream tests pass; full offline regression remains green.

- [ ] **Step 8: Restart and visually verify port 8768**

Restart the existing loopback launch agent using the implementation worktree, then verify:

```bash
curl --fail --silent http://127.0.0.1:8768/ >/dev/null
curl --fail --silent http://127.0.0.1:8768/app.js >/dev/null
```

Use `browser:control-in-app-browser` to inspect desktop and mobile widths. Confirm no overlap, blank state, secret echo, stale run state, or missing dynamic event. Submit one deterministic MCQ and one recorded open exact-answer recovery fixture through the visible UI.

- [ ] **Step 9: Commit WebUI integration**

```bash
git add bayesprobe/webui.py bayesprobe/webui_static/index.html bayesprobe/webui_static/app.js bayesprobe/webui_static/styles.css tests/test_webui.py tests/test_webui_stream.js
git commit -m "feat: expose epistemic kernel in webui"
```

---

### Task 12: Add Recorded Vertical Slices and Close Every Verification Gate

**Files:**
- Create: `tests/test_epistemic_kernel_vertical_slices.py`
- Create: `tests/fixtures/epistemic_kernel/exact_answer_missing_candidate_v0.2.json`
- Create: `tests/fixtures/epistemic_kernel/design_synthesis_v0.2.json`
- Create: `tests/fixtures/epistemic_kernel/admission_reframing_v0.2.json`
- Create: `tests/fixtures/epistemic_kernel/out_of_scope_generation_v0.2.json`
- Create: `tests/fixtures/epistemic_kernel/unavailable_capability_v0.2.json`
- Modify: `tests/test_question_runner.py`
- Modify: `tests/test_synchronized_runner.py`
- Modify: `tests/test_public_api_and_config.py`
- Modify: `tests/evaluation/test_end_to_end.py`
- Modify: `tests/evaluation/test_deepseek_live.py`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `CONTEXT.md`
- Modify: `docs/superpowers/specs/2026-07-11-epistemic-kernel-completion-design.md`

**Interfaces:**
- Produces: public-runner, secret-free recorded proof for every main architecture path.
- Preserves: existing HLE text-MCQ harness and benchmark behavior without a formal HLE run.
- Produces: verified implementation status and review evidence.

- [ ] **Step 1: Write the exact-answer recovery fixture and vertical slice**

Use a self-authored scalar question whose initial candidates are `1`, `2`, and `3`, while recorded discriminating evidence supports unresolved and expansion proposes `4`. Match repeated model requests by task and metadata. Assert:

```python
def test_recorded_exact_answer_recovers_missing_candidate_through_public_runner():
    result = run_recorded_fixture("exact_answer_missing_candidate_v0.2.json")
    assert result.final_belief_state.frame_state.frame_version >= 2
    assert any(h.answer_value == 4 for h in result.final_belief_state.hypotheses)
    assert result.final_answer_projection.projection_mode == ProjectionMode.SELECTION
    assert result.final_answer_projection.typed_answer_value == 4
    assert discovery_event_did_not_confirm_created_candidate(result)
```

- [ ] **Step 2: Add design synthesis, reframing, and capability fixtures**

The design fixture must produce independent-open claims and a synthesis satisfying all required sections. The existential fixture returns `needs_reframing` with clarification questions and no state. The pure creative-generation fixture returns `out_of_scope` with no TaskFrame or state. The unavailable-capability fixture requests search, records the rejection, and projects abstention with `missing_capabilities=["search"]`.

- [ ] **Step 3: Cover all cycle shapes through public runners**

Add active-only, passive-only, active-plus-passive, autonomous, and synchronized recorded slices. Assert boundary closure, common Evidence Gate behavior, memory updates, candidate carry-forward, projection mode, and ledger ordering. Do not assert internal helper calls when the public result proves the contract.

- [ ] **Step 4: Add recursive security and artifact scans**

Scan every generated ledger, report, fixture, config snapshot, model invocation summary, and WebUI serialized result for forbidden key names and secret-like values. Assert Evidence Memory contains fingerprints/refs rather than API keys or provider headers.

- [ ] **Step 5: Run all recorded and offline suites**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_epistemic_kernel_vertical_slices.py tests/test_question_runner.py tests/test_synchronized_runner.py tests/test_public_api_and_config.py tests/evaluation/test_end_to_end.py -q -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
node --test tests/test_webui_stream.js
git diff --check
```

Expected: every recorded slice and the full offline suites pass with no failures.

- [ ] **Step 6: Run the real Docker isolation suite**

Run:

```bash
BAYESPROBE_RUN_DOCKER_TESTS=1 PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/evaluation/test_python_sandbox_integration.py -q -p no:cacheprovider
```

Expected: all real Docker isolation tests pass; no local fallback is accepted.

- [ ] **Step 7: Add and run five self-authored live-provider smokes**

Extend the opt-in DeepSeek live test with exactly five self-authored tasks: MCQ, missing-candidate exact answer, independent claim verification, design synthesis, and needs-reframing admission. Read the key only from `DEEPSEEK_API_KEY`; do not print or persist it.

Run only with an explicitly provided environment key:

```bash
BAYESPROBE_RUN_DEEPSEEK_LIVE=1 PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/evaluation/test_deepseek_live.py -q -p no:cacheprovider
```

Expected: five live smoke cases pass. Without the environment key, this Definition of Done item remains open and the branch must not be described as complete.

- [ ] **Step 8: Update architecture and status documentation**

Update the implemented-state table in `docs/ARCHITECTURE.md`, preserve the distinction between Probe and future Intervention, record SWE-bench Verified, Terminal-Bench, and RE-Bench only as future targets, and keep HLE secondary. Mark the design status `Implemented and verified` only after Steps 5-7 pass. Update `CONTEXT.md` only for terminology changed by actual code.

- [ ] **Step 9: Request two independent whole-branch reviews**

Use `superpowers:requesting-code-review` twice with fresh reviewers. Each reviewer compares the full branch against the design and this plan, prioritizing correctness, state invariants, security, runner consistency, and missing tests. Resolve every Critical or Important finding, rerun affected focused tests, then rerun the full offline suite.

- [ ] **Step 10: Run final clean-room verification**

Run from the implementation worktree:

```bash
git status --short
git diff --check main...HEAD
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
node --test tests/test_webui_stream.js
BAYESPROBE_RUN_DOCKER_TESTS=1 PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/evaluation/test_python_sandbox_integration.py -q -p no:cacheprovider
```

Expected: only intentional uncommitted documentation/review fixes appear before the final commit; all suites pass.

- [ ] **Step 11: Commit verification and documentation**

```bash
git add tests/test_epistemic_kernel_vertical_slices.py tests/fixtures/epistemic_kernel tests/test_question_runner.py tests/test_synchronized_runner.py tests/test_public_api_and_config.py tests/evaluation/test_end_to_end.py tests/evaluation/test_deepseek_live.py docs/ARCHITECTURE.md CONTEXT.md docs/superpowers/specs/2026-07-11-epistemic-kernel-completion-design.md
git commit -m "test: verify epistemic kernel completion"
```

- [ ] **Step 12: Hand off the finished branch**

Use `superpowers:finishing-a-development-branch`. Report exact Python, Node, Docker, and live-smoke counts, review findings resolved, final commit hash, WebUI URL, and remaining future benchmark/Intervention work. Do not merge or push the implementation branch without the user's selected finish option.

---

## Milestone Review Gates

1. After Tasks 1-2: review admission, migration, and no-BeliefState-before-framing invariants.
2. After Tasks 3-4: review mass conservation, frame adequacy, provenance, duplicate neutrality, and correlation credit.
3. After Tasks 5-7: review capability honesty, probe semantics, candidate carry-forward, expansion, discovery evidence, and projection purity.
4. After Tasks 8-9: review autonomous/synchronized behavioral equivalence at the shared core.
5. After Tasks 10-11: review public SDK/config/provider/artifact security and WebUI observability.
6. After Task 12: complete two fresh whole-branch reviews and all verification gates before any completeness claim.

## Spec Coverage Audit

| Approved design sections | Implementation tasks |
| --- | --- |
| 1-5: decisions, scope, invariants, target runtime | Global Constraints, Tasks 0-1, milestone gates |
| 6: Task Admission | Task 2, Tasks 8 and 11 integration, Task 12 recorded outcomes |
| 7-9: TaskFrame, competition/coverage, FrameState | Tasks 1-3 |
| 10-13: evidence judgment, expansion input, provenance, Evidence Memory | Tasks 3-4 and Task 6 discovery handling |
| 14-16: ProbeDesigner, Capability Registry, CandidatePoolManager | Task 5, Tasks 8-9 integration |
| 17: history-aware semantic evolution | Task 6 |
| 18: selection, synthesis, abstention | Task 7, Tasks 8-9 integration |
| 19: runner/progress semantics | Tasks 8-9 and Task 11 streaming |
| 20-23: WebUI, SDK, config, failures, security, migration | Tasks 1-2 and Tasks 10-11 |
| 24-25: tests and delivery milestones | All task-level gates and Task 12 |
| 26-27: future benchmarks and non-goals | Global Constraints, Completion Boundary, Task 12 docs |
| 28: Definition of Done | Task 12 verification, live smoke, reviews, and handoff |

Self-review found no approved design requirement without an owning task. The plan does not claim coding Intervention or public benchmark implementation.

## Completion Boundary

This plan completes the epistemic kernel and its public/WebUI verification surface. It deliberately stops before repository mutation, coding Intervention, real search/retrieval, SWE-bench, Terminal-Bench, RE-Bench, private benchmarks, or formal HLE execution. Those items require separate approved designs and plans after this kernel is frozen.
