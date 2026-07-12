# Open-Question MVP Vertical Slice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make an unseeded open question complete a real BayesProbe autonomous loop with task-specific probes, Core-authorized hypothesis expansion, and an Answer-Contract-facing result in the WebUI.

**Architecture:** Preserve `BayesProbeCore` as the authority for Signal-to-Evidence conversion, belief revision, frame adequacy, and expansion authorization. Add three semantic adapters around that kernel: `ProbeDesigner`, `HypothesisExpansionService`, and `AnswerProjector`. The autonomous runner coordinates them, while the OpenAI-compatible gateway supplies structured model output and recorded fixtures make both MVP paths deterministic offline.

**Tech Stack:** Python 3.11+, Pydantic 2, pytest 8+, existing synchronous `ModelGateway`, OpenAI Responses and Chat Completions adapters, vanilla JavaScript, Node's built-in test runner.

## Global Constraints

- Implement only the autonomous open-question vertical slice described in `docs/superpowers/specs/2026-07-12-open-question-mvp-vertical-slice-design.md`.
- Preserve existing explicit multiple-choice behavior.
- Do not implement search, browsing, document retrieval, repository tools, shell tools, synchronized-runner parity, benchmark execution, or a general plugin registry.
- The WebUI's only executable remote capability is `CapabilityKind.MODEL_REASONING`.
- A model proposes semantics only. It never assigns ids, priors, posteriors, frame status, unresolved mass, cost, information-gain scores, or final priority.
- Only `BayesProbeCore` may admit Evidence, change belief, or authorize hypothesis expansion.
- A Signal used to discover a new hypothesis cannot confirm that hypothesis in the same cycle.
- Provider or schema failure must name the failed stage and must not fall back to generic `source_tracing`, placeholder hypotheses, or top-H answer text.
- API keys must remain request-scoped and absent from prompts, fixtures, ledgers, progress events, exceptions, and commits.
- Keep provider timeouts at or above the existing WebUI minimum of 360 seconds.
- Every product task follows red-green-refactor and ends in one focused commit.
- Stop implementation when the two recorded vertical slices, the full Python suite, the Node WebUI suite, and one user-driven real-provider WebUI smoke test pass.

## Execution Preflight

The worktree currently contains five known, interrupted Task 4 hardening edits that are outside this MVP. Before Task 1, archive and remove only those edits:

```bash
git diff -- \
  .superpowers/sdd/task-4-findings.md \
  bayesprobe/schemas.py \
  tests/test_core_cycles.py \
  tests/test_migrations.py \
  tests/test_schemas.py \
  > /tmp/bayesprobe-interrupted-task4.patch

git apply --reverse /tmp/bayesprobe-interrupted-task4.patch
git status --short
```

Expected: no modified files. If `git status --short` lists any path not named above before cleanup, stop and preserve it as user work.

Baseline verification:

```bash
pytest -q
node --test tests/test_webui_stream.js
```

Expected: the committed baseline passes before MVP changes begin.

## File Map

- Create `bayesprobe/probe_design.py`: semantic proposal validation, deterministic ids/scores, capability filtering, frame-level fallback designer, and model-backed designer.
- Create `bayesprobe/hypothesis_expansion.py`: semantic expansion proposal validation, bounded materialization, unresolved-mass allocation, and model-backed expansion adapter.
- Modify `bayesprobe/projections.py`: task-aware selection, synthesis, and abstention projectors while retaining the compatibility function used by other runners.
- Modify `bayesprobe/schemas.py`: typed `ProbeDesign` semantics and explicit `AnswerProjection` mode/value/sections.
- Modify `bayesprobe/initialization.py`: stop emitting generic open-task per-hypothesis probes.
- Modify `bayesprobe/core.py`: invoke semantic expansion only after `FrameAdequacyPolicy` authorizes it and commit expansion atomically.
- Modify `bayesprobe/question_runner.py`: call the designer, merge every candidate source, use the projector, and emit stage progress.
- Modify `bayesprobe/openai_gateway.py`: strict schemas and prompts for probe design, hypothesis expansion, and answer projection.
- Modify `bayesprobe/recorded_gateway.py`: allow fixture matching by `cycle_id` and `probe_id` metadata.
- Modify `bayesprobe/webui.py`: wire one request-scoped gateway into all three semantic adapters and serialize their progress.
- Modify `bayesprobe/webui_static/app.js`: render projection mode, contract sections, independent credence, unresolved mass, and new progress stages.
- Modify `bayesprobe/__init__.py`: export the new public protocols and adapters.
- Create `tests/test_probe_design.py`: proposal validation, deterministic authority, repair, deduplication, and capability tests.
- Create `tests/test_hypothesis_expansion_service.py`: exact/open and independent/open materialization tests.
- Modify `tests/test_initialization.py`, `tests/test_core_cycles.py`, `tests/test_question_runner.py`, `tests/test_openai_gateway.py`, `tests/test_recorded_model_gateway.py`, `tests/test_webui.py`, and `tests/test_webui_stream.js`.
- Create `tests/fixtures/open_questions/model_scale_open_mvp_v0.1.json` and `tests/fixtures/open_questions/exact_answer_expansion_mvp_v0.1.json`.
- Modify `docs/ARCHITECTURE.md`: mark only this autonomous MVP slice implemented.

---

### Task 1: Add Typed Probe Semantics and the Model Probe Designer

**Files:**
- Modify: `bayesprobe/schemas.py`
- Create: `bayesprobe/probe_design.py`
- Modify: `bayesprobe/openai_gateway.py`
- Modify: `bayesprobe/__init__.py`
- Create: `tests/test_probe_design.py`
- Modify: `tests/test_schemas.py`
- Modify: `tests/test_openai_gateway.py`

**Interfaces:**
- Consumes: `TaskFrame`, `BeliefState`, `ProbePurpose`, `CapabilityKind`, `CapabilityDescriptor`, `CapabilityDecision`, `ModelGateway`, and `StructuredModelRequest`.
- Produces: `ProbeDesignError`, `ProbeDesignContext`, `ProbeDesignResult`, `ProbeDesigner.propose(context)`, `FrameProbeDesigner`, `ModelProbeDesigner`, and `MODEL_REASONING_CAPABILITY`.
- Produces on `ProbeDesign`: `purpose`, `expected_observation`, and `required_capability`.

- [ ] **Step 1: Write failing schema tests for typed probe fields**

Add to `tests/test_schemas.py`:

```python
def test_probe_design_carries_server_typed_semantics():
    probe = ProbeDesign(
        id="P_cycle_1_discriminate",
        cycle_id="cycle_1",
        target_hypotheses=["H1", "H2"],
        inquiry_goal="Distinguish a size effect from a compute-budget confounder.",
        method="model_reasoning",
        purpose=ProbePurpose.HYPOTHESIS_DISCRIMINATION,
        expected_observation="A matched-budget comparison changes the apparent size effect.",
        required_capability=CapabilityKind.MODEL_REASONING,
    )

    assert probe.purpose == ProbePurpose.HYPOTHESIS_DISCRIMINATION
    assert probe.required_capability == CapabilityKind.MODEL_REASONING


def test_probe_design_rejects_blank_expected_observation():
    with pytest.raises(ValueError, match="expected_observation"):
        ProbeDesign(
            id="P1",
            cycle_id="cycle_1",
            target_hypotheses=["H1"],
            inquiry_goal="Test H1.",
            method="model_reasoning",
            expected_observation="   ",
        )
```

- [ ] **Step 2: Run the schema tests and verify RED**

Run:

```bash
pytest tests/test_schemas.py::test_probe_design_carries_server_typed_semantics \
  tests/test_schemas.py::test_probe_design_rejects_blank_expected_observation -q
```

Expected: FAIL because `ProbeDesign` does not yet expose or validate these fields.

- [ ] **Step 3: Add backward-compatible typed fields to `ProbeDesign`**

Add these fields and validators in `bayesprobe/schemas.py`:

```python
class ProbeDesign(BaseModel):
    id: str
    cycle_id: str
    target_hypotheses: list[str]
    inquiry_goal: str
    method: str
    purpose: ProbePurpose = ProbePurpose.HYPOTHESIS_DISCRIMINATION
    expected_observation: str = "A result that changes support for a target hypothesis."
    required_capability: CapabilityKind = CapabilityKind.MODEL_REASONING
    probe_type: str = "discriminative_test"
    support_condition: dict[str, str] = Field(default_factory=dict)
    weaken_condition: dict[str, str] = Field(default_factory=dict)
    reframe_condition: dict[str, str] | None = None
    expected_information_gain: float = 0.5
    decision_relevance: float = 0.5
    cost_estimate: float = 0.5
    priority: float = 0.5
    status: str = "candidate"

    @field_validator("id", "cycle_id", "inquiry_goal", "method", "expected_observation")
    @classmethod
    def clean_probe_text(cls, value: str, info: ValidationInfo) -> str:
        return _required_text(value, info.field_name)
```

Keep the existing score validator unchanged so old fixtures remain readable.

- [ ] **Step 4: Write failing designer tests**

Create `tests/test_probe_design.py` with a v0.2 independent/open state fixture and these tests:

```python
def test_model_probe_designer_materializes_server_owned_candidate(open_state):
    gateway = ScriptedModelGateway(
        {
            "design_probes": {
                "proposals": [
                    {
                        "purpose": "hypothesis_discrimination",
                        "target_hypotheses": ["H1", "H2"],
                        "inquiry_goal": "Compare model sizes under matched inference budgets.",
                        "expected_observation": "The size coefficient survives or collapses after matching.",
                        "support_condition": {"H1": "The matched coefficient remains positive."},
                        "weaken_condition": {"H1": "The matched coefficient is negligible."},
                        "reframe_condition": {"frame": "Neither hypothesis explains task interactions."},
                        "required_capability": "model_reasoning",
                    }
                ]
            }
        }
    )
    designer = ModelProbeDesigner(gateway)

    result = designer.propose(
        ProbeDesignContext(
            run_id="run_open",
            cycle_id="cycle_1",
            task_frame=open_state.task_frame,
            belief_state=open_state,
            available_capabilities=(MODEL_REASONING_CAPABILITY,),
        )
    )

    assert len(result.candidates) == 1
    probe = result.candidates[0].candidate_probe
    assert probe.id.startswith("P_cycle_1_")
    assert probe.priority == 0.85
    assert probe.required_capability == CapabilityKind.MODEL_REASONING
    response = gateway.responses["design_probes"]["proposals"][0]
    assert "id" not in response
    assert "priority" not in response


def test_model_probe_designer_rejects_unavailable_search(open_state):
    gateway = ScriptedModelGateway(
        {"design_probes": {"proposals": [search_proposal()]}}
    )

    result = ModelProbeDesigner(gateway).propose(
        ProbeDesignContext(
            run_id="run_open",
            cycle_id="cycle_1",
            task_frame=open_state.task_frame,
            belief_state=open_state,
            available_capabilities=(MODEL_REASONING_CAPABILITY,),
        )
    )

    assert result.candidates == []
    assert result.capability_decisions[0].kind == CapabilityKind.SEARCH
    assert result.capability_decisions[0].available is False
```

Also test one repair attempt, unknown hypothesis ids, semantic duplicate removal, secret rejection, and the requirement that an initial open design contains a multi-hypothesis discriminator or frame-coverage proposal.

- [ ] **Step 5: Run the designer tests and verify RED**

Run: `pytest tests/test_probe_design.py -q`

Expected: collection fails because `bayesprobe.probe_design` does not exist.

- [ ] **Step 6: Implement the probe-design deep module**

Create `bayesprobe/probe_design.py` with these public contracts:

```python
class ProbeDesignError(ValueError):
    pass


@dataclass(frozen=True)
class ProbeDesignContext:
    run_id: str
    cycle_id: str
    task_frame: TaskFrame
    belief_state: BeliefState
    available_capabilities: tuple[CapabilityDescriptor, ...]


@dataclass(frozen=True)
class ProbeDesignResult:
    candidates: list[ProbeCandidate]
    capability_decisions: list[CapabilityDecision]


class ProbeDesigner(Protocol):
    def propose(self, context: ProbeDesignContext) -> ProbeDesignResult: ...


MODEL_REASONING_CAPABILITY = CapabilityDescriptor(
    kind=CapabilityKind.MODEL_REASONING,
    available=True,
    cost_class="bounded",
    latency_class="interactive",
    epistemic_origin=EpistemicOrigin.MODEL_REASONING,
    quality_caps={"verifiability": 0.45, "independence": 0.25},
    executor_adapter_id="model_probe_gateway:v1",
)
```

Use a private Pydantic `ProbeProposal` with `extra="forbid"` for the eight model-owned semantic fields in Step 4. Materialize candidates by hashing this canonical identity:

```python
identity = {
    "cycle_id": context.cycle_id,
    "purpose": proposal.purpose.value,
    "targets": sorted(proposal.target_hypotheses),
    "goal": " ".join(proposal.inquiry_goal.casefold().split()),
    "capability": proposal.required_capability.value,
}
digest = sha256(
    json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
).hexdigest()[:12]
```

Server-owned scores are fixed for the MVP:

```python
_PRIORITY_BY_PURPOSE = {
    ProbePurpose.HYPOTHESIS_DISCRIMINATION: 0.85,
    ProbePurpose.HYPOTHESIS_FALSIFICATION: 0.80,
    ProbePurpose.FRAME_COVERAGE: 0.82,
    ProbePurpose.SOURCE_VERIFICATION: 0.70,
    ProbePurpose.ANOMALY_CLARIFICATION: 0.78,
    ProbePurpose.ANSWER_CONTRACT_GAP: 0.75,
}
```

`FrameProbeDesigner` creates one frame-level discriminator from the existing hypotheses and is used only for deterministic/explicit compatibility paths. `ModelProbeDesigner` sends `design_probes`, permits one `repair_probe_design` call, validates all target ids against the current active hypotheses, and returns no candidate for an unavailable capability.

`FrameProbeDesigner` reports its deterministic compatibility executor as an
available `CapabilityKind.MODEL_REASONING` decision with
`executor_adapter_id="deterministic_frame_probe_designer:v1"`; it does not
claim an external source. `ModelProbeDesigner` catches gateway failures and
raises only `ProbeDesignError("probe design model gateway call failed")`.

- [ ] **Step 7: Add the strict provider schema**

Add `PROBE_DESIGN_JSON_SCHEMA` to `bayesprobe/openai_gateway.py`:

```python
PROBE_DESIGN_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["proposals"],
    "properties": {
        "proposals": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "purpose", "target_hypotheses", "inquiry_goal",
                    "expected_observation", "support_condition",
                    "weaken_condition", "reframe_condition",
                    "required_capability",
                ],
                "properties": {
                    "purpose": {"type": "string", "enum": [item.value for item in ProbePurpose]},
                    "target_hypotheses": {"type": "array", "minItems": 1, "items": {"type": "string"}},
                    "inquiry_goal": {"type": "string", "minLength": 1},
                    "expected_observation": {"type": "string", "minLength": 1},
                    "support_condition": {"type": "object", "additionalProperties": {"type": "string"}},
                    "weaken_condition": {"type": "object", "additionalProperties": {"type": "string"}},
                    "reframe_condition": {
                        "anyOf": [
                            {"type": "object", "additionalProperties": {"type": "string"}},
                            {"type": "null"},
                        ]
                    },
                    "required_capability": {"type": "string", "enum": [item.value for item in CapabilityKind]},
                },
            },
        }
    },
}
```

Route `design_probes` and `repair_probe_design` through `_instruction_for_task`, `_structured_output_for_task`, `_chat_instruction_for_task`, and `_required_output_for_task`. Tests must prove both Responses and Chat Completions payloads use this exact schema and forbid model-owned `id`, `priority`, `prior`, and `posterior` fields.

- [ ] **Step 8: Run focused tests and commit**

Run:

```bash
pytest tests/test_schemas.py tests/test_probe_design.py tests/test_openai_gateway.py -q
git diff --check
```

Expected: all focused tests pass and no whitespace errors exist.

Commit:

```bash
git add bayesprobe/schemas.py bayesprobe/probe_design.py bayesprobe/openai_gateway.py \
  bayesprobe/__init__.py tests/test_schemas.py tests/test_probe_design.py \
  tests/test_openai_gateway.py
git commit -m "feat: add semantic open-question probe design"
```

---

### Task 2: Wire Probe Design and the Complete Candidate Feedback Path

**Files:**
- Modify: `bayesprobe/initialization.py`
- Modify: `bayesprobe/question_runner.py`
- Modify: `tests/test_initialization.py`
- Modify: `tests/test_question_runner.py`

**Interfaces:**
- Consumes: `ProbeDesigner`, `ProbeDesignContext`, `ProbeDesignResult`, and `CycleResult.probe_candidates` from Task 1 and the existing Core.
- Produces: progress events `probe_design_started`, `probe_design_completed`, and a deterministic `_next_candidate_pool` that includes every candidate source.

- [ ] **Step 1: Write failing initializer and runner tests**

Add these assertions:

```python
def test_open_initializer_does_not_emit_generic_per_hypothesis_probes(open_initializer):
    result = open_initializer.initialize(open_input())
    assert result.task_frame.answer_relationship == AnswerRelationship.SYNTHESIS
    assert result.probe_candidates == []


def test_runner_designs_open_probe_after_belief_initialization(open_runner_fixture):
    result, gateway, progress = open_runner_fixture.run(max_cycles=1)
    tasks = [request.task for request in gateway.requests]
    assert tasks.index("frame_open_question") < tasks.index("design_probes")
    assert tasks.index("design_probes") < tasks.index("execute_probe")
    kinds = [event.kind for event in progress]
    assert AutonomousQuestionProgressKind.PROBE_DESIGN_STARTED in kinds
    assert AutonomousQuestionProgressKind.PROBE_DESIGN_COMPLETED in kinds
    assert len(result.cycle_results[0].probe_set.probes) == 1


def test_next_pool_keeps_core_candidates_before_fresh_and_remaining(runner, candidates):
    pool = runner._next_candidate_pool(
        previous_pool=[candidates.remaining],
        selected_candidates=[candidates.selected],
        core_candidates=[candidates.core],
        designed_candidates=[candidates.designed],
        answer_projection=candidates.projection,
    )
    assert [item.candidate_id for item in pool] == [
        candidates.core.candidate_id,
        candidates.designed.candidate_id,
        candidates.projection_candidate.candidate_id,
        candidates.remaining.candidate_id,
    ]
```

Also prove semantic duplicates keep the earliest item in this order and MCQ initialization still returns its answer-choice discriminator.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
pytest tests/test_initialization.py::test_open_initializer_does_not_emit_generic_per_hypothesis_probes \
  tests/test_question_runner.py -q
```

Expected: FAIL because open initialization still emits generic probes and the runner has no designer dependency.

- [ ] **Step 3: Remove generic open-task initialization probes**

Change `_initial_probe_candidates` so only explicit MCQ receives an initializer-owned candidate:

```python
def _initial_probe_candidates(*, run_id, problem, hypotheses, is_multiple_choice):
    if not is_multiple_choice:
        return []
    return [
        _answer_choice_discriminator_candidate(
            run_id=run_id,
            problem=problem,
            hypotheses=hypotheses,
        )
    ]
```

Delete the now-unused `_probe_candidate` helper. Do not alter TaskFrame or BeliefState construction.

- [ ] **Step 4: Add runner designer dependencies and progress payloads**

Extend `AutonomousQuestionRunner.__init__`:

```python
def __init__(
    self,
    *,
    core: BayesProbeCore,
    initializer: BayesProbeInitializer | None = None,
    planner: ProbePlanner | None = None,
    executor: ProbeExecutor | None = None,
    config: AutonomousQuestionRunConfig | None = None,
    progress_observer: AutonomousQuestionProgressObserver | None = None,
    task_admitter: TaskAdmitter | None = None,
    probe_designer: ProbeDesigner | None = None,
    available_capabilities: tuple[CapabilityDescriptor, ...] = (),
) -> None:
    self.core = core
    self.initializer = initializer or BayesProbeInitializer(ledger=core.ledger)
    self.planner = planner or ProbePlanner(ledger=core.ledger)
    self.executor = executor or ProbeExecutor(
        gateway=DeterministicProbeToolGateway(),
        ledger=core.ledger,
    )
    self.config = config or AutonomousQuestionRunConfig()
    self.progress_observer = progress_observer
    self.task_admitter = task_admitter or ExplicitTaskAdmitter()
    self.probe_designer = probe_designer or FrameProbeDesigner()
    self.available_capabilities = tuple(available_capabilities)
```

Add `probe_candidates` and `capability_decisions` tuples to `AutonomousQuestionProgress`. After `INITIALIZATION_COMPLETED`, use initializer candidates for MCQ; otherwise emit `PROBE_DESIGN_STARTED`, call the designer with `cycle_id="cycle_0"`, emit `PROBE_DESIGN_COMPLETED`, and use only the returned executable candidates.

Pass the same capability descriptors into task admission:

```python
admission = validate_task_admission_decision(
    self.task_admitter.assess(
        _task_admission_input(
            input,
            available_capabilities=self.available_capabilities,
        )
    )
)
```

Update `_task_admission_input` to copy these descriptors into
`TaskAdmissionInput.available_capabilities`. This makes admission, design, and
execution agree about the WebUI's one real capability.

- [ ] **Step 5: Merge all next-cycle candidate sources**

Replace `_next_candidate_pool` with:

```python
def _next_candidate_pool(
    self,
    *,
    previous_pool,
    selected_candidates,
    core_candidates,
    designed_candidates,
    answer_projection,
):
    selected_ids = {item.candidate_id for item in selected_candidates}
    remaining = [item for item in previous_pool if item.candidate_id not in selected_ids]
    ordered = [
        *core_candidates,
        *designed_candidates,
        *answer_projection.change_my_mind_condition.structured_probe_candidates,
        *remaining,
    ]
    return _deduplicate_probe_candidates(ordered)
```

The semantic identity is the tuple `(purpose, sorted targets, required_capability, normalized inquiry_goal)`. Call the designer after each integrated cycle using the updated BeliefState, then pass `core_result.probe_candidates` and the fresh design into this method.

- [ ] **Step 6: Run focused tests and commit**

Run:

```bash
pytest tests/test_initialization.py tests/test_question_runner.py -q
git diff --check
```

Commit:

```bash
git add bayesprobe/initialization.py bayesprobe/question_runner.py \
  tests/test_initialization.py tests/test_question_runner.py
git commit -m "feat: route designed probes through autonomous cycles"
```

---

### Task 3: Implement Bounded Semantic Hypothesis Expansion

**Files:**
- Create: `bayesprobe/hypothesis_expansion.py`
- Modify: `bayesprobe/openai_gateway.py`
- Modify: `bayesprobe/__init__.py`
- Create: `tests/test_hypothesis_expansion_service.py`
- Modify: `tests/test_openai_gateway.py`

**Interfaces:**
- Consumes: `ExpansionPolicy`, `OpenCoveragePolicy`, `FrameAdequacyDecision`, `TaskFrame`, `FrameState`, `Hypothesis`, `EvidenceEvent`, and `ModelGateway`.
- Produces: `HypothesisExpansionError`, `HypothesisExpansionRequest`, `HypothesisExpansionProposal`, `HypothesisExpansionResult`, `HypothesisExpansionAdapter.propose(request)`, `ModelHypothesisExpansionAdapter`, and `HypothesisExpansionService.expand(request, decision)`.

- [ ] **Step 1: Write failing exact/open and independent/open expansion tests**

Create `tests/test_hypothesis_expansion_service.py`:

```python
def test_exact_expansion_transfers_half_unresolved_mass(exact_open_state, expansion_decision):
    exact_open_state = with_unresolved_mass(exact_open_state, 0.60)
    adapter = StaticExpansionAdapter(
        [
            proposal(answer_value=4, statement="The supported tendon count is four."),
            proposal(answer_value=5, statement="The supported tendon count is five."),
        ]
    )
    service = HypothesisExpansionService(adapter=adapter)

    result = service.expand(
        request=expansion_request(exact_open_state),
        decision=expansion_decision,
    )

    new_items = [item for item in result.hypotheses if item.created_by == "spawned"]
    assert [item.answer_value for item in new_items] == [4, 5]
    assert [item.posterior for item in new_items] == [0.15, 0.15]
    assert result.frame_state.unresolved_alternative_mass == 0.30
    assert sum(item.posterior for item in result.hypotheses if item.status == HypothesisStatus.ACTIVE) + 0.30 == pytest.approx(1.0)
    assert result.frame_state.frame_version == 2
    assert result.frame_state.adequacy_status == FrameAdequacyStatus.PROVISIONAL
    assert result.probe_candidates[0].candidate_probe.target_hypotheses[-2:] == [
        new_items[0].id,
        new_items[1].id,
    ]


def test_independent_expansion_adds_claim_without_cross_normalization(independent_state, expansion_decision):
    service = HypothesisExpansionService(
        adapter=StaticExpansionAdapter([proposal(answer_value=None, statement="Task difficulty moderates the scale effect.")])
    )
    result = service.expand(
        request=expansion_request(independent_state),
        decision=expansion_decision,
    )

    added = result.hypotheses[-1]
    assert added.prior == 0.5
    assert added.posterior == 0.5
    assert result.frame_state.unresolved_alternative_mass is None
```

Also test: expansion forbidden when `should_expand` is false; one-to-three proposal limit; duplicate active or historical statements; typed integer/number/short-text validation; max three revisions; max eight active hypotheses; minimum unresolved reserve; server-owned ids; and no BeliefUpdate for a newly created hypothesis.

- [ ] **Step 2: Run and verify RED**

Run: `pytest tests/test_hypothesis_expansion_service.py -q`

Expected: collection fails because the expansion module does not exist.

- [ ] **Step 3: Implement proposal and service contracts**

Create the public dataclasses/protocol and a strict proposal model:

```python
class HypothesisExpansionError(ValueError):
    pass


class HypothesisExpansionProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")
    statement: str
    type: str
    scope: str
    falsifiers: list[str]
    predictions: list[str]
    answer_value: str | int | float | None
    why_current_frame_missed: str
    required_next_probe: str


@dataclass(frozen=True)
class HypothesisExpansionRequest:
    run_id: str
    cycle_id: str
    task_frame: TaskFrame
    frame_state: FrameState
    hypotheses: tuple[Hypothesis, ...]
    triggering_events: tuple[EvidenceEvent, ...]
    expansion_reason: str


@dataclass(frozen=True)
class HypothesisExpansionResult:
    hypotheses: list[Hypothesis]
    frame_state: FrameState
    evolutions: list[HypothesisEvolution]
    probe_candidates: list[ProbeCandidate]
    frame_mass_updates: list[FrameMassUpdate]
    discovery_evidence_ids: list[str]


class HypothesisExpansionAdapter(Protocol):
    def propose(
        self,
        request: HypothesisExpansionRequest,
    ) -> list[HypothesisExpansionProposal]: ...
```

`HypothesisExpansionService.expand` must reject a false `should_expand`, validate proposals, assign ids `H_exp_f{next_frame_version}_{index}`, and enforce `ExpansionPolicy`.

- [ ] **Step 4: Implement deterministic mass and frame transitions**

For `exclusive + open`, use:

```python
available = max(
    current_unresolved - open_policy.minimum_unresolved_reserve,
    0.0,
)
transfer = min(current_unresolved * 0.5, available)
per_candidate = transfer / len(proposals)
next_unresolved = current_unresolved - transfer
```

For `independent + open`, each new claim starts at credence `0.5` and does not change existing credences. In both cases set `frame_version += 1`, `parent_frame_version` to the old version, `revision_count += 1`, and `adequacy_status=PROVISIONAL`.

Create one `EvolutionOperation.SPAWN` audit per new hypothesis and one mandatory follow-up `ProbeCandidate` targeting the new hypotheses plus current active rivals. Use the triggering Evidence ids as discovery ids, but do not create same-cycle BeliefUpdates for the new hypotheses.

- [ ] **Step 5: Implement the model adapter and one repair**

`ModelHypothesisExpansionAdapter.propose` sends:

```python
StructuredModelRequest(
    task="expand_hypotheses",
    input={
        "task_frame": request.task_frame.model_dump(mode="json"),
        "frame_state": request.frame_state.model_dump(mode="json"),
        "hypotheses": [item.model_dump(mode="json") for item in request.hypotheses],
        "triggering_evidence": [_safe_event_summary(item) for item in request.triggering_events],
        "expansion_reason": request.expansion_reason,
        "answer_value_type": request.task_frame.answer_contract.answer_value_type.value,
        "proposal_count": {"minimum": 1, "maximum": 3},
    },
    prompt_id="hypothesis_expansion",
    prompt_version="v0.2",
    schema_name="HypothesisExpansion",
    schema_version="v0.2",
    metadata={"run_id": request.run_id, "cycle_id": request.cycle_id},
)
```

One malformed response triggers `repair_hypothesis_expansion`; a second invalid response raises `HypothesisExpansionError("hypothesis expansion invalid after 1 repair attempt")`.

- [ ] **Step 6: Add and test the provider schema**

Add `HYPOTHESIS_EXPANSION_JSON_SCHEMA` with one required top-level `candidates` array, one to three items, and exactly the eight semantic fields in `HypothesisExpansionProposal`. Route `expand_hypotheses` and `repair_hypothesis_expansion` through all four OpenAI task-routing helpers. Assert that both provider protocols forbid `id`, `prior`, `posterior`, `frame_status`, and `unresolved_mass` in model output.

- [ ] **Step 7: Run focused tests and commit**

Run:

```bash
pytest tests/test_hypothesis_expansion_service.py tests/test_openai_gateway.py -q
git diff --check
```

Commit:

```bash
git add bayesprobe/hypothesis_expansion.py bayesprobe/openai_gateway.py \
  bayesprobe/__init__.py tests/test_hypothesis_expansion_service.py \
  tests/test_openai_gateway.py
git commit -m "feat: add bounded semantic hypothesis expansion"
```

---

### Task 4: Integrate Expansion Atomically Inside the Core

**Files:**
- Modify: `bayesprobe/core.py`
- Modify: `tests/test_core_cycles.py`

**Interfaces:**
- Consumes: `HypothesisExpansionService` and `HypothesisExpansionResult` from Task 3.
- Produces: a `CycleResult` whose BeliefState, FrameState, Evidence Memory, evolutions, frame-mass updates, and probe candidates describe one atomic expanded cycle.

- [ ] **Step 1: Write a failing Core expansion test**

Add a focused test with an exclusive/open state, accepted Evidence that disconfirms `1/2/3` and supports unresolved alternatives, and a static expansion adapter returning `4/5`:

```python
def test_core_authorizes_and_commits_exact_answer_expansion_atomically(tmp_path):
    state = exact_open_state(named={"H1": 0.15, "H2": 0.15, "H3": 0.15}, unresolved=0.55)
    event = unresolved_support_event(
        likelihoods={
            "H1": LikelihoodBand.STRONGLY_DISCONFIRMING,
            "H2": LikelihoodBand.STRONGLY_DISCONFIRMING,
            "H3": LikelihoodBand.STRONGLY_DISCONFIRMING,
        },
        unresolved_likelihood=LikelihoodBand.STRONGLY_CONFIRMING,
    )
    core = core_with_event_and_expander(tmp_path, event, proposals_for_4_and_5())

    result = core.integrate_cycle(cycle(), state, probe_set(), [signal()])

    assert result.frame_adequacy_decision.should_expand is True
    assert {item.answer_value for item in result.belief_state.hypotheses} >= {4, 5}
    assert result.belief_state.frame_state.frame_version == 2
    assert event.id in result.belief_state.evidence_memory.discovery_evidence_ids
    assert all(update.hypothesis_id not in {"H_exp_f2_1", "H_exp_f2_2"} for update in result.belief_updates)
    assert result.probe_candidates[0].candidate_probe.purpose == ProbePurpose.HYPOTHESIS_DISCRIMINATION
```

Add a failure test proving an invalid expansion leaves the prior BeliefState unchanged and writes no integrated cycle or expanded BeliefState ledger record.

- [ ] **Step 2: Run and verify RED**

Run:

```bash
pytest tests/test_core_cycles.py::test_core_authorizes_and_commits_exact_answer_expansion_atomically -q
```

Expected: FAIL because Core records `should_expand` but never invokes an expansion service.

- [ ] **Step 3: Inject the expansion service**

Extend the constructor without changing defaults for existing callers:

```python
def __init__(
    self,
    ledger=None,
    model_gateway=None,
    judgment_repair_policy=None,
    correlation_credit_policy=None,
    hypothesis_expander: HypothesisExpansionService | None = None,
) -> None:
    self._ledger = ledger
    self._model_gateway = model_gateway
    self._judgment_repair_policy = judgment_repair_policy
    self._cycle_allocations: dict[str, int] = {}
    configured_credit_policy = (
        correlation_credit_policy or CorrelationCreditPolicy()
    )
    self._correlation_credit_policy = _copy_correlation_credit_policy(
        configured_credit_policy
    )
    self._evidence_memory_manager = EvidenceMemoryManager(
        _copy_correlation_credit_policy(self._correlation_credit_policy)
    )
    self._evidence_gate = self._create_evidence_integration_gate()
    self._belief_solver = self._create_belief_solver()
    self._frame_policy = self._create_frame_adequacy_policy()
    self._evolution_policy = self._create_hypothesis_evolution_policy()
    self._hypothesis_expander = hypothesis_expander
```

After `FrameAdequacyPolicy.assess`, call the service only when both `should_expand` and `_hypothesis_expander is not None` are true. Pass only accepted triggering events whose ids appear in `trigger_event_ids`.

- [ ] **Step 4: Merge expansion into the atomic cycle payload**

On successful expansion:

```python
evolved_hypotheses = expansion.hypotheses
next_frame_state = expansion.frame_state
evolutions = [*evolutions, *expansion.evolutions]
probe_candidates = [*probe_candidates, *expansion.probe_candidates]
frame_mass_updates = [*frame_mass_updates, *expansion.frame_mass_updates]
next_evidence_memory = next_evidence_memory.model_copy(
    update={
        "discovery_evidence_ids": _append_unique(
            next_evidence_memory.discovery_evidence_ids,
            expansion.discovery_evidence_ids,
        )
    }
)
```

Do not overwrite this expanded `next_frame_state` with the pre-expansion decision state later in the method. Existing ledger append code must receive the merged evolutions, probe candidates, frame-mass updates, memory, and final BeliefState in one commit path.

- [ ] **Step 5: Run Core and frame-policy regression tests**

Run:

```bash
pytest tests/test_core_cycles.py tests/test_frame_policy.py \
  tests/test_hypothesis_expansion_service.py -q
git diff --check
```

Expected: all tests pass; named active posterior plus unresolved mass remains exactly one for every exclusive/open cycle.

- [ ] **Step 6: Commit**

```bash
git add bayesprobe/core.py tests/test_core_cycles.py
git commit -m "feat: integrate authorized frame expansion in core"
```

---

### Task 5: Replace Top-H Text with Task-Aware Projection

**Files:**
- Modify: `bayesprobe/schemas.py`
- Modify: `bayesprobe/projections.py`
- Modify: `bayesprobe/openai_gateway.py`
- Modify: `bayesprobe/__init__.py`
- Create: `tests/test_answer_projection.py`
- Modify: `tests/test_openai_gateway.py`

**Interfaces:**
- Consumes: `TaskFrame.answer_relationship`, `AnswerContract`, final `BeliefState`, `CycleResult`, admitted Evidence summaries, and `ModelGateway`.
- Produces: `AnswerProjectionError`, `AnswerProjectionInput`, `AnswerProjector.project(input)`, `TaskAwareAnswerProjector`, explicit `ProjectionMode`, typed `answer_value`, and `contract_sections`.

- [ ] **Step 1: Write failing selection, synthesis, and abstention tests**

Create `tests/test_answer_projection.py`:

```python
def test_exact_projection_returns_typed_value_from_expanded_hypothesis(exact_cycle_result):
    projection = TaskAwareAnswerProjector().project(projection_input(exact_cycle_result))
    assert projection.mode == ProjectionMode.SELECTION
    assert projection.answer_value == 4
    assert projection.answer == "4"
    assert projection.current_best_hypothesis == "H_exp_f2_1"


def test_exact_projection_abstains_while_unresolved_outranks_named(exact_cycle_result):
    state = with_mass(exact_cycle_result.belief_state, top=0.30, unresolved=0.40)
    projection = TaskAwareAnswerProjector().project(
        projection_input(exact_cycle_result, belief_state=state)
    )
    assert projection.mode == ProjectionMode.ABSTENTION
    assert projection.answer_value is None
    assert "unresolved" in projection.main_uncertainty.lower()


def test_synthesis_projection_satisfies_every_required_section(synthesis_cycle_result):
    gateway = ScriptedModelGateway(
        {
            "project_answer": {
                "answer": "Use a preregistered matched-budget factorial evaluation.",
                "contract_sections": {
                    "hypotheses": "Test scale, budget confounding, and task interaction claims.",
                    "controls": "Hold scaffolding, task set, sampling, and inference budget fixed.",
                    "decision_rule": "Accept a scale effect only when the preregistered effect exceeds the practical threshold.",
                },
                "main_uncertainty": "Deployment distributions may differ.",
                "weakest_assumption": "The frozen task set represents deployment.",
                "cited_evidence_ids": ["E_cycle_1"],
            }
        }
    )
    projection = TaskAwareAnswerProjector(gateway).project(
        projection_input(synthesis_cycle_result)
    )
    assert projection.mode == ProjectionMode.SYNTHESIS
    assert set(projection.contract_sections) == {"hypotheses", "controls", "decision_rule"}
    assert not projection.answer.startswith("Current best hypothesis")


def test_synthesis_projection_abstains_while_expansion_is_pending(synthesis_cycle_result):
    state = synthesis_cycle_result.belief_state.model_copy(
        update={
            "frame_state": synthesis_cycle_result.belief_state.frame_state.model_copy(
                update={"adequacy_status": FrameAdequacyStatus.EXPANDING}
            )
        }
    )
    projection = TaskAwareAnswerProjector().project(
        projection_input(synthesis_cycle_result, belief_state=state)
    )
    assert projection.mode == ProjectionMode.ABSTENTION
    assert "expansion" in projection.main_uncertainty.lower()
```

Also test one repair, unknown Evidence ids, missing required sections, model attempts to set posterior values, and no structured generic `source_tracing` candidate in the change-my-mind condition.

- [ ] **Step 2: Run and verify RED**

Run: `pytest tests/test_answer_projection.py -q`

Expected: collection fails because the task-aware projector and fields do not exist.

- [ ] **Step 3: Extend `AnswerProjection` compatibly**

Update `bayesprobe/schemas.py`:

```python
class AnswerProjection(BaseModel):
    mode: ProjectionMode = ProjectionMode.SELECTION
    answer: str
    answer_value: str | int | float | None = None
    contract_sections: dict[str, str] = Field(default_factory=dict)
    current_best_hypothesis: str | None = None
    posterior_summary: str
    main_uncertainty: str
    weakest_assumption: str
    main_evidence_events: list[str]
    change_my_mind_condition: ChangeMyMindCondition
    answer_utility_notes: str = ""
```

Validate nonblank answer/section text and forbid `answer_value` for abstention.
The projector, which has the TaskFrame, enforces the stronger rule that an
exact scalar or choice-label selection requires a contract-compatible value.
This keeps legacy `AnswerProjection` records readable.

- [ ] **Step 4: Implement `TaskAwareAnswerProjector`**

Use this protocol:

```python
class AnswerProjectionError(ValueError):
    pass


@dataclass(frozen=True)
class AnswerProjectionInput:
    cycle_id: str
    previous_belief_state: BeliefState
    cycle_result: CycleResult
    stop_reason: str | None = None


class AnswerProjector(Protocol):
    def project(self, input: AnswerProjectionInput) -> AnswerProjection: ...
```

Selection is deterministic: rank active hypotheses, require a contract-compatible `answer_value`, reject `INADEQUATE`/`EXPANDING` frame states, and abstain whenever unresolved mass exceeds the top named posterior. Synthesis also abstains for `INADEQUATE`/`EXPANDING` states; otherwise it sends only TaskFrame, hypothesis/belief summaries, sanitized admitted Evidence summaries, and stop reason to the model. Validate that `cited_evidence_ids` is a subset of admitted ids and that `contract_sections` exactly covers `required_sections`.

Gateway failure raises only
`AnswerProjectionError("answer projection model gateway call failed")`; invalid
output after the one repair raises
`AnswerProjectionError("answer projection invalid after 1 repair attempt")`.

Keep `build_answer_projection` as a compatibility wrapper for the synchronized and legacy paths; it may call a deterministic projector, but it must not be used by the new provider-backed autonomous WebUI path.

- [ ] **Step 5: Add the strict projection provider schema**

Add `ANSWER_PROJECTION_JSON_SCHEMA` with exactly:

```text
answer
contract_sections
main_uncertainty
weakest_assumption
cited_evidence_ids
```

Route `project_answer` and `repair_answer_projection` through all OpenAI task helpers. The schema must not contain `mode`, `answer_value`, `current_best_hypothesis`, `posterior`, or `belief_updates`; those remain server-owned.

- [ ] **Step 6: Run focused tests and commit**

Run:

```bash
pytest tests/test_answer_projection.py tests/test_openai_gateway.py \
  tests/test_question_runner.py tests/test_synchronized_runner.py -q
git diff --check
```

Commit:

```bash
git add bayesprobe/schemas.py bayesprobe/projections.py bayesprobe/openai_gateway.py \
  bayesprobe/__init__.py tests/test_answer_projection.py \
  tests/test_openai_gateway.py tests/test_question_runner.py \
  tests/test_synchronized_runner.py
git commit -m "feat: project answers through task contracts"
```

---

### Task 6: Wire the Full Provider Path and Dynamic WebUI Trace

**Files:**
- Modify: `bayesprobe/question_runner.py`
- Modify: `bayesprobe/webui.py`
- Modify: `bayesprobe/webui_static/app.js`
- Modify: `tests/test_question_runner.py`
- Modify: `tests/test_webui.py`
- Modify: `tests/test_webui_stream.js`

**Interfaces:**
- Consumes: the designer from Task 1, Core expansion from Task 4, and projector from Task 5.
- Produces: one request-scoped WebUI runtime whose gateway is shared across admission, framing, probe design, execution, Evidence judgment, expansion, and synthesis projection.
- Produces progress kinds: `probe_design_started`, `probe_design_completed`, `frame_adequacy_assessed`, `hypothesis_expansion_completed`, `answer_projection_started`, and `answer_projection_completed`.

- [ ] **Step 1: Write failing runner and WebUI wiring tests**

Add a provider spy test:

```python
def test_webui_open_question_uses_one_gateway_for_every_semantic_stage():
    status, payload = handle_autonomous_run_request(
        open_provider_payload(max_cycles=1),
        client_factory=OpenQuestionMVPClient,
    )

    assert status == 200
    assert OpenQuestionMVPClient.gateway_count == 1
    assert OpenQuestionMVPClient.tasks == [
        "assess_task_admission",
        "frame_open_question",
        "design_probes",
        "execute_probe",
        "judge_evidence",
        "project_answer",
    ]
    assert payload["final_answer"]["mode"] == "synthesis"
```

Add stream assertions that every new stage appears in order and that `hypothesis_expansion_completed` includes new hypotheses and mandatory follow-up probes. Add JavaScript tests that synthesis sections render before belief metadata and unresolved mass appears for `exclusive + open` frames.

- [ ] **Step 2: Run and verify RED**

Run:

```bash
pytest tests/test_question_runner.py tests/test_webui.py -q
node --test tests/test_webui_stream.js
```

Expected: FAIL because the WebUI does not wire the new components or recognize their progress.

- [ ] **Step 3: Inject the task-aware projector into the runner**

Add `answer_projector: AnswerProjector | None = None` to the runner constructor. Replace the direct `build_answer_projection` call with:

```python
answer_projection = self.answer_projector.project(
    AnswerProjectionInput(
        cycle_id=cycle_id,
        previous_belief_state=previous_belief_state,
        cycle_result=core_result,
        stop_reason=self._prospective_stop_reason(
            previous=previous_belief_state,
            current=core_result.belief_state,
            completed_cycle_count=len(cycle_results) + 1,
        ),
    )
)
```

Emit projection start/completion around the call. Emit frame adequacy after Core integration; when Core evolutions include newly spawned semantic hypotheses, emit expansion completion before projection starts.

Implement the helper exactly once in the runner:

```python
def _prospective_stop_reason(
    self,
    *,
    previous: BeliefState,
    current: BeliefState,
    completed_cycle_count: int,
) -> str | None:
    if completed_cycle_count >= self.config.max_cycles:
        return AutonomousQuestionStopReason.MAX_CYCLES.value
    if self._confidence_reached(current):
        return AutonomousQuestionStopReason.CONFIDENCE_REACHED.value
    if self._posterior_stable(previous=previous, current=current):
        return AutonomousQuestionStopReason.POSTERIOR_STABLE.value
    return None
```

`NO_PROBES` remains a post-projection stop because projection change-my-mind
candidates participate in the completed next pool.

- [ ] **Step 4: Wire one gateway through the WebUI provider runtime**

Inside `_prepare_autonomous_run`, construct:

```python
capability = MODEL_REASONING_CAPABILITY.model_copy(
    update={"executor_adapter_id": model_gateway_adapter_kind(gateway)}
)
expander = HypothesisExpansionService(
    adapter=ModelHypothesisExpansionAdapter(gateway)
)
core = BayesProbeCore(
    model_gateway=gateway,
    hypothesis_expander=expander,
)
runner = AutonomousQuestionRunner(
    core=core,
    initializer=BayesProbeInitializer(
        ledger=core.ledger,
        task_framer=task_framer,
        task_admitter=task_admitter,
    ),
    executor=ProbeExecutor(gateway=ModelBackedProbeToolGateway(gateway), ledger=core.ledger),
    task_admitter=task_admitter,
    probe_designer=ModelProbeDesigner(gateway),
    available_capabilities=(capability,),
    answer_projector=TaskAwareAnswerProjector(gateway),
    config=request["runner_config"],
    progress_observer=progress_observer,
)
```

Do not create another provider client or gateway at any stage.

Add safe stage-aware provider errors in `_provider_runtime_error_message`:

```python
for error_type, stage in (
    (ProbeDesignError, "probe design"),
    (HypothesisExpansionError, "hypothesis expansion"),
    (AnswerProjectionError, "answer projection"),
):
    if isinstance(current, error_type):
        return (
            f"provider request failed during {stage} for {provider_kind}. "
            "Check the provider configuration and retry."
        )
```

The message identifies the stage but does not include raw provider output,
request content, or credentials.

- [ ] **Step 5: Serialize and render explicit stage data**

Extend `_serialize_progress_data` so design events include candidates/capability decisions, adequacy includes the decision, expansion includes semantic evolutions and follow-up candidates, and projection completion includes the `AnswerProjection`.

In `app.js`, add progress labels and update `renderAnswer`:

```javascript
answerPanel.appendChild(kv("Projection", answer.mode));
answerPanel.appendChild(kv("Answer", answer.answer));
for (const [section, content] of Object.entries(answer.contract_sections || {})) {
  answerPanel.appendChild(kv(section, content));
}
answerPanel.appendChild(kv("Belief summary", answer.posterior_summary));
answerPanel.appendChild(kv("Main uncertainty", answer.main_uncertainty));
```

Determine belief semantics from `hypothesis_frame.competition` and `coverage`, not the legacy computed `relation` property. Display `unresolved_alternative_mass` whenever it is numeric.

- [ ] **Step 6: Run focused WebUI tests and commit**

Run:

```bash
pytest tests/test_question_runner.py tests/test_webui.py -q
node --test tests/test_webui_stream.js
git diff --check
```

Commit:

```bash
git add bayesprobe/question_runner.py bayesprobe/webui.py \
  bayesprobe/webui_static/app.js tests/test_question_runner.py \
  tests/test_webui.py tests/test_webui_stream.js
git commit -m "feat: expose open-question loop in webui"
```

---

### Task 7: Add Two Secret-Free Recorded Vertical Slices

**Files:**
- Modify: `bayesprobe/recorded_gateway.py`
- Modify: `tests/test_recorded_model_gateway.py`
- Create: `tests/fixtures/open_questions/model_scale_open_mvp_v0.1.json`
- Create: `tests/fixtures/open_questions/exact_answer_expansion_mvp_v0.1.json`
- Modify: `tests/test_question_runner.py`

**Interfaces:**
- Consumes: the complete autonomous runner path from Tasks 1-6.
- Produces: deterministic provider-shaped evidence that proves both MVP acceptance cases without a network or secret.

- [ ] **Step 1: Extend recorded matching with exact metadata keys**

Write failing tests showing two `execute_probe` responses can be distinguished by `cycle_id` and `probe_id`. Then extend `_matches_request`:

```python
for key in ("cycle_id", "probe_id"):
    expected = match.get(key)
    if expected is not None and expected != request.metadata.get(key):
        return False
```

Reject match objects containing keys outside `task`, `signal_id`, `cycle_id`, and `probe_id` so fixtures cannot silently rely on unsupported matching.

- [ ] **Step 2: Create the explanation/design fixture**

The fixture response sequence must contain:

```text
assess_task_admission -> admitted claim_verification / structured_text synthesis
frame_open_question -> independent + open causal, confounder, and boundary hypotheses
design_probes cycle_0 -> matched-budget factorial discrimination probe
execute_probe cycle_1 -> reasoning signal describing controls, metrics, and interaction checks
judge_evidence for the cycle_1 signal -> explained_by_named with no unresolved likelihood
project_answer cycle_1 -> hypotheses, controls, metrics, decision_rule, and limitations sections
```

Use `max_cycles=1`. The final response must have `mode="synthesis"`, all required sections, and no answer beginning with `Current best hypothesis`.

- [ ] **Step 3: Create the exact-answer expansion fixture**

The fixture response sequence must contain:

```text
assess_task_admission -> admitted exact_answer / integer selection
frame_open_question -> exclusive + open candidates 1, 2, 3; server framing assigns unresolved mass 0.50
design_probes cycle_0 -> frame-coverage discriminator
execute_probe cycle_1 -> signal that 1, 2, and 3 are inconsistent with the anatomical description
judge_evidence for cycle_1 -> all named strongly_disconfirming, unresolved strongly_confirming, supports_unresolved
expand_hypotheses cycle_1 -> candidates 4 and 5 with typed integer values
design_probes cycle_1 -> discriminator between the expanded candidates
execute_probe cycle_2 -> signal supporting 4 over 5
judge_evidence for cycle_2 -> 4 strongly_confirming, 5 strongly_disconfirming, unresolved strongly_disconfirming
```

Use `max_cycles=2`. The final response must have `mode="selection"`, `answer_value=4`, `answer="4"`, `frame_version=2`, and at least one expansion evolution. The cycle-1 discovery Evidence id must be in `EvidenceMemorySnapshot.discovery_evidence_ids` and must not create a same-cycle BeliefUpdate for either new candidate.

- [ ] **Step 4: Write the two end-to-end tests**

Add:

```python
@pytest.mark.parametrize(
    ("fixture_name", "max_cycles", "mode", "answer_value"),
    [
        ("model_scale_open_mvp_v0.1.json", 1, ProjectionMode.SYNTHESIS, None),
        ("exact_answer_expansion_mvp_v0.1.json", 2, ProjectionMode.SELECTION, 4),
    ],
)
def test_recorded_open_question_mvp_vertical_slice(
    fixture_name, max_cycles, mode, answer_value
):
    gateway = RecordedModelGateway.from_json(
        Path("tests/fixtures/open_questions") / fixture_name
    )
    runner, input = recorded_open_mvp_runtime(gateway, max_cycles=max_cycles)

    result = runner.run_question(input)

    assert result.final_answer_projection.mode == mode
    assert result.final_answer_projection.answer_value == answer_value
    assert len(result.cycle_results) == max_cycles
```

Add case-specific assertions from Steps 2 and 3 instead of relying only on this parameterized smoke assertion.

- [ ] **Step 5: Run recorded vertical slices and commit**

Run:

```bash
pytest tests/test_recorded_model_gateway.py tests/test_question_runner.py \
  -k "recorded or open_question_mvp" -q
git diff --check
```

Commit:

```bash
git add bayesprobe/recorded_gateway.py tests/test_recorded_model_gateway.py \
  tests/test_question_runner.py \
  tests/fixtures/open_questions/model_scale_open_mvp_v0.1.json \
  tests/fixtures/open_questions/exact_answer_expansion_mvp_v0.1.json
git commit -m "test: add recorded open-question vertical slices"
```

---

### Task 8: Freeze the MVP Boundary and Verify the User-Facing Run

**Files:**
- Modify: `docs/ARCHITECTURE.md`
- Modify only when a failing acceptance test requires it: files already listed in Tasks 1-7.

**Interfaces:**
- Consumes: all completed tasks.
- Produces: a documented, regression-tested autonomous open-question MVP and a running WebUI URL for manual validation.

- [ ] **Step 1: Update architecture status precisely**

Add an implementation-status section stating:

```text
Implemented: autonomous model-reasoning open-question vertical slice with
task-specific probe design, Core-authorized semantic expansion, task-aware
selection/synthesis/abstention, recorded explanation and exact-answer fixtures,
and dynamic WebUI progress.

Not implemented: external search/retrieval/tools, synchronized parity, coding
interventions, public benchmark execution, and probability calibration claims.
```

- [ ] **Step 2: Run focused acceptance tests**

Run:

```bash
pytest tests/test_probe_design.py tests/test_hypothesis_expansion_service.py \
  tests/test_answer_projection.py tests/test_question_runner.py \
  tests/test_webui.py -q
node --test tests/test_webui_stream.js
```

Expected: all tests pass.

- [ ] **Step 3: Run the complete regression suite**

Run:

```bash
pytest -q
node --test tests/test_webui_stream.js
git diff --check
```

Expected: all Python and Node tests pass, with only documented opt-in live tests skipped.

- [ ] **Step 4: Start the WebUI for real-provider validation**

Run:

```bash
python -m bayesprobe.webui --host 127.0.0.1 --port 8768
```

Expected console output:

```text
BayesProbe WebUI running at http://127.0.0.1:8768
```

Keep the server running. In the WebUI, the user supplies the OpenAI-compatible API key, base URL, and model. Submit the model-scale validation question with `max_cycles=2` and `max_probes_per_cycle=2`.

Manual acceptance requires:

- initial hypotheses are substantive and non-duplicative;
- a task-specific Probe appears before execution;
- Signal and Evidence remain distinct in the trace;
- belief values update after Evidence integration;
- the final answer is a validation protocol with the required sections;
- the answer is understandable without reading H ids; and
- the API key remains populated in the page for another run without appearing in any response or trace.

- [ ] **Step 5: Run the exact-answer manual scenario**

Submit an exact-answer question without answer choices. Manual acceptance requires an `exclusive + open` frame, visible unresolved mass, expansion when all initial candidates are rejected, a follow-up discriminator, and a typed final answer or explicit abstention when the real provider does not produce sufficient Evidence.

The recorded exact-answer fixture, not one nondeterministic live result, is the automated proof that expansion can reach a new correct candidate.

- [ ] **Step 6: Commit documentation and stop**

```bash
git add docs/ARCHITECTURE.md
git commit -m "docs: mark open-question MVP vertical slice complete"
git status --short
```

Expected: clean worktree. Do not begin synchronized parity, external tools, coding-agent work, benchmark runs, or additional hardening in this implementation cycle.
