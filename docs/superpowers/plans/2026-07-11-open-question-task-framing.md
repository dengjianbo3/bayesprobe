# Open-Question Framing and Relation-Aware Belief Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace BayesProbe's silent unseeded support/refute fallback with an explicit TaskFrame and preserve its exclusive-versus-independent semantics through belief revision.

**Architecture:** Add TaskFrame domain contracts, then make `BayesProbeInitializer` consume a `TaskFramer` instead of classifying raw question text itself. Explicit choices and seeds use a deterministic framer; unseeded open questions use the configured `ModelGateway`, with one structured repair attempt and a recorded adapter for offline tests. A relation-aware solver preserves categorical mass for exclusive frames and independent log-odds credences for coexistable hypotheses before the recorded vertical slice is allowed to run.

**Tech Stack:** Python 3.11+, Pydantic 2, pytest 8, existing synchronous `ModelGateway`, vanilla JavaScript WebUI tests.

## Global Constraints

- Do not add a WebUI-only hypothesis generation call.
- Do not create a Belief State before task framing succeeds.
- Do not silently convert an unseeded open question to support/refute H1/H2.
- Preserve deterministic initialization for explicit choices and explicit hypothesis seeds.
- Keep `context` as the compatibility initial passive signal; it is not Task Context.
- Provider keys remain request-scoped and must not enter TaskFrame, ledger records, fixtures, telemetry, or errors.
- Do not use the supplied one-time key or run live provider/HLE experiments in this plan.
- One structured framing repair is allowed; a second invalid response fails the run.
- This plan completes architecture-correction Milestones 1-2 only. Evidence Memory, ProbeDesigner, semantic evolution, and task-aware projection remain separate plans and must not be claimed complete here.

---

## File Map

- Create `bayesprobe/task_framing.py`: framing protocol, explicit/model/recorded adapters, legacy choice parser, validation, and repair policy.
- Modify `bayesprobe/schemas.py`: TaskFrame value objects and optional compatibility attachment on BeliefState.
- Modify `bayesprobe/initialization.py`: materialize only a completed TaskFrame and remove generic open fallback.
- Modify `bayesprobe/benchmark.py` and `bayesprobe/benchmark_io.py`: require explicit benchmark hypothesis seeds.
- Reuse `bayesprobe/model_gateway.py`: framing adapters consume the existing structured request protocol without widening its interface.
- Modify `bayesprobe/openai_gateway.py`: structured schemas and instructions for `frame_open_question` and `repair_task_frame`.
- Modify `bayesprobe/belief.py`: explicit categorical and independent solvers plus relation-aware summaries.
- Modify `bayesprobe/core.py`: explicit legacy-state migration before solver entry and relation propagation.
- Modify `bayesprobe/hypothesis_evolution.py`: relation-aware normalization calls.
- Modify `bayesprobe/projections.py`: relation-neutral belief terminology while task-aware synthesis remains deferred.
- Modify `bayesprobe/runners.py`: prevent categorical confidence stopping on independent frames.
- Modify `bayesprobe/question_runner.py`: framing progress events and TaskFrame result propagation.
- Modify `bayesprobe/probe_executor.py`: pass Task Context to model reasoning without duplicating the compatibility Initial Signal.
- Modify `bayesprobe/webui.py`: provider-backed framer wiring, progress serialization, and safe failure classification.
- Modify `bayesprobe/webui_static/index.html`: separate Task Context from the compatibility Initial Signal field.
- Modify `bayesprobe/webui_static/app.js`: framing progress labels; no Belief State rendering before initialization.
- Modify `bayesprobe/__init__.py`: public TaskFrame and framer exports.
- Create `tests/test_task_framing.py`: adapter and validation contracts.
- Modify `tests/test_schemas.py`, `tests/test_initialization.py`, `tests/test_belief.py`, `tests/test_core_cycles.py`, `tests/test_hypothesis_evolution.py`, `tests/test_controllers.py`, `tests/test_autonomous_runner.py`, `tests/test_question_runner.py`, `tests/test_probe_executor.py`, `tests/test_synchronized_runner.py`, `tests/test_benchmark_harness.py`, `tests/test_benchmark_io.py`, `tests/test_openai_gateway.py`, `tests/test_webui.py`, and `tests/test_webui_stream.js`: framing, relation, fixture, regression, and progress assertions.
- Create `tests/fixtures/open_questions/model_scale_validation_v0.1.json`: secret-free provider-shaped recorded fixture.
- Modify both `fixtures/benchmarks/*.json` datasets: replace implicit H1/H2 reliance with explicit semantic seeds.
- Modify `docs/ARCHITECTURE.md`: record Milestones 1-2 as implemented without upgrading later milestones.

---

### Task 1: Add TaskFrame Domain Contracts

**Files:**
- Modify: `bayesprobe/schemas.py`
- Modify: `tests/test_schemas.py`

**Interfaces:**
- Produces: `TaskKind`, `HypothesisRelation`, `FramingMethod`, `AnswerChoice`, `AnswerContract`, `FramedHypothesis`, `HypothesisFrame`, and `TaskFrame`.
- Produces: `BeliefState.task_frame: TaskFrame | None` as a compatibility field; all new initialization paths populate it.

- [ ] **Step 1: Write failing schema tests**

Add tests that construct a valid independent TaskFrame and reject duplicate hypothesis ids, empty required sections, unknown rivals, and secret-bearing framing traces:

```python
def _open_task_frame() -> TaskFrame:
    return TaskFrame(
        task_frame_id="run_frame_task_frame",
        task_kind=TaskKind.CLAIM_VERIFICATION,
        normalized_question="How should the model-scale claim be tested?",
        task_context="Evaluate on a frozen real-task distribution.",
        answer_contract=AnswerContract(
            objective="Design a discriminating validation protocol.",
            required_sections=["hypotheses", "controls", "decision_rule"],
            decision_form="experimental_protocol",
            permits_synthesis=True,
        ),
        hypothesis_frame=HypothesisFrame(
            frame_id="run_frame_hypothesis_frame",
            relation=HypothesisRelation.INDEPENDENT,
            hypotheses=[
                FramedHypothesis(
                    id="H1",
                    statement="Scale has an independent positive effect.",
                    type="causal_claim",
                    scope="Matched agent and compute conditions.",
                    initial_prior=0.5,
                    falsifiers=["The controlled effect is negligible."],
                    predictions=["Performance rises under matched controls."],
                ),
                FramedHypothesis(
                    id="H2",
                    statement="The apparent effect is caused by confounding.",
                    type="confounding_explanation",
                    scope="Unmatched published comparisons.",
                    initial_prior=0.5,
                    falsifiers=["The effect survives all matched controls."],
                    predictions=["The effect shrinks after matching resources."],
                ),
            ],
            rival_sets={"H1": [], "H2": []},
            coverage_statement="Tests the causal claim and its main confounder.",
            coverage_limitation="Other task-specific interactions may exist.",
        ),
        framing_method=FramingMethod.MODEL,
        framing_trace={"task": "frame_open_question", "schema_version": "v0.1"},
    )


def test_task_frame_accepts_independent_open_hypotheses():
    frame = _open_task_frame()
    assert frame.hypothesis_frame.relation == HypothesisRelation.INDEPENDENT
    assert [item.id for item in frame.hypothesis_frame.hypotheses] == ["H1", "H2"]


@pytest.mark.parametrize(
    "mutator, message",
    [
        (lambda frame: frame.model_copy(update={"answer_contract": frame.answer_contract.model_copy(update={"required_sections": []})}), "required_sections"),
        (lambda frame: frame.model_copy(update={"hypothesis_frame": frame.hypothesis_frame.model_copy(update={"hypotheses": [frame.hypothesis_frame.hypotheses[0], frame.hypothesis_frame.hypotheses[1].model_copy(update={"id": "H1"})]})}), "ids must be unique"),
        (lambda frame: frame.model_copy(update={"hypothesis_frame": frame.hypothesis_frame.model_copy(update={"rival_sets": {"H1": ["missing"], "H2": []}})}), "unknown rival"),
        (lambda frame: frame.model_copy(update={"framing_trace": {"api_key": "forbidden"}}), "secret"),
    ],
)
def test_task_frame_rejects_invalid_contract(mutator, message):
    with pytest.raises(ValueError, match=message):
        TaskFrame.model_validate(mutator(_open_task_frame()).model_dump())
```

- [ ] **Step 2: Run the schema tests and verify RED**

Run: `pytest tests/test_schemas.py -q`

Expected: collection fails because the TaskFrame types do not exist.

- [ ] **Step 3: Implement the domain models**

Add the enums and Pydantic models to `bayesprobe/schemas.py`. Use a model validator on `HypothesisFrame` to enforce 1-6 unique hypotheses and known rival ids, and a model validator on `TaskFrame` to require 2-6 hypotheses for non-legacy frames and reject secret-like keys recursively:

```python
import math
import re
from collections.abc import Mapping

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)


class TaskKind(StrEnum):
    MULTIPLE_CHOICE = "multiple_choice"
    CLAIM_VERIFICATION = "claim_verification"
    EXPLANATION = "explanation"
    DIAGNOSIS = "diagnosis"
    DESIGN = "design"
    DECISION = "decision"


class HypothesisRelation(StrEnum):
    EXCLUSIVE_EXHAUSTIVE = "exclusive_exhaustive"
    INDEPENDENT = "independent"


class FramingMethod(StrEnum):
    EXPLICIT = "explicit"
    MODEL = "model"
    RECORDED = "recorded"
    LEGACY_MIGRATION = "legacy_migration"


class StrictTaskModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AnswerChoice(StrictTaskModel):
    label: str
    text: str

    @field_validator("label")
    @classmethod
    def clean_label(cls, value: str) -> str:
        return _required_text(value, "answer choice label").upper()

    @field_validator("text")
    @classmethod
    def clean_text(cls, value: str) -> str:
        return _required_text(value, "answer choice text")


class AnswerContract(StrictTaskModel):
    objective: str
    required_sections: list[str]
    decision_form: str
    permits_synthesis: bool = False

    @field_validator("objective", "decision_form")
    @classmethod
    def clean_required_text(cls, value: str, info: ValidationInfo) -> str:
        return _required_text(value, info.field_name)

    @field_validator("required_sections")
    @classmethod
    def clean_required_sections(cls, value: list[str]) -> list[str]:
        sections = [_required_text(item, "required section") for item in value]
        if not sections or len(sections) != len(set(sections)):
            raise ValueError("required_sections must be non-empty and unique")
        return sections


class FramedHypothesis(StrictTaskModel):
    id: str
    statement: str
    type: str
    scope: str
    initial_prior: float
    falsifiers: list[str]
    predictions: list[str]

    @field_validator("id", "statement", "type", "scope")
    @classmethod
    def clean_required_text(cls, value: str, info: ValidationInfo) -> str:
        return _required_text(value, info.field_name)

    @field_validator("initial_prior")
    @classmethod
    def validate_initial_prior(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("initial_prior must be between zero and one")
        return value

    @field_validator("falsifiers", "predictions")
    @classmethod
    def clean_semantic_lists(cls, value: list[str], info: ValidationInfo) -> list[str]:
        items = [_required_text(item, info.field_name) for item in value]
        if not items:
            raise ValueError(f"{info.field_name} must not be empty")
        return items
```

Use these validation helpers and model-level invariants:

```python
def _required_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must not be empty")
    return value.strip()


def _normalized_semantic_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _reject_secret_material(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized_key = re.sub(r"[^a-z0-9]", "", str(key).casefold())
            if any(part in normalized_key for part in ("apikey", "authorization", "token", "secret")):
                raise ValueError("framing_trace must not contain secret fields")
            _reject_secret_material(item)
    elif isinstance(value, list):
        for item in value:
            _reject_secret_material(item)
    elif isinstance(value, str) and re.search(r"(?:^|\s)sk-[A-Za-z0-9_-]{12,}", value):
        raise ValueError("framing_trace must not contain secret values")


class HypothesisFrame(StrictTaskModel):
    frame_id: str
    relation: HypothesisRelation
    hypotheses: list[FramedHypothesis]
    rival_sets: dict[str, list[str]]
    coverage_statement: str
    unresolved_alternative_mass: float | None = None
    coverage_limitation: str | None = None

    @field_validator("frame_id", "coverage_statement")
    @classmethod
    def clean_required_text(cls, value: str, info: ValidationInfo) -> str:
        return _required_text(value, info.field_name)

    @field_validator("coverage_limitation")
    @classmethod
    def clean_optional_text(cls, value: str | None) -> str | None:
        return None if value is None else _required_text(value, "coverage_limitation")

    @field_validator("unresolved_alternative_mass")
    @classmethod
    def validate_alternative_mass(cls, value: float | None) -> float | None:
        if value is not None and not 0 <= value <= 1:
            raise ValueError("unresolved_alternative_mass must be between zero and one")
        return value

    @model_validator(mode="after")
    def validate_frame(self) -> "HypothesisFrame":
        if not 1 <= len(self.hypotheses) <= 6:
            raise ValueError("hypothesis frame must contain between 1 and 6 hypotheses")
        ids = [item.id for item in self.hypotheses]
        if len(ids) != len(set(ids)):
            raise ValueError("hypothesis ids must be unique")
        statements = [_normalized_semantic_text(item.statement) for item in self.hypotheses]
        if len(statements) != len(set(statements)):
            raise ValueError("hypothesis statements must be semantically distinct")
        if set(self.rival_sets) != set(ids):
            raise ValueError("rival_sets must contain every hypothesis id exactly once")
        for hypothesis_id, rivals in self.rival_sets.items():
            unknown = set(rivals).difference(ids)
            if unknown:
                raise ValueError(f"unknown rival ids for {hypothesis_id}: {sorted(unknown)}")
            if hypothesis_id in rivals:
                raise ValueError("a hypothesis cannot rival itself")
        if self.relation == HypothesisRelation.EXCLUSIVE_EXHAUSTIVE:
            for hypothesis_id, rivals in self.rival_sets.items():
                if set(rivals) != set(ids).difference({hypothesis_id}):
                    raise ValueError("exclusive frames require all-to-all rival sets")
            if not math.isclose(
                sum(item.initial_prior for item in self.hypotheses),
                1.0,
                abs_tol=1e-6,
            ):
                raise ValueError("exclusive frame initial priors must sum to one")
        return self


class TaskFrame(StrictTaskModel):
    task_frame_id: str
    task_kind: TaskKind
    normalized_question: str
    task_context: str = ""
    answer_contract: AnswerContract
    hypothesis_frame: HypothesisFrame
    framing_method: FramingMethod
    framing_trace: dict[str, Any] = Field(default_factory=dict)

    @field_validator("task_frame_id", "normalized_question")
    @classmethod
    def clean_required_text(cls, value: str, info: ValidationInfo) -> str:
        return _required_text(value, info.field_name)

    @field_validator("task_context")
    @classmethod
    def clean_task_context(cls, value: str) -> str:
        if not isinstance(value, str):
            raise ValueError("task_context must be a string")
        return value.strip()

    @model_validator(mode="after")
    def validate_frame(self) -> "TaskFrame":
        _reject_secret_material(self.framing_trace)
        if (
            self.framing_method != FramingMethod.LEGACY_MIGRATION
            and len(self.hypothesis_frame.hypotheses) < 2
        ):
            raise ValueError("new task frames require at least two hypotheses")
        if (
            self.task_kind == TaskKind.MULTIPLE_CHOICE
            and self.hypothesis_frame.relation
            != HypothesisRelation.EXCLUSIVE_EXHAUSTIVE
        ):
            raise ValueError("multiple-choice tasks require an exclusive frame")
        return self
```

`initial_prior` is a server-authored TaskFrame field; it is deliberately absent from provider output schemas.

Add this field to `BeliefState`:

```python
task_frame: TaskFrame | None = None
```

- [ ] **Step 4: Run focused and schema regression tests**

Run: `pytest tests/test_schemas.py tests/test_public_api_and_config.py -q`

Expected: all selected tests pass.

- [ ] **Step 5: Commit TaskFrame contracts**

```bash
git add bayesprobe/schemas.py tests/test_schemas.py
git commit -m "feat: add task framing domain contracts"
```

---

### Task 2: Replace Binary Fallback with ExplicitTaskFramer

**Files:**
- Create: `bayesprobe/task_framing.py`
- Modify: `bayesprobe/initialization.py`
- Modify: `bayesprobe/benchmark.py`
- Modify: `bayesprobe/benchmark_io.py`
- Modify: `bayesprobe/__init__.py`
- Modify: `fixtures/benchmarks/toy_belief_revision.json`
- Modify: `fixtures/benchmarks/bayesprobe_v0_2_methodology.json`
- Create: `tests/test_task_framing.py`
- Modify: `tests/test_initialization.py`
- Modify: `tests/test_benchmark_harness.py`
- Modify: `tests/test_benchmark_io.py`
- Modify: `tests/test_probe_executor.py`
- Modify: `tests/test_probe_planner.py`
- Modify: `tests/test_public_api_and_config.py`
- Modify: `tests/test_question_runner.py`
- Modify: `tests/test_synchronized_runner.py`
- Modify: `tests/test_webui.py`

**Interfaces:**
- Produces: `TaskFramingInput`, `TaskFramer.frame(input) -> TaskFrame`, `HypothesisSeed`, and `ExplicitTaskFramer`.
- Consumes: TaskFrame domain contracts from Task 1.
- Produces: `InitializeRunInput.answer_choices`, `task_context`, `task_kind`, and `hypothesis_relation`.
- Produces: `InitializationResult.task_frame` and a populated `BeliefState.task_frame`.

- [ ] **Step 1: Write failing explicit-framer tests**

Cover structured choices, English and Chinese legacy choice blocks, explicit seeds, and unseeded open failure:

```python
def test_explicit_framer_uses_structured_choices_without_text_parsing():
    frame = ExplicitTaskFramer().frame(
        TaskFramingInput(
            run_id="run_choices",
            question="Which result follows?",
            answer_choices=[
                AnswerChoice(label="A", text="First result"),
                AnswerChoice(label="B", text="Second result"),
            ],
        )
    )
    assert frame.task_kind == TaskKind.MULTIPLE_CHOICE
    assert frame.hypothesis_frame.relation == HypothesisRelation.EXCLUSIVE_EXHAUSTIVE
    assert [item.id for item in frame.hypothesis_frame.hypotheses] == ["A", "B"]


def test_explicit_framer_parses_chinese_legacy_choices():
    frame = ExplicitTaskFramer().frame(
        TaskFramingInput(
            run_id="run_cn_choices",
            question="哪一项正确？\n答案选项：\nA. 第一项\nB. 第二项",
        )
    )
    assert frame.task_kind == TaskKind.MULTIPLE_CHOICE


def test_explicit_framer_rejects_unseeded_open_question():
    with pytest.raises(TaskFramingError, match="requires a model or recorded task framer"):
        ExplicitTaskFramer().frame(
            TaskFramingInput(
                run_id="run_open",
                question="这个命题应该如何验证？",
            )
        )
```

Replace the old binary-fallback test in `tests/test_initialization.py` with:

```python
def test_initializer_never_creates_generic_binary_hypotheses_for_open_question():
    with pytest.raises(TaskFramingError):
        BayesProbeInitializer().initialize(
            InitializeRunInput(
                run_id="run_open",
                problem="某团队认为模型变大一定提升 agent 表现，应该如何验证？",
            )
        )
```

- [ ] **Step 2: Run explicit-framer tests and verify RED**

Run: `pytest tests/test_task_framing.py tests/test_initialization.py -q`

Expected: imports fail for `ExplicitTaskFramer`, then the old initializer behavior fails the no-fallback assertion.

- [ ] **Step 3: Implement TaskFramer and choice parsing**

Create `bayesprobe/task_framing.py` with these public contracts:

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
class TaskFramingInput:
    run_id: str
    question: str
    task_context: str = ""
    answer_choices: list[AnswerChoice] = field(default_factory=list)
    hypothesis_seeds: list[HypothesisSeed] = field(default_factory=list)
    task_kind: TaskKind | None = None
    hypothesis_relation: HypothesisRelation | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class TaskFramer(Protocol):
    def frame(self, input: TaskFramingInput) -> TaskFrame:
        raise NotImplementedError


class TaskFramingError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedAnswerChoiceFrame:
    stem: str
    choices: list[AnswerChoice]


class ExplicitTaskFramer:
    def frame(self, input: TaskFramingInput) -> TaskFrame:
        parsed = (
            None
            if input.answer_choices
            else parse_legacy_answer_choice_frame(input.question)
        )
        choices = list(input.answer_choices) if input.answer_choices else (
            list(parsed.choices) if parsed is not None else []
        )
        if choices and input.hypothesis_seeds:
            raise TaskFramingError("provide answer choices or hypothesis seeds, not both")
        if choices:
            normalized_question = parsed.stem if parsed is not None else input.question.strip()
            return _frame_choices(input, choices, normalized_question)
        if input.hypothesis_seeds:
            return _frame_seeds(input)
        raise TaskFramingError(
            "unseeded open question requires a model or recorded task framer"
        )
```

Implement parsing and server-side prior assignment with these helpers:

```python
_ANSWER_CHOICES_HEADER_RE = re.compile(
    r"(?:\banswer\s+choices?\s*:|答案选项\s*[：:])",
    re.IGNORECASE,
)
_CHOICE_BLOCK_RE = re.compile(
    r"^\s*([A-Z])\s*[\.\)]\s+(.*?)(?=^\s*[A-Z]\s*[\.\)]\s+|\Z)",
    re.MULTILINE | re.DOTALL,
)
_CHOICE_INLINE_RE = re.compile(
    r"(?:^|\s)([A-Z])\s*[\.\)]\s+(.*?)(?=\s+[A-Z]\s*[\.\)]\s+|\Z)",
    re.DOTALL,
)


def parse_legacy_answer_choice_frame(
    question: str,
) -> ParsedAnswerChoiceFrame | None:
    header = _ANSWER_CHOICES_HEADER_RE.search(question)
    if header is None:
        return None
    stem = " ".join(question[:header.start()].split())
    choice_text = question[header.end():].strip()
    matches = list(_CHOICE_BLOCK_RE.finditer(choice_text))
    if len(matches) < 2:
        matches = list(_CHOICE_INLINE_RE.finditer(choice_text))
    parsed = [
        AnswerChoice(label=match.group(1), text=" ".join(match.group(2).split()))
        for match in matches
    ]
    if (
        not stem
        or len(parsed) < 2
        or len({choice.label for choice in parsed}) != len(parsed)
    ):
        return None
    return ParsedAnswerChoiceFrame(stem=stem, choices=parsed)


def _initial_priors(
    seeds: list[HypothesisSeed],
    relation: HypothesisRelation,
) -> list[float]:
    supplied = [seed.prior is not None for seed in seeds]
    if any(supplied) and not all(supplied):
        raise TaskFramingError("seed priors must be supplied for every seed or none")
    if all(supplied):
        priors = [float(seed.prior) for seed in seeds if seed.prior is not None]
    elif relation == HypothesisRelation.EXCLUSIVE_EXHAUSTIVE:
        priors = [1.0 / len(seeds)] * len(seeds)
    else:
        priors = [0.5] * len(seeds)
    if relation == HypothesisRelation.EXCLUSIVE_EXHAUSTIVE and not math.isclose(
        sum(priors), 1.0, abs_tol=1e-6
    ):
        raise TaskFramingError("exclusive seed priors must sum to one")
    if any(prior < 0 or prior > 1 for prior in priors):
        raise TaskFramingError("seed priors must be between zero and one")
    return priors


def _rival_sets(
    ids: list[str],
    relation: HypothesisRelation,
) -> dict[str, list[str]]:
    if relation == HypothesisRelation.EXCLUSIVE_EXHAUSTIVE:
        return {
            hypothesis_id: [other for other in ids if other != hypothesis_id]
            for hypothesis_id in ids
        }
    return {hypothesis_id: [] for hypothesis_id in ids}
```

The compatibility parser returns the text before the header as `stem`, and `_frame_choices` stores that value as `normalized_question`. Explicit seed frames default to `TaskKind.DECISION` and `HypothesisRelation.EXCLUSIVE_EXHAUSTIVE` only when the versioned input omits them. Every adapter assigns `task_frame_id=f"{run_id}_task_frame"` and `frame_id=f"{run_id}_hypothesis_frame"`; recorded frames are re-scoped to the current run so ledger identities never collide.

- [ ] **Step 4: Refactor the initializer to materialize TaskFrame**

Change construction and result contracts:

```python
@dataclass(frozen=True)
class InitializeRunInput:
    run_id: str
    problem: str
    context: str = ""
    task_context: str = ""
    answer_choices: list[AnswerChoice] = field(default_factory=list)
    regime: RunRegime = RunRegime.AUTONOMOUS
    hypothesis_seeds: list[HypothesisSeed] = field(default_factory=list)
    task_kind: TaskKind | None = None
    hypothesis_relation: HypothesisRelation | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BayesProbeInitializer:
    def __init__(
        self,
        ledger: JsonlLedgerStore | None = None,
        task_framer: TaskFramer | None = None,
    ) -> None:
        self._ledger = ledger
        self._task_framer = task_framer or ExplicitTaskFramer()


@dataclass(frozen=True)
class InitializationResult:
    run: RunRecord
    task_frame: TaskFrame
    belief_state: BeliefState
    probe_candidates: list[ProbeCandidate]
```

`initialize` must call `self._task_framer.frame(framing_input)` first, then convert every `FramedHypothesis` to `Hypothesis` using its server-authored `initial_prior`. `ExplicitTaskFramer` assigns a uniform prior for exclusive choices, validates complete explicit seed priors as a distribution, and defaults independent seed credences to `0.5`. `ModelTaskFramer` assigns uniform priors for `exclusive_exhaustive` output and `0.5` to each independent output. Copy `task_frame` into `BeliefState`, set metadata keys `task_kind`, `hypothesis_relation`, `framing_method`, and use initialization method `task_frame_v0.1`.

Write the validated TaskFrame to the ledger before the RunRecord, then write BeliefState and initial probe candidates. Update the ledger regression test to expect `task_frame`, `run`, `belief_state`, and probe-candidate records in that order.

Delete `_default_seeds` for open questions. Keep the existing answer-choice discriminator and temporary per-hypothesis initial candidates; ProbeDesigner replacement belongs to Milestone 4.

- [ ] **Step 5: Migrate deterministic benchmark fixtures to explicit seeds**

Add this field to `BenchmarkSample` and require at least two seeds:

```python
hypothesis_seeds: list[HypothesisSeed] = field(default_factory=list)
```

In `BenchmarkSample.__post_init__`, raise `ValueError("benchmark samples require at least two explicit hypothesis seeds")` when the list is shorter than two. Pass a copy into `InitializeRunInput` from `_initialize_input`.

Parse seed objects in `benchmark_io.py` with an exact adapter:

```python
def _hypothesis_seed_from_payload(payload: Any) -> HypothesisSeed:
    if not isinstance(payload, Mapping):
        raise ValueError("benchmark hypothesis seed must be an object")
    try:
        return HypothesisSeed(
            id=payload.get("id"),
            statement=payload["statement"],
            scope=payload.get("scope"),
            prior=payload.get("prior"),
            falsifiers=list(payload.get("falsifiers", [])),
            predictions=list(payload.get("predictions", [])),
        )
    except KeyError as error:
        raise ValueError("benchmark hypothesis seed requires statement") from error
```

Require `hypothesis_seeds` in every JSON/JSONL sample and pass the parsed list into `BenchmarkSample`. Update test constructors with a local `benchmark_hypothesis_seeds()` helper. Update both checked-in benchmark fixtures with question-specific H1/H2 statements, scopes, falsifiers, predictions, and priors `0.5`; do not synthesize seeds from `gold_best_hypothesis` or question text at runtime.

Run: `pytest tests/test_benchmark_harness.py tests/test_benchmark_io.py tests/test_experiment_runner.py -q`

Expected: all benchmark paths initialize from explicit, auditable seeds.

- [ ] **Step 6: Migrate deterministic unit fixtures to explicit seeds**

In each deterministic runner/planner/executor test module, add a local helper and pass its result into `InitializeRunInput`:

```python
def explicit_test_hypothesis_seeds() -> list[HypothesisSeed]:
    return [
        HypothesisSeed(
            id="H1",
            statement="The fixture's H1 condition holds.",
            prior=0.5,
            scope="Deterministic test fixture.",
            falsifiers=["The fixture emits a reliable H1 refutation."],
            predictions=["The fixture emits a reliable H1 support cue."],
        ),
        HypothesisSeed(
            id="H2",
            statement="The fixture's H2 condition holds instead.",
            prior=0.5,
            scope="Deterministic test fixture.",
            falsifiers=["The fixture emits a reliable H2 refutation."],
            predictions=["The fixture emits a reliable H2 support cue."],
        ),
    ]
```

Apply this only to tests whose purpose is downstream orchestration. Keep MCQ parsing tests on explicit choice text, keep the no-fallback test unseeded, and keep provider-backed open framing tests unseeded. Update deterministic WebUI tests to send structured `answer_choices` rather than relying on hidden binary initialization.

Run: `pytest tests/test_probe_executor.py tests/test_probe_planner.py tests/test_public_api_and_config.py tests/test_question_runner.py tests/test_synchronized_runner.py tests/test_webui.py -q`

Expected: downstream tests retain their original H1/H2 purpose without exercising a removed fallback.

- [ ] **Step 7: Export the framing API and run tests**

Export the new contracts from `bayesprobe/__init__.py`, while keeping `HypothesisSeed` import-compatible through `bayesprobe.initialization`.

Run: `pytest tests/test_task_framing.py tests/test_initialization.py tests/test_benchmark_harness.py tests/test_benchmark_io.py tests/test_public_api_and_config.py -q`

Expected: all selected tests pass; no unseeded open path produces generic H1/H2.

- [ ] **Step 8: Commit explicit framing and fixture migration**

```bash
git add bayesprobe/task_framing.py bayesprobe/initialization.py bayesprobe/benchmark.py bayesprobe/benchmark_io.py bayesprobe/__init__.py fixtures/benchmarks/toy_belief_revision.json fixtures/benchmarks/bayesprobe_v0_2_methodology.json tests/test_task_framing.py tests/test_initialization.py tests/test_benchmark_harness.py tests/test_benchmark_io.py tests/test_probe_executor.py tests/test_probe_planner.py tests/test_public_api_and_config.py tests/test_question_runner.py tests/test_synchronized_runner.py tests/test_webui.py
git commit -m "feat: require explicit task frames for initialization"
```

---

### Task 3: Add ModelTaskFramer with One Repair

**Files:**
- Modify: `bayesprobe/task_framing.py`
- Modify: `tests/test_task_framing.py`

**Interfaces:**
- Consumes: `ModelGateway.complete_structured(StructuredModelRequest) -> dict[str, Any]`.
- Produces: `TaskFramingRepairPolicy(max_attempts=1, repair_task="repair_task_frame")`.
- Produces: `ModelTaskFramer.frame(input) -> TaskFrame`, `RecordedTaskFramer.frame(input) -> TaskFrame`, and `RoutingTaskFramer.frame(input) -> TaskFrame`.
- Produces: `task_frame_from_mapping(payload, run_id, question, method, trace) -> TaskFrame` for strict validation and stable server-side ids.

- [ ] **Step 1: Write failing model-framer tests**

Use `ScriptedModelGateway` to prove request shape, server-assigned ids, independent priors remaining outside model output, one repair, and fail-closed behavior:

```python
class QueueModelGateway:
    adapter_kind = "queue_test"

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.requests: list[StructuredModelRequest] = []

    def complete_structured(self, request: StructuredModelRequest) -> dict[str, Any]:
        self.requests.append(request)
        if not self.responses:
            raise AssertionError(f"unexpected model task: {request.task}")
        return self.responses.pop(0)


VALID_OPEN_FRAME = {
    "task_kind": "claim_verification",
    "answer_contract": {
        "objective": "Design a discriminating validation protocol.",
        "required_sections": [
            "hypotheses",
            "experimental_design",
            "controls",
            "metrics",
            "decision_rule",
            "limitations",
        ],
        "decision_form": "experimental_protocol",
        "permits_synthesis": True,
    },
    "hypothesis_relation": "independent",
    "hypotheses": [
        {
            "statement": "Scale has an independent positive effect under matched conditions.",
            "type": "causal_claim",
            "scope": "Matched task, scaffold, and inference budget.",
            "falsifiers": ["The controlled effect is negligible or negative."],
            "predictions": ["Performance increases across matched sizes."],
        },
        {
            "statement": "The apparent scale effect is materially confounded.",
            "type": "confounding_explanation",
            "scope": "Comparisons with unmatched resources.",
            "falsifiers": ["The effect survives matched controls."],
            "predictions": ["The effect shrinks after matching resources."],
        },
    ],
    "coverage_statement": "Covers the target effect and the primary confounder.",
    "coverage_limitation": "Conditional task interactions remain possible.",
}


def test_model_task_framer_calls_gateway_before_returning_frame():
    gateway = ScriptedModelGateway({"frame_open_question": VALID_OPEN_FRAME})
    frame = ModelTaskFramer(gateway).frame(
        TaskFramingInput(run_id="run_model", question="这个命题应该如何验证？")
    )
    assert [request.task for request in gateway.requests] == ["frame_open_question"]
    assert gateway.requests[0].input["question"] == "这个命题应该如何验证？"
    assert [item.id for item in frame.hypothesis_frame.hypotheses] == ["H1", "H2"]
    assert frame.framing_method == FramingMethod.MODEL


def test_model_task_framer_repairs_once_then_accepts():
    gateway = QueueModelGateway(
        [
            {"task_kind": "claim_verification", "hypotheses": []},
            VALID_OPEN_FRAME,
        ]
    )
    frame = ModelTaskFramer(gateway).frame(
        TaskFramingInput(run_id="run_repair", question="How should this be tested?")
    )
    assert [request.task for request in gateway.requests] == [
        "frame_open_question",
        "repair_task_frame",
    ]
    assert frame.framing_trace["repair_attempt_index"] == 1


def test_model_task_framer_fails_after_one_invalid_repair():
    gateway = QueueModelGateway(
        [
            {"task_kind": "claim_verification", "hypotheses": []},
            {"task_kind": "claim_verification", "hypotheses": []},
        ]
    )
    with pytest.raises(TaskFramingError, match="invalid after 1 repair attempt"):
        ModelTaskFramer(gateway).frame(
            TaskFramingInput(run_id="run_bad_repair", question="How should this be tested?")
        )


def test_routing_task_framer_keeps_explicit_mcq_off_the_model_gateway():
    gateway = QueueModelGateway([])
    frame = RoutingTaskFramer(
        explicit_framer=ExplicitTaskFramer(),
        open_framer=ModelTaskFramer(gateway),
    ).frame(
        TaskFramingInput(
            run_id="run_routed_mcq",
            question="Which result follows? Answer Choices: A. First B. Second",
        )
    )
    assert frame.task_kind == TaskKind.MULTIPLE_CHOICE
    assert gateway.requests == []
```

- [ ] **Step 2: Run model-framer tests and verify RED**

Run: `pytest tests/test_task_framing.py -q`

Expected: failures report missing `ModelTaskFramer`, repair policy, and mapping parser.

- [ ] **Step 3: Implement strict mapping validation**

Add this repair contract and adapter structure to `task_framing.py`:

```python
@dataclass(frozen=True)
class TaskFramingRepairPolicy:
    max_attempts: int = 1
    repair_task: str = "repair_task_frame"


class ModelTaskFramer:
    def __init__(
        self,
        model_gateway: ModelGateway,
        repair_policy: TaskFramingRepairPolicy | None = None,
    ) -> None:
        self._model_gateway = model_gateway
        self._repair_policy = repair_policy or TaskFramingRepairPolicy()

    def frame(self, input: TaskFramingInput) -> TaskFrame:
        request = _open_frame_request(input)
        payload = self._model_gateway.complete_structured(request)
        try:
            return task_frame_from_mapping(
                payload,
                run_id=input.run_id,
                question=input.question,
                task_context=input.task_context,
                method=FramingMethod.MODEL,
                trace=_trace_for(request, self._model_gateway),
            )
        except (ValueError, ModelGatewayValidationError) as error:
            return self._repair_or_raise(input, request, payload, error)
```

`task_frame_from_mapping` must reject unknown task kinds and relations, require exactly 2-6 hypotheses, assign ids `H1` through `H6` after validation, assign server-authored `initial_prior` values according to relation, normalize statements for duplicate detection, set all-to-all rival ids only for `exclusive_exhaustive`, and set explicit empty rival lists for independent hypotheses. It must never read ids, priors, posteriors, API keys, or provider credentials from model output.

Use this parser shape so provider fields cannot leak into the domain object:

```python
def task_frame_from_mapping(
    payload: Mapping[str, Any],
    *,
    run_id: str,
    question: str,
    task_context: str,
    method: FramingMethod,
    trace: dict[str, Any],
) -> TaskFrame:
    if not isinstance(payload, Mapping):
        raise TaskFramingError("task frame payload must be an object")
    expected_top_level = {
        "task_kind",
        "answer_contract",
        "hypothesis_relation",
        "hypotheses",
        "coverage_statement",
        "coverage_limitation",
    }
    if set(payload) != expected_top_level:
        raise TaskFramingError("task frame payload has missing or unknown fields")
    forbidden = {"id", "prior", "posterior", "api_key", "authorization", "token"}
    expected_hypothesis_fields = {
        "statement",
        "type",
        "scope",
        "falsifiers",
        "predictions",
    }
    raw_hypotheses = payload.get("hypotheses")
    if not isinstance(raw_hypotheses, list) or not 2 <= len(raw_hypotheses) <= 6:
        raise TaskFramingError("task frame must contain between 2 and 6 hypotheses")
    for item in raw_hypotheses:
        if not isinstance(item, Mapping):
            raise TaskFramingError("each framed hypothesis must be an object")
        if forbidden.intersection(item):
            raise TaskFramingError("provider hypotheses cannot assign ids or beliefs")
        if set(item) != expected_hypothesis_fields:
            raise TaskFramingError("provider hypothesis has missing or unknown fields")

    try:
        task_kind = TaskKind(payload["task_kind"])
        relation = HypothesisRelation(payload["hypothesis_relation"])
    except (KeyError, TypeError, ValueError) as error:
        raise TaskFramingError("invalid task kind or hypothesis relation") from error
    if task_kind == TaskKind.MULTIPLE_CHOICE:
        raise TaskFramingError("model framing cannot create a multiple-choice task")

    ids = [f"H{index}" for index in range(1, len(raw_hypotheses) + 1)]
    priors = (
        [1.0 / len(ids)] * len(ids)
        if relation == HypothesisRelation.EXCLUSIVE_EXHAUSTIVE
        else [0.5] * len(ids)
    )
    hypotheses = [
        FramedHypothesis(
            id=ids[index],
            statement=str(item.get("statement", "")),
            type=str(item.get("type", "")),
            scope=str(item.get("scope", "")),
            initial_prior=priors[index],
            falsifiers=list(item.get("falsifiers", [])),
            predictions=list(item.get("predictions", [])),
        )
        for index, item in enumerate(raw_hypotheses)
    ]
    contract_payload = payload.get("answer_contract")
    if not isinstance(contract_payload, Mapping):
        raise TaskFramingError("answer_contract must be an object")
    answer_contract = AnswerContract.model_validate(contract_payload)
    return TaskFrame(
        task_frame_id=f"{run_id}_task_frame",
        task_kind=task_kind,
        normalized_question=question.strip(),
        task_context=task_context.strip(),
        answer_contract=answer_contract,
        hypothesis_frame=HypothesisFrame(
            frame_id=f"{run_id}_hypothesis_frame",
            relation=relation,
            hypotheses=hypotheses,
            rival_sets={
                hypothesis_id: [other for other in ids if other != hypothesis_id]
                if relation == HypothesisRelation.EXCLUSIVE_EXHAUSTIVE
                else []
                for hypothesis_id in ids
            },
            coverage_statement=str(payload.get("coverage_statement", "")),
            coverage_limitation=(
                str(payload["coverage_limitation"])
                if payload.get("coverage_limitation") is not None
                else None
            ),
        ),
        framing_method=method,
        framing_trace=trace,
    )
```

Sanitize malformed payloads before repair or trace use:

```python
def _secret_free_payload(value: Any) -> Any:
    if isinstance(value, Mapping):
        clean: dict[str, Any] = {}
        for key, item in value.items():
            normalized_key = re.sub(r"[^a-z0-9]", "", str(key).casefold())
            if any(part in normalized_key for part in ("apikey", "authorization", "token", "secret")):
                clean[str(key)] = "[REDACTED]"
            else:
                clean[str(key)] = _secret_free_payload(item)
        return clean
    if isinstance(value, list):
        return [_secret_free_payload(item) for item in value]
    if isinstance(value, str):
        return re.sub(r"(?:^|\s)sk-[A-Za-z0-9_-]{12,}", " [REDACTED]", value)
    return value
```

Implement `_repair_or_raise` as one bounded loop over `range(1, max_attempts + 1)`. Each repair request contains `original_request`, `_secret_free_payload(invalid_payload)`, `validation_error`, `attempt_index`, enum values, and required fields. Validate each response with `task_frame_from_mapping`; on exhaustion raise `TaskFramingError(f"task frame invalid after {max_attempts} repair attempt")` from the last validation error. `TaskFramingRepairPolicy` rejects negative or non-integer attempt counts and empty repair task names.

The initial request is:

```python
StructuredModelRequest(
    task="frame_open_question",
    input={
        "question": input.question.strip(),
        "task_context": input.task_context.strip(),
        "supported_task_kinds": [kind.value for kind in TaskKind if kind != TaskKind.MULTIPLE_CHOICE],
        "supported_relations": [relation.value for relation in HypothesisRelation],
        "hypothesis_count": {"minimum": 2, "maximum": 6},
    },
    prompt_id="open_question_task_framing",
    prompt_version="v0.1",
    schema_name="OpenQuestionTaskFrame",
    schema_version="v0.1",
    metadata={"run_id": input.run_id},
)
```

The repair request includes the original request input, secret-free invalid payload, validation error, attempt index, required fields, and allowed enum values. It uses prompt id `open_question_task_framing_repair`, schema `OpenQuestionTaskFrame`, and metadata `repair_attempt_index=1`.

- [ ] **Step 4: Add RecordedTaskFramer**

Implement an immutable adapter used when a caller already has a validated frame fixture:

```python
class RecordedTaskFramer:
    def __init__(self, frame: TaskFrame) -> None:
        self._frame = frame.model_copy(deep=True)

    def frame(self, input: TaskFramingInput) -> TaskFrame:
        hypothesis_frame = self._frame.hypothesis_frame.model_copy(
            deep=True,
            update={"frame_id": f"{input.run_id}_hypothesis_frame"},
        )
        return self._frame.model_copy(
            deep=True,
            update={
                "task_frame_id": f"{input.run_id}_task_frame",
                "normalized_question": input.question.strip(),
                "framing_method": FramingMethod.RECORDED,
                "hypothesis_frame": hypothesis_frame,
            },
        )
```

Do not change `RecordedModelGateway`; replaying provider-shaped framing remains possible by injecting it into `ModelTaskFramer`.

Add the provider router:

```python
class RoutingTaskFramer:
    def __init__(
        self,
        *,
        explicit_framer: ExplicitTaskFramer,
        open_framer: TaskFramer,
    ) -> None:
        self._explicit_framer = explicit_framer
        self._open_framer = open_framer

    def frame(self, input: TaskFramingInput) -> TaskFrame:
        has_explicit_frame = bool(
            input.answer_choices
            or input.hypothesis_seeds
            or parse_legacy_answer_choice_frame(input.question) is not None
        )
        if has_explicit_frame:
            return self._explicit_framer.frame(input)
        return self._open_framer.frame(input)
```

- [ ] **Step 5: Run adapter tests**

Run: `pytest tests/test_task_framing.py tests/test_recorded_model_gateway.py -q`

Expected: all selected tests pass, including one-repair and no-secret assertions.

- [ ] **Step 6: Commit model and recorded framers**

```bash
git add bayesprobe/task_framing.py tests/test_task_framing.py
git commit -m "feat: add model-backed open task framing"
```

---

### Task 4: Teach OpenAI-Compatible Gateways the Framing Tasks

**Files:**
- Modify: `bayesprobe/openai_gateway.py`
- Modify: `tests/test_openai_gateway.py`

**Interfaces:**
- Consumes: `StructuredModelRequest(task="frame_open_question" | "repair_task_frame")`.
- Produces: identical `OpenQuestionTaskFrame` JSON shape for Responses and Chat Completions adapters.

- [ ] **Step 1: Write failing payload tests**

Add one Responses payload test and one Chat Completions repair test:

```python
def _frame_request() -> StructuredModelRequest:
    return StructuredModelRequest(
        task="frame_open_question",
        input={
            "question": "How should this claim be tested?",
            "task_context": "Use a frozen task distribution.",
            "supported_task_kinds": ["claim_verification", "design"],
            "supported_relations": ["exclusive_exhaustive", "independent"],
            "hypothesis_count": {"minimum": 2, "maximum": 6},
        },
        prompt_id="open_question_task_framing",
        prompt_version="v0.1",
        schema_name="OpenQuestionTaskFrame",
        schema_version="v0.1",
    )


def _repair_frame_request() -> StructuredModelRequest:
    return StructuredModelRequest(
        task="repair_task_frame",
        input={
            "original_request": _frame_request().input,
            "invalid_payload": {"hypotheses": []},
            "validation_error": "at least two hypotheses are required",
            "attempt_index": 1,
        },
        prompt_id="open_question_task_framing_repair",
        prompt_version="v0.1",
        schema_name="OpenQuestionTaskFrame",
        schema_version="v0.1",
        metadata={"repair_attempt_index": 1},
    )


def test_build_openai_payload_for_open_question_frame():
    payload = build_openai_request_payload(
        _frame_request(),
        OpenAIModelGatewayConfig(model="test-model"),
    )
    assert payload["text"]["format"]["name"] == "OpenQuestionTaskFrame"
    schema = payload["text"]["format"]["schema"]
    assert schema["properties"]["hypothesis_relation"]["enum"] == [
        "exclusive_exhaustive",
        "independent",
    ]
    assert "prior" not in json.dumps(schema)
    assert "posterior" not in json.dumps(schema)


def test_build_chat_payload_for_task_frame_repair():
    payload = build_openai_chat_completions_payload(
        _repair_frame_request(),
        OpenAIModelGatewayConfig(model="test-model"),
    )
    required_output = json.loads(payload["messages"][1]["content"])["required_output"]
    assert required_output["type"] == "OpenQuestionTaskFrame"
    assert payload["response_format"] == {"type": "json_object"}
```

- [ ] **Step 2: Run gateway tests and verify RED**

Run: `pytest tests/test_openai_gateway.py -q`

Expected: both tasks fail with `unsupported openai model task`.

- [ ] **Step 3: Add the framing JSON schema and instructions**

Define the provider schema without id/prior/posterior fields:

```python
OPEN_QUESTION_TASK_FRAME_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "task_kind",
        "answer_contract",
        "hypothesis_relation",
        "hypotheses",
        "coverage_statement",
        "coverage_limitation",
    ],
    "properties": {
        "task_kind": {
            "type": "string",
            "enum": [
                "claim_verification",
                "explanation",
                "diagnosis",
                "design",
                "decision",
            ],
        },
        "answer_contract": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "objective",
                "required_sections",
                "decision_form",
                "permits_synthesis",
            ],
            "properties": {
                "objective": {"type": "string", "minLength": 1},
                "required_sections": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "minItems": 1,
                    "uniqueItems": True,
                },
                "decision_form": {"type": "string", "minLength": 1},
                "permits_synthesis": {"type": "boolean"},
            },
        },
        "hypothesis_relation": {
            "type": "string",
            "enum": ["exclusive_exhaustive", "independent"],
        },
        "hypotheses": {
            "type": "array",
            "minItems": 2,
            "maxItems": 6,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "statement",
                    "type",
                    "scope",
                    "falsifiers",
                    "predictions",
                ],
                "properties": {
                    "statement": {"type": "string", "minLength": 1},
                    "type": {"type": "string", "minLength": 1},
                    "scope": {"type": "string", "minLength": 1},
                    "falsifiers": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "minItems": 1,
                    },
                    "predictions": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "minItems": 1,
                    },
                },
            },
        },
        "coverage_statement": {"type": "string", "minLength": 1},
        "coverage_limitation": {"type": ["string", "null"]},
    },
}
```

Extend all three dispatch helpers:

```python
if task == "frame_open_question":
    return (
        "Frame the supplied open question for BayesProbe before belief initialization. "
        "Return 2-6 distinct, falsifiable hypotheses and an AnswerContract. "
        "Do not assign ids, priors, posteriors, or claim external evidence."
    )
if task == "repair_task_frame":
    return (
        "Repair the malformed BayesProbe open-question frame using the validation "
        "error. Return one complete frame without ids, priors, or posteriors."
    )
```

For `_structured_output_for_task`, both tasks return `("OpenQuestionTaskFrame", OPEN_QUESTION_TASK_FRAME_JSON_SCHEMA)`. For Chat Completions, require exactly `task_kind`, `answer_contract`, `hypothesis_relation`, `hypotheses`, `coverage_statement`, and `coverage_limitation` as top-level keys. `_required_output_for_task` must include the full schema and notes that hypotheses are semantic candidates rather than answer labels.

- [ ] **Step 4: Run all OpenAI adapter tests**

Run: `pytest tests/test_openai_gateway.py tests/test_openai_live.py -q`

Expected: adapter tests pass; live tests remain skipped unless explicitly enabled.

- [ ] **Step 5: Commit provider task support**

```bash
git add bayesprobe/openai_gateway.py tests/test_openai_gateway.py
git commit -m "feat: support task framing in openai gateways"
```

---

### Task 5: Put Framing Before Initialization in Runners and WebUI

**Files:**
- Modify: `bayesprobe/question_runner.py`
- Modify: `bayesprobe/probe_executor.py`
- Modify: `bayesprobe/webui.py`
- Modify: `bayesprobe/webui_static/index.html`
- Modify: `bayesprobe/webui_static/app.js`
- Modify: `tests/test_question_runner.py`
- Modify: `tests/test_probe_executor.py`
- Modify: `tests/test_webui.py`
- Modify: `tests/test_webui_stream.js`

**Interfaces:**
- Consumes: `InitializationResult.task_frame` and `ModelTaskFramer`.
- Produces: progress kinds `task_framing_started` and `task_framing_completed`.
- Produces: `AutonomousQuestionRunResult.task_frame` and serialized top-level `task_frame`.
- Guarantees: progress events before `initialization_completed` contain no Belief State.

- [ ] **Step 1: Write failing runner progress tests**

Add a gateway that records task calls and returns a valid frame, probe signal, and evidence judgment. Assert the phase boundary:

```python
def valid_open_frame_payload() -> dict[str, Any]:
    return {
        "task_kind": "claim_verification",
        "answer_contract": {
            "objective": "Design a discriminating validation protocol.",
            "required_sections": ["hypotheses", "controls", "decision_rule"],
            "decision_form": "experimental_protocol",
            "permits_synthesis": True,
        },
        "hypothesis_relation": "independent",
        "hypotheses": [
            {
                "statement": "Scale has an independent effect under matched conditions.",
                "type": "causal_claim",
                "scope": "Matched task and resource conditions.",
                "falsifiers": ["The controlled effect is negligible."],
                "predictions": ["Matched performance rises with size."],
            },
            {
                "statement": "The apparent effect is materially confounded.",
                "type": "confounding_explanation",
                "scope": "Unmatched comparisons.",
                "falsifiers": ["The effect survives matched controls."],
                "predictions": ["The effect shrinks after matching."],
            },
        ],
        "coverage_statement": "Covers the effect and its primary confounder.",
        "coverage_limitation": "Task interactions may remain.",
    }


class RecordingOpenQuestionGateway:
    adapter_kind = "recording_open_question_test"

    def __init__(self) -> None:
        self.requests: list[StructuredModelRequest] = []

    def complete_structured(self, request: StructuredModelRequest) -> dict[str, Any]:
        self.requests.append(request)
        if request.task == "frame_open_question":
            return valid_open_frame_payload()
        if request.task == "execute_probe":
            return {"raw_content": "MODEL REASONING: A matched controlled test is required."}
        if request.task == "judge_evidence":
            return {
                "evidence_type": "supporting",
                "likelihoods": {
                    hypothesis_id: "weakly_confirming"
                    for hypothesis_id in request.input["target_hypotheses"]
                },
                "interpretation": "A design suggestion, not an external result.",
                "quality_overrides": {"independence": 0.2, "verifiability": 0.2},
            }
        raise AssertionError(f"unexpected task: {request.task}")


def test_open_question_framing_precedes_belief_initialization():
    gateway = RecordingOpenQuestionGateway()
    observed = []

    def observe(event):
        observed.append((event, [request.task for request in gateway.requests]))

    runner = AutonomousQuestionRunner(
        core=BayesProbeCore(model_gateway=gateway),
        initializer=BayesProbeInitializer(
            task_framer=ModelTaskFramer(gateway),
        ),
        executor=ProbeExecutor(ModelBackedProbeToolGateway(gateway)),
        config=AutonomousQuestionRunConfig(max_cycles=1, max_probes_per_cycle=1),
        progress_observer=observe,
    )

    result = runner.run_question(
        InitializeRunInput(
            run_id="run_open_progress",
            problem="某团队认为模型变大一定提升 agent 表现，应该如何验证？",
        )
    )

    events = [event for event, _ in observed]
    kinds = [event.kind for event in events]
    assert kinds[:4] == [
        AutonomousQuestionProgressKind.RUN_STARTED,
        AutonomousQuestionProgressKind.TASK_FRAMING_STARTED,
        AutonomousQuestionProgressKind.TASK_FRAMING_COMPLETED,
        AutonomousQuestionProgressKind.INITIALIZATION_COMPLETED,
    ]
    assert events[1].belief_state is None
    assert events[2].belief_state is None
    assert events[2].task_frame is not None
    assert observed[1][1] == []
    assert observed[2][1] == ["frame_open_question"]
    assert gateway.requests[0].task == "frame_open_question"
    assert result.initial_belief_state.task_frame == result.task_frame
```

- [ ] **Step 2: Run runner tests and verify RED**

Run: `pytest tests/test_question_runner.py -q`

Expected: missing framing progress enum members and result fields cause failures.

- [ ] **Step 3: Add progress and result contracts**

Extend the enum and dataclasses in `question_runner.py`:

```python
class AutonomousQuestionProgressKind(StrEnum):
    RUN_STARTED = "run_started"
    TASK_FRAMING_STARTED = "task_framing_started"
    TASK_FRAMING_COMPLETED = "task_framing_completed"
    INITIALIZATION_COMPLETED = "initialization_completed"
    CYCLE_STARTED = "cycle_started"
    PROBE_SET_PLANNED = "probe_set_planned"
    PROBE_EXECUTION_STARTED = "probe_execution_started"
    SIGNALS_COLLECTED = "signals_collected"
    EVIDENCE_INTEGRATION_STARTED = "evidence_integration_started"
    CYCLE_INTEGRATED = "cycle_integrated"
    RUN_COMPLETED = "run_completed"


@dataclass(frozen=True)
class AutonomousQuestionProgress:
    kind: AutonomousQuestionProgressKind
    run_id: str
    task_frame: TaskFrame | None = None
    cycle_id: str | None = None
    cycle_index: int | None = None
    run: RunRecord | None = None
    belief_state: BeliefState | None = None
    probe_set: ProbeSet | None = None
    signals: Sequence[ExternalSignal] = ()
    cycle_result: AutonomousQuestionCycleResult | None = None
    result: AutonomousQuestionRunResult | None = None
```

Add `task_frame: TaskFrame` to `AutonomousQuestionRunResult`. In `run_question`, emit `TASK_FRAMING_STARTED`, call `initializer.initialize`, emit `TASK_FRAMING_COMPLETED` with only the TaskFrame, then emit `INITIALIZATION_COMPLETED` with the RunRecord and BeliefState. Pass the frame into every `_result` call.

- [ ] **Step 4: Write failing WebUI framing tests**

Update provider tests so the fake provider returns framing before probe execution. Add assertions for serialized events and deterministic-open failure:

```python
def test_webui_provider_frames_open_question_before_exposing_belief():
    events = []
    status, body = handle_autonomous_stream_request(
        _open_provider_payload(),
        event_sink=events.append,
        client_factory=_framing_client_factory,
    )
    assert status == 200
    assert body is None
    assert [event["event"] for event in events[:4]] == [
        "run_started",
        "task_framing_started",
        "task_framing_completed",
        "initialization_completed",
    ]
    assert "belief_state" not in events[2]["data"]
    assert events[2]["data"]["task_frame"]["task_kind"] == "claim_verification"
    assert events[3]["data"]["belief_state"]["task_frame"] is not None


def test_webui_deterministic_open_question_fails_without_binary_fallback():
    status, payload = handle_autonomous_run_request(
        {
            "question": "某团队认为模型变大一定提升 agent 表现，应该如何验证？",
            "provider": {"kind": "deterministic"},
        }
    )
    assert status == 400
    assert payload["error"]["type"] == "validation_error"
    assert "requires a model or recorded task framer" in payload["error"]["message"]


def test_webui_accepts_structured_answer_choices_without_model_framing():
    status, payload = handle_autonomous_run_request(
        {
            "question": "Which result follows?",
            "answer_choices": [
                {"label": "A", "text": "First result"},
                {"label": "B", "text": "Second result"},
            ],
            "provider": {"kind": "deterministic"},
            "runner": {"max_cycles": 1, "max_probes_per_cycle": 1},
        }
    )
    assert status == 200
    assert payload["task_frame"]["task_kind"] == "multiple_choice"
    assert [item["id"] for item in payload["initial_belief_state"]["hypotheses"]] == ["A", "B"]
```

- [ ] **Step 5: Wire provider-backed framing in WebUI**

In `_prepare_autonomous_run`, construct one request-scoped gateway and inject it into all model-backed roles:

```python
task_framer = (
    RoutingTaskFramer(
        explicit_framer=ExplicitTaskFramer(),
        open_framer=ModelTaskFramer(gateway),
    )
    if provider_kind in OPENAI_COMPATIBLE_PROVIDER_KINDS
    else ExplicitTaskFramer()
)
initializer = BayesProbeInitializer(
    ledger=core.ledger,
    task_framer=task_framer,
)
runner = AutonomousQuestionRunner(
    core=core,
    initializer=initializer,
    executor=executor,
    config=request["runner_config"],
    progress_observer=progress_observer,
)
```

Extend `_parse_autonomous_request` with optional `task_context: str` and structured `answer_choices: list[{label, text}]`. Reject non-array choices, duplicate labels, empty text, or fewer than two choices with HTTP 400. Pass both fields into `InitializeRunInput`; keep the existing `context` field mapped only to the initial passive signal compatibility path.

Use this parser at the HTTP boundary:

```python
def _answer_choices_from_payload(value: Any) -> list[AnswerChoice]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise WebUIError("answer_choices must be an array")
    try:
        choices = [AnswerChoice.model_validate(item) for item in value]
    except (TypeError, ValueError) as error:
        raise WebUIError("answer_choices must contain non-empty label/text objects") from error
    if len(choices) < 2:
        raise WebUIError("answer_choices must contain at least two choices")
    labels = [choice.label for choice in choices]
    if len(labels) != len(set(labels)):
        raise WebUIError("answer_choices labels must be unique")
    return choices
```

Add `task_context` and `answer_choices` to the parsed request mapping, then construct `InitializeRunInput` with both values. Catch `TaskFramingError` separately: deterministic requests map it to `WebUIError`; OpenAI-compatible requests map it to `ProviderError(_provider_error_message(provider_kind))` so provider validation details and malformed payloads remain private.

Replace the probe execution metadata key `initial_context` with `task_context`:

```python
context=ProbeExecutionContext(
    run_id=run.run_id,
    cycle_id=cycle_id,
    belief_state=current_belief_state,
    metadata={
        "problem": run.problem,
        "task_context": input.task_context.strip(),
    },
)
```

In `ModelBackedProbeToolGateway`, emit request input field `task_context` and remove `initial_context`. Resolve Task Context from explicit execution metadata first, then from `belief_state.task_frame.task_context`, so Synchronized execution receives the same semantics. Update `tests/test_probe_executor.py` to assert the task constraint appears exactly once in the provider request and the Initial Signal text is absent. The latter remains present only in `_initial_context_signals` and the Evidence Gate input.

Split the visible fields in `index.html`:

```html
<label>
  Task context
  <textarea id="task-context" placeholder="Optional scope, constraints, audience, or required output"></textarea>
</label>
<label>
  Initial signal
  <textarea id="context" placeholder="Optional observation, source text, log, or expert feedback"></textarea>
</label>
```

Add `task_context: valueOf("task-context")` to `buildPayload` and add `task-context` to the JavaScript test DOM ids. Do not pass the Initial Signal text to `ModelTaskFramer`; it enters cycle 1 through `_initial_context_signals` only.

Serialize `TASK_FRAMING_STARTED` as `{}` and `TASK_FRAMING_COMPLETED` as an object whose only field is the serialized `task_frame`. Add `task_frame` to `serialize_autonomous_run_result`.

For deterministic framing errors, return HTTP 400 or stream `run_failed` with `validation_error`. For OpenAI-compatible framing failures, retain HTTP 502 or stream `provider_error`. Reuse secret-safe diagnostics and never include invalid provider payloads in the response.

- [ ] **Step 6: Update frontend progress behavior**

Add labels:

```javascript
task_framing_started: "Framing task",
task_framing_completed: "Task framed",
```

Do not call `renderBeliefs` for either event. Continue rendering the first Belief State only on `initialization_completed`. Extend `tests/test_webui_stream.js` to feed the four initial events and assert the belief panel remains `Run pending.` through `task_framing_completed`, then changes on `initialization_completed`.

- [ ] **Step 7: Run runner and WebUI tests**

Run: `pytest tests/test_question_runner.py tests/test_webui.py -q`

Run: `node --test tests/test_webui_stream.js`

Expected: all selected Python and JavaScript tests pass; provider request order begins with framing.

- [ ] **Step 8: Commit runner and WebUI framing**

```bash
git add bayesprobe/question_runner.py bayesprobe/probe_executor.py bayesprobe/webui.py bayesprobe/webui_static/index.html bayesprobe/webui_static/app.js tests/test_question_runner.py tests/test_probe_executor.py tests/test_webui.py tests/test_webui_stream.js
git commit -m "feat: expose task framing before belief initialization"
```

---

### Task 6: Preserve Hypothesis Relation Through Belief Revision

**Files:**
- Modify: `bayesprobe/schemas.py`
- Modify: `bayesprobe/task_framing.py`
- Modify: `bayesprobe/belief.py`
- Modify: `bayesprobe/core.py`
- Modify: `bayesprobe/hypothesis_evolution.py`
- Modify: `bayesprobe/initialization.py`
- Modify: `bayesprobe/projections.py`
- Modify: `bayesprobe/runners.py`
- Modify: `bayesprobe/webui_static/app.js`
- Modify: `tests/test_belief.py`
- Modify: `tests/test_autonomous_runner.py`
- Modify: `tests/test_core_cycles.py`
- Modify: `tests/test_controllers.py`
- Modify: `tests/test_hypothesis_evolution.py`
- Modify: `tests/test_initialization.py`
- Modify: `tests/test_question_runner.py`
- Modify: `tests/test_synchronized_runner.py`
- Modify: `tests/test_webui_stream.js`

**Interfaces:**
- Consumes: `BeliefState.task_frame.hypothesis_frame.relation`.
- Produces: `migrate_legacy_belief_state(state) -> BeliefState` as the only compatibility adapter for relation-less legacy state.
- Produces: `normalize_hypotheses(hypotheses, *, relation)`, `solve_updates(run_id, cycle_id, belief_state, events)`, and `summarize_hypotheses(hypotheses, *, relation)`.
- Guarantees: exclusive active posterior mass sums to one; independent credences are never cross-normalized.

- [ ] **Step 1: Write failing relation-aware solver tests**

Build Belief States from explicit TaskFrames with these test helpers:

```python
def belief_state_for_relation(
    relation: HypothesisRelation,
    *,
    posteriors: dict[str, float],
    complexity_penalty: dict[str, float] | None = None,
) -> BeliefState:
    penalty_by_id = complexity_penalty or {}
    ids = list(posteriors)
    hypotheses = [
        Hypothesis(
            id=hypothesis_id,
            statement=f"Semantic statement for {hypothesis_id}.",
            scope="Test scope.",
            prior=posterior,
            posterior=posterior,
            rivals=[other for other in ids if other != hypothesis_id]
            if relation == HypothesisRelation.EXCLUSIVE_EXHAUSTIVE
            else [],
            falsifiers=[f"A result falsifies {hypothesis_id}."],
            predictions=[f"A result supports {hypothesis_id}."],
            complexity_penalty=penalty_by_id.get(hypothesis_id, 0.0),
        )
        for hypothesis_id, posterior in posteriors.items()
    ]
    frame = TaskFrame(
        task_frame_id="run_relation_task_frame",
        task_kind=TaskKind.DECISION,
        normalized_question="Which hypotheses remain credible?",
        task_context="",
        answer_contract=AnswerContract(
            objective="Report the current beliefs.",
            required_sections=["answer", "uncertainty"],
            decision_form="belief_report",
            permits_synthesis=relation == HypothesisRelation.INDEPENDENT,
        ),
        hypothesis_frame=HypothesisFrame(
            frame_id="run_relation_hypothesis_frame",
            relation=relation,
            hypotheses=[
                FramedHypothesis(
                    id=item.id,
                    statement=item.statement,
                    type=item.type,
                    scope=item.scope,
                    initial_prior=item.prior,
                    falsifiers=list(item.falsifiers),
                    predictions=list(item.predictions),
                )
                for item in hypotheses
            ],
            rival_sets={
                hypothesis_id: [other for other in ids if other != hypothesis_id]
                if relation == HypothesisRelation.EXCLUSIVE_EXHAUSTIVE
                else []
                for hypothesis_id in ids
            },
            coverage_statement="Test relation frame.",
        ),
        framing_method=FramingMethod.RECORDED,
    )
    return BeliefState(
        belief_state_id="run_relation_bs_0",
        run_id="run_relation",
        cycle_id="cycle_0",
        hypotheses=hypotheses,
        task_frame=frame,
    )


def evidence_event(
    *,
    event_id: str,
    targets: list[str],
    likelihoods: dict[str, LikelihoodBand],
) -> EvidenceEvent:
    return EvidenceEvent(
        id=event_id,
        derived_from_signal=f"S_{event_id}",
        target_hypotheses=targets,
        evidence_type=EvidenceType.SUPPORTING,
        content="Controlled test signal.",
        reliability=1.0,
        independence=1.0,
        relevance=1.0,
        novelty=1.0,
        likelihoods=likelihoods,
    )
```

Then assert the two relation invariants:

```python
def test_independent_update_does_not_cross_normalize_untargeted_hypothesis():
    state = belief_state_for_relation(
        HypothesisRelation.INDEPENDENT,
        posteriors={"H1": 0.5, "H2": 0.5},
    )
    event = evidence_event(
        event_id="E_independent",
        targets=["H1"],
        likelihoods={"H1": LikelihoodBand.STRONGLY_CONFIRMING},
    )

    hypotheses, updates = solve_updates("run_relation", "cycle_1", state, [event])

    by_id = {item.id: item for item in hypotheses}
    assert by_id["H1"].posterior > 0.5
    assert by_id["H2"].posterior == 0.5
    assert sum(item.posterior for item in hypotheses) > 1.0
    assert [update.hypothesis_id for update in updates] == ["H1"]


def test_exclusive_update_keeps_active_mass_equal_to_one():
    state = belief_state_for_relation(
        HypothesisRelation.EXCLUSIVE_EXHAUSTIVE,
        posteriors={"H1": 0.5, "H2": 0.5},
    )
    event = evidence_event(
        event_id="E_exclusive",
        targets=["H1", "H2"],
        likelihoods={
            "H1": LikelihoodBand.MODERATELY_CONFIRMING,
            "H2": LikelihoodBand.MODERATELY_DISCONFIRMING,
        },
    )

    hypotheses, updates = solve_updates("run_relation", "cycle_1", state, [event])

    assert sum(item.posterior for item in hypotheses) == pytest.approx(1.0)
    assert {update.hypothesis_id for update in updates} == {"H1", "H2"}


def test_static_penalties_are_not_subtracted_again_on_later_events():
    state = belief_state_for_relation(
        HypothesisRelation.INDEPENDENT,
        posteriors={"H1": 0.5, "H2": 0.5},
        complexity_penalty={"H1": 0.2},
    )
    neutral = evidence_event(
        event_id="E_neutral",
        targets=["H1"],
        likelihoods={"H1": LikelihoodBand.NEUTRAL},
    )
    after_first, _ = solve_updates("run_relation", "cycle_1", state, [neutral])
    state_after_first = state.model_copy(update={"hypotheses": after_first})
    after_second, _ = solve_updates("run_relation", "cycle_2", state_after_first, [neutral])
    assert after_second[0].posterior == after_first[0].posterior
```

- [ ] **Step 2: Run solver tests and verify RED**

Run: `pytest tests/test_belief.py -q`

Expected: independent posteriors are incorrectly normalized and the untargeted hypothesis moves.

- [ ] **Step 3: Track applied static penalties**

Add these compatibility fields to `Hypothesis`:

```python
applied_complexity_penalty: float = 0.0
applied_ad_hoc_penalty: float = 0.0
```

Validate both in `[0, 1]` with the existing probability-like validator. In the solver, subtract only:

```python
complexity_delta = max(
    hypothesis.complexity_penalty - hypothesis.applied_complexity_penalty,
    0.0,
)
ad_hoc_delta = max(
    hypothesis.ad_hoc_penalty - hypothesis.applied_ad_hoc_penalty,
    0.0,
)
```

After an admitted event applies the deltas, return the hypothesis with `applied_complexity_penalty=hypothesis.complexity_penalty` and `applied_ad_hoc_penalty=hypothesis.ad_hoc_penalty`. Discarded events apply neither evidence nor penalties.

- [ ] **Step 4: Implement explicit relation-aware math**

Change normalization to require a relation:

```python
def normalize_hypotheses(
    hypotheses: list[Hypothesis],
    *,
    relation: HypothesisRelation,
) -> list[Hypothesis]:
    if relation == HypothesisRelation.INDEPENDENT:
        return list(hypotheses)
    return _normalize_exclusive_hypotheses(hypotheses)
```

Implement independent updates in log-odds space:

```python
def _logit(probability: float) -> float:
    bounded = min(max(probability, _MIN_PROBABILITY), 1.0 - _MIN_PROBABILITY)
    return math.log(bounded / (1.0 - bounded))


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _independent_event_posterior(
    prior: float,
    band: LikelihoodBand,
    weight: float,
    complexity_delta: float,
    ad_hoc_delta: float,
) -> float:
    score = (
        _logit(prior)
        + math.log(likelihood_band_to_lr(band)) * weight
        - complexity_delta
        - ad_hoc_delta
    )
    return round(_sigmoid(score), 4)
```

`solve_updates` must raise `ValueError("belief state requires hypothesis relation metadata")` when `task_frame` is absent. For exclusive frames, retain softmax redistribution across all active hypotheses. For independent frames, iterate only `event.target_hypotheses`, leave untargeted hypotheses byte-for-byte unchanged, and create updates only for targeted active ids.

- [ ] **Step 5: Add explicit legacy-state migration**

Add `FramingMethod.LEGACY_MIGRATION = "legacy_migration"` and this adapter to `task_framing.py`:

```python
def migrate_legacy_belief_state(state: BeliefState) -> BeliefState:
    if state.task_frame is not None:
        return state
    ids = [item.id for item in state.hypotheses]
    if not ids:
        raise ValueError("legacy belief state requires at least one hypothesis")
    prior_total = sum(max(item.prior, 0.0) for item in state.hypotheses)
    priors = (
        [item.prior / prior_total for item in state.hypotheses]
        if prior_total > 0
        else [1.0 / len(ids)] * len(ids)
    )
    frame = TaskFrame(
        task_frame_id=f"{state.run_id}_legacy_task_frame",
        task_kind=TaskKind.DECISION,
        normalized_question="Legacy categorical BayesProbe state.",
        task_context="",
        answer_contract=AnswerContract(
            objective="Preserve legacy categorical belief behavior.",
            required_sections=["answer", "uncertainty"],
            decision_form="legacy_selection",
            permits_synthesis=False,
        ),
        hypothesis_frame=HypothesisFrame(
            frame_id=f"{state.run_id}_legacy_hypothesis_frame",
            relation=HypothesisRelation.EXCLUSIVE_EXHAUSTIVE,
            hypotheses=[
                FramedHypothesis(
                    id=item.id,
                    statement=item.statement,
                    type=item.type,
                    scope=item.scope,
                    initial_prior=priors[index],
                    falsifiers=list(item.falsifiers)
                    or [f"A reliable result falsifies legacy hypothesis {item.id}."],
                    predictions=list(item.predictions)
                    or [f"A reliable result supports legacy hypothesis {item.id}."],
                )
                for index, item in enumerate(state.hypotheses)
            ],
            rival_sets=_rival_sets(ids, HypothesisRelation.EXCLUSIVE_EXHAUSTIVE),
            coverage_statement="Migrated legacy categorical hypothesis set.",
            coverage_limitation="Relation was assigned by the versioned legacy migration.",
        ),
        framing_method=FramingMethod.LEGACY_MIGRATION,
        framing_trace={"migration": "legacy_categorical_v0.1"},
    )
    return state.model_copy(update={"task_frame": frame})
```

Call this migration once at the start of `BayesProbeCore.integrate_cycle`, before invoking the evidence gate or solver. The solver itself never infers a relation. Add a core regression test proving a legacy state returns with `framing_method=legacy_migration` and still sums to one.

- [ ] **Step 6: Make summaries and normalization relation-aware**

Change every call in `core.py` and `hypothesis_evolution.py` to pass the explicit relation. `HypothesisEvolutionEngine.evolve` rejects relation-less direct inputs; update its unit-test fixture to attach an explicit exclusive TaskFrame. Extend summaries:

```python
if relation == HypothesisRelation.INDEPENDENT:
    summary = {
        "hypothesis_relation": relation.value,
        "belief_measure": "credence",
        "top_hypothesis": top.id,
        "top_credence": top.posterior,
        "runner_up_hypothesis": runner_up.id if runner_up else None,
        "credence_gap": round(top.posterior - runner_up.posterior, 6) if runner_up else top.posterior,
        "total_active_credence": round(sum(posteriors), 6),
    }
    uncertainty = (
        f"{top.id} has the highest current credence, but independent hypotheses may coexist; "
        "ranking does not by itself select the answer."
    )
```

Exclusive summaries retain `top_posterior`, `posterior_gap`, entropy, and `total_active_posterior`, and add `hypothesis_relation="exclusive_exhaustive"` plus `belief_measure="posterior_mass"`.

Use the same `summarize_hypotheses(hypotheses, relation=relation)` function when the initializer creates Belief State 0, then merge initialization audit keys into that summary. Add an initializer test proving three independent hypotheses begin at `0.5` each with `total_active_credence == 1.5` and are never displayed as posterior mass.

In `projections.py`, make `_posterior_summary_text` prefix independent values with `Credences (not normalized):` and exclusive values with `Posterior mass:`. For independent frames, `_main_uncertainty_text` returns the relation-aware `belief_state.uncertainty_summary` instead of calculating a categorical posterior gap. This is a terminology correction only; synthesis projection remains deferred.

Both autonomous runner implementations currently interpret `confidence_threshold` as a categorical top-winner condition. Preserve that stop only for `exclusive_exhaustive`. For `independent` frames, `_confidence_reached` returns `False`; max-cycle, no-input/no-probe, and credence-stability stops remain active until task-aware completion criteria are introduced. Add one test to each runner suite with two credences above the threshold and assert the run does not stop with `confidence_reached`.

- [ ] **Step 7: Render credence without implying mass**

In `renderBeliefs`, branch on `beliefState.task_frame.hypothesis_frame.relation`:

```javascript
const relation = beliefState?.task_frame?.hypothesis_frame?.relation ||
  "exclusive_exhaustive";
if (relation === "independent") {
  beliefPanel.appendChild(
    kv("Total credence (not normalized)", formatNumber(summary.total_active_credence))
  );
  beliefPanel.appendChild(
    kv(
      "Top / credence gap",
      `${summary.top_hypothesis || "n/a"} / ${formatNumber(summary.credence_gap)}`
    )
  );
} else {
  beliefPanel.appendChild(
    kv("Posterior mass", formatNumber(summary.total_active_posterior))
  );
  beliefPanel.appendChild(
    kv(
      "Top / posterior gap",
      `${summary.top_hypothesis || "n/a"} / ${formatNumber(summary.posterior_gap)}`
    )
  );
}
```

Change the answer-panel label `Posterior summary` to the relation-neutral `Belief summary`. Add a JavaScript test that renders an independent Belief State with credences `0.8` and `0.7`, then asserts the first label is `Total credence (not normalized)` and the displayed total is `1.500`.

- [ ] **Step 8: Run relation-focused regressions**

Run: `pytest tests/test_belief.py tests/test_core_cycles.py tests/test_hypothesis_evolution.py tests/test_controllers.py tests/test_autonomous_runner.py tests/test_question_runner.py tests/test_synchronized_runner.py -q`

Run: `node --test tests/test_webui_stream.js`

Expected: categorical regressions remain green; independent hypotheses do not sum to one unless coincidentally; legacy states migrate explicitly.

- [ ] **Step 9: Commit relation-aware belief revision**

```bash
git add bayesprobe/schemas.py bayesprobe/task_framing.py bayesprobe/belief.py bayesprobe/core.py bayesprobe/hypothesis_evolution.py bayesprobe/initialization.py bayesprobe/projections.py bayesprobe/runners.py bayesprobe/question_runner.py bayesprobe/webui_static/app.js tests/test_belief.py tests/test_core_cycles.py tests/test_controllers.py tests/test_autonomous_runner.py tests/test_hypothesis_evolution.py tests/test_initialization.py tests/test_question_runner.py tests/test_synchronized_runner.py tests/test_webui_stream.js
git commit -m "feat: preserve hypothesis relation in belief updates"
```

---

### Task 7: Add the Recorded Open-Question Vertical Slice

**Files:**
- Create: `tests/fixtures/open_questions/model_scale_validation_v0.1.json`
- Modify: `tests/test_question_runner.py`
- Modify: `tests/test_recorded_model_gateway.py`
- Modify: `docs/ARCHITECTURE.md`

**Interfaces:**
- Consumes: `RecordedModelGateway` through `ModelTaskFramer`, `ModelBackedProbeToolGateway`, and `BayesProbeCore`.
- Produces: a secret-free end-to-end fixture proving `frame_open_question -> execute_probe -> judge_evidence` through the public autonomous runner.

- [ ] **Step 1: Create the recorded fixture and failing end-to-end test**

Create this provider-shaped fixture:

```json
{
  "fixture_name": "model_scale_validation_v0.1",
  "metadata": {
    "language": "zh-CN",
    "purpose": "open-question task-framing regression"
  },
  "responses": [
    {
      "match": {"task": "frame_open_question"},
      "response": {
        "task_kind": "claim_verification",
        "answer_contract": {
          "objective": "设计能够区分规模效应与混杂因素的验证方案。",
          "required_sections": [
            "hypotheses",
            "experimental_design",
            "controls",
            "metrics",
            "decision_rule",
            "limitations"
          ],
          "decision_form": "experimental_protocol",
          "permits_synthesis": true
        },
        "hypothesis_relation": "independent",
        "hypotheses": [
          {
            "statement": "在任务、脚手架与推理预算匹配时，模型规模仍有独立正向效应。",
            "type": "causal_claim",
            "scope": "匹配资源条件下的真实 agent 任务分布。",
            "falsifiers": ["预注册的规模效应不显著、为负或低于实际意义阈值。"],
            "predictions": ["匹配条件后表现仍随规模提高。"]
          },
          {
            "statement": "观察到的规模收益主要来自数据、脚手架或推理预算混杂。",
            "type": "confounding_explanation",
            "scope": "资源未匹配的模型比较。",
            "falsifiers": ["控制混杂并完成消融后规模效应保持稳定。"],
            "predictions": ["匹配资源后表观规模收益明显缩小。"]
          },
          {
            "statement": "规模效应取决于任务和工具条件，并非单调或普遍成立。",
            "type": "boundary_condition",
            "scope": "不同任务类型、工具可靠性与交互长度。",
            "falsifiers": ["所有预注册任务层都呈现稳定同向效应。"],
            "predictions": ["规模与任务或工具条件存在显著交互。"]
          }
        ],
        "coverage_statement": "覆盖独立规模效应、主要混杂解释和条件性交互。",
        "coverage_limitation": "不穷尽训练数据污染和评测泄漏等失效模式。"
      }
    },
    {
      "match": {"task": "execute_probe"},
      "response": {
        "raw_content": "MODEL REASONING: 应采用匹配脚手架与推理预算的分层随机或配对设计，并预注册交互项。"
      }
    },
    {
      "match": {"task": "judge_evidence"},
      "response": {
        "evidence_type": "supporting",
        "likelihoods": {
          "H1": "weakly_confirming"
        },
        "interpretation": "该信号提出设计方向，但不是外部实验结果。",
        "quality_overrides": {
          "independence": 0.2,
          "verifiability": 0.2
        }
      }
    }
  ]
}
```

Add an end-to-end test:

```python
def test_recorded_open_question_frames_before_running_cycle():
    gateway = RecordedModelGateway.from_json(
        Path("tests/fixtures/open_questions/model_scale_validation_v0.1.json")
    )
    runner = AutonomousQuestionRunner(
        core=BayesProbeCore(model_gateway=gateway),
        initializer=BayesProbeInitializer(task_framer=ModelTaskFramer(gateway)),
        executor=ProbeExecutor(ModelBackedProbeToolGateway(gateway)),
        config=AutonomousQuestionRunConfig(max_cycles=1, max_probes_per_cycle=1),
    )
    result = runner.run_question(
        InitializeRunInput(
            run_id="recorded_model_scale",
            problem="某团队认为‘模型变大一定能提升 agent 的真实任务表现’。这个命题应该如何验证？",
        )
    )

    assert [request.task for request in gateway.requests] == [
        "frame_open_question",
        "execute_probe",
        "judge_evidence",
    ]
    statements = [item.statement for item in result.task_frame.hypothesis_frame.hypotheses]
    assert len(statements) == len(set(statements)) == 3
    assert all("这个命题应该如何验证" not in statement for statement in statements)
    assert result.initial_belief_state.task_frame == result.task_frame
    assert result.cycle_results[0].signals[0].source_type == "model_probe_gateway"
    final_by_id = result.final_belief_state.hypotheses_by_id()
    assert final_by_id["H1"].posterior > 0.5
    assert final_by_id["H2"].posterior == 0.5
    assert final_by_id["H3"].posterior == 0.5
    assert sum(item.posterior for item in final_by_id.values()) > 1.0
```

- [ ] **Step 2: Run the recorded vertical slice and verify RED**

Run: `pytest tests/test_recorded_model_gateway.py tests/test_question_runner.py::test_recorded_open_question_frames_before_running_cycle -q`

Expected: the new test fails until Tasks 1-5 are complete.

- [ ] **Step 3: Make fixture matching and assertions pass**

Use the existing task-only matching behavior in `RecordedModelGateway`; do not add question-text matching or put user text into fixture metadata. Add a fixture test that recursively scans the JSON for secret-like keys and values beginning with `sk-`.

- [ ] **Step 4: Correct architecture status**

In `docs/ARCHITECTURE.md`, record these exact capability boundaries:

```text
Implemented: explicit/model/recorded TaskFrame before Belief State creation.
Implemented: fail-closed open framing with one structured repair.
Implemented: explicit categorical and independent belief-update semantics.
Not yet implemented: cross-cycle Evidence Memory.
Not yet implemented: task-aware ProbeDesigner and open Answer Projection.
```

Do not publish a new global completion percentage in this milestone.

- [ ] **Step 5: Run Milestones 1-2 verification**

Run: `pytest tests/test_task_framing.py tests/test_initialization.py tests/test_question_runner.py tests/test_webui.py tests/test_openai_gateway.py tests/test_recorded_model_gateway.py -q`

Run: `node --test tests/test_webui_stream.js`

Run: `pytest -q`

Run: `git diff --check`

Expected: all offline tests pass, explicitly enabled live tests remain skipped, and the diff check is clean.

- [ ] **Step 6: Commit the recorded vertical slice and status correction**

```bash
git add tests/fixtures/open_questions/model_scale_validation_v0.1.json tests/test_question_runner.py tests/test_recorded_model_gateway.py docs/ARCHITECTURE.md
git commit -m "test: freeze open task framing vertical slice"
```

---

## Milestones 1-2 Exit Gate

Milestones 1-2 are complete only when all of these assertions are evidenced by tests:

1. Provider-backed unseeded open questions call `frame_open_question` before Belief State creation.
2. Explicit MCQ choices and seeded hypotheses remain deterministic and require no framing model call.
3. Deterministic unseeded open questions fail clearly and never become generic H1/H2.
4. Invalid model frames receive exactly one repair and then fail closed.
5. Framing progress contains no Belief State; initialization progress contains the first Belief State.
6. The recorded Chinese regression creates three distinct semantic hypotheses through the public autonomous runner.
7. Independent hypotheses retain independent credences through the recorded cycle and are not cross-normalized.
8. Provider keys and secret-like fields are absent from TaskFrame, fixtures, ledger output, telemetry, stream events, and error payloads.
9. The complete offline suite remains green.

## Subsequent Plans

After this gate passes, write and execute separate plans in this order:

1. Full evidence-judgment hypothesis semantics, Signal Provenance, and cross-cycle Evidence Memory.
2. ProbeDesigner, capability registry, and autonomous/synchronized candidate feedback repair.
3. History-aware semantic evolution and selection/synthesis/abstention projection.
4. One-time-key smoke, open-question pilot, and final architecture status correction.
