# Task 2 Report: Task Admission and Native v0.2 Task Framing

## Status

Implementation complete within the Task 2 ownership boundary. The focused Task 2 suite passes, but the required full offline suite is blocked by 39 legacy public/WebUI tests in files that the brief explicitly excludes from Task 2 ownership.

## Implementation

- Added the task-admission deep module with `TaskAdmissionInput`, `TaskAdmitter`, explicit/model/recorded/routing adapters, exact payload validation, one bounded repair, stable safe errors, and sanitized model traces.
- Made `TaskFramingInput` require an admitted `TaskAdmissionDecision` and made all native explicit, model, and recorded framing writes use `TaskFrame.schema_version="v0.2"` with `admission_decision_id`.
- Upgraded model framing to typed answer contracts, answer relationships, independent competition/coverage, server-owned hypothesis IDs/priors, exact-answer candidate values, and exclusive-open `0.50` unresolved mass with one-to-six candidates.
- Added strict OpenAI Responses and Chat Completions schemas/instructions for admission and v0.2 framing.
- Updated initialization to assess once only when no decision is supplied, reject non-admitted decisions before state creation, create v0.2 `FrameState` and empty `EvidenceMemorySnapshot`, and ledger admission/frame/run/state/probes in order.
- Updated the public runner to admit before progress/framing, pass admitted decisions without reassessment, and return tagged `NeedsReframingResult`/`OutOfScopeResult` values with no `BeliefState`.
- Added a v0.2 recorded fixture containing admission, framing, probe, and evidence responses while leaving the v0.1 fixture unchanged.

## Files

Created:

- `bayesprobe/task_admission.py`
- `tests/test_task_admission.py`
- `tests/fixtures/open_questions/model_scale_validation_v0.2.json`

Modified:

- `bayesprobe/task_framing.py`
- `bayesprobe/initialization.py`
- `bayesprobe/openai_gateway.py`
- `bayesprobe/question_runner.py`
- `bayesprobe/__init__.py`
- `tests/test_task_framing.py`
- `tests/test_initialization.py`
- `tests/test_openai_gateway.py`
- `tests/test_recorded_model_gateway.py`
- `tests/test_question_runner.py`

`bayesprobe/model_gateway.py` required no code change: its existing `StructuredModelRequest`, `ModelInvocationTrace`, `ModelGateway`, and scripted adapter contracts already support the new admission tasks without duplication.

## RED Evidence

Admission RED:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_task_admission.py tests/test_question_runner.py -q -p no:cacheprovider
ERROR tests/test_task_admission.py
ERROR tests/test_question_runner.py
ModuleNotFoundError: No module named 'bayesprobe.task_admission'
exit 2
```

Framing RED:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_task_framing.py -q -p no:cacheprovider -k 'exact_answer_framing_preserves_open_coverage or zero_candidate_exact_frame_fails_after_one_repair'
2 failed, 89 deselected
TypeError: TaskFramingInput.__init__() got an unexpected keyword argument 'admission_decision'
exit 1
```

OpenAI schema RED:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_openai_gateway.py -q -p no:cacheprovider -k task_admission
ImportError: cannot import name 'TASK_ADMISSION_DECISION_JSON_SCHEMA'
exit 2
```

Initialization RED:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_initialization.py -q -p no:cacheprovider -k 'uses_supplied_admission_once or default_admitter_fails_closed'
2 failed, 20 deselected
TypeError: BayesProbeInitializer.__init__() got an unexpected keyword argument 'task_admitter'
exit 1
```

Security self-review RED:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_task_framing.py -q -p no:cacheprovider -k constructed_admission_trace
1 failed, 91 deselected
Failed: DID NOT RAISE TaskFramingError
exit 1
```

## GREEN Evidence

Final focused suite:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_task_admission.py tests/test_task_framing.py tests/test_initialization.py tests/test_openai_gateway.py tests/test_recorded_model_gateway.py tests/test_question_runner.py -q -p no:cacheprovider
239 passed in 0.38s
exit 0
```

The focused command was followed by `git diff --check` in the same shell invocation and exited 0 with no output.

## Full Offline Suite

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
39 failed, 812 passed, 10 skipped in 12.79s
exit 1
```

Failure ownership:

- 1 failure in `tests/test_public_api_and_config.py`, which directly constructs the now-required `TaskFramingInput` without an admission decision.
- 38 failures in `tests/test_webui.py`, whose out-of-scope WebUI code directly constructs `TaskFramingInput` without admission and whose provider fakes still return legacy framing as the first model response rather than `assess_task_admission` followed by native v0.2 framing.

Changing the owned Task 2 implementation to accept those legacy paths would violate the brief's requirements that admission precede framing, that `TaskFramingInput` require an admitted decision, and that new native framing writes v0.2. Fixing the full suite requires authorization to modify `bayesprobe/webui.py`, `tests/test_webui.py`, and `tests/test_public_api_and_config.py`, none of which are in the Task 2 file list.

## Self-Review

- Confirmed non-admitted runner outcomes append only `task_admission` and create no TaskFrame, Run, Cycle, or BeliefState.
- Confirmed runner-supplied decisions are passed to the initializer and never reassessed.
- Confirmed exact-answer zero-candidate responses consume one repair and fail before BeliefState creation.
- Confirmed exclusive-open named priors plus unresolved mass total exactly 1.0 and reserve at least 0.05.
- Confirmed provider-owned IDs/priors/posteriors remain rejected.
- Confirmed admission prompts omit passive compatibility context and model metadata.
- Confirmed fixtures and repair payloads are recursively secret-safe.
- Added and passed a regression for malformed constructed admission decisions carrying credential material.
- Confirmed no runtime dependency was added and the v0.1 fixture was not modified.
- Confirmed only Task 2 listed source/test files plus this required report were edited.

## Concerns

The full suite cannot pass without changing files outside the explicit ownership boundary. No compatibility fallback was added because it would reintroduce framing-before-admission or native v0.1 acceptance, directly undoing Task 2's core guarantees.

## Compatibility Fix

Status: complete. WebUI composition now supplies the same Task Admitter to the autonomous runner and initializer: routing explicit/model admission for OpenAI-compatible providers and explicit-only admission for deterministic runs. Deterministic preflight obtains an explicit admitted decision before framing validation and preserves the existing fail-closed message for unseeded open questions.

Provider-backed WebUI fixtures now return `assess_task_admission` before native v0.2 `frame_open_question` payloads. The public SDK recorded-framing test constructs a real explicit admission decision and passes it to both `TaskFramingInput` and initializer replay. No admission bypass, native v0.1 fallback, Task 11 rendering, static controls, or stream semantics were added.

Compatibility-focused suites:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_webui.py tests/test_public_api_and_config.py -q -p no:cacheprovider
128 passed in 5.61s
exit 0
```

Original Task 2 focused suite:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_task_admission.py tests/test_task_framing.py tests/test_initialization.py tests/test_openai_gateway.py tests/test_recorded_model_gateway.py tests/test_question_runner.py -q -p no:cacheprovider
239 passed in 0.36s
exit 0
```

Full offline suite:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
851 passed, 10 skipped in 7.49s
exit 0
```

Diff hygiene:

```text
git diff --check
exit 0
```

Compatibility fix concerns: none.

## Review Fix 1

### Changed Files

- `bayesprobe/task_framing.py`
- `bayesprobe/initialization.py`
- `tests/test_task_framing.py`
- `tests/test_initialization.py`
- `.superpowers/sdd/task-2-report.md`

### RED Evidence

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_task_framing.py tests/test_initialization.py -q -p no:cacheprovider -k 'exact_answer_candidate_type_mismatch or preserves_framed_answer_value'
7 failed, 2 passed, 119 deselected in 0.20s
exit 1
```

The failures showed that contract/type mismatches were accepted without repair and
that the runtime hypothesis received `answer_value=None` instead of the framed
integer. The two boolean cases passed because the existing scalar validator already
rejected booleans before contract-aware validation.

### GREEN Evidence

Regression slice after implementation and fail-closed enum refinement:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_task_framing.py tests/test_initialization.py -q -p no:cacheprovider -k 'exact_answer_candidate or preserves_framed_answer_value'
14 passed, 114 deselected in 0.14s
exit 0
```

Focused Task 2 suite:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_task_admission.py tests/test_task_framing.py tests/test_initialization.py tests/test_openai_gateway.py tests/test_recorded_model_gateway.py tests/test_question_runner.py -q -p no:cacheprovider
253 passed in 0.38s
exit 0
```

Full offline suite:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
865 passed, 10 skipped in 7.57s
exit 0
```

Diff hygiene:

```text
git diff --check
exit 0
```

### Concerns

None. Initial open-probe discrimination and frame coverage remain unchanged for
Task 5, per the explicit plan-boundary decision.
