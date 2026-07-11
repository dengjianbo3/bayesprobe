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

## Review Fix 2

### Changed Files

- `bayesprobe/task_admission.py`
- `bayesprobe/task_framing.py`
- `tests/test_task_admission.py`
- `tests/test_task_framing.py`
- `.superpowers/sdd/task-2-report.md`

### RED Evidence

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_task_admission.py tests/test_task_framing.py -q -p no:cacheprovider -k 'invalid_explicit_material or simultaneous_choices_and_seeds or valid_hypothesis_seeds or candidates_accept_values_matching_contract_type or provider_replacement_contract or recorded_exact_answer_candidates or recorded_frame_contract'
13 failed, 7 passed, 104 deselected in 0.24s
exit 1
```

The failures demonstrated that underspecified and malformed explicit material
bypassed model admission, simultaneous choices and seeds were silently treated
as MCQ input, provider replacement contracts were accepted, and recorded null,
duplicate, type-mismatched, and admission-incompatible exact-answer frames were
accepted.

### GREEN Evidence

Regression slice:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_task_admission.py tests/test_task_framing.py -q -p no:cacheprovider -k 'invalid_explicit_material or simultaneous_choices_and_seeds or valid_hypothesis_seeds or candidates_accept_values_matching_contract_type or provider_replacement_contract or recorded_exact_answer_candidates or recorded_frame_contract'
20 passed, 104 deselected in 0.14s
exit 0
```

Owned admission and framing tests:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_task_admission.py tests/test_task_framing.py -q -p no:cacheprovider
124 passed in 0.23s
exit 0
```

Complete Task 2 focused suite:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_task_admission.py tests/test_task_framing.py tests/test_initialization.py tests/test_openai_gateway.py tests/test_recorded_model_gateway.py tests/test_question_runner.py -q -p no:cacheprovider
268 passed in 0.40s
exit 0
```

### Full Offline Suite

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
880 passed, 10 skipped in 7.64s
exit 0
```

### Diff Hygiene

```text
git diff --check
exit 0
```

### Self-Review

- Model and recorded framing now share task-kind, contract-continuity,
  hypothesis-count, and exact-answer presence/type/uniqueness validation.
- Exact-answer contracts preserve admitted objective terms and required sections;
  open domain contracts retain their intended outline-to-native refinement.
- Explicit framing materializes its contract from the admitted outline, so a
  recorded native frame cannot inherit a locally replaced contract.
- Routing invokes explicit admission only for a validated two-to-six choice or
  hypothesis frame, routes malformed/underspecified single-mode material to
  model admission, and rejects simultaneous modes without a model call.
- One-repair behavior, recursive secret validation, v0.1 recorded migration
  acceptance, and valid explicit MCQ/hypothesis-seed no-model-call paths remain
  covered and passing.

### Concerns

None. WebUI tagged-result serialization and initialization probe design remain
unchanged under their stated plan boundaries.

## Review Fix 3

### Changed Files

- `bayesprobe/webui.py`
- `bayesprobe/webui_static/app.js`
- `tests/test_webui.py`
- `tests/test_webui_stream.js`
- `.superpowers/sdd/task-2-report.md`

### RED Evidence

Tagged serializer, HTTP, stream, state-absence, ordering, and recursive secret
safety regressions:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_webui.py -q -p no:cacheprovider -k 'tagged_admission_outcome or recursively_redacts_secrets'
7 failed, 87 deselected in 0.29s
exit 1
```

Browser terminal-event compatibility:

```text
node --test tests/test_webui_stream.js
14 passed, 1 failed in 102.02ms
exit 1
```

The Python failures showed completed-run-only attribute access, HTTP 500
responses, and an empty early-outcome stream. The Node failure showed that the
browser rejected terminal `task_admission_completed` at EOF.

### GREEN Evidence

Targeted regression slice:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_webui.py -q -p no:cacheprovider -k 'tagged_admission_outcome or recursively_redacts_secrets'
7 passed, 87 deselected in 0.18s
exit 0
```

Complete WebUI Python suite:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_webui.py -q -p no:cacheprovider
94 passed in 5.56s
exit 0
```

Complete Task 2 focused Python suite:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_task_admission.py tests/test_task_framing.py tests/test_initialization.py tests/test_openai_gateway.py tests/test_recorded_model_gateway.py tests/test_question_runner.py -q -p no:cacheprovider
268 passed in 0.42s
exit 0
```

Node stream suite:

```text
node --test tests/test_webui_stream.js
15 passed in 100.94ms
exit 0
```

### Full Offline Suite

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
887 passed, 10 skipped in 7.62s
exit 0
```

### Diff Hygiene

```text
git diff --check
exit 0
```

### Concerns

None for this compatibility fix. Task 11 still owns admitted-task admission
progress, kernel policy/capability controls, complete v0.2 observability, and
rich admission outcome presentation. The `app.js` change only recognizes the
two tagged admission outcomes as successful terminal events and marks progress
complete.

## Review Fix 4

### Changed Files

- `bayesprobe/task_admission.py`
- `bayesprobe/task_framing.py`
- `tests/test_task_admission.py`
- `tests/test_task_framing.py`
- `tests/test_initialization.py`
- `tests/test_question_runner.py`
- `.superpowers/sdd/task-2-report.md`

### RED Evidence

Initial routing, seed semantics, model/recorded continuity, and full runner
regressions:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_task_admission.py tests/test_task_framing.py tests/test_initialization.py tests/test_question_runner.py -q -p no:cacheprovider -k 'hypothesis_seeds_reject_task_kinds or creates_no_state_for_answer_valued_seed_task_kind or malformed_single_choice_routes or accepts_semantically_paraphrased_objective or accepts_required_section_superset or accepts_paraphrased_objective_and_section_superset'
8 failed, 172 deselected in 0.33s
exit 1
```

The failures showed answer-valued seed kinds were accepted, malformed
single-choice material was recaptured by explicit framing after model
admission, and paraphrased/superset contracts were rejected by both model and
recorded framing.

Router-level fail-closed refinement:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_task_admission.py tests/test_initialization.py -q -p no:cacheprovider -k 'hypothesis_seeds_reject_task_kinds or creates_no_state_for_answer_valued_seed_task_kind'
4 failed, 33 deselected in 0.14s
exit 1
```

This second RED established that returning `can_assess=False` was insufficient:
the routing admitter could still send semantically forbidden seeds to model
admission instead of rejecting them before any model call.

### GREEN Evidence

Focused regressions:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_task_admission.py tests/test_task_framing.py tests/test_initialization.py tests/test_question_runner.py -q -p no:cacheprovider -k 'hypothesis_seeds_reject_task_kinds or creates_no_state_for_answer_valued_seed_task_kind or malformed_single_choice_routes or accepts_semantically_paraphrased_objective or accepts_required_section_superset or accepts_paraphrased_objective_and_section_superset'
8 passed, 172 deselected in 0.19s
exit 0
```

Router-level fail-closed refinement:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_task_admission.py tests/test_initialization.py -q -p no:cacheprovider -k 'hypothesis_seeds_reject_task_kinds or creates_no_state_for_answer_valued_seed_task_kind'
4 passed, 33 deselected in 0.11s
exit 0
```

Complete Task 2 focused suite:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_task_admission.py tests/test_task_framing.py tests/test_initialization.py tests/test_openai_gateway.py tests/test_recorded_model_gateway.py tests/test_question_runner.py -q -p no:cacheprovider
275 passed in 0.41s
exit 0
```

WebUI Python integration suite:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_webui.py -q -p no:cacheprovider
94 passed in 5.57s
exit 0
```

Node stream integration suite:

```text
node --test tests/test_webui_stream.js
15 passed in 101.70ms
exit 0
```

### Full Offline Suite

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
894 passed, 10 skipped in 7.60s
exit 0
```

### Diff Hygiene

```text
git diff --check
exit 0
```

### Concerns

None. Exact-answer continuity now treats objective text as semantic wording,
requires the admitted sections as a subset, and still rejects replacement of
answer value type, decision form, synthesis permission, task kind, answer
relationship, or answer-candidate invariants. Open-task contracts retain their
existing outline-to-native section refinement.

## Review Fix 5

### Changed Files

- `bayesprobe/task_framing.py`
- `tests/test_task_framing.py`
- `.superpowers/sdd/task-2-report.md`

### RED Evidence

Exact-answer shape, admission-owned objective, and finite-number regressions:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_task_framing.py -q -p no:cacheprovider -k 'uses_admitted_objective or invalid_exact_answer_shape or non_finite_number or recorded_exact_answer_rejects_invalid_shape'
15 failed, 9 passed, 114 deselected in 0.31s
exit 1
```

The failures showed that model and recorded contracts retained provider-owned
objectives, model and recorded number candidates accepted `NaN` and infinities,
and recorded exact-answer frames accepted absent or non-default unresolved mass.
The nine passing cases confirmed that existing construction-time validation
already routed invalid competition, coverage, and relationship classifications
through the model repair loop or rejected them for recorded replay.

### GREEN Evidence

Focused approval-gate regressions:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_task_framing.py -q -p no:cacheprovider -k 'uses_admitted_objective or invalid_exact_answer_shape or non_finite_number or recorded_exact_answer_rejects_invalid_shape'
24 passed, 114 deselected in 0.14s
exit 0
```

Complete framing suite:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_task_framing.py -q -p no:cacheprovider
138 passed in 0.23s
exit 0
```

Complete Task 2 focused suite:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_task_admission.py tests/test_task_framing.py tests/test_initialization.py tests/test_openai_gateway.py tests/test_recorded_model_gateway.py tests/test_question_runner.py -q -p no:cacheprovider
297 passed in 0.42s
exit 0
```

WebUI Python integration suite:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_webui.py -q -p no:cacheprovider
94 passed in 5.58s
exit 0
```

Node stream integration suite:

```text
node --test tests/test_webui_stream.js
15 passed in 90.50ms
exit 0
```

Full offline suite:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
916 passed, 10 skipped in 7.63s
exit 0
```

### Concerns

None. Model and recorded framing share one native canonicalization path for
admission continuity, exact-answer shape, and candidate validation. Provider
and recorded `answer_format` values and required-section supersets remain
preserved; only the admitted objective is authoritative in the final contract.

## Review Fix 6

### Changed Files

- `bayesprobe/schemas.py`
- `bayesprobe/task_admission.py`
- `bayesprobe/task_framing.py`
- `tests/test_schemas.py`
- `tests/test_task_admission.py`
- `tests/test_task_framing.py`
- `tests/test_question_runner.py`
- `tests/test_migrations.py`
- `.superpowers/sdd/task-2-report.md`

### RED Evidence

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_schemas.py tests/test_task_admission.py tests/test_task_framing.py tests/test_question_runner.py tests/test_migrations.py -q -p no:cacheprovider -k 'secret_requested_output_shape or secret_material_in_semantic_fields or secret_bearing_decision or secret_bearing_non_admission or admitted_contract_outline or rejects_v01 or skewed_named_priors or recorded_text_answer or equal_float or explicit_migration_keeps'
15 failed, 6 passed, 274 deselected in 0.35s
exit 1
```

The failures showed that the output-shape secret reached the gateway, semantic
decision secrets were accepted and persisted, framing requests omitted the
admitted contract, recorded v0.1 frames were implicitly upgraded, recorded
priors and text values bypassed canonical rules, and numerically equal integer
and float candidates were treated as distinct.

### GREEN Evidence

Focused Review Fix 6 regressions:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_schemas.py tests/test_task_admission.py tests/test_task_framing.py tests/test_question_runner.py tests/test_migrations.py -q -p no:cacheprovider -k 'secret_requested_output_shape or secret_material_in_semantic_fields or secret_bearing_decision or secret_bearing_non_admission or admitted_contract_outline or rejects_v01 or skewed_named_priors or recorded_text_answer or equal_float or explicit_migration_keeps'
21 passed, 274 deselected in 0.22s
exit 0
```

Focused changed suites:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_schemas.py tests/test_task_admission.py tests/test_task_framing.py tests/test_question_runner.py tests/test_migrations.py -q -p no:cacheprovider
295 passed in 0.40s
exit 0
```

Complete Task 2 focused suite:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_task_admission.py tests/test_task_framing.py tests/test_initialization.py tests/test_openai_gateway.py tests/test_recorded_model_gateway.py tests/test_question_runner.py -q -p no:cacheprovider
308 passed in 0.45s
exit 0
```

Migration and schema suites:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_migrations.py tests/test_schemas.py -q -p no:cacheprovider
108 passed in 0.15s
exit 0
```

WebUI Python integration suite:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_webui.py -q -p no:cacheprovider
94 passed in 5.58s
exit 0
```

Node stream integration suite:

```text
node --test tests/test_webui_stream.js
15 passed in 90.41ms
exit 0
```

Full offline suite:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
932 passed, 10 skipped in 7.70s
exit 0
```

Diff hygiene:

```text
git diff --check
exit 0
```

### Concerns

None. Task admission now rejects secret-bearing semantic records before repair
or persistence; framing requests carry only the validated, sanitized admitted
contract outline. Recorded framing is native-v0.2-only while the explicit v0.1
migration API and compatibility wrappers remain covered and unchanged.

## Review Fix 7

### Changed Files

- `bayesprobe/task_admission.py`
- `bayesprobe/question_runner.py`
- `bayesprobe/initialization.py`
- `bayesprobe/task_framing.py`
- `tests/test_question_runner.py`
- `tests/test_initialization.py`
- `tests/test_task_framing.py`
- `.superpowers/sdd/task-2-report.md`

### RED Evidence

Adapter-boundary and hypothesis/answer separation regressions:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_question_runner.py tests/test_initialization.py tests/test_task_framing.py -q -p no:cacheprovider -k 'constructed_secret_bearing_admission or wrong_admitter_return_type or revalidates_direct_admitter_result or revalidates_caller_admission or wrong_direct_admitter_return_type or open_model_hypothesis_answer_value or recorded_open_hypothesis_rejects_answer_value or open_hypotheses_never_acquire'
7 failed, 1 passed, 197 deselected in 0.39s
exit 1
```

The failures showed that bypass-constructed decisions reached framing, wrong
adapter return types reached attribute access, and model and recorded open
hypotheses accepted answer values. The passing runtime regression confirmed
that ordinary explicit open hypotheses already initialized with null answer
values.

### GREEN Evidence

Focused Review Fix 7 regressions:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_question_runner.py tests/test_initialization.py tests/test_task_framing.py -q -p no:cacheprovider -k 'constructed_secret_bearing_admission or wrong_admitter_return_type or revalidates_direct_admitter_result or revalidates_caller_admission or wrong_direct_admitter_return_type or open_model_hypothesis_answer_value or recorded_open_hypothesis_rejects_answer_value or open_hypotheses_never_acquire'
8 passed, 197 deselected in 0.20s
exit 0
```

Owned admission, runner, initialization, and framing suites:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_task_admission.py tests/test_question_runner.py tests/test_initialization.py tests/test_task_framing.py -q -p no:cacheprovider
220 passed in 0.40s
exit 0
```

Complete Task 2 suite:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_task_admission.py tests/test_task_framing.py tests/test_initialization.py tests/test_openai_gateway.py tests/test_recorded_model_gateway.py tests/test_question_runner.py -q -p no:cacheprovider
316 passed in 0.52s
exit 0
```

Migration and schema suites:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_migrations.py tests/test_schemas.py -q -p no:cacheprovider
108 passed in 0.16s
exit 0
```

WebUI Python integration suite:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_webui.py -q -p no:cacheprovider
94 passed in 5.57s
exit 0
```

Node stream integration suite:

```text
node --test tests/test_webui_stream.js
15 passed in 115.96ms
exit 0
```

Full offline suite:

```text
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
940 passed, 10 skipped in 7.72s
exit 0
```

### Concerns

None. Runner and initializer admission boundaries now share one authoritative
revalidation/canonicalization helper with a stable secret-safe failure. Native
model and recorded framing reject answer values for every task kind except
exact-answer and multiple-choice before runtime state creation.
