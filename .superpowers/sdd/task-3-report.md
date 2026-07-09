# Task 3 Report: Evidence Gate Invocation Trace Propagation

## Status

DONE

## Scope Completed

- Added RED tests for:
  - valid direct judgment trace propagation
  - schema-violation trace retention
  - repaired judgment trace propagation
  - projection decomposition preserving empty traces
- Wired direct evidence judgment requests to include prompt/schema metadata.
- Wired repair requests to include prompt/schema metadata plus `metadata["repair_attempt_index"]`.
- Propagated `ModelInvocationTrace` onto direct evidence events, including schema-violation discard events.
- Kept projection decomposition events on empty `model_trace`.

## RED Evidence

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_core_cycles.py::test_direct_signal_valid_judgment_records_model_trace tests/test_core_cycles.py::test_direct_signal_schema_violation_records_judge_model_trace tests/test_core_cycles.py::test_direct_signal_repaired_judgment_records_repair_model_trace tests/test_core_cycles.py::test_projection_decomposition_events_keep_empty_model_trace -q -p no:cacheprovider
```

Result:

- `3 failed, 1 passed`
- Failures showed:
  - `request.prompt_id` was `None` for direct judgment requests
  - `request.prompt_id` was `None` for repair requests
  - `event.model_trace` was empty on schema-violation path

## GREEN Evidence

Focused command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_core_cycles.py -q -p no:cacheprovider
```

Result:

- `36 passed in 0.08s`

Full suite command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
```

Result:

- `217 passed in 0.32s`

## Implementation Notes

- Introduced a private `_EvidenceJudgmentFailure` wrapper to carry both validation error and model trace through discard handling without changing broader interfaces.
- Added `_model_trace_for_request(...)` to centralize `ModelInvocationTrace.from_request(...)` plus adapter-kind resolution.
- Updated direct judgment request construction to set:
  - `prompt_id="evidence_judgment"`
  - `prompt_version="v0.1"`
  - `schema_name="EvidenceJudgment"`
  - `schema_version="v0.1"`
- Updated repair request construction to set:
  - `prompt_id="evidence_judgment_repair"`
  - `prompt_version="v0.1"`
  - `schema_name="EvidenceJudgment"`
  - `schema_version="v0.1"`
  - `metadata={"repair_attempt_index": attempt_index}`
- Ensured:
  - normal direct evidence events carry the judgment trace
  - schema-violation discard events retain the trace from the failing invocation
  - repaired evidence events carry the repair trace
  - projection decomposition events still emit `{}`

## Self-Review

- Stayed within owned files only:
  - `bayesprobe/evidence.py`
  - `tests/test_core_cycles.py`
- Did not change:
  - posterior update math
  - projection decomposition behavior
  - probe planning/execution
  - ledger serialization
- Existing deterministic and scripted gateway payload behavior remains unchanged apart from added request metadata and event trace propagation.
- `StructuredModelRequest(task=..., input=...)` remains backward-compatible.

## Commit

- `feat: trace model invocations on evidence events`
