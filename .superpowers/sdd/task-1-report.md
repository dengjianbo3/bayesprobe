# Task 1 Report

## What changed
- Extended `StructuredModelRequest` with optional prompt/schema metadata and validated frozen construction.
- Added `ModelInvocationTrace` with `from_request(...)`, `to_dict()`, and request metadata extraction for `repair_attempt_index`.
- Added stable `adapter_kind` identities for deterministic and scripted gateways plus `model_gateway_adapter_kind(...)`.
- Exported `ModelInvocationTrace` from the package root.

## Verification
- `pytest tests/test_model_gateway.py -q` -> 47 passed
- `pytest tests/test_public_api_and_config.py -q` -> 19 passed
- `pytest tests/test_model_gateway.py tests/test_public_api_and_config.py -q` -> 66 passed
- `pytest -q` -> 211 passed

## Notes
- Existing deterministic and scripted `complete_structured(...)` behavior was left unchanged.
- No provider adapter, transport, retry, or ledger changes were introduced in this slice.
