# Task 1 Report

## Status

Implemented support for OpenAI Responses `base_url` plus request-scoped API keys without persisting raw keys into experiment config snapshots or manifests.

## Files Changed

- `bayesprobe/openai_gateway.py`
- `bayesprobe/model_gateway.py`
- `bayesprobe/config.py`
- `bayesprobe/experiment_artifacts.py`
- `tests/test_openai_gateway.py`
- `tests/test_public_api_and_config.py`
- `tests/test_experiment_artifacts.py`

## RED

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tests/test_openai_gateway.py::test_openai_model_gateway_config_accepts_base_url \
  tests/test_openai_gateway.py::test_openai_responses_model_gateway_uses_request_scoped_key_and_base_url \
  tests/test_public_api_and_config.py::test_experiment_config_from_mapping_parses_openai_base_url \
  tests/test_experiment_artifacts.py::test_artifact_snapshot_includes_openai_base_url_without_raw_api_key \
  -q -p no:cacheprovider
```

Exit code: `1`

Output:

```text
FFFF                                                                     [100%]
=================================== FAILURES ===================================
TypeError: OpenAIModelGatewayConfig.__init__() got an unexpected keyword argument 'base_url'
AttributeError: 'ModelGatewayConfig' object has no attribute 'base_url'
KeyError: 'base_url'
4 failed in 0.11s
```

## GREEN

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tests/test_openai_gateway.py \
  tests/test_public_api_and_config.py \
  tests/test_experiment_artifacts.py \
  -q -p no:cacheprovider
```

Exit code: `0`

Output:

```text
........................................................................ [ 96%]
...                                                                      [100%]
75 passed in 0.12s
```

## Extra Verification

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
```

Exit code: `0`

Output:

```text
........................................................................ [ 24%]
........................................................................ [ 49%]
..............................................................ss........ [ 74%]
........................................................................ [ 99%]
..                                                                       [100%]
288 passed, 2 skipped in 0.57s
```

## Concerns

None.
