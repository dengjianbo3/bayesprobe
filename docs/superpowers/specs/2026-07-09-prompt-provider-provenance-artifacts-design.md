# Prompt Provider Provenance Artifacts v0.1 Design

Date: 2026-07-09

## Context

BayesProbe now has a stable experiment artifact directory containing manifest,
report, ledger, config snapshot, and dataset snapshot. The evidence pipeline
already records model invocation metadata on `EvidenceEvent.model_trace`,
including task, adapter kind, prompt id/version, schema name/version, repair
attempt index, and request metadata.

The next engineering gap is to surface those invocation traces at the experiment
artifact level. Provider-backed benchmark runs should answer a simple audit
question without requiring manual JSONL inspection: which model adapter,
prompt/schema versions, and model invocation tasks were used in this run?

This slice strengthens reproducibility and provider-backed experiment auditing.
It does not introduce a provider registry, prompt registry, record/replay model
fixtures, or any new BayesProbe control-flow semantics.

## Goals

- Add a `model_invocations.json` artifact derived from `ledger.jsonl`.
- Summarize prompt/schema/model invocation provenance in `manifest.json`.
- Preserve the existing evidence ledger as the source of truth.
- Keep the implementation deterministic and offline-testable.
- Add a default-skipped provider-backed benchmark smoke test for artifact
  provenance.

## Non-Goals

- No changes to `BayesProbeCore`, evidence integration, posterior updates, or
  probe control flow.
- No provider registry redesign.
- No prompt registry implementation.
- No recorded provider-response replay format.
- No default network calls in tests.
- No semantic benchmark scoring changes.

## Artifact Shape

When `artifact_dir` is configured, the artifact bundle gains:

- `model_invocations.json`

The file is derived from the copied `ledger.jsonl` and contains:

```json
{
  "artifact_version": "0.1",
  "invocation_count": 2,
  "invocations": [
    {
      "task": "judge_evidence",
      "adapter_kind": "scripted",
      "prompt_id": "evidence_judgment",
      "prompt_version": "v0.1",
      "schema_name": "EvidenceJudgment",
      "schema_version": "v0.1",
      "repair_attempt_index": null,
      "metadata": {},
      "occurrence_count": 1
    }
  ]
}
```

`invocations` is a stable, sorted list of unique invocation signatures. The
sort order is lexical by task, adapter kind, prompt id/version, schema
name/version, repair attempt index, and sanitized metadata.

## Manifest Additions

`manifest.json` gains:

- `model_invocations_path`
- `model_invocation_count`
- `model_invocation_summary`

`model_invocation_summary` uses the same aggregation records as
`model_invocations.json`, but it may omit verbose fields in future versions. In
v0.1 it mirrors the aggregated invocation list so the manifest is useful on its
own.

## Source of Truth

The provenance artifact is derived only from ledger records:

- include records where `record_type == "evidence_event"`;
- read `payload.model_trace`;
- ignore empty `model_trace` objects;
- ignore records without a model trace.

This preserves the existing separation:

- model invocation trace belongs to evidence production;
- experiment artifacts summarize already-recorded evidence events;
- no artifact code reaches into the core runner internals.

## Secret Handling

Invocation metadata is sanitized before writing artifacts. It reuses the same
secret-key redaction behavior as experiment metadata:

- common key forms such as `api_key`, `apiKey`, `APIKEY`, `token`, and `secret`
  are removed recursively;
- safe metadata fields remain;
- raw model responses are not written.

## Provider-Backed Smoke Test

Add a live OpenAI benchmark smoke test that is skipped unless both are true:

- `BAYESPROBE_RUN_OPENAI_LIVE=1`
- `OPENAI_API_KEY` is set

The smoke test should:

- run the existing toy benchmark or a one-sample fixture through
  `kind="openai"`;
- enable `artifact_dir`;
- assert `model_invocations.json` exists;
- assert manifest model invocation metadata includes `adapter_kind == "openai"`
  or the OpenAI adapter's stable kind;
- avoid asserting exact model text or benchmark accuracy beyond basic run
  completion.

The live test is an integration smoke only. Offline tests remain the primary
regression suite.

## Testing

Tests will be written first.

Coverage:

- artifact writer creates `model_invocations.json` from ledger evidence-event
  model traces;
- empty ledgers produce a valid empty `model_invocations.json`;
- duplicate traces aggregate with `occurrence_count`;
- repair traces preserve `repair_attempt_index`;
- metadata secret fields are redacted from invocation artifacts and manifest;
- manifest includes `model_invocations_path`, `model_invocation_count`, and
  `model_invocation_summary`;
- default-skipped OpenAI smoke does not run without explicit environment opt-in.

## Documentation

Update `docs/ARCHITECTURE.md`:

- mark prompt/model invocation artifact summary v0.1 as implemented;
- keep full prompt registry snapshots and provider registry as future work;
- clarify that provenance artifacts summarize existing ledger model traces.

## Follow-Up Work

- Prompt registry snapshot files with prompt template checksums.
- Provider registry and provider observability metrics.
- Recorded provider-backed benchmark fixtures.
- Dataset splits and sample filters.
- Richer benchmark suites for final-answer utility and belief-state revision
  quality.
