# Task 4 Report: Signal Provenance and Cross-Cycle Evidence Memory

## Status

Complete. Task 4 is implemented on `codex/epistemic-kernel-completion` from
reviewed HEAD `67abac9`.

## Files

Created:

- `bayesprobe/evidence_memory.py`
- `tests/test_evidence_memory.py`
- `.superpowers/sdd/task-4-report.md`

Modified:

- `bayesprobe/kernel_config.py`
- `bayesprobe/schemas.py`
- `bayesprobe/model_gateway.py`
- `bayesprobe/openai_gateway.py`
- `bayesprobe/evidence.py`
- `bayesprobe/core.py`
- `bayesprobe/belief.py`
- `bayesprobe/probe_executor.py`
- `tests/fixtures/open_questions/model_scale_validation_v0.2.json`
- `tests/test_model_gateway.py`
- `tests/test_openai_gateway.py`
- `tests/test_core_cycles.py`
- `tests/test_probe_executor.py`

No Task 2 admission/framing implementation, Task 5 probe design, Task 6
expansion, WebUI, dependency, search, or retrieval files were changed.

## Implementation

- Added the deep `SignalProvenanceNormalizer.normalize(...)` module with
  Unicode/whitespace canonicalization, SHA-256 content identity, stable
  provider/model/session correlation groups, parent-root preservation,
  deterministic origin mapping, and pre-hash recursive secret rejection.
- Added the deep `EvidenceMemoryManager.classify(...)` and `commit(...)`
  module. It detects exact source/root/content repeats before provider calls,
  zeros same-root restatements, tracks neutral-event correlation history,
  requires distinct source and root for independence, and commits compact
  identity/lifecycle refs once.
- Added validated correlation-credit policy and directional keys of
  `group|subject|confirming-or-disconfirming`. Latent unresolved credit uses
  `frame:<version>:unresolved`; `H_other` is rejected. Saturated events retain
  an Evidence Event with zero weight and no belief/frame update.
- Upgraded native Evidence Judgment and Evidence Event writes to v0.2 with
  strict unresolved/frame-fit coherence, full hypothesis/probe/frame/
  provenance/memory request context, origin quality caps, and reduce-only
  model quality overrides.
- Versioned both OpenAI transport schemas and repairs. Native requests use the
  seven-field v0.2 schema, including explicit seeded frames. Exact legacy
  four-field provider payloads use a named compatibility completion only on
  the explicit legacy-migration route before strict validation.
- Integrated normalized ledger signals and the committed memory snapshot into
  the same recursively revalidated final BeliefState before any cycle append.
  Replayed evidence ids neither recommit credit nor append another Evidence
  Event record.
- Kept probe outputs raw through execution and boundary closure. Native probe
  execution advertises v0.2, while all Active and Passive signals enter the
  same provenance and evidence gate.

## RED Evidence

1. Initial Task 4 focused command stopped with `1 error` because
   `bayesprobe.evidence_memory` did not exist.
2. After adding the deep modules, memory tests produced `2 failed, 3 passed`
   because the integration gate did not accept or return provenance/memory.
3. Judgment/context tests produced `7 failed, 76 passed` because native
   semantic context and v0.2 cross-field parsing were absent.
4. OpenAI transport tests stopped with `1 error` because no versioned v0.1/
   v0.2 evidence schema pair existed.
5. Core transaction tests produced `5 failed, 55 passed`, exposing absent
   snapshot persistence, unnormalized ledger signals, and the missing atomic
   invalid-memory failure path.
6. Native schema tests produced `7 failed, 13 passed`; v0.2 Event and secret-
   free memory invariants were not enforced.
7. Security/probe tests produced `3 failed, 139 passed`; secret-bearing model
   structures were accepted and native probe execution still declared v0.1.
8. The first full offline run produced `14 failed, 1009 passed, 10 skipped`,
   identifying explicit legacy provider-shape compatibility routes.
9. Correlation-history self-review produced `2 failed, 20 passed`; neutral
   events and same-source relabeling were not retained as correlated history.
10. Final provenance/quality self-review produced `4 failed, 22 passed`, then
    a cap regression check produced `3 failed`; supplied model groups, known
    parent roots, pre-judgment remaining credit, and origin caps were tightened
    without changing source-claim or deterministic-tool behavior.

## GREEN Evidence

- Exact Task 4 focused command: `242 passed in 0.49s`.
- Full offline Python suite: `1029 passed, 10 skipped in 8.36s`.
- Node WebUI stream regression: `15 passed, 0 failed`.
- `git diff --check`: clean.

## Self-Review

- Cross-cycle exact duplicates: source/root/content identity skips provider
  judgment, records `duplicate_exact`, has weight zero, and moves no mass.
- Paraphrased same-root evidence: records `correlated_restatement`, forces
  independence and effective weight to zero, and preserves derived roots.
- Credit saturation: named and unresolved subjects are direction-specific;
  saturated events remain ledger-visible and produce no solver updates.
- Unresolved credit: keys use `frame:<version>:unresolved`; schema and tests
  reject `H_other` credit subjects.
- Provider calls and semantics: exact repeats make no second call; other
  judgments receive complete descriptors, probe conditions, provenance, and
  prior memory. Partial v0.2 combinations fail before the solver.
- Ledger order and replay: normalized signal precedes canonical Evidence Event,
  updates/adequacy/evolution, and final state; replay ids append no duplicate
  Evidence Event and recommit no credit.
- Atomic failure: an invalid committed memory snapshot fails recursive final-
  state validation with an empty cycle ledger.
- Migration: v0.1 Event weight fallback and exact legacy provider shape remain
  explicit; native writes require provenance and effective weight.
- Secrets: signal and provider payload checks run before hashing, prompting,
  tracing, repair, memory, or ledger persistence. Repair payloads are redacted;
  request-scoped credentials and headers remain outside structured requests.

## Concerns

No blocking concerns. The v0.1 provider compatibility completion is
intentionally limited to payloads containing only the reviewed legacy evidence
fields; any partial or incoherent v0.2 payload still fails closed.

## Review Fix 1

### Status

Complete. All seven Task 4 review findings were addressed as one
memory/provenance hardening wave from reviewed HEAD `d6032b2`.

### Changes

- Canonicalized correlation groups from the first observed source lineage and
  used that group for both classification credit and committed identities, so
  later same-source group relabeling cannot reset directional credit.
- Classified every signal with declared parents as correlated. Unknown parents
  remain ledger-visible with zero independent weight, while known parents still
  enforce derivation-root preservation.
- Neutralized evidence ids already present in `ledger_refs` before memory
  classification, provider judgment, or commit. Explicit migrated v0.1 states
  with empty memory neither call the provider nor create accepted memory ids or
  duplicate evidence records.
- Applied epistemic-origin caps to every evidence type, including provider-
  labeled `source_claim`, while retaining reduce-only quality overrides.
- Tightened memory snapshots to require coherent identity-map keys, canonical
  SHA-256 source/content identities, exact directional credit-key grammar, and
  hypothesis subjects that exist in the enclosing belief state.
- Added a safe model identity seam. Probe provenance records the adapter and
  configured model identity without credentials, endpoints, or headers;
  deterministic and custom gateways use stable adapter fallbacks.
- Made native judgment parsing reject empty/non-string interpretations and
  boolean, string, non-finite, or out-of-range quality overrides before solver
  use. The native OpenAI schema now requires a non-empty interpretation.

### RED Evidence

- Combined review regression command: `30 failed, 239 passed`. Failures covered
  all seven findings, including cumulative same-source saturation, unknown
  parents, migrated replay, source-claim origin caps, strict snapshot grammar,
  two models behind one adapter, and strict native judgment values.

### GREEN Evidence

- Exact Task 4 focused suite: `269 passed`.
- Schema and migration suite: `113 passed`.
- Full offline Python suite: `1056 passed, 10 skipped`.
- Node WebUI stream regression: `15 passed, 0 failed`.
- `git diff --check`: clean.

### Concerns

No blocking concerns. Model identity deliberately excludes base URLs, API-key
environment names, request headers, and credentials; custom gateways without an
explicit safe identity fall back to their stable adapter identity.

## Review Fix 2

### Status

Complete. All five Task 4 re-review findings were addressed from reviewed HEAD
`92b496b` without adding dependencies or Task 5/6, search, retrieval, or WebUI
runtime behavior.

### Changes

- Made persisted canonical groups provable: a source identity and a derivation
  root may each map to exactly one group. Runtime lineage resolution now uses a
  candidate set, so map insertion order cannot select a group and conflicting
  lineage joins fail closed. Reordered snapshots preserve cumulative saturation.
- Replaced native positional Evidence Event ids with scoped cycle ids containing
  a SHA-256 identity over canonical source/content identity and derivation root,
  plus a per-identity occurrence for true duplicates. Inserted or reordered
  unrelated signals cannot change event identity. Already-used migrated cycles
  with empty identity memory now fail before provider judgment or credit commit.
- Reserved `|` in correlation groups and every named hypothesis-id constructor,
  and reserved the exact `frame:<positive-version>:unresolved` namespace from
  named hypotheses. Snapshot grammar and generated `group|subject|direction`
  keys now share the same component domain.
- Removed legacy completion from native v0.2 judgment validation. Native requests
  require the exact seven-field payload; exact four-field completion runs only on
  the explicit v0.1 migration route. Request and repair traces persist the route,
  lifecycle schema, and frame contract. Deterministic and recorded fixtures emit
  the schema version requested without weakening either OpenAI transport schema.
- Replaced colon-concatenated discard history with strict compact JSON pairs
  `[event_id,reason]`. Snapshot validation requires canonical parse/round-trip,
  and idempotency lookup decodes the exact event id even when it contains colons.

### RED Evidence

1. Canonical-lineage regressions produced `2 failed, 1 passed`: conflicting
   source/root groups were accepted before snapshot validation was tightened.
2. Replay regressions produced `3 failed`: native ids moved under insertion and
   reorder, duplicate occurrences lacked stable identity, and migration-empty
   replay still proceeded. A later same-content/different-root regression also
   failed until derivation root entered the hashed identity.
3. Credit-domain regressions produced `5 failed`: correlation groups, named
   hypotheses, seed ids, and MCQ labels accepted reserved key syntax.
4. Route regressions produced `2 failed`: native v0.2 silently completed an exact
   legacy payload and migrated requests lacked an auditable route marker.
5. Discard regressions produced `7 failed, 1 passed`: legacy concatenation was
   accepted and colon-bearing event ids could not be recommitted idempotently.

### GREEN Evidence

- Exact Task 4 focused suite: `278 passed in 0.53s`.
- Schema, migration, and framing compatibility suite: `271 passed in 0.36s`.
- Full offline Python suite: `1076 passed, 10 skipped in 8.55s`.
- Node WebUI stream regression: `15 passed, 0 failed`.
- `git diff --check`: clean.

### Concerns

No blocking concerns. Legacy four-field judgment completion is intentionally
unavailable to every native v0.2 route; stale provider fixtures were upgraded to
the seven-field contract rather than receiving compatibility preprocessing.

## Review Fix 3

### Status

Complete. The final-gate deterministic recomputation finding was addressed
from reviewed HEAD `dd81971` without changing Task 5/6, WebUI, routing, or
dependencies.

### Changes

- Added one shared deterministic-computation root helper in evidence memory.
  It recursively rejects secret keys and values before hashing, sorts object
  keys, preserves exact Unicode string values, includes only a safe tool
  identity and structured computation inputs, and returns a namespaced SHA-256
  root.
- Stamped deterministic probe results with explicit `TOOL_RESULT` provenance.
  Their roots use method, inquiry goal, sorted targets, support/weaken/reframe
  conditions, probe type, and stable gateway identity, excluding probe, cycle,
  and run ids plus rendered result labels.
- Stamped successful Python sandbox results with explicit `TOOL_RESULT`
  provenance. Their roots use executed code, plan purpose, expected
  observation, sorted targets, and immutable image digest, excluding execution,
  probe, cycle, and run ids plus output-wrapper metadata.
- Kept Python reasoning fallback explicitly `MODEL_REASONING` with the existing
  safe provider/model/session identity. It never uses the deterministic tool
  root path.
- Added gate regressions proving repeated deterministic probes and Python
  computations retain one root across cycles, classify as
  `correlated_restatement`, receive zero independence and effective weight,
  and leave correlation credit unchanged. Changed probe semantics, code, plan
  inputs, and image digest produce different roots.
- Corrected the report: explicit seeded frames are native v0.2 and use the
  seven-field judgment contract. Four-field compatibility exists only on the
  explicit legacy-migration route.

### RED Evidence

1. Shared-helper regression stopped with `1 error` because
   `derive_deterministic_computation_root` did not exist.
2. Deterministic-probe regression produced `1 failed` because gateway signals
   had no explicit provenance.
3. Python provenance regressions produced `2 failed` because successful sandbox
   and model-only reasoning signals had no explicit provenance.

### GREEN Evidence

- Focused provenance/probe/Python suites: `92 passed in 0.26s`.
- Exact Task 4 focused suite: `282 passed in 0.53s`.
- Full offline Python suite: `1081 passed, 10 skipped in 8.58s`.
- Node WebUI stream regression: `15 passed, 0 failed`.
- `git diff --check`: clean.

### Concerns

No blocking concerns. Deterministic roots intentionally describe computation
semantics and environment identity, not rendered output or execution-instance
metadata; changed stable inputs split the root.

## Review Fix 4

### Status

Complete. All seven high-intensity final-gate findings were addressed from
reviewed HEAD `813057a` without adding dependencies or Task 5/6, search,
retrieval, or WebUI behavior.

### Changes

- Preserved deterministic-root string values as exact Unicode while retaining
  canonical object-key ordering. Indentation, line boundaries, literal spaces,
  composed/compatibility characters, and tool identity text now split roots;
  secret checks inspect both exact and separately normalized text before hash.
- Added a sandbox-owned structured execution-policy snapshot to every Python
  execution record. Python roots now include the resolved image digest, user,
  CPU/memory/pids, timeout/output limits, network, read-only root, tmpfs,
  security controls, deterministic environment, and interpreter contract while
  excluding execution, run, cycle, and probe ids.
- Made signal-id commitment compare prior source identity, canonical content
  fingerprint, canonical group, and derivation root before event idempotency,
  credit, or identity-map assignment. Exact lineage reuse remains valid;
  mismatches fail closed.
- Added stable `RecordedModelGateway.model_identity`. Explicit safe fixture,
  provider, and model metadata is preferred; otherwise canonical secret-free
  fixture content is fingerprinted. Paths, transport metadata, headers, base
  URLs, question text, and credentials cannot define the identity.
- Populated native exclusive-open projection sender/source-claim events with
  neutral unresolved likelihood and underdetermined frame fit. Non-exclusive
  projections retain null unresolved likelihood, origin caps, and source
  verification candidates.
- Rechecked canonical source identity and canonical content immediately before
  provenance hashing. Unicode-normalized credential forms now fail before a
  provider request, fingerprint, memory snapshot, or ledger write.
- Removed the explicit-seed probe exclusion. Native v0.2 seeded frames advertise
  the v0.2 probe contract; only explicit legacy-migration framing uses v0.1.

### RED Evidence

1. Exact-string, normalized-secret, and signal-lineage regressions produced
   `11 failed, 57 passed`.
2. Python policy regressions produced `9 failed, 16 passed` because neither the
   snapshot interface nor execution-record policy identity existed.
3. Recorded identity and seeded routing regressions produced
   `5 failed, 24 passed`.
4. Projection and canonical-secret atomicity regressions produced `2 failed`.
5. Fallback local-path exclusion and unmigrated v0.1 probe routing each
   produced `1 failed` before their contracts were completed.

### GREEN Evidence

- Combined changed suites: `390 passed`.
- Exact Task 4 focused suite: `298 passed in 0.57s`.
- Schema, migration, framing, and recorded compatibility: `299 passed in 0.40s`.
- Python evaluation callers: `26 passed in 0.82s`.
- Full offline Python suite: `1103 passed, 10 skipped in 8.70s`.
- Node WebUI stream regression: `15 passed, 0 failed`.
- `git diff --check`: clean.

### Concerns

No blocking concerns. Code that intentionally observes nondeterministic state
such as wall time, randomness, or container hostname remains in the same
factual computation lineage when its exact code, plan, and safe sandbox policy
are unchanged. This is conservative: differing outputs remain ledger-visible
but do not claim fresh independent factual credit merely because the execution
instance changed.

## Review Fix 5

### Changes

- Validated reused signal-id source/content/root/supplied-group lineage directly
  after provenance normalization and again at memory commit through one manager
  method, before replay return, provider access, or ledger/memory mutation.
- Kept accepted neutral events ledger-visible without emitting correlation-credit
  deltas, so all existing directional balances remain unchanged.
- Made native exclusive-open schema-violation events explicitly neutral on
  unresolved mass with an `underdetermined` frame fit.

### Verification

- RED: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_evidence_memory.py::test_accepted_neutral_event_preserves_existing_directional_credit tests/test_core_cycles.py::test_native_exclusive_open_schema_violation_is_neutral_and_underdetermined tests/test_core_cycles.py::test_replayed_signal_id_lineage_conflict_is_atomic_and_skips_provider -q -p no:cacheprovider` -> `3 failed`.
- GREEN regression rerun: the same command -> `3 passed in 0.23s`.
- Task 4 focused: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_evidence_memory.py tests/test_model_gateway.py tests/test_openai_gateway.py tests/test_core_cycles.py tests/test_probe_executor.py -q -p no:cacheprovider` -> `301 passed in 0.64s`.
- Compatibility: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_schemas.py tests/test_migrations.py tests/test_task_framing.py tests/test_recorded_model_gateway.py -q -p no:cacheprovider` -> `299 passed in 0.43s`.
- Full offline: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider` -> `1106 passed, 10 skipped in 8.74s`.
- Node: `node --test tests/test_webui_stream.js` -> `15 passed, 0 failed`.
- `git diff --check` -> clean.

### Concerns

No blocking concerns.

## Review Fix 6

### Changes

- Versioned source-content identity memory from the legacy v1 triple to a v2
  four-part value containing source, content fingerprint, stable canonical
  credit group, and supplied provenance group. Legacy v1 triples remain
  recursively valid and upgrade on the next identity write.
- Added one identity-only `EvidenceMemoryManager` operation used by normal
  commit and event-id replay. Replay now persists a previously unseen signal id
  without changing accepted/discard history, directional credit, or provider
  calls.
- Kept canonical source/root accounting and credit keys on the stable canonical
  group while validating signal-id continuity against the separately persisted
  supplied group.

### Verification

- RED: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_evidence_memory.py::test_supplied_group_replay_is_idempotent_while_credit_stays_canonical tests/test_core_cycles.py::test_replayed_native_evidence_id_does_not_recommit_credit_or_ledger_record tests/test_core_cycles.py::test_replayed_new_signal_identity_is_persisted_then_conflict_fails_atomically tests/test_schemas.py::test_v2_evidence_memory_preserves_canonical_and_supplied_groups tests/test_schemas.py::test_v2_evidence_memory_rejects_legacy_three_part_identity tests/test_schemas.py::test_v1_evidence_memory_identity_remains_compatible tests/test_migrations.py::test_migrates_belief_state_with_frame_and_empty_memory -q -p no:cacheprovider` -> `5 failed, 2 passed in 0.33s`.
- GREEN: the same command -> `7 passed in 0.25s`.
- Strengthened regressions: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_core_cycles.py::test_replayed_new_signal_identity_is_persisted_then_conflict_fails_atomically tests/test_evidence_memory.py::test_supplied_group_replay_is_idempotent_while_credit_stays_canonical tests/test_schemas.py::test_v2_evidence_memory_preserves_canonical_and_supplied_groups tests/test_schemas.py::test_v2_evidence_memory_rejects_legacy_three_part_identity tests/test_schemas.py::test_v2_evidence_memory_keeps_canonical_source_group_invariant tests/test_schemas.py::test_v1_evidence_memory_identity_remains_compatible tests/test_migrations.py::test_migrates_belief_state_with_frame_and_empty_memory -q -p no:cacheprovider` -> `10 passed in 0.27s`.
- Task 4 focused: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_evidence_memory.py tests/test_model_gateway.py tests/test_openai_gateway.py tests/test_core_cycles.py tests/test_probe_executor.py -q -p no:cacheprovider` -> `305 passed in 0.69s`.
- Compatibility: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_schemas.py tests/test_migrations.py tests/test_task_framing.py tests/test_recorded_model_gateway.py -q -p no:cacheprovider` -> `303 passed in 0.41s`.
- Full offline: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider` -> `1114 passed, 10 skipped in 8.87s`.
- Node: `node --test tests/test_webui_stream.js` -> `15 passed, 0 failed`.
- `git diff --check` -> clean.

### Self-Review

- New-id event replay writes only the three identity maps and memory version;
  accepted/discard lifecycle refs, directional credit, discovery/counterevidence
  refs, and provider count remain unchanged. The returned belief state carries
  S2, and changed S2 lineage fails before provider or ledger mutation.
- Canonical and supplied groups remain distinct: source/root conflict checks and
  all credit keys use the canonical field, identical supplied-group replay is
  idempotent, and changing the supplied group for a recorded id fails closed.
  V1 migration/default snapshots remain loadable and upgrade to exact v2 shape.

### Concerns

No blocking concerns.

## Review Fix 7

### Changes

- Added optional `supplied_correlation_group` provenance. Model-origin
  `correlation_group` remains the server-derived provider/model/session group;
  caller input is persisted separately for audit and signal-id continuity and
  never selects independence or directional credit keys.
- Restricted evidence memory to versions 1 and 2 at direct and recursive schema
  validation. Identity operations also reject bypass-constructed unsupported
  snapshots, upgrade v1 to v2, and write exactly version 2.
- Normalized every signal exactly once up front and preflighted the full batch
  through a cycle-local identity-only shadow before provider access. The shadow
  is discarded; normal classification, replay, credit, and event ordering still
  start from the prior working memory.

### Verification

- RED: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_evidence_memory.py::test_supplied_model_group_cannot_override_stable_provider_session_group tests/test_evidence_memory.py::test_batch_preflight_normalizes_once_and_stops_before_provider tests/test_evidence_memory.py::test_identity_write_rejects_unsupported_memory_version tests/test_evidence_memory.py::test_v1_identity_write_upgrades_all_identities_to_v2 tests/test_evidence_memory.py::test_native_belief_state_rejects_unsupported_memory_version tests/test_schemas.py::test_evidence_memory_rejects_unsupported_versions tests/test_core_cycles.py::test_model_supplied_group_is_audited_and_changed_reuse_fails_atomically tests/test_core_cycles.py::test_later_cross_cycle_batch_conflict_preflights_before_provider_or_ledger tests/test_core_cycles.py::test_same_batch_reused_signal_conflict_preflights_atomically -q -p no:cacheprovider` -> `13 failed, 3 passed in 0.47s`.
- GREEN: the same command -> `16 passed in 0.28s`.
- Task 4 focused: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_evidence_memory.py tests/test_model_gateway.py tests/test_openai_gateway.py tests/test_core_cycles.py tests/test_probe_executor.py -q -p no:cacheprovider` -> `317 passed in 0.70s`.
- Compatibility: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_schemas.py tests/test_migrations.py tests/test_task_framing.py tests/test_recorded_model_gateway.py -q -p no:cacheprovider` -> `306 passed in 0.41s`.
- Full offline: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider` -> `1129 passed, 10 skipped in 8.89s`.
- Node: `node --test tests/test_webui_stream.js` -> `15 passed, 0 failed`.
- `git diff --check` -> clean.

### Self-Review

- Model provenance now records both facts: distinct caller groups survive in
  ledger and v2 identity field four, while identical provider/model/session
  signals share one `model:` canonical field and only that field prefixes
  credit. Unchanged replay is idempotent; changed caller group fails before
  provider, ledger, or memory mutation; recursive secret checks include both.
- Memory version validation accepts only v1 triples and v2 four-part identities.
  Versions 0, 3, and future values fail directly, through BeliefState recursion,
  and at identity-write defense; supported v1 writes upgrade all identities.
- Full-batch preflight catches prior-state and newly introduced same-batch id
  conflicts across source, content, root, and supplied group with zero new
  provider calls or ledger/state mutation. Successful batches still classify
  against actual working memory in original order, never the preflight shadow.

### Concerns

No blocking concerns.

## Review Fix 8

### Changes

- Added one shared belief-lifecycle resolver with exactly two valid routes:
  native v0.2 and explicit legacy migration. All other lifecycle shapes raise
  before downstream work.
- Resolved the Python augmented gateway lifecycle once at execution entry and
  used that provider version for plan, plan repair, reasoning, and code repair.
- Applied the same resolver to the evidence gate and model-backed probe gateway.
  Direct invalid-state evidence integration now fails before normalization,
  provider access, memory work, or core ledger append.
- Converted legacy direct-gate fixtures into explicit v0.1 migrations so their
  four-field transport and `legacy_v0.1_migration` audit metadata remain covered.

### Verification

- RED: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/evaluation/test_python_probe.py::test_python_augmented_gateway_converts_successful_execution_to_active_signal tests/evaluation/test_python_probe.py::test_invalid_plan_gets_one_plan_repair tests/evaluation/test_python_probe.py::test_reasoning_mode_uses_model_signal_without_starting_sandbox tests/evaluation/test_python_probe.py::test_runtime_failure_gets_one_code_repair_and_second_execution tests/evaluation/test_python_probe.py::test_explicit_migration_uses_v01_for_every_python_model_route tests/evaluation/test_python_probe.py::test_unmigrated_v01_python_gateway_rejects_before_model_or_sandbox tests/test_evidence_memory.py::test_unmigrated_v01_direct_gate_rejects_before_provider_or_memory tests/test_core_cycles.py::test_invalid_lifecycle_fails_before_provider_or_cycle_ledger_append -q -p no:cacheprovider` -> `7 failed, 1 passed in 0.36s`.
- GREEN: the same focused command -> `8 passed in 0.28s`.
- Task 4 focused: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_evidence_memory.py tests/test_model_gateway.py tests/test_openai_gateway.py tests/test_core_cycles.py tests/test_probe_executor.py tests/evaluation/test_python_probe.py -q -p no:cacheprovider` -> `346 passed in 0.77s`.
- Compatibility: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_schemas.py tests/test_migrations.py tests/test_task_framing.py tests/test_recorded_model_gateway.py -q -p no:cacheprovider` -> `306 passed in 0.39s`.
- Full offline: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider` -> `1133 passed, 10 skipped in 8.90s`.
- Node: `node --test tests/test_webui_stream.js` -> `15 passed, 0 failed`.
- `git diff --check` -> clean.

### Self-Review

- Native Python requests remain v0.2 across all retries and fallbacks because
  one immutable resolved version is threaded through every model route. Explicit
  migration uses v0.1 throughout; invalid states cannot reach the model,
  sandbox preflight, execution, observer, or process counters.
- Evidence routing now derives both transport and audit route from the shared
  lifecycle result. Unmigrated v0.1 and incomplete v0.2 states fail before
  provider, accepted/discard event construction, identity memory, or core ledger
  writes. Explicit migration retains the reviewed four-field completion path,
  and native v0.2 retains the exact seven-field contract.

### Concerns

No blocking concerns.

## Review Fix 9

### Changes

- Required every provider-facing lifecycle to be a recursively validated v0.2
  `BeliefState` with a v0.2 `TaskFrame`, `FrameState`, and `EvidenceMemory`.
- Required the legacy route to carry both `LEGACY_MIGRATION` and one exact
  migration-writer marker. Tag-only, schema-mismatched, trace-invalid, missing,
  and structurally incoherent envelopes now fail at lifecycle entry.
- Extracted both recognized marker values into one migration interface consumed
  by the two migration writers and lifecycle resolver.
- Replaced tag-only positive fixtures with validated v0.1 states passed through
  the real migration helper. Both markers now exercise EvidenceGate,
  ModelBackedProbeToolGateway, and PythonAugmentedProbeToolGateway at v0.1;
  existing native coverage remains v0.2.

### Verification

- RED: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_evidence_memory.py::test_explicit_migration_route_completes_exact_legacy_shape_auditably tests/test_evidence_memory.py::test_invalid_migration_envelope_rejects_before_provider_or_memory tests/test_probe_executor.py::test_model_backed_probe_gateway_uses_v01_only_for_explicit_migration tests/test_probe_executor.py::test_model_backed_probe_gateway_rejects_invalid_migration_envelope tests/evaluation/test_python_probe.py::test_explicit_migration_uses_v01_for_every_python_model_route tests/evaluation/test_python_probe.py::test_invalid_python_migration_envelope_rejects_without_side_effects tests/evaluation/test_python_probe.py::test_unmigrated_v01_python_gateway_rejects_before_model_or_sandbox tests/test_core_cycles.py::test_invalid_lifecycle_fails_before_provider_or_cycle_ledger_append -q -p no:cacheprovider` -> `31 failed, 9 passed in 0.73s`.
- GREEN: the same focused command -> `40 passed in 0.38s`.
- Task 4 focused: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_evidence_memory.py tests/test_model_gateway.py tests/test_openai_gateway.py tests/test_core_cycles.py tests/test_probe_executor.py tests/evaluation/test_python_probe.py -q -p no:cacheprovider` -> `380 passed in 0.96s`.
- Compatibility: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_schemas.py tests/test_migrations.py tests/test_task_framing.py tests/test_recorded_model_gateway.py -q -p no:cacheprovider` -> `306 passed in 0.38s`.
- Full offline: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider` -> `1167 passed, 10 skipped in 9.51s`.
- Node: `node --test tests/test_webui_stream.js` -> `15 passed, 0 failed`.
- `git diff --check` -> clean.

### Self-Review

- Envelope checks and recursive reconstruction run before native/legacy route
  selection. Every invalid fixture proves unchanged state and zero provider
  requests; Python additionally proves no sandbox/counter activity, and core
  proves byte-empty ledger output.
- A legacy tag cannot select v0.1 by itself. Only the exact markers emitted by
  `migrate_belief_state_v0_1` and `migrate_task_frame_v0_1` are accepted, and
  the shared constants prevent resolver/writer drift.
- Both positive marker fixtures first validate as v0.1 `BeliefState` objects,
  then migrate to complete v0.2 runtime envelopes before reaching any gateway.

### Concerns

No blocking concerns.

## Review Fix 10

### Changes

- Added a compact v2 evidence-memory map from each accepted or discarded event
  id to the canonical signal-identity digest used by native event-id generation.
  Existing v2 snapshots load with an empty map; v1 cannot claim new bindings.
- Made first commit write lifecycle history and its binding in one snapshot, and
  made known-event commit require a matching persisted binding before identity
  writes or idempotent return. Projection sender/source events bind separately
  to the same signal digest.
- Planned the complete normalized batch before integration. Migration replay
  validates its exact positional event set and every referenced binding against
  an identity-only shadow before provider access or committed state work.
- Replaced the coarse non-empty identity-memory replay guard with per-event
  proof. Historical native or migrated lifecycle ids without a binding now fail
  closed; exact native and migrated replays remain idempotent.

### Verification

- RED: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_schemas.py::test_v2_evidence_memory_defaults_missing_event_signal_bindings tests/test_schemas.py::test_evidence_memory_rejects_invalid_event_signal_binding_grammar tests/test_schemas.py::test_evidence_memory_event_bindings_require_lifecycle_history tests/test_schemas.py::test_evidence_memory_event_bindings_cover_accepted_and_discarded_events tests/test_schemas.py::test_belief_state_recursively_rejects_unowned_event_signal_binding tests/test_evidence_memory.py::test_native_event_id_and_binding_share_canonical_signal_identity_digest tests/test_evidence_memory.py::test_accepted_neutral_event_preserves_existing_directional_credit tests/test_evidence_memory.py::test_discard_history_uses_exact_event_id_with_colons_for_idempotency tests/test_evidence_memory.py::test_direct_commit_rejects_known_event_with_different_signal_binding tests/test_evidence_memory.py::test_direct_commit_rejects_known_event_without_historical_binding tests/test_evidence_memory.py::test_identity_only_write_preserves_event_signal_bindings tests/test_core_cycles.py::test_migrated_positional_replay_conflicts_fail_atomically tests/test_core_cycles.py::test_later_positional_conflict_preflights_before_novel_provider tests/test_core_cycles.py::test_exact_migrated_positional_replay_is_idempotent tests/test_core_cycles.py::test_historical_positional_event_without_binding_fails_with_other_identity_memory tests/test_core_cycles.py::test_external_projection_decomposes_source_claim_and_generates_verification_probe tests/test_core_cycles.py::test_replayed_native_evidence_id_does_not_recommit_credit_or_ledger_record -q -p no:cacheprovider` -> `25 failed in 0.78s`.
- Version-boundary RED: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_schemas.py::test_v1_evidence_memory_rejects_event_signal_bindings -q -p no:cacheprovider` -> `1 failed in 0.13s`.
- GREEN: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_schemas.py::test_v2_evidence_memory_defaults_missing_event_signal_bindings tests/test_schemas.py::test_v1_evidence_memory_rejects_event_signal_bindings tests/test_schemas.py::test_evidence_memory_rejects_invalid_event_signal_binding_grammar tests/test_schemas.py::test_evidence_memory_event_bindings_require_lifecycle_history tests/test_schemas.py::test_evidence_memory_event_bindings_cover_accepted_and_discarded_events tests/test_schemas.py::test_belief_state_recursively_rejects_unowned_event_signal_binding tests/test_evidence_memory.py::test_native_event_id_and_binding_share_canonical_signal_identity_digest tests/test_evidence_memory.py::test_accepted_neutral_event_preserves_existing_directional_credit tests/test_evidence_memory.py::test_discard_history_uses_exact_event_id_with_colons_for_idempotency tests/test_evidence_memory.py::test_direct_commit_rejects_known_event_with_different_signal_binding tests/test_evidence_memory.py::test_direct_commit_rejects_known_event_without_historical_binding tests/test_evidence_memory.py::test_identity_only_write_preserves_event_signal_bindings tests/test_core_cycles.py::test_migrated_positional_replay_conflicts_fail_atomically tests/test_core_cycles.py::test_later_positional_conflict_preflights_before_novel_provider tests/test_core_cycles.py::test_exact_migrated_positional_replay_is_idempotent tests/test_core_cycles.py::test_historical_positional_event_without_binding_fails_with_other_identity_memory tests/test_core_cycles.py::test_external_projection_decomposes_source_claim_and_generates_verification_probe tests/test_core_cycles.py::test_replayed_native_evidence_id_does_not_recommit_credit_or_ledger_record -q -p no:cacheprovider` -> `26 passed in 0.35s`.
- Task 4 focused: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_evidence_memory.py tests/test_model_gateway.py tests/test_openai_gateway.py tests/test_core_cycles.py tests/test_probe_executor.py tests/evaluation/test_python_probe.py -q -p no:cacheprovider` -> `391 passed in 1.04s`.
- Compatibility: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_schemas.py tests/test_migrations.py tests/test_task_framing.py tests/test_recorded_model_gateway.py -q -p no:cacheprovider` -> `316 passed in 0.38s`.
- Full offline: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider` -> `1188 passed, 10 skipped in 9.52s`.
- Node: `node --test tests/test_webui_stream.js` -> `15 passed, 0 failed`.
- `git diff --check` -> clean.

### Self-Review

- Reordered, inserted, deleted, first-changed, and later-changed positional
  batches fail during full-batch preflight with unchanged provider count,
  ledger bytes, and belief memory. Exact replay proves binding equality and
  performs no provider, event-history, directional-credit, or ledger write.
- The shared digest is the sole input to native event-id construction and to
  persisted replay bindings. Accepted, discarded, colon-bearing, and projection
  secondary ids are covered; binding grammar and lifecycle ownership are also
  enforced through recursive `BeliefState` validation.
- All real evidence-memory reconstructions preserve bindings. Missing historical
  proof is never inferred from unrelated source identity memory, while v2
  snapshots without bindings remain loadable and fail only if such an event is
  replayed.

### Concerns

No blocking concerns.

## Review Fix 11

### Changes

- Added one v0.2 `BeliefState` cross-object invariant: accepted evidence ids
  plus decoded discard-history ids must be a subset of
  `ledger_refs["evidence_events"]`. The error is secret-free and never includes
  an event id.
- Kept the relation deliberately one-way. Extra historical ledger evidence ids
  remain valid, while evidence-memory binding ownership continues to be checked
  inside `EvidenceMemorySnapshot`.
- Exercised deep lifecycle revalidation with a bypass-constructed migrated state
  whose bound E1 was removed from ledger refs. It now fails before a changed
  positional signal reaches provider, memory, state, or ledger work.
- Updated direct-gate test transitions to carry the event refs corresponding to
  copied committed memory, and retained ledger-only missing-binding replay as a
  valid load that fails closed only when replayed.

### Verification

- RED: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_schemas.py::test_v02_belief_state_rejects_accepted_memory_event_missing_from_ledger_refs tests/test_schemas.py::test_v02_belief_state_rejects_discarded_memory_event_missing_from_ledger_refs tests/test_schemas.py::test_v02_belief_state_rejects_bound_memory_event_missing_from_ledger_refs tests/test_schemas.py::test_v02_belief_state_accepts_memory_lifecycle_subset_with_extra_ledger_ids tests/test_core_cycles.py::test_bypass_migrated_memory_event_without_ledger_ref_fails_atomically tests/test_core_cycles.py::test_exact_migrated_positional_replay_is_idempotent tests/test_core_cycles.py::test_replayed_native_evidence_id_does_not_recommit_credit_or_ledger_record tests/test_core_cycles.py::test_historical_positional_event_without_binding_fails_with_other_identity_memory -q -p no:cacheprovider` -> `4 failed, 4 passed in 0.33s`.
- Initial GREEN exposed the old historical fixture as memory-owned rather than
  ledger-only: the same command -> `1 failed, 7 passed in 0.28s`.
- GREEN after fixture refactor: the same command -> `8 passed in 0.26s`.
- The first Task 4 run exposed four direct-gate fixtures that copied committed
  memory without its refs: the Task 4 command below -> `4 failed, 388 passed in
  1.14s`; their focused correction check -> `4 passed in 0.21s`.
- Task 4 focused: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_evidence_memory.py tests/test_model_gateway.py tests/test_openai_gateway.py tests/test_core_cycles.py tests/test_probe_executor.py tests/evaluation/test_python_probe.py -q -p no:cacheprovider` -> `392 passed in 1.05s`.
- Compatibility: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_schemas.py tests/test_migrations.py tests/test_task_framing.py tests/test_recorded_model_gateway.py -q -p no:cacheprovider` -> `320 passed in 0.40s`.
- Full offline: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider` -> `1193 passed, 10 skipped in 9.50s`.
- Node: `node --test tests/test_webui_stream.js` -> `15 passed, 0 failed`.
- `git diff --check` -> clean.

### Self-Review

- Direct construction rejects accepted, discarded, and bound lifecycle entries
  missing from evidence ledger refs. Recursive lifecycle resolution converts a
  bypassed version of the same inconsistency into an invalid lifecycle before
  provider access; core assertions prove unchanged state, memory, and ledger
  bytes.
- Normal first migrated commit, exact migrated replay, and exact native replay
  all assert the new subset invariant. The validator does not require equality,
  so historical ledger-only ids remain backward-loadable and their absent
  binding is checked only on replay.
- The invariant applies only to v0.2 envelopes, uses canonical decoded discard
  ids, and does not weaken the existing internal binding-to-lifecycle subset or
  any prior lifecycle, identity, credit, replay, and secret-safety checks.

### Concerns

No blocking concerns.

## Review Fix 12

### Changes

- Made the shared lifecycle resolver inspect migration-key presence after deep
  runtime-envelope validation and before selecting either provider route.
- Kept legacy routing exact: `LEGACY_MIGRATION` requires a string marker from
  `RECOGNIZED_V01_TO_V02_MIGRATION_MARKERS`. Every non-legacy framing method
  now requires the `migration` key to be absent, regardless of its value.
- Added concise parameterized consumer regressions for `EXPLICIT`, `MODEL`, and
  `RECORDED` mutations of a real migrated state, plus recognized, fake, empty,
  and non-string migration values on a native state.
- Retained positive coverage for non-migration native trace metadata at v0.2
  and both migration-writer markers at v0.1 across evidence, model-backed probe,
  and Python-augmented probe routes.

### Verification

- RED: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_evidence_memory.py::test_migrated_marker_with_nonlegacy_method_rejects_before_evidence_side_effects tests/test_evidence_memory.py::test_native_migration_trace_key_rejects_before_evidence_side_effects tests/test_evidence_memory.py::test_native_judgment_request_contains_full_semantics_provenance_and_memory tests/test_evidence_memory.py::test_explicit_migration_route_completes_exact_legacy_shape_auditably tests/test_probe_executor.py::test_model_backed_probe_rejects_migrated_marker_with_nonlegacy_method tests/test_probe_executor.py::test_model_backed_probe_gateway_uses_v01_only_for_explicit_migration tests/evaluation/test_python_probe.py::test_python_gateway_rejects_migrated_marker_with_nonlegacy_method tests/evaluation/test_python_probe.py::test_python_augmented_gateway_converts_successful_execution_to_active_signal tests/evaluation/test_python_probe.py::test_explicit_migration_uses_v01_for_every_python_model_route -q -p no:cacheprovider` -> `13 failed, 8 passed in 0.37s`.
- GREEN: the same focused command -> `21 passed in 0.24s`.
- Task 4 focused: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_evidence_memory.py tests/test_model_gateway.py tests/test_openai_gateway.py tests/test_core_cycles.py tests/test_probe_executor.py tests/evaluation/test_python_probe.py -q -p no:cacheprovider` -> `405 passed in 1.06s`.
- Compatibility: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_schemas.py tests/test_migrations.py tests/test_task_framing.py tests/test_recorded_model_gateway.py -q -p no:cacheprovider` -> `320 passed in 0.39s`.
- Full offline: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider` -> `1206 passed, 10 skipped in 9.64s`.
- Node: `node --test tests/test_webui_stream.js` -> `15 passed, 0 failed`.
- `git diff --check` -> clean.

### Self-Review

- EvidenceGate rejects incoherent states before provenance normalization,
  provider requests, or memory changes. ModelBackedProbe rejects before its
  request, and PythonAugmented rejects before model, sandbox preflight,
  execution, or process counters; every fixture also remains unchanged by
  serialized state comparison.
- The native check uses key membership, so recognized, fake, empty, non-string,
  and null-like future values cannot select or masquerade as native. Other trace
  metadata remains allowed because only the exact `migration` key is reserved.
- Real migrated envelopes retain their original method and recognized marker,
  and both migration paths continue to emit v0.1 requests. Ordinary native
  envelopes without the key continue to emit v0.2 requests.

### Concerns

No blocking concerns.

## Review Fix 13

### Changes

- Centralized reused signal-id and locally known parent-root coherence in
  `EvidenceMemoryManager.validate_signal_lineage`. Both identity writes and
  classification now invoke that path; the duplicate parent check was removed
  from `classify`.
- Split EvidenceGate preflight into two passes. The first builds the identity-only
  shadow for every normalized signal; the second validates every lineage and
  existing event binding against the completed shadow before formal processing.
- Preserved formal semantics: classification and commitment still begin from the
  original working memory and proceed in signal order; the preflight shadow is
  never used as classification memory.
- Added prior-memory and both-order same-batch conflict regressions, both-order
  matching-root success, direct manager checks, and an explicit regression that
  unknown external parents remain non-independent.

### Verification

- RED: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_evidence_memory.py::test_direct_memory_operations_reject_known_parent_root_mismatch tests/test_evidence_memory.py::test_unknown_external_parent_remains_correlated_and_nonindependent tests/test_evidence_memory.py::test_unknown_parent_is_ledger_visible_but_receives_zero_independent_credit tests/test_evidence_memory.py::test_batch_preflight_normalizes_once_and_stops_before_provider tests/test_core_cycles.py::test_prior_known_parent_root_conflict_preflights_before_novel_provider tests/test_core_cycles.py::test_same_batch_parent_root_conflict_preflights_before_provider tests/test_core_cycles.py::test_matching_parent_root_succeeds_with_zero_independence_in_both_orders tests/test_core_cycles.py::test_later_cross_cycle_batch_conflict_preflights_before_provider_or_ledger tests/test_core_cycles.py::test_same_batch_reused_signal_conflict_preflights_atomically tests/test_core_cycles.py::test_later_positional_conflict_preflights_before_novel_provider tests/test_core_cycles.py::test_replayed_new_signal_identity_is_persisted_then_conflict_fails_atomically -q -p no:cacheprovider` -> `4 failed, 16 passed in 0.45s`.
- GREEN: the same focused command -> `20 passed in 0.36s`.
- Task 4 focused: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_evidence_memory.py tests/test_model_gateway.py tests/test_openai_gateway.py tests/test_core_cycles.py tests/test_probe_executor.py tests/evaluation/test_python_probe.py -q -p no:cacheprovider` -> `412 passed in 1.11s`.
- Compatibility: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_schemas.py tests/test_migrations.py tests/test_task_framing.py tests/test_recorded_model_gateway.py -q -p no:cacheprovider` -> `320 passed in 0.40s`.
- Full offline: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider` -> `1213 passed, 10 skipped in 9.68s`.
- Node: `node --test tests/test_webui_stream.js` -> `15 passed, 0 failed`.
- `git diff --check` -> clean.

### Self-Review

- A child mismatching a prior parent, an earlier same-batch parent, or a later
  same-batch parent fails before any new provider request. Core assertions prove
  unchanged belief state, evidence memory, and ledger bytes.
- Matching-root parent/child batches succeed in both orders. The parent remains
  novel only when formally processed first and becomes correlated when processed
  after the child, proving the completed preflight shadow does not leak into
  classification. The child is always a zero-independence correlated restatement.
- Unknown parent ids absent from prior and full-batch memory remain allowed but
  classify as correlated restatements with zero independence and zero effective
  weight. Earlier signal-id, event-binding, and batch-atomicity regressions remain
  included in the focused GREEN command.

### Concerns

No blocking concerns.

## Review Fix 14

### Changes

- Centralized exact-plus-NFKC secret detection in the shared schema predicates
  and exposed one recursive inspector used by redaction, schema validation, and
  `ExternalSignal` normalization. The full signal payload is now rejected at
  normalization entry before identity hashing or provider-request construction.
- Added one canonical event-binding id helper for non-empty, trim-exact,
  secret-free ids. Evidence-memory binding validation and EvidenceGate planning
  now share it; every planned primary and projection-secondary id is checked
  before provider access.
- Added atomic regressions for whitespace and NFKC-secret run/cycle namespaces,
  every string-bearing signal/provenance field, pre-hash ordering, recursive
  schema/redaction behavior, persisted bindings, and projection-secondary ids.

### Verification

- RED: `pytest -q tests/test_schemas.py::test_canonical_event_binding_id_helper_enforces_exact_secret_free_text tests/test_schemas.py::test_evidence_memory_rejects_nfkc_secret_event_signal_binding_id tests/test_schemas.py::test_secret_predicates_recognize_nfkc_equivalent_forms tests/test_schemas.py::test_shared_redaction_recognizes_nfkc_secret_keys_and_values tests/test_schemas.py::test_task_frame_recursive_secret_validation_recognizes_nfkc_forms tests/test_evidence_memory.py::test_projection_secondary_event_id_is_validated_during_batch_planning tests/test_core_cycles.py::test_nfkc_secret_anywhere_in_signal_fails_atomically tests/test_core_cycles.py::test_noncanonical_planned_event_namespace_fails_atomically` -> `10 failed, 18 passed in 0.52s`.
- Pre-hash RED: `pytest -q tests/test_evidence_memory.py::test_recursive_signal_secret_validation_precedes_identity_hash` -> `4 failed in 0.15s`.
- GREEN: `pytest -q tests/test_schemas.py::test_canonical_event_binding_id_helper_enforces_exact_secret_free_text tests/test_schemas.py::test_evidence_memory_rejects_nfkc_secret_event_signal_binding_id tests/test_schemas.py::test_secret_predicates_recognize_nfkc_equivalent_forms tests/test_schemas.py::test_shared_redaction_recognizes_nfkc_secret_keys_and_values tests/test_schemas.py::test_task_frame_recursive_secret_validation_recognizes_nfkc_forms tests/test_evidence_memory.py::test_recursive_signal_secret_validation_precedes_identity_hash tests/test_evidence_memory.py::test_projection_secondary_event_id_is_validated_during_batch_planning tests/test_core_cycles.py::test_nfkc_secret_anywhere_in_signal_fails_atomically tests/test_core_cycles.py::test_noncanonical_planned_event_namespace_fails_atomically` -> `32 passed in 0.14s`.
- Task 4 focused: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_evidence_memory.py tests/test_model_gateway.py tests/test_openai_gateway.py tests/test_core_cycles.py tests/test_probe_executor.py tests/evaluation/test_python_probe.py -q -p no:cacheprovider` -> `438 passed in 1.02s`.
- Compatibility: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_schemas.py tests/test_migrations.py tests/test_task_framing.py tests/test_recorded_model_gateway.py -q -p no:cacheprovider` -> `326 passed in 0.35s`.
- Full offline: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider` -> `1245 passed, 10 skipped in 9.50s`.
- Node: `node --test tests/test_webui_stream.js` -> `15 passed, 0 failed`.
- `git diff --check` -> clean.

### Self-Review

- Signal validation recursively examines every mapping key and nested string in
  the serialized `ExternalSignal`, including ids, cycle/probe linkage, targets,
  source/content fields, and all provenance lists and scalars. Errors remain
  generic and tests prove neither exact nor normalized credential text is echoed.
- Planning validates the scoped namespace plus every complete primary and
  projection-secondary event id before lineage preflight or provider calls.
  Persistence invokes the same helper, so planned and stored binding grammar
  cannot drift; prior event-set and binding preflights remain unchanged.
- Exact-form behavior remains covered, NFKC strengthening flows through existing
  redaction and global schema callers, immediate source/content pre-hash checks
  remain in place, and empty signal batches retain their prior behavior.

### Concerns

No blocking concerns.

## Review Fix 15

### Design Decision

- Legacy provider downgrade authority is now an identity-based, non-serialized
  receipt attached only by `migrate_belief_state_v0_1`. Shallow and deep
  `model_copy` preserve an authentic in-memory receipt; public
  `model_dump`/`model_validate` round-trips intentionally drop it and therefore
  cannot select the legacy route. Core explicitly carries the receipt across
  its own recursively validated state reconstruction.
- Provider identity uses one shared exact-plus-NFKC secret-free validator.
  Model-backed and Python-augmented executors resolve it before their first
  provider request; Python threads the same resolved identity to reasoning
  provenance across planning and all repair paths.
- Core validates every new native event against a recursively valid memory
  snapshot, matching accepted/discard lifecycle ownership, and the exact
  canonical digest of the normalized signal named by the event before solver
  application or ledger writes. Native plain-list results fail closed.
- Python plan, execution-request, and code-repair validation test nonblankness
  without replacing the executable string. Sandbox stdin, execution records,
  SHA-256 audit, and deterministic-root inputs receive the original text.

### Changes

- Added private migration receipt creation, lifecycle resolution, deep-copy
  stability, and core propagation while retaining both reviewed v0.1 migration
  markers and strict native seven-field evidence transport.
- Moved model identity resolution ahead of `complete_structured` in
  `ModelBackedProbeToolGateway` and ahead of all Python planning, reasoning,
  plan-repair, and code-repair activity. Exact and NFKC credential-like values
  raise one generic error without provider, sandbox, process, or state effects.
- Replaced native core memory fallback with an ownership preflight for new
  events. Test-only static gates now return complete normalized signals and
  bound lifecycle memory; the legacy list-return compatibility test remains
  limited to an explicitly migrated v0.1 input.
- Preserved leading indentation, trailing newlines, line boundaries, and NFKC-
  compatible code characters through both normal planning and repaired-code
  execution, with distinct deterministic roots for every byte-distinct pair.

### Verification

- RED: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_migrations.py::test_explicit_migration_receipt_survives_copy_but_not_public_round_trip tests/test_migrations.py::test_native_public_fields_cannot_forge_legacy_migration_authority tests/test_evidence_memory.py::test_invalid_migration_envelope_rejects_before_provider_or_memory tests/test_probe_executor.py::test_model_backed_probe_gateway_rejects_invalid_migration_envelope tests/test_probe_executor.py::test_model_backed_probe_rejects_secret_identity_before_provider_call tests/evaluation/test_python_probe.py::test_invalid_python_migration_envelope_rejects_without_side_effects tests/evaluation/test_python_probe.py::test_python_gateway_rejects_secret_identity_before_every_route tests/evaluation/test_python_probe.py::test_python_plan_path_preserves_byte_exact_code_and_distinct_roots tests/evaluation/test_python_probe.py::test_python_repair_path_preserves_byte_exact_code_and_distinct_roots tests/test_core_cycles.py::test_invalid_lifecycle_fails_before_provider_or_cycle_ledger_append tests/test_core_cycles.py::test_native_plain_list_gate_fails_before_state_or_ledger_mutation -q -p no:cacheprovider` -> `22 failed, 37 passed in 0.64s`.
- Binding-coherence RED: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_core_cycles.py::test_native_gate_wrong_event_signal_binding_fails_before_ledger -q -p no:cacheprovider` -> `1 failed in 0.22s`.
- Deep-copy RED: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_migrations.py::test_explicit_migration_receipt_survives_copy_but_not_public_round_trip -q -p no:cacheprovider` -> `2 failed in 0.08s`.
- GREEN: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_migrations.py::test_explicit_migration_receipt_survives_copy_but_not_public_round_trip tests/test_migrations.py::test_native_public_fields_cannot_forge_legacy_migration_authority tests/test_evidence_memory.py::test_invalid_migration_envelope_rejects_before_provider_or_memory tests/test_probe_executor.py::test_model_backed_probe_gateway_rejects_invalid_migration_envelope tests/test_probe_executor.py::test_model_backed_probe_rejects_secret_identity_before_provider_call tests/evaluation/test_python_probe.py::test_invalid_python_migration_envelope_rejects_without_side_effects tests/evaluation/test_python_probe.py::test_python_gateway_rejects_secret_identity_before_every_route tests/evaluation/test_python_probe.py::test_python_plan_path_preserves_byte_exact_code_and_distinct_roots tests/evaluation/test_python_probe.py::test_python_repair_path_preserves_byte_exact_code_and_distinct_roots tests/test_core_cycles.py::test_invalid_lifecycle_fails_before_provider_or_cycle_ledger_append tests/test_core_cycles.py::test_native_plain_list_gate_fails_before_state_or_ledger_mutation tests/test_core_cycles.py::test_native_gate_wrong_event_signal_binding_fails_before_ledger -q -p no:cacheprovider` -> `60 passed in 0.45s`.
- Core ownership/refactor: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_core_cycles.py -q -p no:cacheprovider` -> `120 passed in 0.68s`.
- Task 4 focused: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_evidence_memory.py tests/test_model_gateway.py tests/test_openai_gateway.py tests/test_core_cycles.py tests/test_probe_executor.py tests/evaluation/test_python_probe.py -q -p no:cacheprovider` -> `462 passed in 1.31s`.
- Compatibility: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_schemas.py tests/test_migrations.py tests/test_task_framing.py tests/test_recorded_model_gateway.py -q -p no:cacheprovider` -> `329 passed in 0.36s`.
- Full offline: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider` -> `1272 passed, 10 skipped in 9.89s`.
- Node: `node --test tests/test_webui_stream.js` -> `15 passed, 0 failed`.
- `git diff --check 67abac9..HEAD` -> clean before and after the Review Fix 15 commit; `git diff --check` and `git diff --cached --check` -> clean before commit.

### Self-Review

- A native state forged with `LEGACY_MIGRATION` and either recognized marker
  now fails in EvidenceGate, model-backed probe, Python-augmented probe, and
  core before provider, normalization, sandbox, memory, state, or ledger work.
  Both real v0.1 migration paths retain v0.1 provider compatibility in memory.
- Provider identity is resolved once before any route-specific `try` block, so
  an invalid value cannot be converted into an unverified signal after a model
  call. Error text contains neither the exact nor NFKC-normalized credential.
- Native event ownership checks accepted versus discarded lifecycle semantics,
  recursively validates the memory snapshot, requires a binding, resolves the
  event's normalized source signal, and recomputes the shared canonical digest.
  The solver and ledger are downstream of this complete check.
- No executable-code path assigns a stripped or normalized value. Tests observe
  exact code in sandbox requests and compare roots after normal execution and
  after code repair for all four opaque-text distinctions.

### Concerns

No blocking concerns. A serialized migrated v0.2 envelope deliberately loses
legacy downgrade authority and fails lifecycle resolution; callers that require
the v0.1 provider contract must retain the in-memory migrated state or re-enter
through the explicit v0.1 migration function.

## Review Fix 16

### Design Decision

- Replaced the transferable singleton migration capability with a private,
  non-serialized receipt containing a SHA-256 digest of the exact public
  `BeliefState` envelope. Lifecycle resolution recomputes and compares that
  digest; unchanged shallow/deep copies retain authority, while any public
  mutation or field transfer invalidates it. Core issues a newly bound receipt
  only after source authorization, solver/policy work, recursive final-state
  validation, and the complete internal transition succeed.
- Moved native Evidence Memory ownership into one
  `EvidenceMemoryManager.validate_transition` transaction. It recursively
  validates both snapshots, reconstructs exact signal identities, requires
  append-only lifecycle/history and binding changes, preserves discovery and
  counterevidence refs, and computes the exact cumulative directional-credit
  result from current accepted events. The same check runs for new, replay-only,
  existing-event-only, and no-event integration results.
- Added a native event gate before solver/ledger work. Every native input event,
  including replays, is recursively revalidated as v0.2, must carry effective
  weight and provenance, must match its normalized signal origin/root, and must
  carry the unresolved-likelihood shape required by the current frame. Authentic
  explicit migration remains on the reviewed v0.1 compatibility route.

### Verification

- Receipt RED: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_migrations.py::test_migration_receipt_is_bound_to_the_exact_public_envelope tests/test_migrations.py::test_authentic_receipt_cannot_be_transferred_to_another_public_envelope tests/test_evidence_memory.py::test_invalid_migration_envelope_rejects_before_provider_or_memory tests/test_probe_executor.py::test_model_backed_probe_gateway_rejects_invalid_migration_envelope tests/evaluation/test_python_probe.py::test_invalid_python_migration_envelope_rejects_without_side_effects tests/test_core_cycles.py::test_invalid_lifecycle_fails_before_provider_or_cycle_ledger_append -q -p no:cacheprovider` -> `6 failed, 37 passed in 0.46s`.
- Transition/event RED: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_evidence_memory.py::test_memory_transition_validator_accepts_production_and_identity_only_replay tests/test_evidence_memory.py::test_memory_transition_validator_rejects_replay_only_credit_replacement tests/test_core_cycles.py::test_native_memory_replacement_regressions_fail_before_solver_or_ledger tests/test_core_cycles.py::test_existing_event_only_result_cannot_rewrite_directional_credit tests/test_core_cycles.py::test_native_event_contract_fails_before_solver_or_ledger -q -p no:cacheprovider` -> `16 failed in 0.71s`.
- Credit-omission RED found during self-review: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_core_cycles.py::test_new_directional_event_cannot_skip_its_credit_commit -q -p no:cacheprovider` -> `1 failed in 0.23s`.
- Receipt GREEN: the receipt RED command above -> `43 passed in 0.42s`.
- Transition/event GREEN: the transition/event RED command above -> `16 passed in 0.31s`; the expanded no-event/existing-event replacement matrix -> `28 passed in 0.35s`.
- Credit-omission GREEN: its RED command above -> `1 passed in 0.20s`.
- Task 4 focused: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_evidence_memory.py tests/test_model_gateway.py tests/test_openai_gateway.py tests/test_core_cycles.py tests/test_probe_executor.py tests/evaluation/test_python_probe.py -q -p no:cacheprovider` -> `502 passed in 1.58s`.
- Compatibility: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_schemas.py tests/test_migrations.py tests/test_task_framing.py tests/test_recorded_model_gateway.py -q -p no:cacheprovider` -> `331 passed in 0.37s`.
- Full offline: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider` -> `1314 passed, 10 skipped in 10.36s`.
- Node: `node --test tests/test_webui_stream.js` -> `15 passed, 0 failed`.
- `git diff --check 67abac9..HEAD`, `git diff --check 67abac9`, and `git diff --check` -> clean with the complete pre-commit patch.

### Self-Review

- Receipt tests cover unchanged shallow/deep copies, direct and in-place public
  mutation, all-public-field transfer, public serialization round-trip, and
  zero-side-effect rejection in EvidenceGate, model-backed probe, Python probe,
  and Core. Existing exact migrated replay proves Core rebinds authority after a
  valid internal state update; both real migration markers retain v0.1 routing.
- Transition tests use recursively valid adversarial snapshots. They drop or
  rewrite identities, accepted/discard history, bindings, discovery refs,
  counterevidence refs, and credit; the complete matrix runs for both no-event
  and existing-event-only results. Real production commits and replay identity-
  only additions remain accepted, while omitted new credit is also rejected.
- Native event tests cover new and replayed v0.1 events plus bypass-constructed
  v0.2 events with null effective weight. Correct bindings cannot authorize
  either shape, and state/ledger/provider assertions prove failure precedes the
  solver and commit. Existing authentic migration tests retain v0.1 event
  compatibility.

### Concerns

No blocking concerns. Serialized migrated envelopes intentionally remain
unauthorized for the v0.1 provider route because the receipt is runtime-only.

## Review Fix 17

### Design Decision

- Made `EvidenceMemoryManager.validate_transition` reconstruct the sole valid
  native transition by replaying the shared classification policy and
  `EvidenceMemoryManager.commit` from the prior snapshot. Each signal captures
  its classification snapshot before any of that signal's events are committed,
  preserving production projection decomposition semantics without a second
  credit algorithm.
- Moved `SignalQuality`, `SignalQualityAssessor`, and the quality metric set into
  the cycle-safe evidence-memory layer. `bayesprobe.evidence` imports and
  re-exports the same public assessor, so production event construction and
  transition verification cannot drift while existing imports remain valid.
- Treats historical event binding proof as a batch-wide first preflight. Every
  event id already in `ledger_refs.evidence_events` must have a prior binding
  equal to the current normalized signal digest before any identity remembering
  or classification. Only after all bindings pass does full-batch lineage
  validation and formal ordered reconstruction begin.

### Changes

- Replaced `_expected_transition_credit`, append-only field checklists, and
  event-declared weight summation with exact classify/commit replay. New events
  must match independently derived correlation status, quality-product weight,
  remaining-credit cap, and policy discard semantics; the supplied candidate
  memory must equal the reconstructed snapshot exactly.
- Enforced source/type/origin and low-reliability quality ceilings on all six
  event quality fields. Lower provider overrides remain valid. Exact duplicates
  use duplicate caps, while same-root restatements must retain zero independence
  and bounded novelty.
- Added ordered event grouping for one-event signals and projection primary plus
  `_source` secondary pairs. Both projection decisions use the signal-start
  snapshot, while commits remain in event order.
- Moved EvidenceGate's historical binding check ahead of its identity-only
  preflight shadow. Existing events can now perform only exact identity-only
  remembering; lifecycle, binding, credit, discovery, and counterevidence state
  must remain unchanged.
- Added atomic adversarial regressions for self-consistent inflated and over-cap
  credit, exact/same-root events mislabeled novel, inflated model-origin quality,
  changed or missing replay bindings, and preflight ordering. Added positive
  coverage for lower quality, production transitions, projection two-event
  reconstruction, identity-only replay, and saturation. Updated the test-only
  `StaticEventGate` to synthesize policy-valid native quality and weight.

### Verification

- RED: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_core_cycles.py::test_self_consistent_inflated_credit_transition_fails_atomically tests/test_core_cycles.py::test_repeat_mislabeled_novel_with_positive_weight_fails_atomically tests/test_core_cycles.py::test_inflated_model_origin_quality_and_matching_memory_fail_atomically tests/test_core_cycles.py::test_valid_lower_model_quality_transition_is_accepted tests/test_core_cycles.py::test_existing_event_changed_signal_binding_fails_atomically tests/test_core_cycles.py::test_existing_event_missing_historical_binding_fails_atomically tests/test_evidence_memory.py::test_existing_binding_preflight_precedes_identity_or_classification tests/test_evidence_memory.py::test_memory_transition_validator_accepts_projection_two_event_reconstruction -q -p no:cacheprovider` -> `9 failed, 2 passed in 0.64s`.
- GREEN: the same focused command -> `11 passed in 0.28s`.
- Task 4 focused: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_evidence_memory.py tests/test_model_gateway.py tests/test_openai_gateway.py tests/test_core_cycles.py tests/test_probe_executor.py tests/evaluation/test_python_probe.py -q -p no:cacheprovider` -> `513 passed in 1.63s`.
- Compatibility: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_schemas.py tests/test_migrations.py tests/test_task_framing.py tests/test_recorded_model_gateway.py -q -p no:cacheprovider` -> `331 passed in 0.37s`.
- Full offline: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider` -> `1325 passed, 10 skipped in 10.60s`.
- Node: `node --test tests/test_webui_stream.js` -> `15 passed, 0 failed`.
- `git diff --check`, `git diff --check 67abac9..HEAD` -> clean before commit; the exact range check is repeated after commit.

### Self-Review

- A later replay conflict is checked against prior memory before any earlier new
  event can be remembered or classified. Missing and changed bindings therefore
  fail before solver and ledger work; Core tests prove unchanged input state,
  memory, and ledger bytes.
- Every new event's base weight comes only from its recursively validated quality
  fields. Classification independently applies duplicate/root/source grouping,
  directional remaining credit, and the configured cumulative cap. Commit then
  owns accepted/discard history, bindings, credit, and counterevidence, and exact
  snapshot equality rejects every forged matching event/memory pair.
- Production direct evidence, neutral/schema-discard behavior, projection pairs,
  exact replay identity additions, same-root zero-independence handling, and
  saturated ledger-visible events all remain covered by the focused GREEN suite.
  The preflight shadow is used only for lineage validation; formal classification
  still starts from prior memory in normalized signal order.

### Concerns

No blocking concerns.

## Review Fix 18

### Design Decision

- Moved the cycle-local source/content signature and seen-set mutation into
  `bayesprobe.evidence_memory`. The signature retains the existing semantics:
  trim and lowercase `signal.source`, lowercase `raw_content`, and collapse its
  whitespace. EvidenceGate and transition reconstruction now call the same
  operation exactly once per normalized signal in order.
- Kept cycle-local duplicate quality separate from memory correlation status.
  A later matching signature can remain `correlated_novel` because its supplied
  root/group differs, while still receiving the duplicate independence/novelty
  cap. Exact memory duplicates are capped through the same quality path even
  when they are the first cycle-local signature occurrence.
- Added one optional `correlation_credit_policy` to `BayesProbeCore`. Core creates
  one `EvidenceMemoryManager` before invoking the overridable gate factory,
  passes that object to its production EvidenceGate, and reuses it for native
  transition validation. Custom gates remain free to construct their result,
  but a result built under an inconsistent policy fails Core validation.

### Changes

- Added and exported `cycle_signal_source_content_signature` and
  `observe_cycle_signal_duplicate`; removed EvidenceGate's local signature and
  duplicate helpers.
- Recorded every production signal signature before replay/classification early
  returns. Transition replay maintains an equivalent cycle-local seen set and
  combines that duplicate flag with independently reconstructed exact-duplicate
  status when selecting deterministic quality ceilings.
- Threaded Core's owned manager through `_create_evidence_integration_gate` and
  `_resolve_next_evidence_memory`; removed the per-cycle default manager from
  transition validation. Initialization order now lets subclass factories use
  `self._evidence_memory_manager` directly without gate-private inspection.
- Added regressions for normalized same-signature/different-lineage batches,
  uncapped self-consistent candidate memory, distinct and first occurrences,
  exact/projection/low-reliability controls, manager construction order, custom
  cap acceptance/saturation, default behavior, and mismatched custom policy
  rejection before solver or ledger work. Updated `StaticEventGate` to represent
  each fixture event's actual content and use the shared duplicate policy.

### Verification

- RED: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_evidence_memory.py::test_same_batch_source_content_duplicate_uses_shared_quality_cap tests/test_evidence_memory.py::test_distinct_cycle_signatures_keep_standard_quality tests/test_evidence_memory.py::test_exact_cross_cycle_repeat_produces_no_update_or_provider_call tests/test_evidence_memory.py::test_memory_transition_validator_accepts_projection_two_event_reconstruction tests/test_core_cycles.py::test_core_constructs_one_memory_manager_before_gate_factory tests/test_core_cycles.py::test_core_custom_credit_policy_is_shared_with_production_gate tests/test_core_cycles.py::test_core_default_credit_policy_behavior_is_unchanged tests/test_core_cycles.py::test_core_rejects_transition_built_under_a_different_credit_policy tests/test_core_cycles.py::test_uncapped_same_batch_duplicate_transition_fails_atomically tests/test_core_cycles.py::test_low_reliability_signal_caps_quality_scores -q -p no:cacheprovider` -> `4 failed, 6 passed in 0.45s`.
- GREEN: the same focused command -> `10 passed in 0.31s`.
- Directly affected files: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_evidence_memory.py tests/test_core_cycles.py -q -p no:cacheprovider` -> `281 passed in 1.29s`.
- Task 4 focused: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_evidence_memory.py tests/test_model_gateway.py tests/test_openai_gateway.py tests/test_core_cycles.py tests/test_probe_executor.py tests/evaluation/test_python_probe.py -q -p no:cacheprovider` -> `520 passed in 1.72s`.
- Compatibility: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/test_schemas.py tests/test_migrations.py tests/test_task_framing.py tests/test_recorded_model_gateway.py -q -p no:cacheprovider` -> `331 passed in 0.36s`.
- Full offline: `PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider` -> `1332 passed, 10 skipped in 10.64s`.
- Node: `node --test tests/test_webui_stream.js` -> `15 passed, 0 failed`.
- `git diff --check`, `git diff --check 67abac9..HEAD` -> clean before commit; the exact range check is repeated after commit.

### Self-Review

- The normalized duplicate regression uses case/outer-whitespace differences in
  source plus case/line/whitespace differences in content, and different
  supplied roots/groups. Production classifies the second signal
  `correlated_novel` but caps it to independence/novelty `0.25` and weight
  `0.045`; transition reconstruction accepts that exact result and rejects the
  internally matching uncapped `0.4608` event/memory pair before the solver.
- Signature observation occurs once per signal rather than once per event, so a
  projection primary/secondary pair shares one flag. Existing event replays also
  contribute to the cycle-local set before a later signal. Exact duplicates use
  the duplicate quality cap regardless of their cycle-local position, while
  first and distinct signatures keep the original quality.
- Core's custom `0.2` policy limits the first production event and its persisted
  directional credit to `0.2`; the next same-direction event is ledger-visible
  with zero weight and `correlation_credit_saturated`. A default Core remains at
  `0.4608`. A default-policy custom result supplied to a `0.2` Core is rejected
  with zero solver calls, unchanged state, and empty ledger bytes.
- Production Core constructs one manager before its overridable gate factory,
  passes that same object explicitly, and performs no manager construction in
  `_resolve_next_evidence_memory`. No gate private state is inspected.

### Concerns

No blocking concerns.
