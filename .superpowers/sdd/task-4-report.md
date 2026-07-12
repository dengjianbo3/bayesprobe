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
