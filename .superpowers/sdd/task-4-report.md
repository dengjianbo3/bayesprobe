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
  seven-field v0.2 schema; exact legacy four-field provider payloads use a
  named compatibility completion before strict validation. Seeded-hypothesis
  evidence remains the explicit v0.1 route.
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
