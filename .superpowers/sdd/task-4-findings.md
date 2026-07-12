# Task 4 Review Findings

1. Same-source signals cannot reset directional/subject credit by supplying a new correlation group. Classification and commitment must use a stable canonical group associated with the source lineage.
2. Any declared parent relationship prevents independent accumulation even when the parent is not yet known locally; unknown-parent signals must fail closed or receive correlated zero-independence treatment.
3. Existing evidence ids from `ledger_refs` must be detected before memory classification/credit commitment, including explicit v0.1 migration states whose memory starts empty.
4. Epistemic-origin quality caps apply to every evidence type, including provider-labeled `source_claim`; provider labels cannot elevate origin quality.
5. `EvidenceMemorySnapshot` must validate coherent identity-map key sets, exact canonical source-content identity structure, and the grammar of directional correlation-credit keys. Invalid persisted memory must fail recursive state validation rather than be ignored.
6. Model-backed probe provenance must include the configured provider/model identity, not only adapter kind, so correlation groups distinguish actual models while preserving run-session grouping.
7. Native judgment parsing must reject non-string interpretation and boolean/non-numeric quality overrides exactly as the v0.2 transport schema does.

## Re-review findings

1. Persisted memory must reject conflicting canonical groups for the same source identity or derivation root (or store one explicit canonical mapping); canonical group selection may not depend on map insertion order.
2. Migration-empty replay cannot rely on positional event ids. Use stable signal-to-event identity for native writes or fail closed when an already-used cycle lacks sufficient identity memory and the batch can no longer be verified.
3. Composite credit-key components must be domain-safe. Reserve `|` from correlation groups and hypothesis ids, and reserve the `frame:<version>:unresolved` namespace from named hypotheses, or use an unambiguous structured encoding consistent with the specified key format.
4. Native v0.2 judgment must require the exact seven-field contract. Legacy four-field completion is allowed only through an explicit legacy/migration route, never as preprocessing on a native request.
5. Discard history must use an unambiguous canonical encoding so event ids containing colons remain idempotent and schema-valid.

## Final-gate finding

1. Deterministic recomputation must preserve a stable factual derivation root derived from canonical computation inputs, not volatile probe ids, cycle ids, execution ids, or rendered output metadata. Repeating the same deterministic probe/Python computation across cycles must classify as same-root correlated evidence with zero independence rather than spend fresh credit.

## High-intensity final-gate findings

1. Deterministic root canonicalization must preserve opaque executable/source strings byte-for-byte (or hash exact bytes) so indentation, line boundaries, string-literal whitespace, and Unicode compatibility characters cannot collapse semantically distinct programs.
2. Python computation roots must include every material safe sandbox-policy input that can affect observable execution: image digest, user, CPU/memory/pids limits, timeout/output limits, network/read-only/tmpfs configuration, and other execution environment policy. Volatile execution ids remain excluded.
3. Reusing an existing signal id with different source/content/root/group identity must fail closed; memory commit may not overwrite prior lineage.
4. RecordedModelGateway requires a stable safe model identity derived from explicit safe fixture metadata or a canonical secret-free fixture fingerprint so distinct recorded providers/models do not collapse to adapter kind.
5. Native exclusive-open projection/source-claim events must populate neutral unresolved likelihood and coherent underdetermined frame fit before v0.2 validation/memory/solver.
6. Secret detection must run against the exact canonical source/content values immediately before hashing, including forms revealed only by Unicode normalization.
7. Native explicit-seed probe execution must advertise v0.2; only explicit legacy-migration lifecycle may use v0.1 probe metadata. Correct the report accordingly.

## Approval-gate findings

1. Validate signal-id lineage immediately after provenance normalization and before ledger-replay early return. Reused ids compare prior identity against the supplied normalized source/content/root/group; canonical-group fallback may not hide a changed supplied group.
2. Accepted evidence with no confirming/disconfirming credit subjects must not consume or rewrite any existing directional correlation credit. Neutral events may remain ledger-visible but directional balances stay unchanged.
3. Native exclusive-open schema-violation events must carry neutral unresolved likelihood and underdetermined frame fit even though discarded.

## Full-range re-review findings

1. An event-id replay with a previously unseen signal id must persist that signal's normalized identity in evidence memory before the replay early return. The identity-only operation must not recommit event history or directional credit; a later reuse of that ledgered signal id with changed source/content/root/group must fail closed before provider, ledger, or memory mutation.
2. Supplied provenance correlation group and canonical credit group are distinct facts. Persist enough identity to validate unchanged supplied-group replay while continuing to account correlation credit under the stable canonical group; an identical replay of a same-source signal whose supplied group differs from the canonical group must remain idempotent.

## Second full-range re-review findings

1. Model-origin normalization must not erase caller-supplied correlation-group continuity. Preserve the supplied group as signal identity while still forcing all model reasoning from the same provider/model/session into one server-derived canonical credit group. Reusing a model-origin signal id with a changed supplied group must fail closed.
2. `EvidenceMemorySnapshot` must reject unsupported memory versions rather than treating every version above one as v2 or preserving unknown versions during writes. Only explicitly supported v1/v2 shapes are valid.
3. Preflight the complete normalized signal batch against prior and cycle-local identity memory before any provider call. A later lineage conflict in the same batch must leave provider requests, ledger bytes, and committed belief memory unchanged; normal integration order and credit semantics must remain unchanged after preflight succeeds.

## Third full-range re-review findings

1. Every model request made by `PythonAugmentedProbeToolGateway` must derive its prompt/schema version from the belief lifecycle. Native v0.2 planning, planning repair, reasoning execution, and code repair advertise v0.2; v0.1 is allowed only for an explicit `FramingMethod.LEGACY_MIGRATION` state. Unmigrated v0.1 states fail before any model or sandbox call.
2. Evidence judgment routing must distinguish native v0.2, explicit legacy migration, and invalid lifecycle. A state that is merely non-v0.2 must not be labeled `legacy_v0.1_migration`; direct gate calls with unmigrated v0.1 state fail before provider, ledger, or memory mutation, while explicitly migrated states retain reviewed four-field compatibility.

## Fourth full-range re-review finding

1. A `LEGACY_MIGRATION` tag alone is not proof of migration. The legacy provider route requires the complete validated migration envelope produced by the migration lifecycle: v0.2 BeliefState and TaskFrame, non-null coherent FrameState and EvidenceMemory, and a recognized explicit migration trace. Tag-only native states and incomplete v0.1 states fail before provider, sandbox, ledger, or memory work. Legacy-route tests must use the real migration helper, not `model_copy` to change only the framing tag.

## Fifth full-range re-review finding

1. Positional Evidence Event ids in the explicit migration route require an explicit persisted event-to-canonical-signal identity binding. First write records the binding atomically; replay validates it before returning or remembering a new signal id. Reordered, inserted, deleted, or changed signals must not rebind an existing event id. A historical positional event id without a provable binding fails closed even when other signal identity memory is non-empty. Native signal-derived ids may also record the binding for one uniform invariant.

## Sixth full-range re-review finding

1. A v0.2 BeliefState must not allow memory-owned accepted/discarded event lifecycle ids or event-signal bindings to exist without corresponding `ledger_refs.evidence_events`. Enforce the cross-object subset invariant during recursive BeliefState validation (memory-owned ids must be ledger-referenced; extra historical ledger ids may remain for backward fail-closed replay). Gate lifecycle revalidation must reject a bypass-constructed inconsistent state before provider, normalization, memory, or ledger work.

## Seventh full-range re-review finding

1. Framing method and migration trace must be coherent in both directions. `LEGACY_MIGRATION` requires one recognized migration marker; every non-legacy framing method requires the migration marker to be absent. A migrated envelope whose method is changed to native/explicit, or a native envelope carrying any migration marker, is invalid and fails before provider, sandbox, memory, or ledger work.

## Eighth full-range re-review finding

1. Whole-batch preflight must validate parent/derivation-root coherence after the full identity shadow is built, before any provider call. Known parents from prior memory or anywhere in the current batch must share the child's derivation root regardless of batch order. Reuse one shared lineage validator in preflight, `remember_signal_identity`, and `classify`; unknown external parents remain correlated/non-independent per policy rather than claiming fresh independence.

## Ninth full-range re-review findings

1. Every newly planned primary and projection-secondary Evidence Event id must pass the same exact, secret-free canonical validation used by persisted event bindings before any provider call. Invalid run/cycle-derived ids fail during event planning/preflight, not after judgment at memory commit.
2. Secret detection must recursively inspect every persisted or provider-requested ExternalSignal field, including signal id, cycle id, probe id, targets, source fields, content, and provenance. Both exact and Unicode NFKC-normalized keys and string values must be checked centrally; normalized credential forms fail before request construction, hashing, memory, or ledger serialization without echoing secret text.

## Tenth full-range re-review findings

1. A native state must not be able to select the legacy v0.1 provider contract by relabeling its framing method and inserting a publicly recognized migration marker. The legacy route requires migration proof that cannot be forged by mutating those public fields; forged native envelopes fail before provider, sandbox, memory, or ledger work.
2. Provider/model identity must be validated as secret-free before every provider call that can later place it into signal provenance, including model-backed probe execution and both Python planning/reasoning routes. Errors stay generic and prove zero provider calls.
3. Native evidence integration must not accept a legacy list-returning gate result without an atomic Evidence Memory result. Every newly ledgered/applied native event must be owned by memory lifecycle and event-signal bindings; compatibility adapters must fail closed or perform a coherent memory commit before ledger/state mutation.
4. Python plan and repair parsing must preserve opaque executable code byte-for-byte for execution and deterministic-root hashing. Leading indentation, trailing newlines, line boundaries, and Unicode compatibility characters must remain distinct; validation may test non-blankness without rewriting the code.

## Eleventh full-range re-review findings

1. The private migration receipt must be bound to the exact public BeliefState envelope it authorizes. Unchanged shallow/deep copies may retain authority, but `model_copy(update=...)` or any public-field mutation that changes the envelope must invalidate the receipt. Core may issue a fresh receipt only after a verified internal state transition from an authentic source.
2. Native Core must validate Evidence Memory as an atomic transition from prior memory, not merely as a valid replacement snapshot. Prior identity maps, event lifecycle/history, event-signal bindings, discovery/counterevidence references, and directional-credit state cannot be erased or rebound; replay-only results are subject to the same transition check. Legitimate additions and policy-defined credit consumption remain allowed.
3. Every Evidence Event applied by a native v0.2 Core cycle must itself be schema v0.2 and satisfy the native effective-weight/provenance contract. A coherent memory binding cannot authorize a v0.1 event or legacy quality-product fallback outside an authentic explicit migration lifecycle.

## Twelfth full-range re-review findings

1. Native memory transition validation must independently enforce correlation classification, deterministic quality-derived base weight, remaining directional-credit limits, and the configured cumulative cap. An event and candidate memory that agree with each other cannot self-authorize an inflated weight, label an exact/same-root repeat as novel, or raise cumulative used credit above policy.
2. Every existing/replayed event must compare its current normalized signal identity digest with the preserved historical event binding. A different signal identity or missing historical binding fails closed before remembering identity, solving, or ledger work; preserving the old binding map without comparing it is insufficient.

## Thirteenth full-range re-review findings

1. Transition reconstruction must use the same cycle-local source/content signature duplicate detection as production EvidenceGate. Two same-batch signals with the same source/content but different supplied roots/groups remain correlated-novel for memory classification yet must receive the duplicate quality cap in both construction and validation.
2. Core and its production EvidenceGate must share one explicitly owned EvidenceMemoryManager and CorrelationCreditPolicy. Transition validation may not instantiate a default manager when the gate committed under a configured non-default cumulative credit cap; custom policy transitions must validate and default behavior must remain unchanged.

## Fourteenth full-range re-review findings

1. Native Core requires explicit closed-signal ownership from the gate. `normalized_signals=None` or an empty/missing/mismatched list cannot fall back to raw inbox signals; returned closed signals must correspond one-to-one with the cycle inputs, carry validated provenance, and be recursively secret-free before transition, solver, or ledger work. A nonempty native cycle with zero owned signals/events fails atomically.
2. The exported PythonExecutionRecord constructor must retain backward-compatible construction after adding policy metadata. Legacy construction may use an explicit safe compatibility representation, while current sandbox execution still requires the complete resolved policy before producing trusted evidence.
3. Python execution policy metadata must be deeply immutable at the record/observer boundary. An observer or external holder cannot mutate nested network/resource/interpreter fields before deterministic provenance hashing; the hash must describe the policy actually executed.
4. EvidenceMemorySnapshot must require accepted evidence IDs and decoded discarded-history event IDs to be disjoint. Contradictory lifecycle ownership fails recursive snapshot and BeliefState validation.

## Fifteenth full-range re-review finding

1. Copying a PythonExecutionRecord must preserve deep policy immutability. `copy.copy` and `copy.deepcopy` of the record cannot thaw nested policy mappings, while `dataclasses.asdict` must continue to return independent JSON-compatible dictionaries/lists for artifact serialization. Directly deep-copying the policy value may produce an independent mutable serialization but must never mutate or replace the policy attached to a record.

## Sixteenth full-range re-review finding

1. Core must preserve immutable authoritative pre-gate snapshots. The EvidenceGate receives isolated deep copies of BeliefState, closed signals, cycle, and ProbeSet; all post-gate ownership, memory-transition, solving, state construction, and ledger work use untouched authoritative snapshots. In-place gate mutation cannot redefine the validation baseline, erase prior memory/ledger history, rewrite signal content, or alter cycle/probe records.

## Seventeenth full-range re-review findings

1. Core's authoritative Evidence Memory transition validator and the production EvidenceGate cannot share a mutable EvidenceMemoryManager authority. A gate-side mutation of the manager policy or validator must not redefine the policy Core uses after the gate call. Preserve the configured CorrelationCreditPolicy consistently while isolating the gate's mutable manager object from Core's authoritative validator.
2. Native transition validation must bind every Evidence Event's content exactly to the authoritative closed signal identified by `derived_from_signal`. A gate cannot return the owned signal unchanged while rewriting only `EvidenceEvent.content`; such a transition must fail before solver, state, or ledger mutation.
