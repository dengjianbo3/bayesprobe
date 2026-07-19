# Terminal Initial Probe Slot Design

Status: Approved for specification review

Date: 2026-07-19

Scope: Terminal-Bench adapter only

## 1. Decision

For the first cycle of an open Terminal-Bench hypothesis frame, BayesProbe's
control policy owns the Probe's structural coverage. The model owns the Probe's
semantic inquiry. The adapter combines both into the existing public
`ProbeDesign` payload before the public `ModelProbeDesigner` validates and
materializes it.

The initial structural slot is fixed before the provider call:

```text
purpose = frame_coverage
target_hypotheses = every active hypothesis in frame order
required_capability = repository_read
terminal_plan_mode = inspect
```

This is the normal initial-open protocol, not a fallback after provider
failure. No deterministic Probe is created after repair exhaustion.

## 2. Evidence for the Redesign

The Stage 0 qualification attempts reached two provider-contract failures that
were not failures of the BayesProbe loop:

1. The provider had to reproduce `target_hypotheses` and two maps with exactly
   the same key set. That cross-field equality is not expressible by the
   provider's JSON-object response mode. Adapter-owned key normalization removed
   this failure.
2. The provider then had to ensure that at least one proposal in the first open
   cycle was a multi-hypothesis discriminator or frame-coverage Probe. This is a
   cross-proposal control invariant, not free-form semantic content. The same
   provider returned three contract-invalid responses despite receiving the
   policy in the initial and repair prompts.

The corrected-v3 qualification stopped before any terminal action:

- task frame: one valid response;
- Probe design: three root-level invalid responses;
- provider calls: four;
- provider tokens: 13,003;
- terminal actions: zero;
- complete BayesProbe cycles: zero;
- verifier: not reached.

Prompt-only retries therefore do not reliably enforce the initial-open control
invariant. Continuing to tune the prompt would measure provider obedience, not
the BayesProbe mechanism.

## 3. Goals

- Make initial open-frame coverage true by construction.
- Preserve the public `Belief State -> Probe -> Signal -> Evidence -> Update`
  runtime without a benchmark-private loop.
- Keep LLM participation in the Probe stage through a required inquiry goal and
  expected observation.
- Keep LLM interpretation mandatory in the Evidence stage.
- Keep all benchmark-specific behavior under `benchmarks/terminal_bench`.
- Make every server-owned transformation visible and content-addressed.
- Reach the next Stage 0 live qualification point with a clean, locked adapter.

## 4. Non-Goals

- No changes under `bayesprobe/`.
- No deterministic solution, patch, command sequence, Signal, Evidence, or
  posterior update.
- No semantic classifier that guesses whether prose is a good Probe by regex.
- No new OpenAI-compatible provider implementation.
- No unbounded provider repair.
- No reward optimization against the frozen qualification tasks.
- No Stage 1 paired experiment before Stage 0 passes.

## 5. Ownership Model

### 5.1 Server-owned fields

For an initial open cycle, the Terminal-Bench adapter owns:

- the existence of exactly one normalized initial proposal;
- `purpose=frame_coverage`;
- all active hypothesis IDs as `target_hypotheses`, preserving frame order;
- `required_capability=repository_read`;
- exact support/weaken condition key sets;
- the read-only terminal-plan mode;
- public Probe priority, which remains owned by the existing public designer.

These fields describe control and admissible capability. They do not claim that
an observation occurred or that any hypothesis is true.

### 5.2 Model-owned fields

The provider must supply non-empty:

- `inquiry_goal`;
- `expected_observation`.

The provider may supply target-specific support and weaken condition text. When
a target entry is absent, the adapter projects the target's already-framed
prediction or falsifier into the corresponding condition. This projection is
Probe metadata, not Evidence.

The provider's returned `purpose`, `target_hypotheses`, and
`required_capability` fields are syntactically required by the public wire
schema but are non-authoritative for the initial slot.

### 5.3 Planner-owned fields

The terminal planner owns the concrete read-only actions used to answer the
inquiry. The adapter validates that an initial frame-coverage plan:

- has `mode=inspect`;
- contains only inspect steps;
- contains only actions classified as read-only;
- contains no transition predictions or mutation.

The planner is not given permission to intervene during this first coverage
Probe. A later cycle may plan an intervention under the existing causal rules.

### 5.4 Evidence and Update ownership

No ownership changes after execution:

- the environment return is recorded as a Signal;
- the LLM interprets the Signal as candidate Evidence;
- the causal guard decides admissibility;
- only admitted Evidence may reach the public Update path.

## 6. Adapter Flow

The initial open Probe request follows this sequence:

```text
ProbeDesignContext
  -> derive immutable InitialOpenProbeSlot
  -> attach slot to terminal_policy
  -> call the existing budgeted provider
  -> validate model-owned semantic fields
  -> fill the server-owned slot
  -> normalize condition maps
  -> validate the normalized public ProbeDesign payload
  -> return it to the public ModelProbeDesigner
```

The adapter emits exactly one normalized proposal for this request. Additional
provider proposals are ignored for the initial slot, because the first cycle
has one declared control objective: frame coverage. Their raw response remains
represented by the provider-response hash.

For later cycles, the existing model-owned proposal flow remains in place:
known targets, available capabilities, bounded proposal count, normalized
condition keys, and bounded repair continue to apply.

## 7. Module Shape

The implementation remains local to the existing provider-contract and planner
modules.

### 7.1 `InitialOpenProbeSlot`

A private frozen model in `provider_contract.py` contains:

```python
class InitialOpenProbeSlot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    purpose: Literal["frame_coverage"]
    target_hypotheses: tuple[str, ...]
    required_capability: Literal["repository_read"]
    plan_mode: Literal["inspect"]
```

Its interface is the immutable policy value. Callers do not construct provider
payloads themselves.

### 7.2 Slot derivation

One private function derives a slot only when:

- `task_frame.coverage == "open"`;
- the request is for the first cycle;
- at least two distinct active hypothesis IDs are available.

Missing or duplicate active IDs are an adapter error because they originate
before the provider seam. They are not repaired by the model.

### 7.3 Slot filling

One private function accepts the validated provider response, the immutable
slot, and the request hypothesis records. It returns a new JSON-compatible
mapping. It never mutates the raw provider response.

The function copies model-owned semantic text, installs server-owned structural
fields, normalizes condition keys, and preserves an optional reframe condition
only when its keys and values are valid non-empty text.

### 7.4 Planner enforcement

`planning.py` receives the public Probe and applies one additional exact rule:
every `frame_coverage` Probe requiring `repository_read` must validate as an
inspect plan. The rule is visible in the initial and repair planner policies.

This rule applies before execution and cannot turn a mutating action into a
read-only action.

## 8. Failure Policy

Initial response plus at most two repairs remains the provider budget.

A repair receives:

- the original request;
- the same immutable slot;
- the redacted shape of the invalid provider payload;
- safe field errors;
- the attempt index.

The slot cannot change across attempts. Missing or empty `inquiry_goal`, missing
or empty `expected_observation`, malformed proposal structure, unavailable
provider response, and invalid condition text remain provider-contract errors.

Repair exhaustion fails closed. The adapter does not invent an inquiry, an
expected observation, or a terminal plan.

## 9. Observability and Locking

Each provider-contract attempt records:

- raw provider-response SHA-256;
- normalized-response SHA-256 when normalization succeeds;
- whether an initial slot was applied;
- the canonical slot SHA-256;
- the names of server-owned fields;
- validation status and safe field errors.

No raw invalid provider payload or hidden reasoning is persisted.

`contract_identity()` includes the canonical slot policy and normalized schema.
`plan_contract_identity()` includes the frame-coverage inspect rule. A live run
therefore requires a new qualification lock bound to the implementation commit,
adapter tree, provider identity, and both updated contract hashes.

## 10. Verification

### 10.1 Provider-contract tests

Tests must prove that:

- an initial open request derives one immutable slot;
- provider-returned single-target or wrong-purpose structure is replaced by the
  declared slot;
- provider inquiry and expected-observation text are preserved exactly after
  whitespace normalization;
- raw provider objects are not mutated;
- support/weaken maps exactly match slot targets;
- unknown extra condition keys are removed;
- missing condition values use framed predictions/falsifiers;
- invalid semantic text still enters bounded repair and can fail closed;
- non-initial requests retain model-owned purpose and targets;
- the public `ModelProbeDesigner` accepts the normalized result;
- raw and normalized hashes plus slot ownership are recorded without secrets.

### 10.2 Planner tests

Tests must prove that:

- a read-only initial frame-coverage plan passes;
- intervention and verify modes fail for the initial slot;
- a write hidden inside an inspect step fails;
- all three failed provider attempts produce `TerminalPlanError` without
  execution;
- later non-frame-coverage Probes retain the existing inspect/intervene/verify
  policy.

### 10.3 Regression verification

Before live use:

- focused provider, planner, runner-factory, public-reuse, lock, and
  qualification tests pass;
- the complete Terminal-Bench suite passes;
- the complete repository suite passes;
- `git diff --check` is clean;
- a secret scan finds no experiment key in repository content;
- the adapter worktree is clean and pushed.

## 11. Next Stage 0 Qualification Point

After implementation and offline verification:

1. Reuse the authorized experiment key from the process environment only.
2. Reuse the frozen provider identity only if model, base URL, protocol,
   temperature, returned model, and fingerprint remain unchanged.
3. Write a new causal qualification lock for the implementation commit.
4. Run `terminal-bench/break-filter-js-from-html` first.
5. Inspect only completion/error class while the trial runs.
6. Require at least one complete five-stage cycle and the official verifier.
7. Run the remaining two frozen tasks only after the first task passes the
   engineering and causal gate.

The implementation phase stops at this live qualification point if a new
system-level contract failure appears. It does not begin another prompt-patch
loop.

## 12. Research Interpretation

The change does not make the benchmark easier by supplying a solution. It makes
an existing BayesProbe invariant explicit at the correct seam. The first Probe
still has model-generated inquiry semantics, model-planned terminal actions,
real environment Signals, LLM-interpreted Evidence, and public posterior Update.

The experimental variable remains the explicit BayesProbe epistemic process.
Provider compliance with a hidden cross-proposal invariant is removed as an
unintended confounder.
