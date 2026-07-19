# Terminal Initial Probe Slot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the first Probe of an open Terminal-Bench frame a server-owned, auditable, read-only frame-coverage slot while preserving model-generated inquiry semantics and the public BayesProbe five-stage loop.

**Architecture:** Extend the benchmark-local `TerminalContractModelGateway` so it derives one immutable initial slot from the public Probe-design request, attaches that slot to the provider policy, and fills it with the provider's inquiry goal and expected observation before returning a normal public `ProbeDesign` payload. Extend the benchmark-local terminal planner so a frame-coverage Probe requiring repository read can only produce an inspect plan. Keep `bayesprobe/` unchanged and bind both policies into the qualification lock through existing contract hashes.

**Tech Stack:** Python 3.12+, Pydantic 2, BayesProbe public interfaces, OpenAI-compatible Chat Completions, Harbor 0.18.0, pytest, Git, standard-library `hashlib` and `json`.

## Global Constraints

- Do not modify any file under `bayesprobe/`.
- Do not introduce a second BayesProbe loop or a benchmark-private posterior update.
- The server may own Probe control structure but may not invent `inquiry_goal`, `expected_observation`, terminal actions, Signal, Evidence, or posterior values.
- The initial slot is the normal first-open-cycle protocol, not a fallback after provider failure.
- Keep initial response plus at most two targeted repairs.
- Keep API keys process-environment-only and never place them in commands, configs, artifacts, locks, tests, docs, or Git.
- Preserve the user-owned untracked `reports/` directory.
- Do not run a live Terminal-Bench trial until every offline test, secret scan, commit, push, and qualification-lock check passes.
- Stop at the next live qualification position; do not start a new prompt-patch loop if another system-level contract failure appears.

## File Map

| File | Responsibility |
| --- | --- |
| `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/provider_contract.py` | Derive, publish, fill, validate, normalize, hash, and audit the initial Probe slot |
| `benchmarks/terminal_bench/tests/test_provider_contract.py` | Slot derivation, normalization, immutability, bounded repair, public-designer compatibility, and secret tests |
| `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/actions.py` | Validate a probe-specific required terminal plan mode through Pydantic context |
| `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/planning.py` | Expose Probe purpose/capability/mode to the planner and bind the inspect-only policy into prompt identity |
| `benchmarks/terminal_bench/tests/test_actions.py` | Exact plan-mode context validation |
| `benchmarks/terminal_bench/tests/test_planning.py` | Planner payload, repair, telemetry, and inspect-only behavior |
| `benchmarks/terminal_bench/tests/test_runner_factory.py` | Real adapter composition into the public runner |
| `benchmarks/terminal_bench/tests/test_qualification.py` | Updated contract identities and Stage 0 lock binding |
| `docs/superpowers/specs/2026-07-19-terminal-initial-probe-slot-design.md` | Implemented-state record after all offline checks pass |

---

### Task 1: Fill an immutable initial-open Probe slot

**Files:**

- Modify: `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/provider_contract.py`
- Modify: `benchmarks/terminal_bench/tests/test_provider_contract.py`

**Interfaces:**

- Consumes: `StructuredModelRequest.input` from the public `ModelProbeDesigner`, including `cycle_id`, `task_frame.coverage`, ordered `hypotheses`, and `available_capabilities`.
- Produces: private `InitialOpenProbeSlot`, `_initial_open_probe_slot(request_input)`, and a normalized public Probe-design mapping returned by `TerminalContractModelGateway.complete_structured()`.

- [ ] **Step 1: Write the failing slot-normalization tests**

Add tests that submit an initial open request whose provider response has the wrong purpose, one target, and the wrong capability. Require the returned proposal to use the immutable slot while preserving model semantic text and leaving the raw response unchanged:

```python
def test_initial_open_probe_fills_server_owned_slot_without_mutating_provider_payload(
    tmp_path,
) -> None:
    provider_payload = _probe(targets=["H1"])
    proposal = provider_payload["proposals"][0]
    proposal["purpose"] = "hypothesis_falsification"
    proposal["required_capability"] = "test_execution"
    proposal["inquiry_goal"] = "  Inspect the implementation against every framed claim.  "
    proposal["expected_observation"] = "  Repository facts distinguish the frame.  "
    before = deepcopy(provider_payload)
    delegate = RecordingGateway([provider_payload])
    gateway = TerminalContractModelGateway(
        delegate,
        artifacts=TrialArtifactStore(tmp_path, restricted_values=()),
    )

    normalized = gateway.complete_structured(_probe_request())

    assert provider_payload == before
    assert normalized["proposals"] == [
        {
            **normalized["proposals"][0],
            "purpose": "frame_coverage",
            "target_hypotheses": ["H1", "H2"],
            "required_capability": "repository_read",
            "inquiry_goal": "Inspect the implementation against every framed claim.",
            "expected_observation": "Repository facts distinguish the frame.",
        }
    ]
```

Also add one test proving that multiple provider proposals normalize to exactly one slot-filled proposal, and one test proving that a later-cycle request preserves model-owned purpose, targets, and capability.

- [ ] **Step 2: Write the failing bounded-semantic-repair test**

Require empty model-owned semantic text to consume the initial attempt and two repairs without producing a deterministic inquiry:

```python
def test_initial_slot_does_not_invent_missing_model_semantics(tmp_path) -> None:
    invalid = _probe()
    invalid["proposals"][0]["inquiry_goal"] = ""
    delegate = RecordingGateway([deepcopy(invalid) for _ in range(3)])
    gateway = TerminalContractModelGateway(
        delegate,
        artifacts=TrialArtifactStore(tmp_path, restricted_values=()),
    )

    with pytest.raises(ProviderContractError) as raised:
        gateway.complete_structured(_probe_request())

    assert raised.value.attempts == 3
    assert [request.task for request in delegate.requests] == [
        "design_probes",
        "repair_probe_design",
        "repair_probe_design",
    ]
```

- [ ] **Step 3: Run the focused tests and confirm RED**

Run:

```bash
cd benchmarks/terminal_bench
uv run pytest tests/test_provider_contract.py -q
```

Expected: the new tests fail because initial purpose, targets, capability, and proposal count remain provider-owned.

- [ ] **Step 4: Implement the private slot model and derivation**

Add the frozen model and exact derivation helper:

```python
class InitialOpenProbeSlot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    purpose: Literal["frame_coverage"] = "frame_coverage"
    target_hypotheses: tuple[str, ...] = Field(min_length=2)
    required_capability: Literal["repository_read"] = "repository_read"
    plan_mode: Literal["inspect"] = "inspect"


def _initial_open_probe_slot(
    request_input: Mapping[str, Any],
) -> InitialOpenProbeSlot | None:
    context = _probe_validation_context(request_input)
    if not context["requires_initial_open_coverage"]:
        return None
    targets = context["ordered_known_targets"]
    if len(targets) < 2 or len(targets) != len(set(targets)):
        raise ValueError("initial open Probe requires distinct active hypotheses")
    if "repository_read" not in context["available_capabilities"]:
        raise ValueError("initial open Probe requires repository_read")
    return InitialOpenProbeSlot(target_hypotheses=targets)
```

Preserve hypothesis order in `_probe_validation_context()` instead of deriving the slot from a sorted set.

- [ ] **Step 5: Implement slot publication and filling**

Attach the canonical slot to `terminal_policy`. Before `_TerminalProbeResponse` validation, copy the first provider proposal and install the server-owned fields:

```python
def _fill_initial_open_probe_slot(
    response: Any,
    slot: InitialOpenProbeSlot,
) -> Any:
    if not isinstance(response, Mapping):
        return response
    proposals = response.get("proposals")
    if not isinstance(proposals, list) or not proposals:
        return response
    first = proposals[0]
    if not isinstance(first, Mapping):
        return response
    normalized = dict(first)
    normalized.update(
        purpose=slot.purpose,
        target_hypotheses=list(slot.target_hypotheses),
        required_capability=slot.required_capability,
    )
    return {"proposals": [normalized]}
```

Pass the copied payload through the existing semantic validators and condition-map normalization. Do not mutate or return the raw provider object.

- [ ] **Step 6: Run the focused tests and confirm GREEN**

Run:

```bash
uv run pytest tests/test_provider_contract.py -q
```

Expected: all provider-contract tests pass.

- [ ] **Step 7: Run public-composition regressions**

Run:

```bash
uv run pytest tests/test_public_reuse.py tests/test_runner_factory.py -q
```

Expected: the normalized payload is accepted by the public `ModelProbeDesigner`; no public-core file changes.

- [ ] **Step 8: Commit Task 1**

```bash
git add benchmarks/terminal_bench/src/bayesprobe_terminal_bench/provider_contract.py benchmarks/terminal_bench/tests/test_provider_contract.py
git commit -m "feat(terminal-bench): fill initial probe slot"
```

### Task 2: Make slot normalization auditable and lock-bound

**Files:**

- Modify: `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/provider_contract.py`
- Modify: `benchmarks/terminal_bench/tests/test_provider_contract.py`
- Modify: `benchmarks/terminal_bench/tests/test_qualification.py`
- Modify: `benchmarks/terminal_bench/tests/test_experiment_lock.py`

**Interfaces:**

- Consumes: raw provider response, normalized response, and optional `InitialOpenProbeSlot`.
- Produces: extended `ContractAttempt` audit fields and updated `contract_identity()` hashes consumed by Stage 0 locks.

- [ ] **Step 1: Write failing audit tests**

Extend the valid initial Probe test to require:

```python
attempt = _attempts(tmp_path / "provider_contract.jsonl")[0]
raw_sha256 = sha256(
    json.dumps(
        provider_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
normalized_sha256 = sha256(
    json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
assert attempt["response_sha256"] == raw_sha256
assert attempt["normalized_response_sha256"] == normalized_sha256
assert attempt["control_policy"] == "initial_open_frame_coverage"
assert attempt["control_policy_sha256"].startswith("sha256:")
assert attempt["server_owned_fields"] == [
    "proposals.0.purpose",
    "proposals.0.required_capability",
    "proposals.0.target_hypotheses",
]
```

Add a non-initial test requiring `normalized_response_sha256` to equal the validated response hash, with no control policy or server-owned fields. Add a secret-shaped semantic-text test proving no raw content is persisted.

- [ ] **Step 2: Run focused tests and confirm RED**

Run:

```bash
uv run pytest tests/test_provider_contract.py tests/test_qualification.py tests/test_experiment_lock.py -q
```

Expected: audit fields and the new slot identity are absent.

- [ ] **Step 3: Extend `ContractAttempt` and recording**

Add fields with deterministic defaults:

```python
normalized_response_sha256: str | None = None
control_policy: Literal["initial_open_frame_coverage"] | None = None
control_policy_sha256: str | None = None
server_owned_fields: tuple[str, ...] = ()
```

Pass the normalized response and slot into `_record_attempt()`. Hash raw and normalized payloads separately. Hash `slot.model_dump(mode="json")` through `_identity_sha256()` and persist only hashes and field names.

- [ ] **Step 4: Bind slot policy into contract identity**

Add a canonical slot-policy object to the terminal Probe prompt identity:

```python
"initial_open_slot": {
    "purpose": "frame_coverage",
    "required_capability": "repository_read",
    "plan_mode": "inspect",
    "proposal_count": 1,
    "targets": "all_active_in_frame_order",
}
```

Do not hard-code resulting SHA values in tests. Require valid `sha256:<64 hex>` values, stable repeated calls, and inequality from the pre-slot fixture identity when a fixture exists.

- [ ] **Step 5: Run focused tests and confirm GREEN**

Run:

```bash
uv run pytest tests/test_provider_contract.py tests/test_qualification.py tests/test_experiment_lock.py -q
```

Expected: all tests pass and a Stage 0 lock includes the updated Probe prompt/schema identities.

- [ ] **Step 6: Commit Task 2**

```bash
git add benchmarks/terminal_bench/src/bayesprobe_terminal_bench/provider_contract.py benchmarks/terminal_bench/tests/test_provider_contract.py benchmarks/terminal_bench/tests/test_qualification.py benchmarks/terminal_bench/tests/test_experiment_lock.py
git commit -m "feat(terminal-bench): audit initial probe ownership"
```

### Task 3: Require an inspect-only terminal plan for frame coverage

**Files:**

- Modify: `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/actions.py`
- Modify: `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/planning.py`
- Modify: `benchmarks/terminal_bench/tests/test_actions.py`
- Modify: `benchmarks/terminal_bench/tests/test_planning.py`

**Interfaces:**

- Consumes: public `ProbeDesign.purpose`, `ProbeDesign.required_capability`, and Pydantic validation context.
- Produces: `required_plan_mode="inspect"` in the model-facing planner input and an exact validation error when the returned plan violates it.

- [ ] **Step 1: Write failing action-model tests**

Require a valid intervention plan to fail only when context declares inspect mode:

```python
def test_probe_required_plan_mode_rejects_intervention() -> None:
    payload = _valid_intervention_payload()

    with pytest.raises(ValidationError, match="required Probe plan mode"):
        TerminalProbePlan.model_validate(
            payload,
            context={
                "target_hypotheses": ("H1", "H2"),
                "required_plan_mode": "inspect",
            },
        )

    assert TerminalProbePlan.model_validate(
        payload,
        context={"target_hypotheses": ("H1", "H2")},
    ).mode == "intervene"
```

Add a passing inspect-plan test under the same required mode.

- [ ] **Step 2: Write failing planner tests**

Create a frame-coverage Probe with repository-read capability:

```python
coverage_probe = probe.model_copy(
    update={
        "purpose": ProbePurpose.FRAME_COVERAGE,
        "required_capability": CapabilityKind.REPOSITORY_READ,
        "target_hypotheses": ["H1", "H2"],
    }
)
```

Require `terminal_plan_input()` to expose `purpose`, `required_capability`, and `required_plan_mode="inspect"`. Feed the planner two intervention responses followed by `VALID_PLAN`; require two repairs, the safe error `plan:required_probe_mode`, and final success. Confirm the same intervention response remains valid for the existing non-frame-coverage fixture.

- [ ] **Step 3: Run focused tests and confirm RED**

Run:

```bash
uv run pytest tests/test_actions.py tests/test_planning.py -q
```

Expected: the planner neither exposes nor validates the required Probe mode.

- [ ] **Step 4: Add probe-specific validation context**

Change `TerminalProbePlan.validate_mode()` to accept `ValidationInfo` and fail before mode-specific validation when the declared mode differs:

```python
@model_validator(mode="after")
def validate_mode(self, info: ValidationInfo) -> "TerminalProbePlan":
    context = info.context if isinstance(info.context, Mapping) else {}
    required_mode = context.get("required_plan_mode")
    if required_mode is not None and self.mode != required_mode:
        raise ValueError("plan mode must equal the required Probe plan mode")
    # Existing inspect, verify, and intervene validation follows unchanged.
```

- [ ] **Step 5: Expose and enforce planner policy**

Add helpers:

```python
def _required_plan_mode(probe: ProbeDesign) -> Literal["inspect"] | None:
    if (
        probe.purpose is ProbePurpose.FRAME_COVERAGE
        and probe.required_capability is CapabilityKind.REPOSITORY_READ
    ):
        return "inspect"
    return None
```

Include Probe purpose, required capability, and required plan mode in
`terminal_plan_input()`. Pass the required mode into
`TerminalProbePlan.model_validate_json()` context. Add
`"plan mode must equal the required Probe plan mode": "plan:required_probe_mode"`
to the safe semantic error map.

Extend `_planner_instruction()` with one exact rule: when
`probe.required_plan_mode` is present, output `mode` must equal it. This changes
the existing planner prompt identities automatically.

- [ ] **Step 6: Run focused tests and confirm GREEN**

Run:

```bash
uv run pytest tests/test_actions.py tests/test_planning.py -q
```

Expected: all tests pass; the initial slot cannot produce an intervention plan.

- [ ] **Step 7: Run execution-path regressions**

Run:

```bash
uv run pytest tests/test_gateway.py tests/test_causal.py tests/test_trajectory.py -q
```

Expected: existing inspect/intervene/verify execution and causal lineage remain unchanged outside the required-mode case.

- [ ] **Step 8: Commit Task 3**

```bash
git add benchmarks/terminal_bench/src/bayesprobe_terminal_bench/actions.py benchmarks/terminal_bench/src/bayesprobe_terminal_bench/planning.py benchmarks/terminal_bench/tests/test_actions.py benchmarks/terminal_bench/tests/test_planning.py
git commit -m "feat(terminal-bench): require initial inspection plan"
```

### Task 4: Verify the public composition and freeze implemented status

**Files:**

- Modify: `benchmarks/terminal_bench/tests/test_runner_factory.py`
- Modify: `docs/superpowers/specs/2026-07-19-terminal-initial-probe-slot-design.md`

**Interfaces:**

- Consumes: the real benchmark-local provider-contract and planner adapters composed by `build_live_session()`.
- Produces: one offline proof that the public runner receives the slot-filled Probe and one implemented-state record.

- [ ] **Step 1: Add the failing real-composition assertion**

Extend the existing real contract-composition test or add one that uses
`TerminalContractModelGateway` followed by the public `ModelProbeDesigner`.
Require the resulting `ProbeCandidate.candidate_probe` to have:

```python
assert candidate.purpose is ProbePurpose.FRAME_COVERAGE
assert candidate.target_hypotheses == ["H1", "H2"]
assert candidate.required_capability is CapabilityKind.REPOSITORY_READ
assert candidate.inquiry_goal == provider_inquiry_goal
```

Assert that no files under `bayesprobe/` are changed in the task diff.

- [ ] **Step 2: Run focused integration tests and confirm RED or existing coverage**

Run:

```bash
uv run pytest tests/test_runner_factory.py tests/test_public_reuse.py -q
```

Expected: the new assertion fails before the Task 1 implementation or passes if Task 1 already exercises the exact composition. In the latter case, retain the assertion as regression coverage.

- [ ] **Step 3: Make only the minimal integration adjustment if required**

The expected composition remains:

```text
OpenAIChatCompletionsModelGateway
-> BudgetedModelGateway
-> TerminalContractModelGateway
-> CausalEvidenceModelGateway
-> public ModelProbeDesigner / BayesProbeCore
```

Do not add a benchmark-local runner or ProbeDesigner if the gateway seam already satisfies the test.

- [ ] **Step 4: Run complete offline verification**

From `benchmarks/terminal_bench`:

```bash
uv run pytest -q
```

From the repository root:

```bash
uv run pytest -q
git diff --check
git status --short
```

Expected: all Terminal-Bench tests pass; all repository tests pass with only documented skips; diff check is clean; only task files plus the user-owned `reports/` directory appear.

- [ ] **Step 5: Scan for the experiment key without exposing it**

In the existing secure shell, run:

```bash
rg -l --hidden --fixed-strings "$BAYESPROBE_BENCH_API_KEY" . -g '!**/.git/**' -g '!**/.venv/**'
```

Expected: no matching paths and no key value in output.

- [ ] **Step 6: Update design status and commit**

Change the design status to `Implemented and offline verified`, append exact test totals, and commit:

```bash
git add benchmarks/terminal_bench/tests/test_runner_factory.py docs/superpowers/specs/2026-07-19-terminal-initial-probe-slot-design.md
git commit -m "test(terminal-bench): verify initial probe slot composition"
```

- [ ] **Step 7: Push the branch**

```bash
git push origin codex/paradigm-conformance-kernel
```

Expected: local HEAD equals `origin/codex/paradigm-conformance-kernel`.

### Task 5: Prepare the next Stage 0 live qualification point

**Files:**

- Generate, ignored: a UTC- and commit-named directory under `benchmarks/terminal_bench/.runs/qualification/`
- Update, ignored: `benchmarks/terminal_bench/.runs/causal-qualification.lock.json`

**Interfaces:**

- Consumes: clean pushed Git identity, unchanged provider identity artifact, frozen Oracle job, and updated prompt/schema identities.
- Produces: one content-addressed Stage 0 lock and a ready, not-yet-started Harbor command for the first frozen task.

- [ ] **Step 1: Confirm provider identity remains reusable**

Read the existing provider identity artifact and require:

```text
configured model = deepseek-v4-flash
returned model = deepseek-v4-flash
base URL = https://api.deepseek.com
protocol = openai_chat_completions
temperature = 0
system fingerprint = fp_8b330d02d0_prod0820_fp8_kvcache_20260402
```

Do not make a new provider call unless one of these values changed.

- [ ] **Step 2: Write a fresh qualification lock**

Run from `benchmarks/terminal_bench` with a unique UTC directory:

```bash
RUN_STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
SHORT_SHA="$(git rev-parse --short=7 HEAD)"
RUN_DIR=".runs/qualification/stage0-initial-slot-${RUN_STAMP}-${SHORT_SHA}"
mkdir -p "$RUN_DIR"
uv run --frozen python scripts/write_causal_qualification_lock.py \
  --oracle-job .runs/harbor/causal-qualification/oracle/bayesprobe-terminal-bench-oracle-causal-qualification \
  --provider-identity .runs/qualification/stage0-corrected-20260719T031402Z-cddd396/provider/947834da31f0ffbcfcf275738bee05e0384fbe9d0555fa72f8cc29eab011b0c8.json \
  --output "$RUN_DIR/causal-qualification.lock.json"
```

- [ ] **Step 3: Activate and inspect the lock**

Copy the generated lock to `.runs/causal-qualification.lock.json`, then verify:

```bash
shasum -a 256 .runs/causal-qualification.lock.json
jq '{root_git_sha,adapter_tree_sha,model,budgets,prompt_schema_hashes,tasks}' .runs/causal-qualification.lock.json
```

Expected: Git identities equal clean pushed HEAD; three frozen tasks and budgets are unchanged; Probe and planner hashes differ from corrected-v3 because the slot and inspect policy are now bound.

- [ ] **Step 4: Confirm the secure experiment environment**

In the existing secure shell, require the variable without printing it:

```bash
test -n "$BAYESPROBE_BENCH_API_KEY"
```

Expected: exit code 0.

- [ ] **Step 5: Stop at the live test position**

Prepare, but do not execute, the first-task command:

```bash
HARBOR_TELEMETRY=off \
BAYESPROBE_BENCH_TASK_TIMEOUT_SECONDS=1200 \
uv run --frozen harbor run \
  -c configs/bayesprobe-causal-qualification.yaml \
  --jobs-dir ".runs/harbor/causal-qualification-initial-slot/${SHORT_SHA}/bayesprobe" \
  --dataset terminal-bench/terminal-bench-2@sha256:c6fc2e2382c1dbae99b2d5ecd2f4f4a60c3c01e0d84642d69b4afd92e99d078b \
  --include-task-name terminal-bench/break-filter-js-from-html \
  --job-name bayesprobe-causal-qualification-break-filter-js-from-html \
  --yes
```

Report the implementation commit, test totals, lock SHA-256, prompt/schema hashes,
and exact stop position. Do not start Harbor until the user confirms the live
qualification run.
