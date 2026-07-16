# Terminal-Bench Causal Paired Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the Terminal-Bench adapter so one real BayesProbe coding-task run faithfully executes `Belief State -> Probe -> Signal -> Evidence -> Update`, qualify that path on the frozen three-task regression set, and only then prepare a reproducible 30-task paired Terminal-Bench 2.1 experiment against a resource-matched reactive control.

**Architecture:** Keep `bayesprobe/` unchanged. Add two benchmark-local `ModelGateway` decorators: one enforces bounded, field-specific provider contracts for terminal task framing and Probe design; the other admits or rejects LLM Evidence judgments using request-bound terminal action lineage. The existing public `BayesProbeCore`, `AutonomousQuestionRunner`, `ModelTaskFramer`, `ModelProbeDesigner`, `ProbeExecutor`, and official Harbor verifier remain authoritative. Experiment code under `benchmarks/terminal_bench` freezes qualification, holdout selection, paired execution, ATIF trajectories, and analysis without introducing a second BayesProbe loop.

**Tech Stack:** Python 3.12+, Pydantic 2, BayesProbe public API, Harbor 0.18.0, OpenAI-compatible Chat Completions, Docker, pytest, standard-library `hashlib`, `json`, `random`, `statistics`, and `subprocess`.

## Global Constraints

- Do not modify any file under `bayesprobe/` or root package behavior. If an adapter requirement cannot be met through the public API, stop and reopen the design.
- Keep all runtime, experiment, and test implementation under `benchmarks/terminal_bench/`; this plan and its approved specification are the only root-level documentation changes.
- Preserve the atomic paradigm: Belief State, Probe, Signal, Evidence, and Update must remain separately observable and causally linked.
- Implementation alternatives, patches, and commands are policies/actions. They must not become rival world hypotheses merely because they are possible solutions.
- LLM interpretation remains mandatory for Evidence. Deterministic code may reject an inadmissible route, but may not invent Evidence, select a patch, or directly alter posterior values.
- A mutation acknowledgement is procedurally neutral. Current code identity cannot refute an unexecuted policy.
- Use initial response plus at most two targeted repairs for terminal task framing, Probe design, terminal planning, and reactive step planning. Exhaustion is `provider_contract_error`; there is no deterministic fallback.
- Keep API keys environment-only. Never place credentials in configs, locks, fixtures, command arguments, reports, trajectories, or Git.
- Do not add Tavily, web search, or online solution retrieval to either Terminal-Bench arm.
- Both experimental arms use the same model, provider endpoint, temperature `0`, task timeout, command timeout, provider timeout, action policy, action ceiling, model-call ceiling, provider-token ceiling, and official verifier.
- Stage 0 uses `max_total_actions=24`, `max_model_calls=72`, `max_provider_tokens=160000`, `max_output_tokens=8192` per request, command timeout `120`, provider timeout `360`, and model-facing terminal-output limit `32768` bytes. The official per-task Harbor timeout remains task-specific and is never extended.
- Stage 1 may not run unless every Stage 0 engineering and causal qualification condition passes.
- Live Stage 0 provider use requires a fresh explicit user authorization. The 180-trial Stage 1 run requires a second explicit authorization after the final lock and projected cost are shown.
- Do not inspect aggregate Stage 1 reward before all paired blocks finish. The orchestrator may inspect completion and error classes only.
- Preserve the user-owned untracked `reports/` directory.
- Commit after each coherent task. Never include `.runs/`, downloaded datasets, provider keys, or generated experiment results in commits.

## File Map

| File | Responsibility |
| --- | --- |
| `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/provider_contract.py` | Terminal framing/Probe-design schemas, bounded repair, response hashes, field-level diagnostics |
| `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/actions.py` | Causally explicit terminal plan steps and intervention predictions |
| `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/planning.py` | Initial plus two-repair terminal planner with exact diagnostics |
| `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/causal.py` | Action bindings, request/state registry, and Evidence admissibility gateway |
| `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/gateway.py` | Execute one frozen plan, register actions, and emit one Signal per completed action |
| `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/signals.py` | Request-bound Signal payloads and fingerprints |
| `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/conformance.py` | Reusable trace classifier and mechanism metrics |
| `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/trajectory.py` | ATIF-v1.7 export for BayesProbe and reactive trials |
| `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/config.py` | Shared action, call, provider-token, and timeout budgets |
| `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/deadline.py` | Shared monotonic task deadline and per-request remaining-time bounds |
| `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/artifacts.py` | Contract, causal, and execution-journal artifacts |
| `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/runner_factory.py` | Public-core composition of the two model decorators and terminal registry |
| `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/agent.py` | Classified BayesProbe Harbor termination and ATIF emission |
| `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/react.py` | Matched reactive controller with the same bounded contract policy |
| `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/direct_agent.py` | Reactive Harbor termination, accounting, and ATIF emission |
| `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/experiment_lock.py` | Stage 0 qualification and Stage 1 immutable experiment locks |
| `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/selection.py` | Exposure exclusion, deterministic stratification, replacement, and randomization |
| `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/orchestration.py` | Serial paired-block execution, retry rules, stop rules, and hash-chained journal |
| `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/analysis.py` | Paired reward statistics, mechanism metrics, and blinded-audit manifests |
| `benchmarks/terminal_bench/scripts/freeze_historical_traces.py` | Produce redacted, content-addressed historical fixtures from old local runs |
| `benchmarks/terminal_bench/scripts/capture_provider_identity.py` | Authorized canary call that freezes actual provider model/fingerprint |
| `benchmarks/terminal_bench/scripts/write_causal_qualification_lock.py` | Freeze the qualified three-task Stage 0 protocol |
| `benchmarks/terminal_bench/scripts/validate_causal_qualification.py` | Run offline replay and validate three live Stage 0 trials |
| `benchmarks/terminal_bench/scripts/prepare_tb21_experiment.py` | Resolve Terminal-Bench 2.1, select 30 tasks, and write a provisional Oracle manifest |
| `benchmarks/terminal_bench/scripts/finalize_tb21_experiment.py` | Apply same-stratum Oracle replacements and freeze the final paired lock |
| `benchmarks/terminal_bench/scripts/run_tb21_experiment.py` | Execute or resume locked paired blocks without reading rewards |
| `benchmarks/terminal_bench/scripts/analyze_tb21_experiment.py` | Unblind completed results and write final statistical reports |
| `benchmarks/terminal_bench/configs/oracle-causal-qualification.yaml` | Frozen three-task Oracle qualification |
| `benchmarks/terminal_bench/configs/bayesprobe-causal-qualification.yaml` | Frozen three-task live BayesProbe qualification |
| `benchmarks/terminal_bench/tests/fixtures/historical_traces/` | Redacted old failures plus fixture manifest and hashes |
| `benchmarks/terminal_bench/tests/fixtures/causal_traces/` | Synthetic conformant and deliberately broken causal traces |
| `benchmarks/terminal_bench/tests/test_provider_contract.py` | Contract validation, repair, telemetry, and secret tests |
| `benchmarks/terminal_bench/tests/test_causal.py` | Action lineage and deterministic Evidence-admission tests |
| `benchmarks/terminal_bench/tests/test_trajectory.py` | ATIF schema and linkage tests |
| `benchmarks/terminal_bench/tests/test_qualification.py` | Historical replay and Stage 0 gate tests |
| `benchmarks/terminal_bench/tests/test_selection.py` | Holdout/exposure/stratification/randomization tests |
| `benchmarks/terminal_bench/tests/test_orchestration.py` | Serial pairing, retry, resume, and stop-rule tests |
| `benchmarks/terminal_bench/tests/test_analysis.py` | Paired estimand, bootstrap, sign-flip, and audit tests |

---

## Milestone A: Correct the Adapter Without Touching the Core

### Task 1: Freeze the historical negative traces as immutable fixtures

**Files:**

- Create: `benchmarks/terminal_bench/scripts/freeze_historical_traces.py`
- Create: `benchmarks/terminal_bench/tests/fixtures/historical_traces/manifest.json`
- Create: `benchmarks/terminal_bench/tests/fixtures/historical_traces/break-filter-js-from-html/`
- Create: `benchmarks/terminal_bench/tests/fixtures/historical_traces/cancel-async-tasks/`
- Create: `benchmarks/terminal_bench/tests/fixtures/historical_traces/log-summary-date-ranges/`
- Create: `benchmarks/terminal_bench/tests/test_historical_fixtures.py`

**Fixture contract:**

```python
class HistoricalTraceManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    schema_version: Literal["terminal_historical_trace:v1"]
    source_commit: str
    traces: tuple[HistoricalTraceRef, ...]

class HistoricalTraceRef(BaseModel):
    task_id: str
    expected_classification: Literal[
        "provider_contract_error", "causal_conformance_error"
    ]
    files: dict[str, str]  # relative path -> sha256:<64 hex>
```

- [ ] Write `test_historical_fixtures.py` first. It must require exactly the three frozen task IDs, verify every file digest, reject absolute paths, reject symlinks, scan decoded text for secret-shaped values, and assert expected classifications are two provider-contract failures plus one causal-conformance failure.
- [ ] Run `uv run pytest tests/test_historical_fixtures.py -q` from `benchmarks/terminal_bench` and confirm RED because the freezer and fixtures do not exist.
- [ ] Implement `freeze_historical_traces.py` with explicit `--source-job`, `--output`, and `--source-commit` arguments. Copy only `bayesprobe_ledger.jsonl`, `provider_telemetry.jsonl`, `plans.jsonl`, `environment_actions.jsonl`, `errors.jsonl`, and `summary.json` when present; reject any source file containing a configured restricted value or a secret pattern; normalize JSON/JSONL using sorted keys; write through a temporary directory and atomic rename.
- [ ] Run the freezer against `.runs/harbor/gate/bayesprobe/bayesprobe-terminal-bench-bayesprobe-gate`, using source commit `12288ad29d162fd9fc8afa296f5f7ec930da9cd0` and no provider key in the command line.
- [ ] Run `uv run pytest tests/test_historical_fixtures.py -q`; expected result: all tests pass and the manifest hashes match the committed fixture bytes.
- [ ] Run `git diff --check` and `git status --short`; confirm `.runs/` and `reports/` remain untouched.
- [ ] Commit: `test(terminal-bench): freeze historical causal failures`

### Task 2: Add the adapter-owned structured provider contract

**Files:**

- Create: `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/provider_contract.py`
- Modify: `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/artifacts.py`
- Create: `benchmarks/terminal_bench/tests/test_provider_contract.py`
- Modify: `benchmarks/terminal_bench/tests/test_artifacts.py`

**Interfaces:**

```python
TERMINAL_HYPOTHESIS_TYPES = frozenset({
    "root_cause",
    "current_behavior",
    "invariant",
    "postcondition",
    "causal_effect",
})

class ProviderContractError(RuntimeError):
    def __init__(self, *, stage: str, attempts: int) -> None: ...

class ContractAttempt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    stage: Literal["terminal_task_frame", "terminal_probe_design"]
    attempt_index: int
    request_task: str
    response_sha256: str | None
    required_keys_present: tuple[str, ...]
    validation: Literal["valid", "invalid", "provider_error", "empty"]
    field_errors: tuple[str, ...]

class TerminalContractModelGateway:
    def complete_structured(self, request: StructuredModelRequest) -> dict[str, Any]: ...
```

`TerminalContractModelGateway` passes unrelated tasks through unchanged. For `frame_open_question` and `design_probes`, it adds a terminal policy object to a copied request, validates the exact semantic payload, and performs at most two repair calls through its delegate. Repair tasks are `repair_task_frame` and `repair_probe_design`; attempts are numbered `1` and `2`. Every delegated call consumes the shared model-call and provider-token budgets.

The terminal frame validator requires `task_kind="design"`, `answer_relationship="synthesis"`, an open hypothesis frame, null `answer_value`, two to six semantically distinct hypotheses, and a hypothesis `type` from `TERMINAL_HYPOTHESIS_TYPES`. It rejects provider-assigned IDs/beliefs and any explicit `implementation_policy`/`patch_choice` hypothesis type. It must not use wording regexes to pretend it can solve semantic classification; mislabeled policy statements are measured by the blinded mechanism audit. The Probe validator requires one to three proposals, known target IDs, exact target-keyed support/weaken conditions, an available terminal capability, and an initial multi-hypothesis discriminator or frame-coverage Probe for an open frame.

Field diagnostics are generated only from Pydantic locations and error types:

```python
def safe_field_errors(error: ValidationError) -> tuple[str, ...]:
    return tuple(
        sorted({
            f"{'.'.join(str(part) for part in item['loc'])}:{item['type']}"
            for item in error.errors(include_url=False, include_input=False)
        })
    )[:32]
```

- [ ] Write tests covering a valid diagnostic frame, rejection of TaskGroup/Semaphore alternatives explicitly labeled `implementation_policy`, unknown Probe targets, missing fields, invalid JSON-shaped mappings, first- and second-repair success, three consecutive failures, provider exceptions, response hashes, required-key telemetry, differential acceptance by the existing public `ModelTaskFramer` and `ModelProbeDesigner`, and secret non-disclosure.
- [ ] Run `uv run pytest tests/test_provider_contract.py tests/test_artifacts.py -q` and confirm RED.
- [ ] Add `append_contract_attempt()` to `TrialArtifactStore`, writing `provider_contract.jsonl` through the existing redactor.
- [ ] Implement `TerminalContractModelGateway`. Persist only response hash, required-key presence, and safe field errors; never persist raw invalid payloads. The repair request may contain the redacted invalid payload in memory, but the artifact may not.
- [ ] Ensure `adapter_kind`, `model_identity`, `config`, and `invocation_observer` delegate exactly as `BudgetedModelGateway` currently does.
- [ ] Run the focused tests, then `uv run pytest tests/test_runner_factory.py tests/test_public_reuse.py -q`.
- [ ] Commit: `feat(terminal-bench): enforce bounded provider contracts`

### Task 3: Make one terminal Probe a causally attributable plan

**Files:**

- Modify: `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/actions.py`
- Modify: `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/planning.py`
- Modify: `benchmarks/terminal_bench/tests/test_actions.py`
- Modify: `benchmarks/terminal_bench/tests/test_planning.py`

**Plan schema:**

```python
class TerminalPlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    role: Literal["inspect", "intervene", "verify"]
    action: TerminalAction
    verification_target: str | None = Field(default=None, max_length=4096)

class TransitionPrediction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    hypothesis_id: str
    expected_transition: str = Field(min_length=1, max_length=4096)

class TerminalProbePlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    mode: Literal["inspect", "intervene", "verify"]
    steps: tuple[TerminalPlanStep, ...] = Field(min_length=1, max_length=3)
    expected_observation: str = Field(min_length=1, max_length=4096)
    transition_predictions: tuple[TransitionPrediction, ...] = ()
```

Validation is exact:

- `inspect`: every step is `inspect`, every action is provably read-only, no transition predictions;
- `verify`: every step is `verify`, every action is `ShellAction`, every step has a non-empty verification target, no transition predictions;
- `intervene`: role order is optional `inspect`, exactly one `intervene`, exactly one or more trailing `verify`; there is exactly one intended mutation; inspect actions are provably read-only; the intervention action may mutate; verification actions are shell commands and have verification targets;
- transition predictions are optional, but when present their IDs equal the Probe targets and their normalized texts are distinct. Without complete differentiated predictions, later verification may update postconditions but not a pre-intervention root-cause or causal-effect hypothesis.

**Repair loop:**

```python
for attempt_index in range(3):
    result = complete(payload, repair=attempt_index > 0)
    try:
        return validate_terminal_plan(result, probe=probe)
    except ValidationError as error:
        record_attempt(attempt_index, error)
        payload = repair_payload(original_input, result, error, attempt_index + 1)
raise TerminalPlanError(category="provider_contract_error", attempts=3)
```

- [ ] Replace current action-plan tests with tests for valid inspect, verify, and `inspect -> intervene -> verify` plans; two mutations; missing verification; verify-before-intervene; write in verify; incomplete/duplicate transition predictions; and immutable tuple normalization.
- [ ] Update planner tests to require initial plus two repairs, field-level errors in repair input, response hashes, shared budget consumption on every attempt, and no imagined fallback.
- [ ] Run `uv run pytest tests/test_actions.py tests/test_planning.py -q` and confirm RED.
- [ ] Implement the models and planner loop. Remove the old `actions` plan field rather than supporting two concurrent plan schemas; this is a locked adapter schema version change to `terminal_probe_plan:v1`.
- [ ] Update `_planner_instruction()` so it explains that writes/patches are interventions, successful mutation output is only acknowledgement, verification must follow the mutation, and transition predictions must be declared before execution.
- [ ] Run focused tests plus `tests/test_environment.py`.
- [ ] Commit: `feat(terminal-bench): make probe plans causally attributable`

### Task 4: Bind every completed action to one Signal and one environment lineage

**Files:**

- Create: `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/causal.py`
- Modify: `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/gateway.py`
- Modify: `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/signals.py`
- Modify: `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/artifacts.py`
- Create: `benchmarks/terminal_bench/tests/test_causal.py`
- Modify: `benchmarks/terminal_bench/tests/test_gateway.py`

**Causal records:**

```python
class CausalActionRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    run_id: str
    cycle_id: str
    probe_id: str
    plan_id: str
    policy_attempt_id: str
    action_id: str
    step_index: int
    action_role: Literal["inspect", "intervene", "verify"]
    request_fingerprint: str
    pre_environment_state_id: str
    post_environment_state_id: str
    subject_environment_state_id: str
    intervention_generation: int
    verification_target: str | None
    transition_predictions: dict[str, str]
    observation: ActionObservation

class CausalTraceRegistry:
    def register_plan(self, *, probe: ProbeDesign, context: ProbeExecutionBrief,
                      plan: TerminalProbePlan) -> RegisteredPlan: ...
    def register_action(self, *, plan: RegisteredPlan, step_index: int,
                        observation: ActionObservation) -> CausalActionRecord: ...
    def bind_signal(self, *, action_id: str, signal_id: str) -> None: ...
    def record_for_signal(self, signal_id: str) -> CausalActionRecord: ...
```

IDs and fingerprints use canonical JSON with sorted keys and SHA-256. `plan_id` hashes run/cycle/Probe/plan, `policy_attempt_id` hashes run/cycle/Probe/intervention plan, and `action_id` hashes plan/step/action-index/request fingerprint. The registry rejects duplicate IDs, a second mutation in one plan, missing or non-linear environment states, and multiple Signals for one action.

Signal raw content becomes:

```json
{
  "action_index": 4,
  "causal_binding": {
    "action_id": "A_...",
    "action_role": "verify",
    "plan_id": "PL_...",
    "policy_attempt_id": "PA_...",
    "request_fingerprint": "sha256:...",
    "subject_environment_state_id": "env:1",
    "verification_target": "the cancellation cleanup invariant"
  },
  "executed_request": {},
  "observation": "...",
  "pre_environment_state_id": "env:1",
  "post_environment_state_id": "env:2"
}
```

- [ ] Write tests for deterministic IDs, exact request binding, one Signal per completed action, no Signal for a rejected/non-executed action, state lineage, mutation acknowledgement role, verify subject state, duplicate binding, output truncation, and large write/patch secrecy.
- [ ] Run `uv run pytest tests/test_causal.py tests/test_gateway.py -q` and confirm RED.
- [ ] Add `append_causal_action()` and `append_causal_decision()` to the artifact store.
- [ ] Implement `CausalTraceRegistry`; keep `ActionObservation` and `HarborEnvironmentBridge` shared with the reactive arm rather than embedding BayesProbe metadata in the low-level bridge.
- [ ] Update `HarborProbeToolGateway` to register the frozen plan before execution, execute `plan.steps` serially, register each completed observation, build and bind exactly one Signal, and re-raise contract/budget errors after recording their stable category.
- [ ] Update signal schema version to `harbor-observation:v3` and tests for canonical fingerprints.
- [ ] Run focused tests plus `tests/test_conformance.py` to expose the expected old-validator failures.
- [ ] Commit: `feat(terminal-bench): bind terminal signals to causal actions`

### Task 5: Gate LLM Evidence judgments by causal admissibility

**Files:**

- Modify: `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/causal.py`
- Modify: `benchmarks/terminal_bench/tests/test_causal.py`
- Modify: `benchmarks/terminal_bench/tests/test_conformance.py`

**Gateway contract:**

```python
class CausalDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    signal_id: str
    action_id: str
    action_role: Literal["inspect", "intervene", "verify"]
    decision: Literal["admit", "discard"]
    reason_code: Literal[
        "state_scoped_inspection",
        "neutral_mutation_acknowledgement",
        "verified_postcondition",
        "preregistered_causal_transition",
        "unbound_signal",
        "stale_state",
        "nonneutral_mutation_acknowledgement",
        "unexecuted_policy_comparison",
        "missing_transition_predictions",
        "target_mismatch",
    ]
    subject_environment_state_id: str
    judgment_response_sha256: str

class CausalEvidenceModelGateway:
    def complete_structured(self, request: StructuredModelRequest) -> dict[str, Any]: ...
```

The gateway delegates first so the LLM still creates the semantic Evidence judgment. It then applies these deterministic rules to `judge_evidence` and `repair_evidence_judgment` responses:

1. Signal ID must resolve to exactly one registered action and request fingerprint.
2. Judgment likelihood keys must equal the request targets.
3. `inspect` may be non-neutral only for a compatible current-state target.
4. `intervene` must be `evidence_type="neutral"`, all target likelihoods `neutral`, and `frame_fit="underdetermined"`; otherwise discard.
5. `verify` may update `current_behavior`, `invariant`, or `postcondition` hypotheses for its subject state.
6. `verify` may update `root_cause` or `causal_effect` only when the plan contains complete, pairwise-distinct transition predictions for every targeted causal hypothesis.
7. No hypothesis of an implementation-policy type is admitted; the terminal frame contract should make this unreachable, and the guard treats it as `unexecuted_policy_comparison` if encountered.
8. A discarded judgment raises `ModelGatewayValidationError("causal_admissibility:<reason_code>")`. The unchanged public core converts this into its existing discarded fail-closed Evidence event, so no contribution delta or posterior Update can be caused by it.

An inspection collected before the single declared intervention in the same plan remains admissible for the plan's pre-intervention diagnosis even though all Signals are integrated after plan completion. A Signal from an older intervention generation or a different policy attempt is stale. Verification is scoped to the intervention generation and to the environment state that existed immediately before the verification command; incidental test-cache writes do not turn the test into a second solution intervention.

An expected guard discard is a valid, observable Evidence outcome, increments `discarded_evidence`, and does not by itself classify the trial as `causal_conformance_error`. Causal conformance fails only when an inadmissible route causes a non-neutral contribution or Update, required lineage is missing/ambiguous/contradictory, the recorded guard decision disagrees with the trace, or the five-stage causal contract is otherwise violated.

- [ ] Write table-driven tests for every reason code. Include the historical Semaphore acknowledgement interpreted against an unexecuted TaskGroup policy and assert `discard`, no accepted contribution delta, and unchanged posterior.
- [ ] Add a test proving valid post-intervention test output can update a declared postcondition.
- [ ] Add a test proving a root-cause update is admitted only with differentiated preregistered transition predictions.
- [ ] Add tests proving same-plan pre-intervention inspection remains admissible after the declared mutation, while a cross-plan/cross-generation inspection is discarded as stale.
- [ ] Add a test proving the delegate LLM is always called before the deterministic decision and that the guard never fabricates a replacement judgment.
- [ ] Run `uv run pytest tests/test_causal.py tests/test_conformance.py -q` and confirm RED.
- [ ] Implement `CausalEvidenceModelGateway`, preserving delegate identity properties and writing one `causal_decisions.jsonl` record per Evidence request.
- [ ] Run focused tests and root `tests/test_paradigm_conformance.py` to confirm the unchanged core contract still passes.
- [ ] Commit: `feat(terminal-bench): reject causally invalid evidence`

### Task 6: Compose the corrected adapter and enforce shared resource accounting

**Files:**

- Modify: `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/config.py`
- Create: `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/deadline.py`
- Modify: `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/runner_factory.py`
- Modify: `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/react.py`
- Modify: `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/agent.py`
- Modify: `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/direct_agent.py`
- Modify: `benchmarks/terminal_bench/tests/test_config.py`
- Modify: `benchmarks/terminal_bench/tests/test_runner_factory.py`
- Modify: `benchmarks/terminal_bench/tests/test_react.py`
- Modify: `benchmarks/terminal_bench/tests/test_agent.py`
- Modify: `benchmarks/terminal_bench/tests/test_direct_agent.py`

**Composition order:**

```python
provider = OpenAIChatCompletionsModelGateway(...)
budgeted = BudgetedModelGateway(provider, budget)
contracted = TerminalContractModelGateway(
    delegate=budgeted,
    artifacts=artifacts,
)
guarded = CausalEvidenceModelGateway(
    delegate=contracted,
    registry=registry,
    artifacts=artifacts,
)
core = BayesProbeCore(
    ledger=ledger,
    model_gateway=guarded,
    judgment_repair_policy=EvidenceJudgmentRepairPolicy(max_attempts=2),
    hypothesis_expander=HypothesisExpansionService(
        adapter=ModelHypothesisExpansionAdapter(guarded)
    ),
)
```

`ModelTaskFramer(guarded)` and `ModelProbeDesigner(guarded)` remain the public core components. No benchmark-local task loop, evidence integrator, belief solver, or posterior updater is permitted.

Extend `RunBudget` with `max_provider_tokens`, `provider_tokens_used`, and `record_provider_usage(total_tokens)`. Missing/non-integer usage is `provider_identity_error`; exceeding the cap is `budget_error`. `ArtifactInvocationObserver` records usage into the shared budget synchronously, and `BudgetedModelGateway` checks the budget immediately after the delegate returns. Terminal and reactive planners record their SDK usage before returning a plan.

Add one `TrialDeadline` created from the locked official agent timeout at Harbor agent start. Generated Harbor configs pass that locked value as `BAYESPROBE_BENCH_TASK_TIMEOUT_SECONDS`; `TerminalBenchConfig` rejects a mismatch. Provider and command timeouts are computed for every call as `min(configured_timeout, floor(deadline.remaining_seconds()))`, with a five-second completion margin. Inject a small OpenAI client proxy into the public `OpenAIChatCompletionsModelGateway`; its `.create()` calls `base_client.with_options(timeout=remaining, max_retries=0)` so core model requests obey the same live deadline. `HarborEnvironmentBridge`, terminal planning, and reactive planning use the same deadline. Non-positive remaining time raises `budget_error` before a request or action starts.

Stable trial error categories are:

```text
provider_contract_error
provider_transport_error
provider_identity_error
budget_error
adapter_error
causal_conformance_error
policy_error
```

- [ ] Write tests for the exact decorator order, every delegated call consuming one logical call, provider token accumulation across core and terminal planner calls, missing usage, token overflow, model/fingerprint drift, deadline reduction on successive calls, no call after deadline exhaustion, plan failure propagation, and category persistence after an agent exception.
- [ ] Change the reactive planner to the same initial-plus-two-repair contract and field-level telemetry; it remains reactive because it has no explicit Belief, Probe, Signal, Evidence, or posterior objects.
- [ ] Run the focused tests and confirm RED.
- [ ] Implement budget/accounting changes, gateway composition, and classified termination. Ensure an initialized `TrialArtifactStore` writes the error even when `runner.run_question()` raises.
- [ ] Update all locked terminal schema versions from `v0.1` to the new `v1` identities; do not silently accept old locks.
- [ ] Run `uv run pytest tests/test_config.py tests/test_runner_factory.py tests/test_agent.py tests/test_react.py tests/test_direct_agent.py tests/test_public_reuse.py -q`.
- [ ] Commit: `feat(terminal-bench): compose causal adapter and shared budgets`

### Task 7: Export complete ATIF-v1.7 trajectories

**Files:**

- Create: `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/trajectory.py`
- Modify: `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/agent.py`
- Modify: `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/direct_agent.py`
- Create: `benchmarks/terminal_bench/tests/test_trajectory.py`

**ATIF contract:**

- Write `agent/trajectory.json`, which is `self.logs_dir / "trajectory.json"`, not inside `agent/bayesprobe/` or `agent/direct/`.
- Set both Harbor agent classes to `SUPPORTS_ATIF = True` only after trajectory validation passes.
- Step 1 is the user instruction.
- Each terminal action is an agent step with one `ToolCall`, one request-bound `ObservationResult`, and `extra` references to plan/action/Signal IDs. Do not include hidden reasoning.
- BayesProbe deterministic state transitions may be represented as `source="agent"`, `llm_call_count=0` steps with `extra` references to Probe, Evidence, and Update IDs.
- The final step records stop reason and artifact identity, not official reward.
- Provider tokens in `final_metrics` must equal the shared budget record.

```python
trajectory = Trajectory(
    schema_version="ATIF-v1.7",
    session_id=session_id,
    trajectory_id=f"trajectory:{run_id}",
    agent=Agent(name=arm_name, version=adapter_version, model_name=model),
    steps=steps,
    extra={"experiment_id": experiment_id, "artifact_schema": "terminal:v1"},
)
TrajectoryValidator().validate(trajectory.to_json_dict())
```

- [ ] Write tests for a successful BayesProbe trajectory, discarded causal Evidence, a provider-contract failure with a terminal system step, a reactive trajectory, sequential step IDs, tool-call/result linkage, no secret values, no hidden evaluator paths, and exact token totals.
- [ ] Run `uv run pytest tests/test_trajectory.py -q` and confirm RED.
- [ ] Implement atomic trajectory writing and validate before replacement. A validation failure is `adapter_error` and must fail Stage 0.
- [ ] Integrate emission into both agents on success and classified failure.
- [ ] Run focused tests plus Harbor's own `TrajectoryValidator` against generated test files.
- [ ] Commit: `feat(terminal-bench): emit ATIF causal trajectories`

---

## Milestone B: Prove Offline Causal Conformance

### Task 8: Replace the script-local validator with a reusable causal validator

**Files:**

- Create: `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/conformance.py`
- Modify: `benchmarks/terminal_bench/scripts/validate_smoke_run.py`
- Modify: `benchmarks/terminal_bench/scripts/validate_paired_gate.py`
- Modify: `benchmarks/terminal_bench/tests/test_conformance.py`
- Create: `benchmarks/terminal_bench/tests/fixtures/causal_traces/conformant-inspect-intervene-verify/`
- Create: `benchmarks/terminal_bench/tests/fixtures/causal_traces/broken-bindings/`

**Public benchmark-local API:**

```python
class TraceClassification(StrEnum):
    CONFORMANT = "conformant"
    PROVIDER_CONTRACT_ERROR = "provider_contract_error"
    CAUSAL_CONFORMANCE_ERROR = "causal_conformance_error"
    BUDGET_ERROR = "budget_error"
    ADAPTER_ERROR = "adapter_error"

class ConformanceReport(BaseModel):
    classification: TraceClassification
    complete_cycles: int
    plans: int
    actions: int
    signals: int
    evidence_events: int
    admitted_evidence: int
    discarded_evidence: int
    nonneutral_updates: int
    violations: tuple[str, ...]
    mechanism_metrics: dict[str, float | int]

def validate_trial_trace(artifact_root: Path) -> ConformanceReport: ...
```

Validation must check contract attempts, exact plan/action/Signal cardinality, request fingerprints, environment lineage, Evidence decisions, Evidence discard behavior, contribution/update causes, prompt/schema provenance, budgets, provider identity, ATIF validity, and absence of evaluator/secret access. A non-neutral Update without exactly one admitted causal route is a violation.

Discarded Evidence with a matching guard decision and no contribution or Update remains conformant. The validator must not turn ordinary fail-closed rejection into a run-level causal error.

- [ ] Write fixture tests first. Required expectations: the two old pre-Probe traces are `provider_contract_error`; the old completed Semaphore/TaskGroup trace is `causal_conformance_error`; the synthetic valid trace is `conformant`; each synthetic broken identity/state/update case is `causal_conformance_error`.
- [ ] Run `uv run pytest tests/test_conformance.py tests/test_historical_fixtures.py -q` and confirm RED.
- [ ] Move generic request/Signal/Evidence/update checks out of `validate_smoke_run.py` into `conformance.py`, then add the causal checks. Keep the scripts as Harbor job/result wrappers.
- [ ] Make classification precedence deterministic: secret/evaluator access, then causal error, provider contract, budget, adapter, conformant.
- [ ] Run focused tests and the entire nested suite.
- [ ] Run `git diff -- bayesprobe` and confirm no output.
- [ ] Commit: `feat(terminal-bench): validate causal trace conformance`

### Task 9: Add the Stage 0 qualification lock and offline gate

**Files:**

- Modify: `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/experiment_lock.py`
- Create: `benchmarks/terminal_bench/scripts/write_causal_qualification_lock.py`
- Create: `benchmarks/terminal_bench/scripts/capture_provider_identity.py`
- Create: `benchmarks/terminal_bench/scripts/validate_causal_qualification.py`
- Create: `benchmarks/terminal_bench/configs/oracle-causal-qualification.yaml`
- Create: `benchmarks/terminal_bench/configs/bayesprobe-causal-qualification.yaml`
- Create: `benchmarks/terminal_bench/tests/test_qualification.py`
- Modify: `benchmarks/terminal_bench/tests/test_experiment_lock.py`

**Lock fields:**

```python
class LockedBudgets(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)
    max_total_actions: int = Field(ge=1)
    max_model_calls: int = Field(ge=1)
    max_provider_tokens: int = Field(ge=1)
    max_output_tokens: int = Field(ge=256)
    command_timeout_seconds: int = Field(ge=1, le=120)
    provider_timeout_seconds: int = Field(ge=1)
    signal_output_bytes: int = Field(ge=1)

class CausalQualificationLock(BaseModel):
    schema_version: Literal["terminal_bench_causal_qualification:v1"]
    harbor_version: Literal["0.18.0"]
    dataset_name: Literal["terminal-bench/terminal-bench-2"]
    dataset_revision: str
    tasks: tuple[GateTask, GateTask, GateTask]
    root_git_sha: str
    adapter_tree_sha: str
    model: str
    base_url: str | None
    provider_protocol: Literal["openai_chat_completions"]
    temperature: Literal[0]
    budgets: LockedBudgets
    prompt_schema_hashes: dict[str, str]
    expected_provider_model: str
    expected_system_fingerprint: str | None
```

The Stage 0 lock validator accepts only `max_total_actions=24`, `max_model_calls=72`, `max_provider_tokens=160000`, `max_output_tokens=8192`, `command_timeout_seconds=120`, `provider_timeout_seconds=360`, and `signal_output_bytes=32768`; the fields are explicit rather than schema defaults. Each `GateTask` also locks its official `agent.timeout_sec` as `agent_timeout_seconds`, because wall time belongs to the task rather than to the shared budget object.

The lock writer requires the three Oracle rewards to be `1.0`, a clean committed adapter tree, exact old task refs/image digests/timeouts, `terminal_probe_plan:v1`, `harbor-observation:v3`, and the canonical hashes returned by `provider_contract.contract_identity()` and `planning.plan_contract_identity()`. It also requires a content-addressed local provider identity artifact from `capture_provider_identity.py`; model and `system_fingerprint` are read from that artifact and may not be entered manually. `expected_system_fingerprint=None` is legal only when the canary response omitted that field; availability as well as value is locked, and every later response must match the canary behavior exactly.

The qualification validator first runs historical replay. It then requires each live BayesProbe task to reach the verifier, contain at least one complete cycle, have complete provenance for every non-neutral Update, stay within all budgets, have one valid ATIF trajectory, and contain zero provider/adapter/budget/causal errors. Official reward is reported but never gates qualification.

- [ ] Write lock and validator tests for Oracle failure, dirty tree, missing/tampered provider canary, stale prompt/schema hash, old plan schema, one live reward-zero conformant pass, one reward-one causal failure, provider-contract failure, missing verifier, missing ATIF, and retry eligibility.
- [ ] Run `uv run pytest tests/test_qualification.py tests/test_experiment_lock.py -q` and confirm RED.
- [ ] Implement the lock, writer, configs, and validator. External 429/5xx/network/Docker failures are marked retryable once; provider-contract, budget, adapter, agent, policy, and conformance failures are never retryable.
- [ ] Run the offline gate command below; it must not call a provider:

```bash
uv run python scripts/validate_causal_qualification.py \
  --historical-fixtures tests/fixtures/historical_traces \
  --offline-only
```

- [ ] Expected JSON: `historical_replay_passed=true`, two `provider_contract_error`, one `causal_conformance_error`, and one conformant synthetic fixture.
- [ ] Run all nested tests, root tests, and `git diff --check`.
- [ ] Commit: `feat(terminal-bench): add causal qualification gate`

### Hard Gate A: Stop before live qualification

- [ ] Show the user the complete offline report, test totals, exact Stage 0 model/base URL, locked budgets, expected maximum provider tokens, and the three tasks.
- [ ] Obtain explicit authorization and a currently valid environment-only provider key.
- [ ] Run exactly one minimal JSON canary through `capture_provider_identity.py`; show its model, fingerprint availability/value, token use, and artifact hash. Retain this immutable artifact as an input to the final Stage 0 lock after Oracle qualification.
- [ ] Do not proceed to Tasks 11-14 if Task 10 has not produced `qualification_passed=true`.

### Task 10: Run live Stage 0 only after authorization

- [ ] Run the frozen Oracle config. Only when all three Oracle rewards are `1.0`, write the final qualification lock from the committed adapter identity, Oracle report, frozen budgets, and the previously captured provider canary.
- [ ] Export `BAYESPROBE_BENCH_API_KEY` in the process environment; never echo it or place it in shell history embedded in the report.
- [ ] Run BayesProbe once per task with `n_concurrent_trials=1`.
- [ ] For one external retryable failure, delete no artifacts; run a fresh retry job and record both trial IDs. Do not retry any internal/system failure.
- [ ] Run `validate_causal_qualification.py` against the lock and live job.
- [ ] Stop immediately on any non-qualification classification. Reward `0` alone does not stop or fail qualification.
- [ ] Write `.runs/qualification/<experiment-id>/qualification-report.json` and report per-task reward, cycles, actions, model calls, provider tokens, causal decisions, discarded Evidence, and provider fingerprint.
- [ ] Do not commit live artifacts or credentials.

---

## Milestone C: Prepare the Locked Terminal-Bench 2.1 Paired Experiment

### Task 11: Resolve the 2.1 holdout and select 30 tasks deterministically

**Files:**

- Create: `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/selection.py`
- Create: `benchmarks/terminal_bench/scripts/prepare_tb21_experiment.py`
- Create: `benchmarks/terminal_bench/tests/test_selection.py`
- Create: `benchmarks/terminal_bench/tests/fixtures/tb21_selection/metadata.json`

**Dataset resolution:**

Use Harbor's public package client, pinned by returned content hash:

```python
client = PackageDatasetClient()
metadata = asyncio.run(
    client.get_dataset_metadata("terminal-bench/terminal-bench-2-1@latest")
)
items = asyncio.run(
    client.download_dataset(
        f"{metadata.name}@{metadata.version}",
        output_dir=dataset_cache,
        export=False,
    )
)
```

Require 89 unique task IDs and a `sha256:` dataset version. Parse each downloaded `task.toml` for `metadata.category`, `metadata.difficulty`, and the official `agent.timeout_sec`; compute the instruction hash from normalized `instruction.md` and retain the package digest. `SelectedTask.agent_timeout_seconds` is mandatory and becomes the only task wall-time value used by generated arm configs.

**Exposure ledger:**

Scan all prior Harbor configs/results under `.runs`, committed historical fixtures, and the known disqualified `terminal-bench/build-cython-ext`. Exclude by task ID when known and by instruction hash otherwise. The generated exposure ledger records source path hash and reason, never raw instructions.

**Selection algorithm:**

1. Group eligible tasks by category.
2. Assign one slot to every category when the number of categories is at most 30.
3. Allocate remaining slots proportionally by category using largest remainder; tie-break category names lexicographically.
4. Within each category, allocate its quota across difficulty strata by largest remainder; tie-break difficulty lexicographically.
5. Within `(category, difficulty)`, sort by `sha256("bayesprobe-tb21-v1:" + task_id)` and select from the front.
6. Keep the unselected ordered tail as the only legal same-stratum Oracle replacement queue.

- [ ] Write tests for exposure by ID/hash, exactly 89 source tasks, duplicate IDs/digests, category minimums, largest-remainder ties, deterministic hash order, 30 unique selected tasks, and same-stratum replacement queues.
- [ ] Run `uv run pytest tests/test_selection.py -q` and confirm RED.
- [ ] Implement immutable Pydantic models `TaskMetadata`, `ExposureRecord`, `SelectedTask`, and `ProvisionalSelection` plus canonical JSON hashing.
- [ ] Implement `prepare_tb21_experiment.py`; output only under `.runs/tb21/<experiment-id>/provisional-selection.json` and generated Oracle config.
- [ ] Run with the fixture metadata offline and confirm byte-identical output across two directories.
- [ ] Commit: `feat(terminal-bench): select deterministic tb21 holdout`

### Task 12: Oracle-qualify the holdout and freeze the paired experiment lock

**Files:**

- Modify: `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/experiment_lock.py`
- Create: `benchmarks/terminal_bench/scripts/finalize_tb21_experiment.py`
- Modify: `benchmarks/terminal_bench/tests/test_experiment_lock.py`
- Create: `benchmarks/terminal_bench/tests/test_tb21_finalize.py`

**Final lock:**

```python
class PairedExperimentLock(BaseModel):
    schema_version: Literal["terminal_bench_paired_experiment:v1"]
    experiment_id: str
    qualification_report_sha256: str
    dataset_name: Literal["terminal-bench/terminal-bench-2-1"]
    dataset_revision: str
    exposure_ledger_sha256: str
    tasks: tuple[SelectedTask, ...]  # exactly 30
    repeats: Literal[3]
    blocks: tuple[PairedBlock, ...]  # exactly 90
    arms: dict[Literal["reactive", "bayesprobe"], str]
    model: Literal["deepseek-v4-flash"]
    expected_provider_model: str
    expected_system_fingerprint: str | None
    budgets: LockedBudgets
    max_experiment_provider_tokens: int
    cost_ceiling_usd: Decimal
    max_usd_per_million_tokens: Decimal
    prompt_schema_hashes: dict[str, str]
    root_git_sha: str
    adapter_tree_sha: str
```

Oracle runs all provisional selections before either real arm. A failed Oracle task is replaced only by the next eligible task in the same `(category, difficulty)` queue; replacements finish before final lock creation. No task may be replaced after the first agent trial.

Provider token ceiling is:

```python
trial_cap = min(
    300_000,
    max(
        160_000,
        math.ceil((1.25 * max_stage0_trial_tokens) / 10_000) * 10_000,
    ),
)
```

`max_experiment_provider_tokens = trial_cap * 180`. `--cost-ceiling-usd` and `--max-usd-per-million-tokens` are required positive arguments supplied during explicit run authorization; neither has a hidden default. The orchestrator uses the conservative upper bound `provider_tokens_used * max_usd_per_million_tokens / 1_000_000` and stops before the next call when that bound reaches the cost ceiling.

The final lock copies the Stage 0 values for action, model-call, output-token, command-timeout, provider-timeout, and signal-output ceilings, but replaces `budgets.max_provider_tokens` with `trial_cap`. The runtime validates each generated trial against its own selected task's `agent_timeout_seconds`; neither arm may substitute a global timeout.

Block order sorts `sha256("bayesprobe-tb21-block-v1:" + task_id + ":" + repeat)`. Arm order uses the low bit of `sha256("bayesprobe-tb21-arm-v1:" + block_id)`. Every task has exactly three repeats and each block contains both arms.

- [ ] Write tests for Oracle replacement, no cross-stratum replacement, replacement exhaustion, 30 final tasks, 90 blocks, deterministic arm order, token formula boundaries, required cost ceiling/rate, stale qualification report, dirty adapter, and lock immutability.
- [ ] Run focused tests and confirm RED.
- [ ] Implement finalization and lock loading. The runtime loader must reject any code, prompt, schema, model, fingerprint, budget, dataset, task, or arm drift.
- [ ] Run focused tests plus existing old-gate lock tests to retain historical-reader compatibility only where explicitly named.
- [ ] Commit: `feat(terminal-bench): freeze paired tb21 experiment`

### Task 13: Execute paired blocks serially with fail-closed resume semantics

**Files:**

- Create: `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/orchestration.py`
- Create: `benchmarks/terminal_bench/scripts/run_tb21_experiment.py`
- Create: `benchmarks/terminal_bench/tests/test_orchestration.py`

**Execution protocol:**

```python
for block in lock.blocks:
    verify_hash_chained_journal()
    for arm in block.arm_order:
        run_fresh_harbor_trial(block, arm)
    pair_class = classify_pair_without_reward(block)
    if pair_class is EXOGENOUS and block.retry_index == 0:
        rerun_both_arms_in_fresh_containers(block)
    elif pair_class is EXOGENOUS:
        mark_both_missing(block)
    elif pair_class is STOP_RULE:
        stop_experiment(block)
```

Generate one one-task/one-attempt Harbor YAML per arm under the experiment `.runs` directory. Invoke Harbor with an argument vector, never `shell=True`. Pair arms are adjacent and serial; blocks are serial; every trial gets a fresh container. The journal stores previous-entry hash, lock hash, block ID, arm, trial/job IDs, start/end timestamps, completion/error class, resource use, and artifact hashes. It does not read or store reward until analysis.

Exogenous categories are provider 429/5xx, network transport, Docker daemon/image pull, and Harbor orchestration failure. System categories are provider contract, provider identity, budget, agent, adapter, causal conformance, evaluator access, and incomplete trace; these count as reward zero during analysis and are never retried.

Stop immediately for first causal-conformance error, lock/hash drift, secret/evaluator access, three consecutive identical exogenous failures, provider identity/token-accounting drift, or authorized cost ceiling breach. There is no efficacy early stopping.

- [ ] Write tests with a fake subprocess runner for serial order, arm adjacency, fresh job names, one whole-pair retry, second exogenous failure becoming paired missing, system error no retry, three consecutive exogenous stop, causal stop, cost stop, journal tampering, safe resume, and reward-file non-access.
- [ ] Run `uv run pytest tests/test_orchestration.py -q` and confirm RED.
- [ ] Implement orchestration and CLI modes `--dry-run`, `--next-block`, and `--resume`. `--dry-run` validates all 180 planned trials without invoking Harbor.
- [ ] Run dry-run against a test lock and inspect the journal for zero reward fields.
- [ ] Commit: `feat(terminal-bench): orchestrate serial paired trials`

---

## Milestone D: Analyze Without Moving the Goalposts

### Task 14: Implement preregistered paired statistics and blinded mechanism audit

**Files:**

- Create: `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/analysis.py`
- Create: `benchmarks/terminal_bench/scripts/analyze_tb21_experiment.py`
- Create: `benchmarks/terminal_bench/tests/test_analysis.py`

**Primary estimand:**

For each task and arm, average official reward over valid repeats. Exogenous missing pairs are absent. System failures are reward `0`. Exclude a task from the primary estimand when fewer than two valid paired repeats remain. Compute:

```python
task_delta[task_id] = mean(bp_rewards) - mean(reactive_rewards)
primary_delta = mean(task_delta.values())
```

Use 10,000 task-level paired bootstrap resamples with deterministic seed derived from `bayesprobe-tb21-bootstrap-v1`. Report percentile 95% interval. Use 100,000 two-sided task-level sign-flip draws with seed `bayesprobe-tb21-signflip-v1`; report `(extreme + 1) / (draws + 1)`.

**Mechanism metrics:**

- complete five-stage cycles per task;
- discriminative Probe rate;
- request-bound Signal rate;
- admitted/discarded Evidence rate by reason;
- non-neutral Update causal-link rate;
- failed-intervention recovery rate;
- provider-contract, budget, adapter, and conformance failure rates;
- actions, model calls, provider tokens, and wall time by arm.

**Blinded audit:**

Select 18 BayesProbe traces by sorting `sha256("bayesprobe-tb21-audit-v1:" + trial_id)`. Emit packets with randomized audit IDs, no task reward, no aggregate result, and no original trial ID. Two reviewers independently score fixed fields: Probe discrimination, Signal binding, Evidence admissibility, state compatibility, and recovery quality. A separate key file maps audit ID to trial ID and is withheld until both review files validate.

- [ ] Write tests for task means, system-zero handling, exogenous missing handling, `<2` repeat exclusion, bootstrap determinism, sign-flip determinism, all-equal deltas, audit selection, blinding, duplicate reviewer rejection, and secret scans.
- [ ] Run `uv run pytest tests/test_analysis.py -q` and confirm RED.
- [ ] Implement analysis with standard-library random generators initialized from SHA-256-derived integer seeds. Never use global random state.
- [ ] Make the CLI refuse to unblind when any planned block lacks a terminal journal state or when the lock/artifact hashes drift.
- [ ] Write machine-readable `results.json`, `mechanisms.json`, `audit-packet.json`, and, after review, `report.md`. Every claim must state that 30 tasks are a preliminary effect estimate, not a universal benchmark conclusion.
- [ ] Commit: `feat(terminal-bench): analyze paired causal experiment`

### Task 15: Reconcile documentation and verify the implementation milestone

**Files:**

- Modify: `benchmarks/terminal_bench/README.md`
- Modify: `benchmarks/terminal_bench/DESIGN.md`
- Modify: `docs/superpowers/specs/2026-07-16-terminal-bench-causal-paired-experiment-design.md` only if implementation discovered an approved, explicitly recorded correction

- [ ] Rewrite the nested README around three commands: offline conformance, authorized Stage 0 qualification, and separately authorized Stage 1 preparation/execution/analysis. Mark the old three-task paired gate as historical and invalid for paradigm effect estimation.
- [ ] Add a causal mapping table to `DESIGN.md` showing terminal diagnosis, Probe plan, action, Signal, Evidence decision, and Update linkage.
- [ ] Document all stable error classes, retry rules, resource ceilings, ATIF location, exposure ledger, no-reward-monitoring rule, and experiment invalidation conditions.
- [ ] Run nested formatting/static checks available in the project and `uv run pytest -q` from `benchmarks/terminal_bench`.
- [ ] Run the root test suite from the repository root.
- [ ] Run `git diff --check`.
- [ ] Run `git diff --name-only -- bayesprobe` and require empty output.
- [ ] Scan the complete staged diff for API-key patterns, Tavily keys, hidden evaluator paths in generated fixtures, unfinished markers, and accidental `.runs` paths. Expected: no secrets and no `TODO`, `TBD`, or `FIXME` in production or experiment protocol files.
- [ ] Run the offline qualification command again and require byte-identical classifications.
- [ ] Commit: `docs(terminal-bench): document causal qualification protocol`

### Hard Gate B: Stop before the 180-trial run

- [ ] Present the Stage 0 qualification report and final Stage 1 lock hash to the user.
- [ ] Present the exact 30 task IDs/categories/difficulties, exposure exclusions, Oracle replacements, block order, arm order, per-trial token ceiling, maximum total provider tokens, user-supplied USD cost ceiling, and conservative maximum USD-per-million-token rate.
- [ ] Confirm adapter Git tree is clean and the lock resolves to current `HEAD` and adapter tree SHA.
- [ ] Obtain explicit authorization for the 180-trial run and a currently valid environment-only provider key.
- [ ] Any code, prompt, schema, model, fingerprint, budget, task, or policy change after authorization invalidates the experiment ID and requires a new lock and full restart.

## Final Completion Criteria

Implementation is complete only when:

1. all nested and root tests pass;
2. historical replay produces exactly the preregistered two provider-contract errors and one causal-conformance error;
3. a synthetic valid inspect/intervene/verify trace passes;
4. no `bayesprobe/` file changed;
5. Stage 0 live qualification passes all three tasks if live execution was authorized;
6. Stage 1 infrastructure can dry-run all 180 locked trials without reading reward;
7. no live Stage 1 result is claimed unless Hard Gate B was separately authorized and the full locked run completed.
