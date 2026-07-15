# Terminal-Bench Paired Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a budget-matched Direct/ReAct control and a preregistered three-task Terminal-Bench gate without changing the BayesProbe kernel.

**Architecture:** The nested benchmark project owns both Harbor adapters. BayesProbe continues to call the public `AutonomousQuestionRunner`; the Direct arm uses a benchmark-local `Thought -> Action -> Observation` controller. Both arms share `TerminalBenchConfig`, `RunBudget`, `ActionPolicy`, `HarborEnvironmentBridge`, low-level action contracts, provider settings, and the official Harbor verifier. A gate lock freezes task identities and controls before either real arm runs.

**Tech Stack:** Python 3.12+, Harbor 0.18.0, Pydantic 2, OpenAI-compatible Chat Completions, pytest, Docker.

## Global Constraints

- Do not modify `bayesprobe/` or root package behavior.
- Keep all new benchmark behavior under `benchmarks/terminal_bench/`.
- The Direct arm must not import or imitate BayesProbe belief, probe, signal, evidence, or update components.
- Both arms use the same provider, model, base URL, temperature `0`, token limit, provider timeout, command timeout, action policy, action budget, model-call budget, task timeout, and verifier.
- Keep `max_total_actions=24`, `max_model_calls=72`, command timeout `120`, provider timeout `360`, and model-facing observation limit `32768` bytes for this gate.
- Freeze these tasks before experimental-arm outcomes: `terminal-bench/break-filter-js-from-html`, `terminal-bench/cancel-async-tasks`, and `terminal-bench/log-summary-date-ranges`.
- Record `terminal-bench/build-cython-ext` as disqualified during Oracle qualification because its upstream repository test failed before either experimental arm ran.
- The already-observed zero reward on `break-filter-js-from-html` remains part of the gate and may not be replaced.
- Oracle must pass all three tasks before either experimental arm runs.
- The gate passes only when both arms reach the verifier on all three tasks and BayesProbe earns reward `1` on at least one task.
- Only an infrastructure failure before any agent action may be retried once. Reward `0` is a task result, never a retry reason.
- Secrets remain environment-only and must not appear in configs, locks, logs, summaries, or committed files.
- Preserve the user-owned untracked `reports/` directory.

## File Map

| File | Responsibility |
| --- | --- |
| `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/react.py` | Direct/ReAct step schema, provider planner, and bounded controller |
| `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/direct_agent.py` | Harbor entry point for the Direct arm |
| `benchmarks/terminal_bench/src/bayesprobe_terminal_bench/experiment_lock.py` | Immutable paired-gate task and fairness contract |
| `benchmarks/terminal_bench/scripts/write_paired_gate_lock.py` | Resolve official task refs, validate Oracle results, and write the gate lock |
| `benchmarks/terminal_bench/scripts/validate_paired_gate.py` | Validate paired completion, rewards, fairness, traces, and secret hygiene |
| `benchmarks/terminal_bench/configs/oracle-gate.yaml` | Three-task Oracle bootstrap |
| `benchmarks/terminal_bench/configs/direct-gate.yaml` | Three-task Direct/ReAct job |
| `benchmarks/terminal_bench/configs/bayesprobe-gate.yaml` | Three-task BayesProbe job |
| `benchmarks/terminal_bench/tests/test_react.py` | Direct controller unit and budget tests |
| `benchmarks/terminal_bench/tests/test_direct_agent.py` | Harbor async bridge and metadata tests |
| `benchmarks/terminal_bench/tests/test_experiment_lock.py` | Lock immutability and fairness tests |
| `benchmarks/terminal_bench/tests/test_paired_gate.py` | Offline result validator tests |

---

### Task 1: Add the Direct/ReAct controller

**Interfaces:**

- `ReActStep`: strict JSON with `thought_summary`, `actions`, `done`, and `completion_summary`.
- `OpenAICompatibleReActPlanner.next_step(instruction, history) -> ReActStep` reserves exactly one logical model call and allows one schema-repair call.
- `ReActController.run(instruction) -> ReActRunResult` executes only shared `TerminalAction` values through `HarborEnvironmentBridge` and stops on `done` or a hard budget.

- [ ] Write tests proving strict schema validation, bounded/redacted history, one repair, shared action and model budgets, policy rejection behavior, and stop behavior.
- [ ] Run `uv run pytest tests/test_react.py -q` and confirm failures are caused by missing Direct components.
- [ ] Implement the minimal planner and controller in `react.py`; reuse action models, bridge, policy, budget, and artifact store.
- [ ] Run `uv run pytest tests/test_react.py -q` and the existing planning/environment/gateway tests.

### Task 2: Add the Direct Harbor adapter

**Interfaces:**

- `DirectHarborAgent(BaseAgent)` reads the same `TerminalBenchConfig.from_sources` inputs as `BayesProbeHarborAgent`.
- It runs the synchronous controller in `asyncio.to_thread`, updates `AgentContext.metadata`, and writes a redacted summary.
- Metadata keys are `experiment_arm`, `stop_reason`, `terminal_actions`, and `model_calls`.

- [ ] Write async tests for successful execution, cancellation, provider failure, metadata bounds, and secret non-disclosure.
- [ ] Run `uv run pytest tests/test_direct_agent.py -q` and observe RED.
- [ ] Implement `direct_agent.py` and the small Direct session factory needed by the tests.
- [ ] Run the new tests plus `tests/test_agent.py` and `tests/test_public_reuse.py`.

### Task 3: Freeze the paired three-task gate

**Interfaces:**

- `PairedGateLock` records the dataset digest, ordered task refs, repository identities, model controls, budgets, arm import paths, and repetition count.
- `load_paired_gate_lock(path, config, arm, runtime_identity)` rejects changed order, refs, controls, dirty adapter state, unknown arm, or secret-shaped fields.
- Existing one-task `benchmark.lock.json` remains valid for the engineering smoke.

- [ ] Write tests for exact task order, immutable refs, fairness equality, runtime Git identity, and secret-shaped fields.
- [ ] Run `uv run pytest tests/test_experiment_lock.py -q` and observe RED.
- [ ] Implement `experiment_lock.py` without changing the one-task lock parser.
- [ ] Add `oracle-gate.yaml`, `direct-gate.yaml`, and `bayesprobe-gate.yaml` with identical dataset, tasks, provider controls, budgets, and `n_attempts=1`.
- [ ] Run config and lock tests.

### Task 4: Add reproducible lock generation and gate validation

**Interfaces:**

- `write_paired_gate_lock.py --oracle-job PATH --output PATH` reads Harbor's resolved lock/results, requires reward `1` for all three Oracle trials, resolves image/task digests, and atomically writes JSON.
- `validate_paired_gate.py --lock PATH --direct-job PATH --bayesprobe-job PATH` emits JSON containing per-task rewards, completion/error class, aggregate reward, budget use, and `gate_passed`.
- The validator fails when tasks differ, a verifier did not complete, controls differ, traces are malformed, or a secret pattern is present.

- [ ] Write fixture-based tests for a passing gate, BayesProbe `0/3`, missing verifier, task substitution, provider error, infrastructure retry eligibility, and leaked secret.
- [ ] Run `uv run pytest tests/test_paired_gate.py -q` and observe RED.
- [ ] Implement both scripts with atomic writes and deterministic JSON ordering.
- [ ] Run all nested benchmark tests.

### Task 5: Document and verify the offline milestone

- [ ] Update `benchmarks/terminal_bench/README.md` with exact Oracle, Direct, BayesProbe, and validation commands plus the gate interpretation.
- [ ] Update `benchmarks/terminal_bench/DESIGN.md` to reconcile the fixed smoke limits (`3` cycles, `72` model calls) and record the frozen three tasks.
- [ ] Run `uv run pytest -q` in the nested project.
- [ ] Run the root test suite and `git diff --check`.
- [ ] Confirm `git diff -- bayesprobe pyproject.toml` is empty and scan tracked/untracked benchmark artifacts for secrets without reading or modifying `reports/`.

### Task 6: Run the live gate

- [ ] Obtain explicit authorization for provider use for this gate; do not reuse a historical one-time key implicitly.
- [ ] Run Oracle on the three frozen tasks and write `.runs/paired-gate.lock.json` only if all rewards are `1`.
- [ ] Run Direct and BayesProbe once each with `n_concurrent_trials=1`.
- [ ] Validate and report every task result, failure classification, action/model-call use, latency, and total token usage.
- [ ] Advance to the 10-task pilot only if the preregistered gate criteria pass.
