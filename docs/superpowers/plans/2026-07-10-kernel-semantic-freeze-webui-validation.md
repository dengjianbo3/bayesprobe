# Kernel Semantic Freeze and WebUI Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make BayesProbe's kernel state, audit trail, public run interfaces, and WebUI traces semantically consistent enough for direct manual testing.

**Architecture:** Keep `BayesProbeCore.integrate_cycle(...)` as the deep transition module. Add categorical belief helpers behind the existing belief seam, make Core own terminal cycle records and canonical ledger objects, and let runners own terminal run records. Provider and WebUI changes validate and expose those domain results without bypassing Core.

**Tech Stack:** Python 3.11+, Pydantic 2, dataclasses, pytest 8, stdlib HTTP server, vanilla JavaScript.

## Global Constraints

- BayesProbe remains a complete paradigm, not a ReAct/ReWOO wrapper.
- All external information remains `ExternalSignal` until the Evidence Integration Gate.
- `BayesProbeCore.integrate_cycle(...)` remains the only belief-revision transition used by supported runners.
- No search/retrieval adapter, persistence expansion, provider registry, or networked multi-agent transport in this milestone.
- Provider credentials remain request-scoped and must never enter reports, ledgers, fixtures, or docs.
- Every behavior change starts with a failing test and ends with focused plus full-suite verification.

---

### Task 1: Normalize Rival Beliefs and Rebuild Belief-State Summaries

**Files:**
- Create: `tests/test_belief.py`
- Modify: `bayesprobe/belief.py`
- Modify: `bayesprobe/core.py`
- Modify: `bayesprobe/hypothesis_evolution.py`
- Test: `tests/test_core_cycles.py`
- Test: `tests/test_hypothesis_evolution.py`

**Interfaces:**
- Produces: `normalize_hypotheses(hypotheses: list[Hypothesis]) -> list[Hypothesis]`
- Produces: `summarize_hypotheses(hypotheses: list[Hypothesis]) -> tuple[dict[str, Any], str]`
- Preserves: `solve_updates(...) -> tuple[list[Hypothesis], list[BeliefUpdate]]`

- [ ] **Step 1: Write failing categorical-update tests**

```python
def test_solve_updates_normalizes_exclusive_rivals():
    hypotheses, updates = solve_updates(
        run_id="run",
        cycle_id="cycle_1",
        belief_state=make_three_way_belief_state(),
        events=[support_h2_event()],
    )
    posterior = {hypothesis.id: hypothesis.posterior for hypothesis in hypotheses}
    assert sum(posterior.values()) == pytest.approx(1.0)
    assert posterior["H2"] > posterior["H1"]
    assert posterior["H2"] > posterior["H3"]
    assert {update.hypothesis_id for update in updates} == {"H1", "H2", "H3"}


def test_neutral_likelihood_applies_complexity_penalty():
    state = make_two_way_belief_state(complexity_penalties={"H1": 0.0, "H2": 0.2})
    hypotheses, _ = solve_updates("run", "cycle_1", state, [neutral_event()])
    posterior = {hypothesis.id: hypothesis.posterior for hypothesis in hypotheses}
    assert posterior["H1"] > posterior["H2"]
    assert sum(posterior.values()) == pytest.approx(1.0)
```

- [ ] **Step 2: Run the new tests and verify the independent-sigmoid implementation fails**

Run: `pytest tests/test_belief.py -q`

Expected: FAIL because current rival posteriors are not normalized and non-target rivals do not receive updates.

- [ ] **Step 3: Implement categorical scoring, stable softmax, and exact rounded normalization**

```python
def normalize_hypotheses(hypotheses: list[Hypothesis]) -> list[Hypothesis]:
    active = [hypothesis for hypothesis in hypotheses if _participates_in_distribution(hypothesis)]
    if not active:
        return list(hypotheses)
    total = sum(max(hypothesis.posterior, _MIN_PROBABILITY) for hypothesis in active)
    normalized = {
        hypothesis.id: max(hypothesis.posterior, _MIN_PROBABILITY) / total
        for hypothesis in active
    }
    rounded = _round_distribution(normalized)
    return [
        hypothesis.model_copy(update={"posterior": rounded[hypothesis.id]})
        if hypothesis.id in rounded
        else hypothesis
        for hypothesis in hypotheses
    ]


def _event_distribution(hypotheses: list[Hypothesis], event: EvidenceEvent) -> dict[str, float]:
    weight = event.reliability * event.independence * event.relevance * event.novelty
    scores = {}
    for hypothesis in hypotheses:
        likelihood = event.likelihoods.get(hypothesis.id, LikelihoodBand.NEUTRAL)
        scores[hypothesis.id] = (
            math.log(max(hypothesis.posterior, _MIN_PROBABILITY))
            + math.log(likelihood_band_to_lr(likelihood)) * weight
            - hypothesis.complexity_penalty
            - hypothesis.ad_hoc_penalty
        )
    return _softmax(scores)
```

Create one `BeliefUpdate` per participating hypothesis for every accepted event so normalization-induced rival movement is auditable.

- [ ] **Step 4: Add failing state-summary and evolution-normalization tests**

```python
def test_integrated_belief_state_has_current_summary():
    result = integrate_supportive_cycle()
    total = sum(h.posterior for h in result.belief_state.hypotheses)
    assert result.belief_state.belief_state_id.endswith("_bs_1")
    assert result.belief_state.posterior_summary["total_active_posterior"] == pytest.approx(1.0)
    assert result.belief_state.posterior_summary["top_hypothesis"] == "H1"
    assert "no external signals" not in result.belief_state.uncertainty_summary
    assert total == pytest.approx(1.0)


def test_anomaly_spawn_renormalizes_active_hypotheses():
    result = evolve_anomaly_fixture()
    assert sum(h.posterior for h in result.hypotheses if h.status != HypothesisStatus.RETIRED) == pytest.approx(1.0)
```

- [ ] **Step 5: Rebuild summaries in Core after normalized evolution**

Use `summarize_hypotheses(...)` to populate:

```python
posterior_summary = {
    "top_hypothesis": top.id,
    "top_posterior": top.posterior,
    "runner_up_hypothesis": runner_up.id if runner_up else None,
    "posterior_gap": round(top.posterior - runner_up.posterior, 6) if runner_up else top.posterior,
    "entropy": round(-sum(p * math.log(p) for p in active_posteriors if p > 0), 6),
    "total_active_posterior": round(sum(active_posteriors), 6),
}
```

Set `belief_state_id=f"{cycle.run_id}_bs_{cycle.cycle_index}"` and replace the stale uncertainty text with a current ranked-rival summary.

- [ ] **Step 6: Run focused tests**

Run: `pytest tests/test_belief.py tests/test_core_cycles.py tests/test_hypothesis_evolution.py -q`

Expected: PASS.

- [ ] **Step 7: Commit Task 1**

```bash
git add bayesprobe/belief.py bayesprobe/core.py bayesprobe/hypothesis_evolution.py tests/test_belief.py tests/test_core_cycles.py tests/test_hypothesis_evolution.py
git commit -m "fix: normalize rival belief states"
```

---

### Task 2: Close Cycle and Run Lifecycles and Enforce Regimes

**Files:**
- Modify: `bayesprobe/core.py`
- Modify: `bayesprobe/question_runner.py`
- Modify: `bayesprobe/synchronized_runner.py`
- Modify: `bayesprobe/benchmark.py`
- Test: `tests/test_core_cycles.py`
- Test: `tests/test_question_runner.py`
- Test: `tests/test_synchronized_runner.py`

**Interfaces:**
- Core returns `CycleResult.cycle` with `BoundaryStatus.INTEGRATED`.
- Supported runners return a terminal `RunRecord` with final cycle and regime.

- [ ] **Step 1: Write failing lifecycle tests**

```python
def test_core_returns_integrated_cycle():
    result = integrate_supportive_cycle()
    assert result.cycle.boundary_status == BoundaryStatus.INTEGRATED
    assert result.cycle.boundary_closed_at is not None
    assert result.cycle.completed_at is not None
    assert result.cycle.started_at <= result.cycle.boundary_closed_at <= result.cycle.completed_at


def test_question_runner_returns_completed_run():
    result = run_one_autonomous_cycle()
    assert result.run.regime == RunRegime.AUTONOMOUS
    assert result.run.status == RunStatus.COMPLETED
    assert result.run.current_cycle_id == result.final_belief_state.cycle_id
    assert result.run.metadata["stop_reason"] == result.stop_reason.value


def test_synchronized_runner_forces_synchronized_regime():
    result = run_one_synchronized_round()
    assert result.run.regime == RunRegime.SYNCHRONIZED
    assert result.run.status == RunStatus.COMPLETED
```

- [ ] **Step 2: Run lifecycle tests and verify they fail**

Run: `pytest tests/test_core_cycles.py tests/test_question_runner.py tests/test_synchronized_runner.py -q`

Expected: FAIL on open cycles, running runs, and autonomous synchronized records.

- [ ] **Step 3: Make Core return a terminal copied cycle**

```python
closed_cycle = cycle.model_copy(
    update={
        "boundary_status": BoundaryStatus.CLOSED,
        "boundary_closed_at": utc_now(),
    }
)
# Integrate using closed_cycle.
integrated_cycle = closed_cycle.model_copy(
    update={
        "boundary_status": BoundaryStatus.INTEGRATED,
        "completed_at": utc_now(),
    }
)
```

Validate that input cycles are open and enforce exact signal composition for
active-only, passive-only, and active-plus-passive cycles.

- [ ] **Step 4: Close runs at the runner interface**

```python
completed_run = run.model_copy(
    update={
        "status": RunStatus.COMPLETED,
        "current_cycle_id": final_belief_state.cycle_id,
        "updated_at": utc_now(),
        "metadata": {**run.metadata, "stop_reason": stop_reason.value},
    }
)
```

Use `dataclasses.replace(initialize_input, regime=RunRegime.SYNCHRONIZED)` for
new synchronized runs and reject resumed runs whose regime is not synchronized.

- [ ] **Step 5: Run focused lifecycle tests**

Run: `pytest tests/test_core_cycles.py tests/test_question_runner.py tests/test_synchronized_runner.py tests/test_benchmark_harness.py -q`

Expected: PASS.

- [ ] **Step 6: Commit Task 2**

```bash
git add bayesprobe/core.py bayesprobe/question_runner.py bayesprobe/synchronized_runner.py bayesprobe/benchmark.py tests/test_core_cycles.py tests/test_question_runner.py tests/test_synchronized_runner.py tests/test_benchmark_harness.py
git commit -m "fix: close BayesProbe run lifecycles"
```

---

### Task 3: Make Ledger Records Canonical and Exactly Once

**Files:**
- Modify: `bayesprobe/probe_planner.py`
- Modify: `bayesprobe/probe_executor.py`
- Test: `tests/test_probe_planner.py`
- Test: `tests/test_probe_executor.py`
- Test: `tests/test_question_runner.py`
- Test: `tests/test_synchronized_runner.py`

**Interfaces:**
- Planner writes `probe_planning` diagnostics, never canonical `probe_set`.
- Executor writes `probe_execution` diagnostics, never canonical `external_signal`.
- Core remains canonical owner of `probe_set` and `external_signal` records.

- [ ] **Step 1: Write failing exact-count tests**

```python
def test_question_runner_writes_canonical_cycle_objects_once(tmp_path):
    ledger, result = run_one_cycle_with_ledger(tmp_path)
    rows = ledger.read_all()
    counts = Counter(row["record_type"] for row in rows)
    assert counts["cycle"] == 1
    assert counts["probe_set"] == 1
    assert counts["external_signal"] == len(result.cycle_results[0].signals)
    assert counts["probe_execution"] == 1
    assert counts["probe_planning"] == 1
```

- [ ] **Step 2: Run focused ledger tests and verify duplicate counts fail**

Run: `pytest tests/test_probe_planner.py tests/test_probe_executor.py tests/test_question_runner.py tests/test_synchronized_runner.py -q`

Expected: FAIL because `probe_set` and active `external_signal` are written twice.

- [ ] **Step 3: Replace planner canonical writes with diagnostics**

```python
self._ledger.append(
    "probe_planning",
    {
        "run_id": run_id,
        "cycle_id": cycle_id,
        "probe_set_id": result.probe_set.probe_set_id,
        "selected_candidate_ids": [candidate.candidate_id for candidate in result.selected_candidates],
        "rejected_candidate_ids": [item.candidate.candidate_id for item in result.rejected_candidates],
    },
)
```

- [ ] **Step 4: Remove executor signal writes while retaining execution diagnostics**

Delete the `for signal in result.signals: ledger.append("external_signal", signal)`
loop. Do not change the returned `ProbeExecutionResult`.

- [ ] **Step 5: Run focused tests**

Run: `pytest tests/test_probe_planner.py tests/test_probe_executor.py tests/test_question_runner.py tests/test_synchronized_runner.py -q`

Expected: PASS with exact canonical counts.

- [ ] **Step 6: Commit Task 3**

```bash
git add bayesprobe/probe_planner.py bayesprobe/probe_executor.py tests/test_probe_planner.py tests/test_probe_executor.py tests/test_question_runner.py tests/test_synchronized_runner.py
git commit -m "fix: record cycle objects exactly once"
```

---

### Task 4: Enforce Request-Aware Provider Judgments and Honest Fixtures

**Files:**
- Modify: `bayesprobe/model_gateway.py`
- Modify: `bayesprobe/evidence.py`
- Modify: `bayesprobe/recorded_gateway.py`
- Modify: `fixtures/benchmarks/bayesprobe_v0_2_methodology.json`
- Modify: `fixtures/providers/deepseek_chat_evidence_v0_1.json`
- Modify: `bayesprobe/benchmark.py`
- Test: `tests/test_model_gateway.py`
- Test: `tests/test_core_cycles.py`
- Test: `tests/test_recorded_model_gateway.py`
- Test: `tests/test_experiment_runner.py`
- Test: `tests/test_benchmark_harness.py`

**Interfaces:**
- Produces: request-aware judgment validation inside `EvidenceIntegrationGate`.
- Preserves: `evidence_judgment_from_mapping(...)` as shape-level parser.
- Recorded adapters replay raw provider-shaped mappings, including malformed mappings used to exercise repair.

- [ ] **Step 1: Write failing target and quality validation tests**

```python
def test_missing_target_likelihood_becomes_schema_violation():
    gateway = ScriptedModelGateway({"judge_evidence": {
        "evidence_type": "supporting",
        "likelihoods": {"H1": "moderately_confirming"},
        "interpretation": "H2 was omitted",
    }})
    result = integrate_two_target_signal(gateway)
    assert result.evidence_events[0].discard_reason.startswith("schema_violation:")
    assert result.belief_updates == []


@pytest.mark.parametrize("value", [-0.1, 1.1, float("inf"), float("nan")])
def test_quality_override_must_be_finite_probability(value):
    with pytest.raises(ModelGatewayValidationError):
        evidence_judgment_from_mapping(valid_payload(quality_overrides={"reliability": value}))
```

- [ ] **Step 2: Run provider-contract tests and verify they fail**

Run: `pytest tests/test_model_gateway.py tests/test_core_cycles.py -q`

Expected: FAIL because missing/unknown likelihood targets and invalid overrides are accepted.

- [ ] **Step 3: Add strict quality parsing and request-target validation**

```python
_QUALITY_METRICS = {
    "reliability", "independence", "relevance",
    "novelty", "specificity", "verifiability",
}


def _validate_judgment_targets(judgment: EvidenceJudgment, request: StructuredModelRequest) -> None:
    expected = {str(item) for item in request.input.get("target_hypotheses", [])}
    actual = set(judgment.likelihoods)
    if actual != expected:
        raise ModelGatewayValidationError(
            f"evidence judgment likelihood targets must equal {sorted(expected)}; got {sorted(actual)}"
        )
```

Apply conservative `min(base_quality, override)` behavior for
`source_type="model_probe_gateway"` so a model cannot inflate the reliability or
independence of its own generated signal.

- [ ] **Step 4: Write failing projection-decomposition and real-repair fixture tests**

```python
def test_v02_projection_fixture_generates_source_claim_and_verification_candidate(...):
    assert {event.evidence_type for event in events} >= {
        EvidenceType.SENDER_JUDGMENT,
        EvidenceType.SOURCE_CLAIM,
    }
    assert verification_candidates


def test_v02_repair_fixture_records_repair_attempt(...):
    assert repaired_event.discard_reason is None
    assert repaired_event.model_trace["task"] == "repair_evidence_judgment"
    assert repaired_event.model_trace["repair_attempt_index"] == 1
```

- [ ] **Step 5: Make recorded fixtures capable of replaying malformed provider output**

Keep `_validate_entry(...)` responsible for fixture structure and secret
rejection, but remove eager `EvidenceJudgment` parsing. Add an invalid
`judge_evidence` response for `S_v02_schema_repair_passive` and a valid
`repair_evidence_judgment` response matched by task.

Change the projection benchmark source type to `external_agent_projection`.
Enable one repair attempt in the recorded experiment test.

- [ ] **Step 6: Score net update direction and meaningful revision per evidence**

For each hypothesis, compare the first update prior with the final update
posterior. Compute belief revision efficiency as total-variation movement per
accepted evidence event:

```python
movement = 0.5 * sum(abs(last.posterior - first.prior) for first, last in update_ranges)
efficiency = movement / accepted_evidence_count if accepted_evidence_count else 0.0
```

- [ ] **Step 7: Run focused provider and benchmark tests**

Run: `pytest tests/test_model_gateway.py tests/test_core_cycles.py tests/test_recorded_model_gateway.py tests/test_experiment_runner.py tests/test_benchmark_harness.py -q`

Expected: PASS and the fixture repair trace uses attempt index 1.

- [ ] **Step 8: Commit Task 4**

```bash
git add bayesprobe/model_gateway.py bayesprobe/evidence.py bayesprobe/recorded_gateway.py bayesprobe/benchmark.py fixtures/benchmarks/bayesprobe_v0_2_methodology.json fixtures/providers/deepseek_chat_evidence_v0_1.json tests/test_model_gateway.py tests/test_core_cycles.py tests/test_recorded_model_gateway.py tests/test_experiment_runner.py tests/test_benchmark_harness.py
git commit -m "fix: enforce provider evidence contracts"
```

---

### Task 5: Publish Supported Run Interfaces and Expose Semantics in WebUI

**Files:**
- Modify: `bayesprobe/__init__.py`
- Modify: `bayesprobe/webui.py`
- Modify: `bayesprobe/webui_static/app.js`
- Test: `tests/test_public_api_and_config.py`
- Test: `tests/test_webui.py`

**Interfaces:**
- Package root exports supported Core, initialization, autonomous, synchronized,
  tool, ledger, and domain interfaces.
- WebUI response includes `run` plus terminal cycle and current belief summaries.

- [ ] **Step 1: Write failing package-root and WebUI serialization tests**

```python
def test_package_root_exports_supported_agent_interfaces():
    from bayesprobe import (
        AutonomousQuestionRunConfig,
        AutonomousQuestionRunner,
        BayesProbeCore,
        InitializeRunInput,
        JsonlLedgerStore,
        SynchronizedRoundInput,
        SynchronizedRoundRunner,
        SynchronizedRunInput,
    )


def test_webui_serializes_terminal_run_and_cycle():
    status, payload = handle_autonomous_run_request(deterministic_payload())
    assert status == 200
    assert payload["run"]["status"] == "completed"
    assert payload["run"]["regime"] == "autonomous"
    assert payload["cycles"][0]["cycle"]["boundary_status"] == "integrated"
    assert payload["final_belief_state"]["posterior_summary"]["total_active_posterior"] == pytest.approx(1.0)
```

- [ ] **Step 2: Run public/WebUI tests and verify missing exports and fields fail**

Run: `pytest tests/test_public_api_and_config.py tests/test_webui.py -q`

Expected: FAIL on package-root imports and missing `payload["run"]`.

- [ ] **Step 3: Export the stable run surface and serialize the terminal run**

Add supported names to `bayesprobe/__init__.py` and add:

```python
"run": _dump_domain(result.run),
```

to `serialize_autonomous_run_result(...)`.

- [ ] **Step 4: Render lifecycle and normalized belief metadata**

Update `app.js` so successful status reads from the domain record:

```javascript
setStatus(
  `${payload.run?.status || "completed"}: ${payload.run?.regime || "autonomous"} / ${payload.stop_reason}`,
  "ok"
);
```

Show `posterior_summary.total_active_posterior` and current uncertainty above the
hypothesis rows. Include `boundary_status` in each cycle summary and add a
`Cycle lifecycle` trace block.

- [ ] **Step 5: Run focused public/WebUI tests**

Run: `pytest tests/test_public_api_and_config.py tests/test_webui.py -q`

Expected: PASS.

- [ ] **Step 6: Commit Task 5**

```bash
git add bayesprobe/__init__.py bayesprobe/webui.py bayesprobe/webui_static/app.js tests/test_public_api_and_config.py tests/test_webui.py
git commit -m "feat: expose semantic run state in webui"
```

---

### Task 6: Regress, Document, Restart, and Manually Verify the WebUI

**Files:**
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/superpowers/specs/2026-07-10-kernel-semantic-freeze-webui-validation-design.md`
- Test: all tests

**Interfaces:**
- No new runtime interface. This task verifies and documents the frozen surface.

- [ ] **Step 1: Run the complete test suite**

Run: `pytest -q`

Expected: all tests pass; live-provider tests may skip unless explicitly enabled.

- [ ] **Step 2: Run repository hygiene checks**

Run: `git diff --check`

Expected: no output.

Run: `rg -n "sk-[A-Za-z0-9]" . --glob '!*.pyc' --glob '!.git/**'`

Expected: no real API key in tracked project files.

- [ ] **Step 3: Update architecture truthfully**

Record the implemented categorical belief family, terminal lifecycle rules,
exactly-once ledger ownership, provider target contract, public run exports, and
remaining limitation that model-backed probes are closed-book model signals,
not verified retrieval.

Mark the design spec `Status: Implemented` only after verification succeeds.

- [ ] **Step 4: Restart the local WebUI**

Run the existing WebUI entry point on `127.0.0.1:8766`. If the port is occupied
by the previous BayesProbe process, stop that process cleanly first and start the
new code on the same port.

- [ ] **Step 5: Verify deterministic MCQ behavior in the browser**

Submit a two- or five-choice question and assert visually and through the API:

- the answer is a concrete choice rather than generic H1 prose;
- posterior mass totals 1.000;
- run status is completed/autonomous;
- cycle boundary is integrated;
- signal -> evidence -> belief update remains visible;
- no text overlaps at desktop and mobile widths.

- [ ] **Step 6: Verify OpenAI-compatible failure and success behavior**

Use request-scoped credentials only. Confirm invalid provider configuration
returns a concise provider error without a fake result. When valid credentials
are available, confirm the provider-backed run follows the same terminal domain
path.

- [ ] **Step 7: Commit and push the completed milestone**

```bash
git add docs/ARCHITECTURE.md docs/superpowers/specs/2026-07-10-kernel-semantic-freeze-webui-validation-design.md
git commit -m "docs: record kernel semantic freeze"
git push origin main
```
