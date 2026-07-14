# Tavily Search Evidence Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Tavily retrieval as a real BayesProbe probe capability and run a fair two-arm HLE search experiment that combines with the frozen no-web checkpoint into a 2 x 2 report.

**Architecture:** A typed Tavily HTTP client returns sanitized result records. BayesProbe plans a query during Probe execution and turns every returned URL into a `RETRIEVED_SOURCE` signal. Direct gets two adaptive search rounds under the same budget. A dedicated search-matrix runner copies and binds the frozen 30-case checkpoint without modifying it.

**Tech Stack:** Python 3.11, stdlib `urllib`, Pydantic, pytest, existing OpenAI-compatible gateway and restricted artifacts.

## Global Constraints

- `TAVILY_API_KEY` is environment-only and never serialized or logged.
- Each search arm has at most two Tavily `advanced` calls per case.
- Every request uses `general`, `max_results=5`, `chunks_per_source=3`, `include_answer=false`, `include_raw_content=false`, and a 60-second timeout.
- Search failure, empty response, invalid query, and exhausted budget create no evidence and never fall back to model reasoning.
- Reuse exactly the frozen 30 checkpoint cases and the existing no-web results.
- Use test-first red-green-refactor for every production module.

---

### Task 1: Implement and test the Tavily client

**Files:**

- Create: `bayesprobe/tavily_search.py`
- Create: `tests/test_tavily_search.py`
- Modify: `bayesprobe/__init__.py`

**Interfaces:**

- `TavilySearchConfig`, `TavilySearchRequest`, `TavilySearchResult`, `TavilySearchResponse`, `TavilySearchExecutionRecord`, `TavilySearchError`, `TavilySearchClient`.
- `TavilySearchClient.search(request) -> TavilySearchResponse` has an injected transport for deterministic tests.

- [ ] **Step 1: Write failing request, validation, and secret-redaction tests**

```python
def test_client_posts_frozen_payload_and_does_not_record_authorization_token():
    client, captured = client_with_transport(success_response())
    response = client.search(TavilySearchRequest(query="test query"))
    assert response.results[0].url == "https://source.test/a"
    assert captured["payload"]["include_answer"] is False
    assert "tvly-" not in json.dumps(client.execution_records())

def test_client_rejects_blank_and_overlong_queries_before_transport():
    with pytest.raises(ValueError, match="query"):
        client.search(TavilySearchRequest(query=" " * 401))
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_tavily_search.py -q`

Expected: import failure because `bayesprobe.tavily_search` is absent.

- [ ] **Step 3: Implement the minimal typed client**

```python
@dataclass(frozen=True)
class TavilySearchConfig:
    api_key_env: str = "TAVILY_API_KEY"
    endpoint: str = "https://api.tavily.com/search"
    search_depth: str = "advanced"
    max_results: int = 5
    chunks_per_source: int = 3
    timeout_seconds: int = 60

class TavilySearchClient:
    def search(self, request: TavilySearchRequest) -> TavilySearchResponse:
        """Issue exactly one logical Tavily request and sanitize its response."""
```

Use stdlib HTTP, an injectable transport, bounded normalized content, and outcome
categories `success`, `authentication`, `rate_limit`, `timeout`,
`provider_error`, and `invalid_response`.

- [ ] **Step 4: Verify GREEN and regressions**

Run: `pytest tests/test_tavily_search.py tests/test_model_gateway.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

Run: `git add bayesprobe/tavily_search.py bayesprobe/__init__.py tests/test_tavily_search.py && git commit -m "feat: add Tavily search client"`

### Task 2: Add structured search-query and search-answer tasks

**Files:**

- Modify: `bayesprobe/openai_gateway.py`
- Modify: `tests/test_openai_gateway.py`
- Modify: `tests/test_model_gateway.py`

**Interfaces:**

- `plan_web_search` produces exactly `{"query": str}`.
- `answer_multiple_choice_with_search` uses the existing `MultipleChoiceAnswer` schema.

- [ ] **Step 1: Write failing task-dispatch tests**

```python
def test_plan_web_search_uses_the_query_schema_and_strict_instruction():
    assert _structured_output_for_task("plan_web_search")[0] == "WebSearchQuery"

def test_search_answer_reuses_multiple_choice_output_schema():
    assert _structured_output_for_task("answer_multiple_choice_with_search")[0] == "MultipleChoiceAnswer"
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_openai_gateway.py -q`

Expected: failure because the task names are unsupported.

- [ ] **Step 3: Add schema, instruction, and response parsing support**

```python
WEB_SEARCH_QUERY_JSON_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["query"],
    "properties": {"query": {"type": "string"}},
}
```

The query instruction must prohibit answer selection. The search-answer instruction
must use only supplied question, choices, and search packets.

- [ ] **Step 4: Verify GREEN**

Run: `pytest tests/test_openai_gateway.py tests/test_model_gateway.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

Run: `git add bayesprobe/openai_gateway.py tests/test_openai_gateway.py tests/test_model_gateway.py && git commit -m "feat: add structured web search prompts"`

### Task 3: Implement Probe-to-Signal Tavily gateway

**Files:**

- Create: `bayesprobe/tavily_probe.py`
- Create: `tests/test_tavily_probe.py`
- Modify: `bayesprobe/__init__.py`

**Interfaces:**

- `TavilyProbeToolGateway(model_gateway, tavily_client, max_search_calls=2)` implements `ProbeToolGateway`.
- `execute_probe(probe, context)` returns one `ExternalSignal` for each valid URL.
- `process_metrics` includes calls, successes, failures, empty searches, result count, unique URLs, malformed queries, and budget exhaustion.

- [ ] **Step 1: Write failing lifecycle tests**

```python
def test_probe_gateway_turns_each_url_into_a_retrieved_source_signal():
    signals = gateway.execute_probe(probe=probe, context=context)
    assert all(s.provenance.epistemic_origin is EpistemicOrigin.RETRIEVED_SOURCE for s in signals)
    assert signals[0].provenance.citations == ["https://source.test/a"]

def test_budget_failure_and_search_failure_return_no_reasoning_signal():
    assert gateway.execute_probe(probe=probe, context=context) == []
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/test_tavily_probe.py -q`

Expected: import failure because the gateway is absent.

- [ ] **Step 3: Implement query planning and provenance conversion**

```python
class TavilyProbeToolGateway:
    def execute_probe(self, *, probe: ProbeDesign, context: ProbeExecutionBrief) -> list[ExternalSignal]:
        query = self._plan_query(probe=probe, context=context)
        response = self._client.search(TavilySearchRequest(query=query))
        return [self._signal_from_result(item, query, probe, context) for item in response.results]
```

Use `RETRIEVED_SOURCE`, a canonical URL citation, URL-derived identity and
correlation group, and a content fingerprint. Return `[]` for all operational or
planning failures; do not invoke the model-backed reasoning gateway.

- [ ] **Step 4: Verify integration with ProbeExecutor and evidence admission**

Run: `pytest tests/test_tavily_probe.py tests/test_probe_executor.py -q`

Expected: PASS with citations retained in the resulting evidence events.

- [ ] **Step 5: Commit**

Run: `git add bayesprobe/tavily_probe.py bayesprobe/__init__.py tests/test_tavily_probe.py && git commit -m "feat: emit Tavily results as probe signals"`

### Task 4: Implement equal-budget Direct and BayesProbe search arms

**Files:**

- Create: `bayesprobe/evaluation/search_arms.py`
- Create: `tests/evaluation/test_search_arms.py`
- Modify: `bayesprobe/evaluation/__init__.py`

**Interfaces:**

- `DirectSearchArm(model_gateway, tavily_client, max_search_calls=2)`.
- `BayesProbeSearchArm(model_gateway, tavily_client, max_search_calls=2, ...)`.
- Both return `ArmCaseResult` with comparable search metrics.

- [ ] **Step 1: Write failing fairness tests**

```python
def test_direct_second_query_receives_first_packet_and_stays_within_budget():
    result = DirectSearchArm(...).run_case(case)
    assert result.process_metrics["search_calls"] == 2
    assert planner_requests[1].input["prior_search_packets"]

def test_bayesprobe_search_calls_tavily_only_inside_probe_execution():
    result = BayesProbeSearchArm(...).run_case(case)
    assert result.process_metrics["search_calls"] <= 2
    assert observed_signal_origins == {EpistemicOrigin.RETRIEVED_SOURCE}

def test_all_operational_search_failures_are_treatment_not_delivered():
    assert result.error_category == "search_treatment_not_delivered"
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/evaluation/test_search_arms.py -q`

Expected: import failure because the search arms are absent.

- [ ] **Step 3: Implement both arms**

Direct follows `plan_web_search -> Tavily -> plan_web_search -> Tavily ->
answer_multiple_choice_with_search`. BayesProbe uses the existing core with one
probe per cycle, a four-cycle ceiling, the two-call Tavily gateway, no Python,
and no model-reasoning probe fallback.

- [ ] **Step 4: Verify GREEN and existing arm behavior**

Run: `pytest tests/evaluation/test_search_arms.py tests/evaluation/test_bayesprobe_arm.py tests/evaluation/test_arms.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

Run: `git add bayesprobe/evaluation/search_arms.py bayesprobe/evaluation/__init__.py tests/evaluation/test_search_arms.py && git commit -m "feat: add fair Tavily search arms"`

### Task 5: Implement artifacts, matrix commands, and score report

**Files:**

- Modify: `bayesprobe/evaluation/artifacts.py`
- Modify: `bayesprobe/evaluation/runner.py`
- Create: `bayesprobe/evaluation/search_matrix.py`
- Modify: `bayesprobe/evaluation/cli.py`
- Create: `configs/hle-search-matrix-v0.1.example.json`
- Create: `tests/evaluation/test_search_matrix.py`
- Modify: `tests/evaluation/test_artifacts.py`
- Modify: `tests/evaluation/test_runner.py`
- Modify: `tests/evaluation/test_evaluation_cli.py`

**Interfaces:**

- Artifact stores accept experiment-owned arm names while retaining existing defaults.
- CLI commands: `search-prepare`, `search-run`, `search-score`, and `search-report`.
- Shareable matrix binds source checkpoint, copied baseline snapshot, Tavily policy, and new arm results.

- [ ] **Step 1: Write failing matrix and leak-scan tests**

```python
def test_search_prepare_copies_exact_checkpoint_selection_and_baseline(tmp_path):
    prepared = prepare_search_matrix(config_path, source_checkpoint)
    assert prepared.sample_count == 30
    assert prepared.baseline["direct_no_web"]["correct"] == 7

def test_matrix_report_rejects_source_manifest_mismatch(tmp_path):
    with pytest.raises(ValueError, match="manifest"):
        score_search_matrix(experiment_path)

def test_shareable_matrix_excludes_question_url_snippet_and_credentials(tmp_path):
    assert_shareable_payload_safe(report, restricted_values=restricted, provider_secrets=[model_key, tavily_key])
```

- [ ] **Step 2: Verify RED**

Run: `pytest tests/evaluation/test_search_matrix.py -q`

Expected: import failure because the search matrix module is absent.

- [ ] **Step 3: Generalize only artifact names and schedule inputs**

Keep default artifact arms `direct_flash` and `bayesprobe_python`. Add per-store
arm names, `search_executions.jsonl`, and deterministic scheduling over a supplied
ordered arm tuple. Existing runners retain their current defaults.

- [ ] **Step 4: Implement prepare, run, score, and report**

Preparation verifies and copies the 30-case source manifest and gold store, stores
baseline correctness without source answers, and binds all non-secret policy hashes.
Scoring reports the 2 x 2 matrix, paired search comparison, no-web-to-search
transitions, BayesProbe cycle transitions, and search coverage without restricted
text.

- [ ] **Step 5: Verify GREEN**

Run: `pytest tests/evaluation/test_search_matrix.py tests/evaluation/test_artifacts.py tests/evaluation/test_runner.py tests/evaluation/test_evaluation_cli.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

Run: `git add bayesprobe/evaluation/artifacts.py bayesprobe/evaluation/runner.py bayesprobe/evaluation/search_matrix.py bayesprobe/evaluation/cli.py configs/hle-search-matrix-v0.1.example.json tests/evaluation && git commit -m "feat: add Tavily HLE search matrix"`

### Task 6: Live smoke, docs, full verification, and matrix run

**Files:**

- Modify: `README.md`
- Modify: `docs/ARCHITECTURE.md`
- Create: `tests/evaluation/test_tavily_live.py`

- [ ] **Step 1: Write the opt-in smoke test**

```python
@pytest.mark.skipif(not os.environ.get("BAYESPROBE_RUN_TAVILY_LIVE"), reason="opt-in")
def test_tavily_live_search_returns_a_sanitized_result():
    response = TavilySearchClient(...).search(TavilySearchRequest(query="Tavily official documentation"))
    assert response.outcome == "success"
```

- [ ] **Step 2: Verify default skip**

Run: `pytest tests/evaluation/test_tavily_live.py -q`

Expected: `1 skipped`.

- [ ] **Step 3: Document configuration and lifecycle semantics**

Document `TAVILY_API_KEY`, the two-call ceiling, Probe-to-Signal behavior, and
the matrix command sequence. No credential appears in examples.

- [ ] **Step 4: Verify the full implementation**

Run: `python -m compileall bayesprobe && pytest -q && git diff --check`

Expected: exit code `0` with all tests passing.

- [ ] **Step 5: Run a one-case live smoke, then prepare and run 30 cases**

Set the Tavily and model credentials only in process environment. Run the two-arm
one-case smoke first, inspect restricted artifacts and leak scan, then run the
30-case matrix from a clean committed worktree. Preserve partial artifacts and
report coverage if a provider limit blocks completion.

## Plan Self-Review

- Tasks 1-4 cover the signal lifecycle and equal-budget arms.
- Task 5 binds the prior checkpoint and produces the matrix without modifying it.
- Task 6 handles live verification, documentation, full tests, and the approved run.
- The plan introduces no placeholders, no credential persistence, and no fallback from failed web retrieval to internal reasoning.
