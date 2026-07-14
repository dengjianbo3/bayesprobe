# Tavily Search Evidence Matrix Design

**Status:** Approved on 2026-07-14

## 1. Objective

Add a Tavily-backed retrieval capability that preserves the BayesProbe lifecycle:

```text
Belief State -> Probe -> Tavily retrieval -> Signal -> Evidence judgment -> Update
```

Then run a frozen HLE comparison with two new arms:

- `direct_search`: Direct Flash with Tavily search.
- `bayesprobe_search`: BayesProbe with Tavily search as its probe capability.

The existing 30-case checkpoint supplies the two no-web baseline cells. The new
experiment reports a 2 x 2 matrix without rerunning those baseline cells.

## 2. Research Question

For the same model, the same 30 HLE cases, the same Tavily configuration, and the
same per-case search-call ceiling:

1. Does web retrieval improve Direct accuracy?
2. Does web retrieval improve BayesProbe accuracy?
3. Does BayesProbe use the retrieved evidence more effectively than Direct?
4. Does real retrieval reduce the correct-to-wrong oscillation observed in the
   no-web BayesProbe checkpoint?

The fourth question is diagnostic. This MVP does not claim a general benchmark
result from 30 cases.

## 3. Fixed Experimental Policy

### 3.1 Cases and baselines

- Reuse exactly the checkpoint's frozen 30 sample IDs, questions, choices, and
  gold labels.
- Copy and hash-bind the checkpoint manifest and gold store during preparation.
- Snapshot per-case correctness from the checkpoint's completed
  `direct_flash` and `bayesprobe_python` results.
- Label the old BayesProbe cell accurately as `no_web_python`; it is not a pure
  no-tool arm because the earlier run allowed bounded Python execution.
- Never mutate or overwrite the source checkpoint artifacts.

### 3.2 Model policy

- Use the same OpenAI-compatible DeepSeek model policy as the checkpoint.
- Use temperature `0`, the existing structured-output gateway, and the existing
  evidence judgment and answer projection contracts.
- Provider credentials remain environment-only.

### 3.3 Search policy

Both search arms receive exactly the same ceiling and search parameters:

| Parameter | Frozen value |
|---|---|
| API endpoint | `https://api.tavily.com/search` |
| API key environment | `TAVILY_API_KEY` |
| Maximum logical calls per case | `2` |
| Search depth | `advanced` |
| Topic | `general` |
| Maximum results per call | `5` |
| Chunks per source | `3` |
| Include Tavily answer | `false` |
| Include raw page content | `false` |
| Include images | `false` |
| Maximum query length | `400` characters |
| Request timeout | `60` seconds |
| Automatic retry | none |

At most 120 logical search calls and 240 Tavily credits are required for the
30-case two-arm run. Provider-side retries are not requested because they could
make treatment cost and request counts ambiguous.

## 4. Epistemic Boundary

### 4.1 What is and is not a signal

A model-generated search query is probe planning, not a signal. Tavily search
results are signals. The model's interpretation of a result is evidence judgment,
not raw signal production.

The BayesProbe search gateway must never fall back to `execute_probe` model
reasoning. If search cannot run, it returns no signals and records the operational
outcome separately.

The following do not become evidence:

- authentication failures;
- timeouts;
- rate limits;
- invalid Tavily responses;
- empty result sets;
- search-budget exhaustion;
- malformed model-generated queries.

### 4.2 One source per signal

Each accepted Tavily result becomes one `ExternalSignal` with:

- `epistemic_origin = RETRIEVED_SOURCE`;
- `source_type = retrieved_web_source`;
- `source = <canonical result URL>`;
- `raw_content = title + bounded content chunks`;
- `citations = [<canonical result URL>]`;
- a URL-derived source identity and correlation group;
- a query-, parameter-, URL-, and content-derived derivation root;
- a canonical content fingerprint.

One Tavily response containing five URLs therefore produces up to five signals.
Repeated retrieval of the same canonical URL resolves to the same contribution
root, so it can revise or confirm that source instead of receiving independent
credit. Different URLs remain distinct roots in this MVP. Cross-site syndication
and proposition-level dependency modeling remain out of scope.

### 4.3 Search relevance is not truth

Tavily ranking score is retained as retrieval metadata but does not directly set
Bayesian likelihood or evidence quality. The existing evidence judge evaluates
relevance, reliability, and likelihood effects from the bounded source content
and provenance.

## 5. Components

### 5.1 Tavily client

Create `bayesprobe/tavily_search.py` with a small HTTP client and typed contracts:

- `TavilySearchConfig`
- `TavilySearchRequest`
- `TavilySearchResult`
- `TavilySearchResponse`
- `TavilySearchExecutionRecord`
- `TavilySearchClient`

The client uses a replaceable transport for deterministic tests. It sends the API
key only in the `Authorization: Bearer` header. Exceptions and execution records
must never contain request headers or the key.

The client validates response status and JSON shape before exposing results.
Content is whitespace-normalized and bounded before it reaches a model prompt or
ledger.

### 5.2 Structured search-query planning

Extend the OpenAI-compatible gateway with `plan_web_search` returning exactly:

```json
{"query": "focused query under 400 characters"}
```

The request includes the problem, choices or hypotheses, the current probe goal,
prior search packets when available, and the remaining logical-call budget. Query
planning output is never persisted as evidence.

Extend the gateway with `answer_multiple_choice_with_search`, which reuses the
existing `MultipleChoiceAnswer` output schema while explicitly allowing only the
question, choices, and supplied search packets as its factual basis.

### 5.3 BayesProbe search gateway and arm

Create `TavilyProbeToolGateway`, implementing `ProbeToolGateway.execute_probe`:

1. Reject execution when the run-scoped two-call budget is exhausted.
2. Ask the model for one focused query derived from the selected `ProbeDesign`.
3. Execute one Tavily request.
4. Convert each valid result URL into a retrieved-source signal.
5. Return those signals to `ProbeExecutor`.

`BayesProbeSearchArm` uses the existing initializer, planner, core, evidence gate,
root reconciler, belief solver, and answer projector unchanged. It uses:

- `max_cycles = 4`;
- `max_probes_per_cycle = 1`;
- `max_search_calls = 2`;
- no Python gateway;
- no model-reasoning probe fallback.

One probe per cycle ensures the second search can be selected after the first
search has updated the belief state. Once the search budget is exhausted, an
empty execution produces no evidence and the existing stagnation behavior may
stop the run.

### 5.4 Direct search arm

`DirectSearchArm` performs two bounded, adaptive search rounds:

1. Plan the first query from the question and choices.
2. Search Tavily and retain a normalized search packet.
3. Plan the second query with access to the first packet.
4. Search Tavily again.
5. Produce one structured multiple-choice answer from the question, choices, and
   both packets.

Direct receives the same maximum number of Tavily calls and the same result shape
as BayesProbe. It does not receive BayesProbe belief state, evidence likelihoods,
or posterior values.

If every Tavily request fails operationally, the case is terminal-failed with
`search_treatment_not_delivered`. A successful request with zero results counts as
delivered treatment and may still lead to an answer.

### 5.5 Experiment artifacts

Extend the artifact store and runner only where needed to support experiment-owned
arm names and concurrency. Existing defaults remain:

```text
direct_flash, bayesprobe_python
```

The search experiment uses:

```text
direct_search, bayesprobe_search
```

Each search case has a restricted `search_executions.jsonl` containing sanitized
query, fixed parameters, outcome category, response metadata, URLs, scores, and
bounded snippets. It contains no authorization headers or credentials.

Preparation creates a new immutable identity bound to:

- current clean Git SHA;
- source checkpoint identity and manifest hashes;
- the copied 30-case manifest and gold hash;
- model and prompt registry hashes;
- frozen Tavily policy hash;
- frozen baseline-result snapshot hash.

## 6. Reporting

The shareable report contains no questions, sample IDs, snippets, or secrets. It
reports:

```text
                         No web baseline       Tavily search
Direct                   7/30                  X/30
BayesProbe               3/30                  Y/30
```

It also reports:

- accuracy and Wilson interval for both search arms;
- paired `direct_search` versus `bayesprobe_search` outcomes;
- within-method no-web-to-search correctness transitions;
- BayesProbe cycle-one-to-final correctness transitions;
- search attempts, successes, failures, empty searches, result count, unique URL
  count, and budget exhaustion;
- citations admitted into the BayesProbe signal trail;
- top-answer reversals and stop reasons;
- model token, latency, and cost telemetry already available from the provider
  observer.

The matrix is descriptive. The earlier BayesProbe baseline includes Python, and
the search run changes the available epistemic substrate, so the within-method
difference is not presented as a pure causal estimate.

## 7. Failure Handling

- Missing `TAVILY_API_KEY`: preflight failure before any case starts.
- Missing model key: preflight failure before any case starts.
- Unauthorized Tavily response: sanitized `authentication` outcome, no signal.
- Rate limit: sanitized `rate_limit` outcome, no signal.
- Timeout: sanitized `timeout` outcome, no signal.
- Malformed response: sanitized `invalid_response` outcome, no signal.
- Duplicate URL: retain the audit record, reuse the URL contribution root.
- Query over 400 characters: reject before HTTP, no signal.
- All calls operationally failed: terminal-failed treatment delivery.
- Partial success: continue with successful packets and expose coverage metrics.

Failures never become evidence content and never trigger an internal-reasoning
fallback.

## 8. Security and Privacy

- The Tavily key is accepted only through `TAVILY_API_KEY`.
- The key is never placed in JSON config, model requests, ledgers, telemetry,
  status files, result files, reports, exception messages, or Git history.
- The supplied one-time key is used only for live smoke and the approved run.
- Search artifacts remain under the ignored restricted artifact root with private
  file permissions.
- Report generation scans for both the model-provider key and Tavily key.
- Tests include canary credentials and verify full redaction.

## 9. Test Strategy

Implementation follows test-driven development.

### 9.1 Unit tests

- exact Tavily HTTP request without credential leakage;
- status and response-schema validation;
- query length and empty-query rejection;
- URL canonicalization and bounded content;
- one retrieved-source signal per URL;
- stable same-URL contribution root across repeated retrieval;
- no signal for failure, empty results, malformed query, or exhausted budget;
- Direct second query sees first-round packets;
- both arms enforce the same two-call ceiling;
- structured gateway tasks and schema validation;
- process metrics and secret redaction.

### 9.2 Integration tests

- BayesProbe path is exactly Probe -> Tavily -> Signal -> Evidence -> Update;
- model reasoning is absent from search-arm probe signals;
- evidence events preserve Tavily citations;
- search experiment preparation binds the exact source checkpoint;
- resumable two-arm execution does not overwrite terminal cases;
- matrix scoring rejects sample, gold, baseline, or policy mismatches.

### 9.3 Live verification

- A one-query Tavily smoke test is opt-in and requires
  `BAYESPROBE_RUN_TAVILY_LIVE=1` plus `TAVILY_API_KEY`.
- Run a one-case end-to-end smoke for both arms before the full 30-case run.
- Run the full test suite and leak scan before experiment preparation.
- Prepare and run only from a committed, clean worktree.

## 10. Non-Goals

- Tavily answer generation or Tavily Research API.
- Raw full-page extraction.
- General browser automation.
- Native provider-specific function calling.
- More than two searches per case.
- Source-authority allowlists or domain-specific legal/medical ranking.
- Proposition-level cross-source dependency graphs.
- Replacing the existing evidence judge or root-reconciliation kernel.
- Claiming benchmark superiority from this 30-case exploratory matrix.

## 11. Acceptance Criteria

The feature is complete when:

1. Tavily can be invoked through the standalone client without secret leakage.
2. BayesProbe invokes Tavily only during probe execution.
3. Tavily results enter the core only as retrieved-source signals.
4. Operational failures and budget exhaustion create no evidence.
5. Direct and BayesProbe each enforce two advanced searches per case at most.
6. The same frozen 30 HLE cases and gold labels are used.
7. The two new search arms complete a one-case live smoke.
8. The full 30-case search run can resume safely.
9. A shareable 2 x 2 matrix report is generated and passes leak scanning.
10. All automated tests pass from a clean committed worktree.
