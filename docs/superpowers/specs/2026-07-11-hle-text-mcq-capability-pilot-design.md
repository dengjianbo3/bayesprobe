# HLE Text-MCQ Python-Augmented Capability Pilot Design

Date: 2026-07-11
Status: Ready for user review

## 1. Context

BayesProbe can now execute real provider-backed autonomous runs through the
local WebUI. Two public FrontierMath Tier 4 adaptations were answered correctly
with DeepSeek V4 Flash, but those runs are not sufficient evidence of general
capability or methodology effectiveness:

- both source problems and their solutions are public;
- both were adapted into multiple-choice form;
- the current WebUI probe path uses repeated calls to the same model rather
  than an independently verifiable computational tool;
- the existing benchmark fixtures are too small to estimate capability;
- no controlled HLE experiment currently exists.

Humanity's Last Exam (HLE) is a gated benchmark containing 2,500 expert-level
multiple-choice and exact-answer questions, with an optional image on some
items. The official public evaluation uses a fixed test split, asks for an
answer and confidence, defaults temperature to zero, and reports accuracy and
calibration. The dataset owners explicitly request that the data not be
redistributed.

Official references:

- https://github.com/centerforaisafety/hle
- https://huggingface.co/datasets/cais/hle
- https://www.lastexam.ai/

The current BayesProbe benchmark layer is not ready to run this experiment
unchanged. `BenchmarkHarness` passes a model gateway to the evidence gate but
does not inject the provider-backed probe executor used by the WebUI, so its
active probes remain deterministic. The initializer also supports explicit
multiple-choice hypotheses but treats non-multiple-choice questions as binary
claims. The first HLE experiment must therefore be a bounded vertical slice,
not a direct import of the full dataset.

## 2. Decision Summary

Build an independent capability-evaluation layer around the existing public
BayesProbe runner and tool seams. Do not alter BayesProbe's core belief,
evidence, posterior, hypothesis-evolution, or projection semantics.

### 2.1 Approaches Considered

Three implementation shapes were considered:

- Extend `BenchmarkHarness` directly. This is the smallest initial change, but
  it would mix HLE access control, two experiment arms, provider telemetry,
  Python execution, resume, and scoring into a fixture-oriented harness.
- Add an independent capability-evaluation layer. This keeps dataset adapters,
  experiment arms, tools, scoring, and restricted artifacts outside the core
  while reusing public BayesProbe runner seams.
- Add a standalone HLE script. This would produce a number quickly but would
  duplicate provider, retry, artifact, and scoring behavior and would not form
  a trustworthy experimental foundation.

The independent capability-evaluation layer is selected. Its additional
initial engineering cost is justified by gold isolation, repeatability, future
exact-answer/multimodal extension, and clean separation from BayesProbe core.

The first experiment is named:

```text
BayesProbe HLE Text-MCQ-100 Python-Augmented Capability Pilot v0.1
```

Its frozen high-level contract is:

- gated `cais/hle` test split at a pinned revision;
- 100 deterministically selected text-only multiple-choice questions;
- DeepSeek V4 Flash only;
- one Direct Flash arm;
- one BayesProbe + restricted Python arm;
- temperature zero and explicit maximum reasoning effort;
- one system run per arm and sample;
- exploratory analysis with no preregistered accuracy threshold;
- exact label scoring without an LLM judge;
- restricted local artifacts plus a non-sensitive aggregate report.

This is a capability pilot, not a causal methodology study. Direct Flash is a
low-cost reference arm, not a compute-matched control.

## 3. Claims and Non-Claims

### 3.1 Permitted Claim

The strongest permitted result statement is:

```text
BayesProbe + DeepSeek V4 Flash + restricted Python achieved X accuracy on a
fixed, revision-pinned, public HLE Text-MCQ-100 exploratory subset.
```

The report may additionally describe Direct Flash accuracy, the paired
difference, calibration, operational reliability, token use, latency, and
cost.

### 3.2 Prohibited Claims

The pilot must not be described as:

- an official complete HLE score;
- a closed-book HLE score;
- a multimodal HLE score;
- evidence that BayesProbe causally improves DeepSeek;
- evidence that BayesProbe is superior to self-consistency, ReAct, ReWOO, or
  another agent method;
- evidence of autonomous research ability or AGI;
- a contamination-free estimate of unseen-question performance.

### 3.3 Non-Goals

- Do not support HLE exact-answer items in this slice.
- Do not support HLE image items in this slice.
- Do not add web search, retrieval, browser, or remote code tools.
- Do not compare multiple model providers.
- Do not implement a compute-matched self-consistency arm.
- Do not change BayesProbe core update rules to improve the score.
- Do not publish or commit HLE questions, answers, rationales, or raw model
  responses.

## 4. Experiment Identity and Frozen Configuration

Every run has an immutable experiment id derived from:

```text
experiment_name
code_git_sha
dataset_revision_sha
selection_manifest_sha256
config_sha256
prompt_registry_sha256
python_image_digest
```

The v0.1 provider policy is:

```json
{
  "kind": "openai_chat_completions",
  "base_url": "https://api.deepseek.com",
  "model": "deepseek-v4-flash",
  "temperature": 0,
  "top_p": 1,
  "thinking": "enabled",
  "reasoning_effort": "max",
  "max_output_tokens": 65536,
  "timeout_seconds": 900
}
```

The v0.1 BayesProbe policy is:

```json
{
  "max_cycles": 4,
  "max_probes_per_cycle": 2,
  "stop_on_no_probes": true,
  "confidence_threshold": null,
  "posterior_delta_threshold": null
}
```

There is no correctness-conditioned retry, adaptive token increase, or manual
question-specific configuration. The API key is read only from
`DEEPSEEK_API_KEY`. Hugging Face access is read only from the standard local
Hugging Face credential mechanism or `HF_TOKEN`.

## 5. Dataset Contract

### 5.1 Source and Revision

The adapter loads:

```python
load_dataset("cais/hle", split="test", revision=<full commit sha>)
```

The revision must be a full immutable commit SHA. Branch names such as `main`
are invalid for a frozen experiment. Dataset access remains subject to the
conditions accepted by the operator on Hugging Face.

### 5.2 Eligible Pool

An HLE row is eligible only when all of the following hold:

1. `answer_type == "multipleChoice"`.
2. `image` is absent, `None`, or the dataset's documented empty value.
3. `id`, `question`, `answer`, and `category` are non-empty strings.
4. The question contains at least two uniquely labelled answer choices.
5. The gold answer can be mapped unambiguously to exactly one choice label.
6. The canonicalized question can be parsed by BayesProbe's multiple-choice
   initializer into the same labels and choice texts.

The adapter may add the literal `Answer Choices:` delimiter required by the
current initializer, but it must not paraphrase the stem or choices. It records
the original and canonicalized question hashes in the restricted manifest.

Rows rejected during validation are counted by reason before sampling. The
rejection summary is shareable; rejected row content is not.

Preparation fails before writing a manifest when fewer than 100 eligible rows
remain. It never silently reduces the requested sample count.

### 5.3 Deterministic Stratified Selection

The fixed seed is the string `20260711`.

Selection is deterministic:

1. Stable-sort the eligible pool by HLE id.
2. Group rows by exact `category` value.
3. Compute proportional category quotas for 100 items.
4. Assign each category its floor quota.
5. Assign remaining slots by descending fractional remainder, breaking ties by
   category name.
6. Within each category, rank rows by
   `SHA256("20260711:" + sample_id)` and take the allocated count.
7. Stable-sort the final manifest by the same seeded hash.

The selection algorithm, eligible-pool counts, category quotas, and manifest
hash are persisted. A changed dataset revision produces a new manifest and a
new experiment id.

### 5.4 Runtime/Gold Separation

The adapter writes two restricted files:

```text
selection_manifest.json
gold_store.json
```

`selection_manifest.json` contains the runtime case:

```text
sample id
canonicalized question and choices
category
answer type
dataset revision
question hashes
```

`gold_store.json` contains only:

```text
sample id
canonical gold label
```

The rationale and canary fields are never copied from the Hugging Face cache.
Experiment arms receive only runtime cases and have no gold-store path in their
configuration. Scoring is a separate post-run phase.

## 6. Evaluation Architecture

Add a dedicated evaluation package rather than extending the fixture-oriented
`BenchmarkHarness`:

```text
bayesprobe/evaluation/
  contracts.py
  hle.py
  arms.py
  runner.py
  python_probe.py
  provider_telemetry.py
  scoring.py
  statistics.py
  artifacts.py
```

The logical flow is:

```text
Gated HLE dataset
  -> HLEDatasetAdapter
  -> RestrictedSampleManifest + EvaluationGoldStore
  -> CapabilityExperimentRunner
      -> DirectFlashArm
      -> BayesProbePythonArm
  -> terminal per-arm results
  -> MCQScorer joins gold
  -> restricted artifacts + shareable aggregate report
```

Core interfaces:

```python
class ExperimentArm(Protocol):
    def run_case(self, case: EvaluationCase) -> ArmCaseResult: ...


class HLEDatasetAdapter:
    def prepare(self, config: HLESelectionConfig) -> PreparedEvaluationSet: ...


class MCQScorer:
    def score(
        self,
        results: Sequence[ArmCaseResult],
        gold: EvaluationGoldStore,
    ) -> EvaluationScoreReport: ...
```

`BenchmarkHarness` remains unchanged for existing deterministic methodology
fixtures. The new runner reuses public BayesProbe initialization, autonomous
runner, core, projections, and tool-gateway contracts.

## 7. Experiment Arms

### 7.1 DirectFlashArm

Direct Flash makes one structured provider request per case. The model receives
the canonicalized question and choices, but no gold, rationale, dataset
metadata, category, source name, or HLE identifier.

The response schema is:

```json
{
  "answer_label": "C",
  "choice_probabilities": {
    "A": 0.05,
    "B": 0.10,
    "C": 0.70,
    "D": 0.10,
    "E": 0.05
  },
  "answer_summary": "Concise final justification."
}
```

Validation requires:

- `answer_label` is one of the supplied labels;
- probability keys exactly match the supplied labels;
- all probabilities are finite and in `[0, 1]`;
- probabilities sum to one within `1e-3`, after which they are deterministically
  normalized for scoring;
- `answer_summary` is a non-empty string.

One schema-repair request is permitted. It receives the malformed structured
output and schema error but not the gold. A second failure is terminal.

### 7.2 BayesProbePythonArm

This arm constructs the same provider-backed BayesProbe path used by the WebUI,
but injects a Python-augmented `ProbeToolGateway`:

```text
BayesProbeInitializer
  -> ProbePlanner
  -> PythonAugmentedProbeToolGateway
  -> ExternalSignal
  -> BayesProbeCore.integrate_cycle(...)
  -> EvidenceEvent
  -> BeliefState
  -> AnswerProjection
```

The initializer produces one exclusive hypothesis per answer choice with a
uniform prior. The scored answer is the final best hypothesis id. The complete
posterior distribution is retained for calibration.

The runner uses at most four cycles and two probes per cycle. It may stop
earlier only when no probe remains. Confidence and posterior-stability stopping
are disabled for this capability pilot.

### 7.3 Arm Asymmetry

Direct Flash has no Python access. The BayesProbe arm includes both structured
deliberation and Python. Therefore the paired delta measures the combined
system difference, not the isolated effect of BayesProbe control flow.

## 8. Python-Augmented Probe Gateway

### 8.1 Probe Planning Protocol

One model request converts a selected `ProbeDesign` into a structured plan:

```json
{
  "mode": "python",
  "purpose": "Compute the discriminating quantity for choices B and C.",
  "target_hypotheses": ["B", "C"],
  "expected_observation": "The computed value should match one candidate.",
  "code": "print(...)"
}
```

Allowed `mode` values are:

- `python`: execute code in the sandbox;
- `reasoning`: decline Python because computation is not useful and return a
  model-generated informational signal.

Python is available, not mandatory. This prevents the tool from forcing
meaningless code onto humanities or conceptual questions. A reasoning-mode
signal retains `source_type=model_probe_gateway` and the existing conservative
quality baseline.

### 8.2 Docker-Only Sandbox

The benchmark must not execute model-generated code directly on the host.
`DockerPythonSandbox` is the only production backend. There is no host-process
fallback.

The repository supplies a dedicated image definition with pinned package
versions:

```text
Python 3.12
galois==0.4.4
gmpy2==2.2.1
mpmath==1.3.0
networkx==3.4.2
numpy==2.1.3
scipy==1.15.2
sympy==1.13.3
```

The built image digest is written into the experiment identity. A mutable tag
without a resolved digest is rejected by preflight.

Each execution uses:

```text
--network=none
--read-only
--user=<non-root uid>
--cap-drop=ALL
--security-opt=no-new-privileges
--pids-limit=64
--memory=1g
--cpus=1
--tmpfs=/tmp:rw,nosuid,nodev,size=64m
```

Additional limits:

- one complete script per execution;
- code delivered through stdin, with no host mount;
- 30-second wall timeout;
- 64 KiB combined captured output limit;
- `PYTHONHASHSEED=0`;
- numerical-library thread counts fixed at one;
- no access to host files, environment secrets, sockets, or HLE gold data.

### 8.3 Execution Record

The immutable `PythonExecutionRecord` contains:

```text
execution id
run / cycle / probe id
code and code SHA-256
container image digest
started/completed timestamps
wall duration
exit code
stdout/stderr
truncation flag
timeout flag
repair attempt index
```

The code and output are available only in restricted artifacts and to the
Evidence Integration Gate for the current run.

### 8.4 Repair and Failure

One repair is allowed only for a syntax error, non-zero runtime error, or empty
output when the plan explicitly required output. The repair request receives
the original code and sanitized execution error. It never receives gold or
correctness feedback.

A timeout, policy violation, or second failure yields a low-quality external
signal describing the failure. It does not silently disappear and does not
directly update belief.

### 8.5 Signal Semantics

A successful execution becomes:

```text
signal_kind=active
source_type=python_sandbox
source=<pinned image digest>
generated_by_probe=<probe id>
raw_content=<purpose, code, stdout, stderr, exit metadata>
```

Python output has high verifiability but is not fully independent because the
same model generated the code. It must pass through the same Evidence
Integration Gate as every other signal. The gateway must never choose an answer
or modify posterior values directly.

## 9. Provider Request Controls

Extend the reusable OpenAI-compatible configuration with explicit, validated
optional fields:

```text
temperature
top_p
thinking
reasoning_effort
```

Only non-null fields are sent. The HLE pilot requires all four fields to be
explicit. For the official DeepSeek endpoint:

```text
temperature=0
top_p=1
thinking={"type": "enabled"}
reasoning_effort="max"
```

The same provider policy applies to direct answering, probe planning, code
repair, evidence judgment, and structured-output repair. The 65,536-token cap
is fixed across tasks; billing uses actual tokens, not the cap.

No provider seed is assumed because the official interface does not document a
portable deterministic seed. Temperature zero reduces sampling variance but
does not guarantee bit-for-bit provider determinism.

## 10. Provider Telemetry

The OpenAI-compatible adapter accepts an optional invocation observer. The
observer is called after every provider attempt, including failures, without
changing the `ModelGateway.complete_structured(...)` return contract.

Each `ProviderInvocationRecord` contains:

```text
experiment / arm / sample id
run / cycle / probe id when present
task and attempt index
adapter kind / base host / model
prompt id and version
schema name and version
sanitized request hash
started/completed timestamps and latency
input / cached-input / reasoning / output / total tokens
finish reason
provider response id and system fingerprint when available
normalized outcome and error category
```

It excludes:

- API keys and authorization headers;
- raw HLE question text in shareable records;
- raw reasoning content;
- raw provider response bodies in shareable records.

Raw token counts are authoritative. Estimated cost is derived from a dated,
immutable `pricing_snapshot.json`; the report does not present a cost without
also reporting the rate snapshot and token totals.

## 11. Retry and Error Semantics

### 11.1 Retryable Provider Failures

The following receive at most two retries:

- HTTP 429;
- HTTP 5xx;
- connection establishment/reset failures;
- provider read timeout.

Retries use exponential backoff with bounded jitter and honor `Retry-After`.
Every attempt is recorded.

### 11.2 Non-Retryable Outcomes

The following do not receive a transport retry:

- wrong or low-confidence answer;
- an answer unfavourable to the current top hypothesis;
- low posterior margin;
- `finish_reason=length`;
- sandbox policy violation;
- a valid but unhelpful Python result.

Structured-output repair and Python-code repair are separate, single-attempt
policies described above. They are not transport retries.

### 11.3 Scored Failures

If an arm cannot produce a valid terminal answer after its permitted retries
and repairs, the case becomes `terminal_failed`. It remains in the denominator
and is scored incorrect. Operational failure categories are reported
separately.

## 12. Scheduling, Atomicity, and Resume

The direct provider concurrency is eight. The BayesProbe sample concurrency and
Docker-container concurrency are both four.

The runner creates a deterministic 200-task work schedule. For each sample, the
arm order is chosen from the low bit of a SHA-256 hash of the experiment id and
sample id. This avoids placing the same arm first for every sample while
remaining reproducible. Direct and BayesProbe share no response state.

Each arm/sample pair owns an independent restricted directory and ledger:

```text
arms/<arm>/<sample_hmac>/
  status.json
  result.json
  ledger.jsonl
  provider_invocations.jsonl
  python_executions.jsonl
```

Valid states are:

```text
pending -> running -> completed
                   -> terminal_failed
```

Results are written to a temporary file, flushed, and atomically renamed.
Completed and terminal-failed cases are immutable. Resume executes only pending
or stale-running cases. It never inspects gold or correctness when deciding
what to resume.

## 13. Two-Phase Scoring

The experiment runner and scorer are separate commands and processes.

Phase one runs both arms without access to the gold-store path. Phase two is
allowed only after all 200 arm/sample pairs are terminal. It loads the gold
store, validates experiment and manifest hashes, scores the results once, and
writes a scoring-complete marker.

Multiple-choice scoring is deterministic:

```text
correct = terminal answer label exactly equals canonical gold label
```

No LLM judge is used. An invalid/missing label or terminal failure is incorrect.
Human adjudication may flag a dataset issue after the report, but it does not
silently alter the frozen v0.1 result.

## 14. Metrics and Statistical Analysis

### 14.1 Primary Endpoint

BayesProbe accuracy is:

```text
correct BayesProbe cases / 100
```

Terminal failures remain in the denominator. Report a 95% Wilson confidence
interval.

### 14.2 Reference and Paired Metrics

Report Direct accuracy with its Wilson interval and the paired table:

```text
both correct
BayesProbe only correct
Direct only correct
both wrong
```

Also report:

- BayesProbe minus Direct accuracy difference;
- a 10,000-resample paired bootstrap 95% interval with seed `20260711`;
- an exact McNemar test over the discordant pairs.

The pilot is exploratory. These statistics describe the observed difference;
they do not convert the study into a preregistered causal test.

### 14.3 Calibration Metrics

For a valid completed output, let `p_k` be the probability or posterior for
choice `k`, and let `y_k` be the one-hot gold vector.

Report:

- multiclass Brier score `sum_k((p_k - y_k)^2)`;
- log loss `-log(max(p_gold, 1e-6))`;
- top-choice ECE with up to ten equal-frequency bins, using
  `min(10, completed_result_count)` non-empty bins;
- mean top confidence conditioned on correct and incorrect answers;
- probability entropy;
- top-two probability margin;
- calibration coverage.

Calibration is computed only for terminal completed results with a valid full
distribution. Coverage is reported prominently. Failures still count as wrong
in accuracy and are not replaced with an arbitrary probability distribution.

BayesProbe posterior values are labelled internal belief mass until empirical
calibration is established.

### 14.4 Process and Operational Metrics

Report at least:

- cycle and probe counts;
- Python plans, executions, successful executions, repairs, timeouts, and
  policy failures;
- active signals and accepted/discarded evidence events;
- schema violations and evidence-judgment repairs;
- cycle at which the final answer first became top-ranked;
- number of top-answer reversals;
- stop-reason distribution;
- input, cached-input, reasoning, output, and total tokens;
- latency and estimated cost;
- tokens, latency, and cost per correct answer;
- completion and terminal-failure rates.

Category-level accuracy is reported only when the category has at least five
selected cases. Smaller groups remain in the overall metric but receive no
standalone interpretation.

## 15. Artifact and Data Governance

### 15.1 Restricted Artifacts

Restricted artifacts live outside tracked repository content:

```text
artifacts/restricted/hle-pilot-v0.1/
  dataset_revision.json
  selection_manifest.json
  gold_store.json
  config_snapshot.json
  prompt_registry_snapshot.json
  pricing_snapshot.json
  arms/
  score_details.json
```

Directories default to mode `0700`; files default to `0600`. These artifacts
may contain HLE questions, answers, model responses, BayesProbe ledgers, Python
code, and stdout. They must never be committed, uploaded, or copied by the
existing general experiment-artifact writer.

### 15.2 Shareable Report

Shareable outputs contain no raw benchmark content:

```text
reports/hle-pilot-v0.1/
  summary.json
  summary.md
  paired_metrics.json
  provenance.json
```

Per-sample pseudonyms, when needed for paired metrics, use HMAC-SHA256 with a
random experiment secret stored only in the restricted directory. A plain hash
of a public HLE id is not considered irreversible.

Shareable outputs may contain:

- aggregate and category metrics;
- HMAC pseudonyms;
- correctness and terminal state;
- token, latency, and cost totals;
- dataset revision and manifest hash;
- code Git SHA, config hash, prompt versions, and image digest.

They may not contain questions, choices, gold labels, rationales, canary values,
raw model responses, Python source/output, provider secrets, or reversible HLE
identifiers.

### 15.3 Leak Prevention

Implementation must:

- add explicit `.gitignore` rules for HLE cache paths, restricted manifests,
  gold stores, and restricted artifacts;
- scan tracked changes and shareable outputs for the HLE canary;
- recursively verify that no shareable string contains any exact restricted
  question, choice, answer, canary, or provider-secret value;
- prove captured provider requests contain no gold, rationale, or canary;
- sanitize all provider exception messages before persistence.

## 16. CLI and Configuration Surface

Add benchmark-independent evaluation commands:

```text
bayesprobe eval prepare --config <path>
bayesprobe eval run --config <path>
bayesprobe eval score --experiment <restricted-dir>
bayesprobe eval report --experiment <restricted-dir>
```

`prepare` performs gated loading, validates the eligible pool, creates the
manifest/gold split, resolves the Python image digest, and freezes provenance.

`run` performs provider and Docker preflight, then executes or resumes the two
arms without loading gold.

`score` refuses to run until all arm/sample pairs are terminal and writes the
one-time score marker.

`report` emits only the shareable aggregate files and runs the leak scan.

The HLE adapter is an optional dependency surface so normal BayesProbe installs
and tests do not require Hugging Face datasets or gated access.

## 17. Implementation Milestones

### M1: Provider Experiment Controls

- Add typed temperature, top-p, thinking, and reasoning-effort configuration.
- Assemble compatible Chat Completions payloads.
- Add provider invocation observation and usage extraction.
- Persist sanitized provider-policy snapshots.

### M2: Restricted HLE Adapter

- Add optional HLE dependency support.
- Pin revision and validate schema.
- Filter and canonicalize eligible text MCQ rows.
- Generate deterministic stratified manifests and isolated gold stores.

### M3: Experiment Arms and Runner

- Add direct structured MCQ answering.
- Construct the provider-backed BayesProbe runner explicitly.
- Add deterministic scheduling, per-sample artifacts, atomic state, and resume.

### M4: Docker Python Probe

- Build and pin the sandbox image.
- Add structured plan and one-repair protocols.
- Enforce resource/network/filesystem constraints.
- Convert execution records into external signals.

### M5: Scoring and Reporting

- Add deterministic MCQ scorer.
- Add paired and calibration statistics.
- Add restricted/shareable artifact separation and leak scanning.
- Generate JSON and Markdown summaries.

## 18. Verification Strategy

### 18.1 Unit Tests

Use synthetic, non-HLE fixtures to test:

- eligibility filtering and exclusion reasons;
- deterministic category quotas and seeded selection;
- question canonicalization and label mapping;
- runtime/gold separation;
- direct response-schema validation and repair;
- probability validation;
- accuracy, Wilson interval, Brier, log loss, ECE, bootstrap, and McNemar;
- retry classification and attempt bounds;
- per-sample state transitions and atomic resume;
- provider telemetry extraction and secret redaction;
- restricted/shareable artifact schemas and leak rejection.

No HLE question, answer, rationale, id, or canary is committed as a test fixture.

### 18.2 Sandbox Tests

The test suite proves that:

- outbound networking fails;
- a known host file is unavailable;
- host environment secrets are unavailable;
- process-count limits contain fork attempts;
- timeouts terminate the container;
- output limits truncate and mark the record;
- identical deterministic code returns identical output;
- Docker unavailability causes preflight failure;
- no unsafe host-execution fallback exists.

### 18.3 Integration Tests

- Fake providers complete Direct and BayesProbe arms end to end.
- Recorded provider fixtures reproduce the same artifacts offline.
- A real Docker probe solves a synthetic math problem.
- A real DeepSeek live smoke uses only a self-authored, non-HLE problem and is
  enabled explicitly by environment variable.
- Resume after interruption neither duplicates completed work nor skips pending
  work.

### 18.4 Preflight Gate

Before preparing the formal manifest:

```text
all tests pass
git diff --check is clean
Docker image digest resolves
provider preflight passes
sandbox isolation tests pass
restricted paths are ignored
shareable leak scan passes
```

## 19. Formal Execution Protocol

1. Complete implementation and verification using synthetic and self-authored
   samples only.
2. Record the implementation Git SHA.
3. Run `eval prepare` once against a pinned HLE revision.
4. Review only eligibility counts, category quotas, hashes, and preflight
   status; do not inspect selected questions or gold.
5. Freeze config, prompt registry, image digest, manifest, and pricing snapshot.
6. Run both arms to terminal states, resuming only by the formal state machine.
7. Monitor operational health without viewing correctness.
8. Run `eval score` once after all 200 arm/sample pairs are terminal.
9. Generate restricted and shareable reports.
10. Record deviations and limitations before interpreting results.

If prompts, budgets, package versions, model policy, dataset revision, choice
canonicalization, scoring, or retry rules change after the manifest is frozen,
the experiment is invalidated. A corrected run receives a new experiment id;
results from different ids are never merged.

## 20. Resource Estimate

The direct arm makes approximately 100 initial provider calls, plus format
repairs when needed.

The BayesProbe arm can select at most eight probes per sample. In the normal
path, each probe uses one planning call and produces one evidence-judgment call,
for at most approximately 1,600 provider calls across 100 samples. Repair calls
occur only on format or execution failure.

With concurrency eight for Direct and four for BayesProbe, expected wall time
is approximately 8 to 24 hours, depending on reasoning-token use and provider
latency. Current DeepSeek V4 Flash pricing suggests a cost in the tens of US
dollars, but the report treats token telemetry and the frozen pricing snapshot
as authoritative rather than this estimate.

## 21. Definition of Done

Implementation is ready for the formal pilot when:

1. The provider policy is explicit, validated, snapshotted, and observable.
2. The HLE adapter pins revision, filters correctly, selects deterministically,
   and isolates gold.
3. Direct and BayesProbe arms run through the new evaluation runner.
4. BayesProbe active probes use the restricted Python gateway or an explicit
   model-reasoning fallback, never the deterministic fixture gateway.
5. Python execution is Docker-isolated with no host fallback.
6. Per-sample results are atomic, resumable, and correctness-blind.
7. All 100 samples can reach `completed` or `terminal_failed` in both arms.
8. Exact scoring and all preregistered metrics are generated.
9. Restricted artifacts are complete and shareable artifacts pass leak scans.
10. The final report states the experiment's exploratory, text-MCQ,
    Python-augmented, public-set limitations.
