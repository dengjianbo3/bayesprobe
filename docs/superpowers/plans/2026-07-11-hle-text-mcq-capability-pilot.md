# HLE Text-MCQ Python-Augmented Capability Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the frozen two-arm HLE text-only multiple-choice capability pilot so Direct Flash and Python-augmented BayesProbe can be run, resumed, scored, and reported without exposing gold during inference or benchmark content in shareable artifacts.

**Architecture:** A new `bayesprobe.evaluation` package owns restricted dataset preparation, arm execution, atomic artifacts, two-phase scoring, and aggregate reporting. It reuses the public BayesProbe initializer/core/autonomous runner interfaces, while a generic provider observer and explicit request controls deepen the existing OpenAI-compatible gateway. Model-generated Python runs only through a pinned Docker sandbox and returns an `ExternalSignal`; it never selects an answer or mutates posterior state directly.

**Tech Stack:** Python 3.11+, dataclasses and Pydantic already used by BayesProbe, OpenAI-compatible Chat Completions, optional Hugging Face `datasets`, Docker Engine, stdlib statistics/concurrency/filesystem APIs, pytest.

## Global Constraints

- Preserve the BayesProbe atomic control flow: initialize beliefs, plan probes, collect external signals, judge evidence, integrate posterior, project an answer.
- Do not extend the fixture-oriented `BenchmarkHarness` for this experiment.
- Never pass a gold label, HLE id, category, rationale, canary, dataset name, or gold-store path to either arm.
- Never copy HLE rationale or canary fields into any BayesProbe artifact.
- Keep restricted data outside tracked content with directory mode `0700` and file mode `0600`.
- Treat terminal failures as incorrect and keep them in the accuracy denominator.
- Make resume decisions only from per-arm state, never from correctness or gold.
- Use the same frozen provider policy for every model task in both arms.
- Execute model-generated code only through `DockerPythonSandbox`; there is no host fallback.
- Keep the existing `ModelGateway.complete_structured(...) -> dict[str, Any]` contract unchanged.
- Add no HLE question, answer, id, rationale, or canary to source control or tests.
- Use only synthetic, self-authored fixtures until the formal `eval prepare` command is explicitly run by the operator.

---

## File Map

- Create `bayesprobe/provider_telemetry.py`: provider attempt records, observer protocol, usage/error extraction, JSONL observer.
- Modify `bayesprobe/model_gateway.py`: reusable provider controls in `ModelGatewayConfig` and observer-aware factory.
- Modify `bayesprobe/openai_gateway.py`: validated sampling/reasoning controls, retry policy, and attempt observation.
- Modify `bayesprobe/config.py`: parse and validate the new provider fields without accepting raw keys.
- Modify `bayesprobe/experiment_artifacts.py`: sanitize and snapshot the expanded provider policy.
- Create `bayesprobe/evaluation/contracts.py`: cases, gold records, arm results, state and report contracts.
- Create `bayesprobe/evaluation/statistics.py`: deterministic accuracy, confidence, paired, and calibration metrics.
- Create `bayesprobe/evaluation/hle.py`: gated HLE loading, validation, canonicalization, deterministic stratified selection.
- Create `bayesprobe/evaluation/artifacts.py`: restricted/shareable stores, atomic state, HMAC pseudonyms, leak scanner.
- Create `bayesprobe/evaluation/arms.py`: direct structured MCQ arm and provider-backed BayesProbe arm.
- Create `bayesprobe/evaluation/python_probe.py`: planning schema, Docker sandbox, execution records, and probe gateway.
- Create `bayesprobe/evaluation/runner.py`: frozen identity, preflight, deterministic scheduling, concurrency, resume.
- Create `bayesprobe/evaluation/scoring.py`: one-time gold join, metrics, report rendering, score marker.
- Create `bayesprobe/evaluation/config.py`: strict JSON configuration and frozen defaults for the pilot.
- Create `bayesprobe/evaluation/cli.py`: `prepare`, `run`, `score`, and `report` command handlers.
- Create `bayesprobe/evaluation/__init__.py`: intentional public evaluation API.
- Create `docker/hle-python-sandbox/Dockerfile`: pinned non-root Python image.
- Create `configs/hle-pilot-v0.1.example.json`: secret-free frozen configuration example.
- Modify `bayesprobe/cli.py`: mount the `eval` command group without changing `run`.
- Modify `bayesprobe/__init__.py`: export stable evaluation contracts only.
- Modify `pyproject.toml`: optional `hle` dependency extra.
- Modify `.gitignore`: explicit restricted HLE and local cache exclusions.
- Modify `docs/ARCHITECTURE.md`: document the evaluation boundary and formal-run protocol.
- Test in `tests/evaluation/` plus focused existing provider/config/artifact/CLI tests.

---

### Task 1: Freeze Provider Request Controls

**Files:**
- Modify: `bayesprobe/model_gateway.py`
- Modify: `bayesprobe/openai_gateway.py`
- Modify: `bayesprobe/config.py`
- Modify: `bayesprobe/experiment_artifacts.py`
- Test: `tests/test_openai_gateway.py`
- Test: `tests/test_model_gateway.py`
- Test: `tests/test_public_api_and_config.py`
- Test: `tests/test_experiment_artifacts.py`

**Contract:**

```python
@dataclass(frozen=True)
class ProviderRequestControls:
    temperature: float | None = None
    top_p: float | None = None
    thinking: str | None = None
    reasoning_effort: str | None = None
```

`thinking="enabled"` serializes as `{"type": "enabled"}`. Null values are omitted. Temperature must be finite and non-negative, `top_p` finite in `(0, 1]`, and non-null string controls must be non-empty.

- [ ] Add failing tests that the Chat Completions payload includes exactly the four explicit controls and omits all four when unset.
- [ ] Add failing tests for invalid finite/range/string values in both direct construction and JSON config loading.
- [ ] Add failing artifact tests proving `api_key` is absent while the explicit policy is present.
- [ ] Run RED:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tests/test_openai_gateway.py tests/test_model_gateway.py \
  tests/test_public_api_and_config.py tests/test_experiment_artifacts.py \
  -q -p no:cacheprovider
```

Expected: assertions fail because the request controls do not exist.

- [ ] Implement `ProviderRequestControls`; embed it in `ModelGatewayConfig` and `OpenAIModelGatewayConfig`; pass it through `build_model_gateway(...)`.
- [ ] Update `build_openai_chat_completions_payload(...)` to emit:

```python
if controls.temperature is not None:
    payload["temperature"] = controls.temperature
if controls.top_p is not None:
    payload["top_p"] = controls.top_p
if controls.thinking is not None:
    payload["thinking"] = {"type": controls.thinking}
if controls.reasoning_effort is not None:
    payload["reasoning_effort"] = controls.reasoning_effort
```

- [ ] Verify GREEN with the focused command above and ensure deterministic/OpenAI Responses tests remain unchanged.
- [ ] Commit:

```bash
git add bayesprobe/model_gateway.py bayesprobe/openai_gateway.py bayesprobe/config.py \
  bayesprobe/experiment_artifacts.py tests/test_openai_gateway.py \
  tests/test_model_gateway.py tests/test_public_api_and_config.py \
  tests/test_experiment_artifacts.py
git commit -m "feat: freeze provider experiment controls"
```

---

### Task 2: Observe Every Provider Attempt

**Files:**
- Create: `bayesprobe/provider_telemetry.py`
- Modify: `bayesprobe/model_gateway.py`
- Modify: `bayesprobe/openai_gateway.py`
- Test: `tests/test_provider_telemetry.py`
- Test: `tests/test_openai_gateway.py`

**Contracts:**

```python
@dataclass(frozen=True)
class ProviderInvocationContext:
    experiment_id: str | None = None
    arm: str | None = None
    sample_id: str | None = None
    run_id: str | None = None
    cycle_id: str | None = None
    probe_id: str | None = None
    attempt_index: int = 1

@dataclass(frozen=True)
class ProviderInvocationRecord:
    task: str
    adapter_kind: str
    model: str
    base_host: str | None
    request_sha256: str
    started_at: str
    completed_at: str
    latency_seconds: float
    input_tokens: int | None
    cached_input_tokens: int | None
    reasoning_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    finish_reason: str | None
    response_id: str | None
    system_fingerprint: str | None
    outcome: str
    error_category: str | None
    context: ProviderInvocationContext

class ProviderInvocationObserver(Protocol):
    def observe(self, record: ProviderInvocationRecord) -> None: ...
```

- [ ] Write RED tests for mapping/object response usage, DeepSeek/OpenAI usage detail variants, finish reason, response id, and system fingerprint.
- [ ] Write RED tests proving a successful attempt and an exception each notify the observer exactly once.
- [ ] Write RED tests proving request hashes are stable, Authorization/API key values never enter records, and observer errors cannot alter model results.
- [ ] Implement normalization helpers and a lock-safe `JsonlProviderInvocationObserver` with atomic line append and `0600` creation.
- [ ] Wrap both OpenAI gateway calls with monotonic timing and `try/except/finally` observation while preserving parsed return values.
- [ ] Add bounded transport retries: at most two retries for 429, 5xx, connect/reset, and read timeout; no retry for `finish_reason=length` or schema errors. Inject sleeper/random functions for deterministic tests and honor integer/date `Retry-After`.
- [ ] Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tests/test_provider_telemetry.py tests/test_openai_gateway.py \
  -q -p no:cacheprovider
```

Expected: all tests pass with exactly one record per attempt.

- [ ] Commit:

```bash
git add bayesprobe/provider_telemetry.py bayesprobe/model_gateway.py \
  bayesprobe/openai_gateway.py tests/test_provider_telemetry.py \
  tests/test_openai_gateway.py
git commit -m "feat: record provider invocation telemetry"
```

---

### Task 3: Add Evaluation Contracts and Statistics

**Files:**
- Create: `bayesprobe/evaluation/__init__.py`
- Create: `bayesprobe/evaluation/contracts.py`
- Create: `bayesprobe/evaluation/statistics.py`
- Test: `tests/evaluation/test_contracts.py`
- Test: `tests/evaluation/test_statistics.py`

**Core contracts:**

```python
@dataclass(frozen=True)
class EvaluationCase:
    sample_id: str
    question: str
    choices: dict[str, str]

@dataclass(frozen=True)
class ArmCaseResult:
    sample_id: str
    arm: str
    state: Literal["completed", "terminal_failed"]
    answer_label: str | None
    probabilities: dict[str, float] | None
    error_category: str | None = None
    process_metrics: dict[str, int | float | str | None] = field(default_factory=dict)
```

- [ ] Write RED validation tests for unique labels, complete finite distributions, normalization within `1e-3`, invalid terminal states, and failures without fabricated probabilities.
- [ ] Write table-driven RED tests for Wilson 95% intervals, paired contingency counts, exact two-sided McNemar, seeded paired bootstrap, multiclass Brier, clipped log loss, equal-frequency ECE, entropy, and top-two margin.
- [ ] Implement pure functions with no provider/filesystem imports. Use `statistics.NormalDist().inv_cdf(0.975)` and `math.comb` rather than adding a statistics dependency.
- [ ] Ensure bootstrap results are byte-stable for seed string `20260711` by deriving an integer through SHA-256.
- [ ] Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/evaluation/test_contracts.py \
  tests/evaluation/test_statistics.py -q -p no:cacheprovider
```

- [ ] Commit:

```bash
git add bayesprobe/evaluation tests/evaluation/test_contracts.py \
  tests/evaluation/test_statistics.py
git commit -m "feat: add capability evaluation contracts and statistics"
```

---

### Task 4: Prepare a Restricted, Gold-Isolated HLE Set

**Files:**
- Create: `bayesprobe/evaluation/hle.py`
- Create: `bayesprobe/evaluation/artifacts.py`
- Modify: `pyproject.toml`
- Modify: `.gitignore`
- Test: `tests/evaluation/test_hle.py`
- Test: `tests/evaluation/test_artifacts.py`

**Selection config:**

```python
@dataclass(frozen=True)
class HLESelectionConfig:
    revision: str
    sample_count: int = 100
    seed: str = "20260711"
```

- [ ] Write synthetic RED tests for every eligibility rejection reason, choice parsing, literal `Answer Choices:` canonicalization, ambiguous/missing gold, and fewer-than-100 failure.
- [ ] Write RED tests for floor-plus-largest-remainder category quotas, category-name tie-breaks, seeded within-category ranking, and final manifest order.
- [ ] Write RED tests proving runtime cases omit gold/category/source metadata and `gold_store.json` contains only sample id plus canonical label.
- [ ] Write RED tests that a non-full revision SHA is rejected and that lazy import emits an actionable `bayesprobe[hle]` error.
- [ ] Implement a row-oriented pure `prepare_rows(...)` path first; make `load_dataset("cais/hle", split="test", revision=...)` a thin lazy adapter so unit tests never access Hugging Face.
- [ ] Write restricted files with `os.open(..., 0o600)`, fsync, and atomic rename; create directories with `0o700`.
- [ ] Add optional dependency:

```toml
hle = [
  "datasets>=3,<5",
]
```

- [ ] Add ignore rules:

```gitignore
artifacts/restricted/
.cache/huggingface/
**/selection_manifest.json
**/gold_store.json
```

- [ ] Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/evaluation/test_hle.py \
  tests/evaluation/test_artifacts.py -q -p no:cacheprovider
git check-ignore artifacts/restricted/hle-pilot-v0.1/gold_store.json
```

- [ ] Commit:

```bash
git add bayesprobe/evaluation/hle.py bayesprobe/evaluation/artifacts.py \
  pyproject.toml .gitignore tests/evaluation/test_hle.py \
  tests/evaluation/test_artifacts.py
git commit -m "feat: prepare restricted HLE evaluation sets"
```

---

### Task 5: Implement Structured Direct MCQ Answering

**Files:**
- Modify: `bayesprobe/openai_gateway.py`
- Create: `bayesprobe/evaluation/arms.py`
- Test: `tests/evaluation/test_arms.py`
- Test: `tests/test_openai_gateway.py`

**Tasks and schema:**

```text
answer_multiple_choice
repair_multiple_choice_answer
```

```json
{
  "answer_label": "C",
  "choice_probabilities": {"A": 0.05, "B": 0.10, "C": 0.70, "D": 0.10, "E": 0.05},
  "answer_summary": "Concise final justification."
}
```

- [ ] Write RED payload tests proving only question and choices are sent and all benchmark metadata/gold fields are absent.
- [ ] Write RED response tests for exact label keys, finite/range/sum validation, deterministic normalization, one schema repair, and terminal failure after a second invalid response.
- [ ] Add generic task schemas/instructions for direct MCQ and repair without weakening evidence-judgment validation.
- [ ] Implement `DirectFlashArm.run_case(...)` using one initial request and at most one schema-repair request; context metadata supplies telemetry ids but not raw benchmark identity in shareable records.
- [ ] Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/evaluation/test_arms.py \
  tests/test_openai_gateway.py -q -p no:cacheprovider
```

- [ ] Commit:

```bash
git add bayesprobe/openai_gateway.py bayesprobe/evaluation/arms.py \
  tests/evaluation/test_arms.py tests/test_openai_gateway.py
git commit -m "feat: add direct multiple choice evaluation arm"
```

---

### Task 6: Build the Docker-Only Python Probe Gateway

**Files:**
- Create: `docker/hle-python-sandbox/Dockerfile`
- Create: `bayesprobe/evaluation/python_probe.py`
- Test: `tests/evaluation/test_python_probe.py`
- Test: `tests/evaluation/test_python_sandbox_integration.py`

**Contracts:**

```python
@dataclass(frozen=True)
class PythonProbePlan:
    mode: Literal["python", "reasoning"]
    purpose: str
    target_hypotheses: tuple[str, ...]
    expected_observation: str
    code: str | None

class DockerPythonSandbox:
    def preflight(self) -> ResolvedSandboxImage: ...
    def execute(self, request: PythonExecutionRequest) -> PythonExecutionRecord: ...
```

- [ ] Write RED schema tests for `python` requiring code, `reasoning` forbidding code, supplied target ids only, and one plan repair.
- [ ] Write RED command-construction tests asserting `--network=none`, `--read-only`, non-root user, all caps dropped, no-new-privileges, pids/memory/cpu/tmpfs limits, fixed thread/hash env, stdin delivery, and no host mounts.
- [ ] Write RED behavior tests for 30-second timeout, 64 KiB combined output cap, truncation, exit metadata, immutable execution id/hash, and no host fallback when Docker is missing.
- [ ] Add a pinned Python 3.12 image with `galois==0.4.4`, `gmpy2==2.2.1`, `mpmath==1.3.0`, `networkx==3.4.2`, `numpy==2.1.3`, `scipy==1.15.2`, and `sympy==1.13.3`; create and switch to an unprivileged user.
- [ ] Implement code repair only for syntax error, non-zero runtime error, or required-but-empty output. Timeout/policy failure never repairs.
- [ ] Convert reasoning output or Python execution output to an `ExternalSignal` with `source_type` equal to `model_probe_gateway` or `python_sandbox`; do not create evidence or posterior updates in this module.
- [ ] Run unit tests, then build and resolve the image digest:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/evaluation/test_python_probe.py \
  -q -p no:cacheprovider
docker build -t bayesprobe-hle-python:v0.1 docker/hle-python-sandbox
docker image inspect bayesprobe-hle-python:v0.1 --format '{{index .RepoDigests 0}}'
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tests/evaluation/test_python_sandbox_integration.py -q -p no:cacheprovider
```

Expected: network/host-file/secret access tests fail inside the container; deterministic math succeeds; a digest is resolved.

- [ ] Commit:

```bash
git add docker/hle-python-sandbox/Dockerfile \
  bayesprobe/evaluation/python_probe.py tests/evaluation/test_python_probe.py \
  tests/evaluation/test_python_sandbox_integration.py
git commit -m "feat: add isolated Python probe gateway"
```

---

### Task 7: Construct the Provider-Backed BayesProbe Arm

**Files:**
- Modify: `bayesprobe/evaluation/arms.py`
- Modify: `bayesprobe/evaluation/python_probe.py`
- Test: `tests/evaluation/test_bayesprobe_arm.py`

- [ ] Write a RED end-to-end test with a scripted provider proving one choice hypothesis per label, uniform prior, provider-backed probe planning, Python/reasoning signal generation, evidence judgment, posterior update, and final answer projection.
- [ ] Assert the test fails if the deterministic probe executor is accidentally used.
- [ ] Write RED tests for fixed `max_cycles=4`, `max_probes_per_cycle=2`, `stop_on_no_probes=True`, disabled confidence/stability stops, full final posterior distribution, process counters, reversals, and first-final-top cycle.
- [ ] Construct explicitly:

```text
BayesProbeInitializer
ProbePlanner
PythonAugmentedProbeToolGateway
ProbeExecutor
BayesProbeCore
AutonomousQuestionRunner
```

- [ ] Keep the answer label equal to the projected best hypothesis id. Do not allow the arm or gateway to inspect gold.
- [ ] Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tests/evaluation/test_bayesprobe_arm.py tests/evaluation/test_arms.py \
  -q -p no:cacheprovider
```

- [ ] Commit:

```bash
git add bayesprobe/evaluation/arms.py bayesprobe/evaluation/python_probe.py \
  tests/evaluation/test_bayesprobe_arm.py
git commit -m "feat: add Python-augmented BayesProbe evaluation arm"
```

---

### Task 8: Add Frozen Config, Experiment Identity, and Resumable Runner

**Files:**
- Create: `bayesprobe/evaluation/config.py`
- Create: `bayesprobe/evaluation/runner.py`
- Modify: `bayesprobe/evaluation/artifacts.py`
- Create: `configs/hle-pilot-v0.1.example.json`
- Test: `tests/evaluation/test_config.py`
- Test: `tests/evaluation/test_runner.py`

- [ ] Write RED config tests for exact v0.1 defaults: DeepSeek endpoint/model, temperature 0, top-p 1, thinking enabled, max reasoning, 65,536 output tokens, 900-second timeout, 4 cycles, 2 probes, and environment-variable key names only.
- [ ] Write RED identity tests over code SHA, dataset revision, manifest/config/prompt hashes, pricing snapshot, and resolved image digest.
- [ ] Write RED scheduling tests for deterministic 200-task order and hash-balanced arm-first ordering.
- [ ] Write RED state tests for `pending -> running -> completed|terminal_failed`, atomic temp/fsync/rename, immutable terminal results, stale-running recovery, and pending-only resume.
- [ ] Write RED tests proving runner construction receives runtime manifest only and has no gold-store property/path.
- [ ] Implement direct concurrency 8, BayesProbe concurrency 4, and sandbox semaphore 4 with injectable executors for deterministic unit tests.
- [ ] Preflight must reject missing API key, mutable/unresolved image tag, unavailable Docker, incomplete manifest, dirty/floating provenance, or tracked restricted path.
- [ ] Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/evaluation/test_config.py \
  tests/evaluation/test_runner.py -q -p no:cacheprovider
```

- [ ] Commit:

```bash
git add bayesprobe/evaluation/config.py bayesprobe/evaluation/runner.py \
  bayesprobe/evaluation/artifacts.py configs/hle-pilot-v0.1.example.json \
  tests/evaluation/test_config.py tests/evaluation/test_runner.py
git commit -m "feat: run resumable paired capability experiments"
```

---

### Task 9: Score Once and Produce Leak-Safe Reports

**Files:**
- Create: `bayesprobe/evaluation/scoring.py`
- Modify: `bayesprobe/evaluation/artifacts.py`
- Test: `tests/evaluation/test_scoring.py`
- Test: `tests/evaluation/test_leak_scan.py`

- [ ] Write RED tests that scoring refuses incomplete experiments, manifest/gold hash mismatch, second scoring, invalid labels, and wrong experiment identity.
- [ ] Write RED metric tests over a known paired fixture including terminal failures, Wilson intervals, paired table/delta/bootstrap/McNemar, calibration coverage, process totals, and category suppression below five cases.
- [ ] Write RED HMAC tests proving pseudonyms are stable within one experiment, differ across secrets, and cannot be plain sample hashes.
- [ ] Write RED recursive leak tests for exact restricted question, choice, answer, canary, API key, raw response, Python source/output, and reversible id in every shareable JSON/Markdown string.
- [ ] Implement score details in restricted storage and aggregate-only `summary.json`, `summary.md`, `paired_metrics.json`, and `provenance.json` in shareable storage.
- [ ] Include the exploratory, public-set, text-MCQ, and Python-arm asymmetry limitations in generated Markdown.
- [ ] Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/evaluation/test_scoring.py \
  tests/evaluation/test_leak_scan.py -q -p no:cacheprovider
```

- [ ] Commit:

```bash
git add bayesprobe/evaluation/scoring.py bayesprobe/evaluation/artifacts.py \
  tests/evaluation/test_scoring.py tests/evaluation/test_leak_scan.py
git commit -m "feat: score and report capability experiments"
```

---

### Task 10: Expose the Four-Phase Evaluation CLI

**Files:**
- Create: `bayesprobe/evaluation/cli.py`
- Modify: `bayesprobe/cli.py`
- Modify: `bayesprobe/evaluation/__init__.py`
- Modify: `bayesprobe/__init__.py`
- Test: `tests/evaluation/test_cli.py`
- Modify: `tests/test_cli.py`

- [ ] Write RED parser/handler tests for:

```text
bayesprobe eval prepare --config PATH
bayesprobe eval run --config PATH
bayesprobe eval score --experiment PATH
bayesprobe eval report --experiment PATH
```

- [ ] Test stable exit codes and sanitized stderr for config, gated access, preflight, provider, Docker, incomplete-run, and leak-scan failures.
- [ ] Implement thin handlers that delegate to package services. Keep legacy `bayesprobe run --config` behavior unchanged.
- [ ] Export only stable contracts and service entry points; do not expose HLE raw row types or gold data through top-level convenience APIs.
- [ ] Run:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/evaluation/test_cli.py \
  tests/test_cli.py -q -p no:cacheprovider
python3 -m bayesprobe.cli --help
python3 -m bayesprobe.cli eval --help
```

- [ ] Commit:

```bash
git add bayesprobe/evaluation/cli.py bayesprobe/cli.py \
  bayesprobe/evaluation/__init__.py bayesprobe/__init__.py \
  tests/evaluation/test_cli.py tests/test_cli.py
git commit -m "feat: expose capability evaluation commands"
```

---

### Task 11: Prove the Full Workflow Without HLE Content

**Files:**
- Create: `tests/evaluation/fixtures/synthetic_mcq_rows.json`
- Create: `tests/evaluation/test_end_to_end.py`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/superpowers/specs/2026-07-11-hle-text-mcq-capability-pilot-design.md`

- [ ] Add 100 self-authored synthetic MCQs with synthetic ids and categories; no HLE-derived wording, answers, ids, metadata, or canary.
- [ ] Run prepare -> paired fake-provider execution -> interruption -> resume -> score -> report end to end.
- [ ] Assert 200 terminal arm/sample results, no duplicate provider/Python records, stable experiment identity, one score marker, correct aggregate metrics, and a clean leak scan.
- [ ] Add an opt-in live DeepSeek smoke over one self-authored problem gated by `BAYESPROBE_RUN_DEEPSEEK_LIVE=1` and `DEEPSEEK_API_KEY`; never load HLE in the smoke.
- [ ] Document architecture, restricted/shareable paths, exact formal protocol, resource expectations, and claims that remain prohibited.
- [ ] Mark the design status implemented only after every verification command succeeds.
- [ ] Run focused integration tests:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest tests/evaluation/test_end_to_end.py \
  -q -p no:cacheprovider
```

- [ ] Run the entire offline suite:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest -q -p no:cacheprovider
node --test tests/test_webui_stream.js
git diff --check
git status --short
```

Expected: all Python and Node tests pass, live tests skip by default, diff check is clean, and only intentional files are modified.

- [ ] Run Docker isolation suite again after the full regression suite:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m pytest \
  tests/evaluation/test_python_sandbox_integration.py -q -p no:cacheprovider
```

- [ ] Commit:

```bash
git add tests/evaluation/fixtures/synthetic_mcq_rows.json \
  tests/evaluation/test_end_to_end.py docs/ARCHITECTURE.md \
  docs/superpowers/specs/2026-07-11-hle-text-mcq-capability-pilot-design.md
git commit -m "test: verify HLE capability pilot workflow"
```

---

## Formal-Run Boundary

Implementation completion does **not** run or score HLE. After this plan passes on synthetic and self-authored fixtures, the operator separately supplies a full immutable HLE dataset revision and accepted gated access, then follows:

```bash
bayesprobe eval prepare --config configs/hle-pilot-v0.1.json
bayesprobe eval run --config configs/hle-pilot-v0.1.json
bayesprobe eval score --experiment artifacts/restricted/hle-pilot-v0.1/<experiment-id>
bayesprobe eval report --experiment artifacts/restricted/hle-pilot-v0.1/<experiment-id>
```

No prompt, provider control, budget, dependency, image, dataset revision, selection, scoring, or retry change is permitted after `prepare`. Any such change creates a new experiment id and a separate run.
