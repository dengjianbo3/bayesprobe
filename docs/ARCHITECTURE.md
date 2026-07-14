# BayesProbe Architecture

Date: 2026-07-13
Status: living architecture document

This document is the engineering architecture entry point for the current
BayesProbe repository. It combines two views:

- **Target architecture**: the intended shape of BayesProbe as a complete agent
  paradigm.
- **Current implementation map**: what exists in the code today, where it lives,
  and what remains to be deepened.

BayesProbe should be implemented as a complete method for signal-grounded belief
revision over evolving hypotheses. ReAct, ReWOO, ToT, GoT, Reflexion, and related
systems are comparison baselines, not internal execution layers.

## 1. Architectural Intent

BayesProbe's primitive state is a **Belief State**, not an answer, plan, chain of
thought, or action trace. An answer is an **Answer Projection**: a task-facing
compression of the current Belief State.

The core philosophy is epistemic humility:

```text
The agent does not directly possess an answer.
The agent maintains a revisable belief state under uncertainty.
External information changes the belief state only after becoming evidence.
```

This gives the architecture five hard commitments:

1. **Signal before evidence**: raw external information enters as
   `ExternalSignal`.
2. **Evidence through one gate**: signals become `EvidenceEvent`s only through
   the Evidence Integration Gate.
3. **Belief before answer**: posterior changes update `BeliefState`; user-facing
   output is projected from that state.
4. **Hypotheses evolve**: anomaly and pressure can spawn, reframe, weaken, or
   retire hypotheses rather than forcing all signals into the old frame.
5. **One contribution per evidence root**: `EvidenceEvent`s remain audit
   interpretations, while each independent `EvidenceRootContribution` owns one
   current likelihood contribution. Reassessment revises that contribution
   instead of adding another copy.

## 2. Target Architecture

The target runtime flow is:

```text
Question + Task Context / Existing BeliefState
  -> Task Framing
  -> TaskFrame
  -> Initial BeliefState materialization
  -> Probe Set Design
  -> Signal Inbox
  -> Signal Collection Boundary
  -> Evidence Integration Gate
  -> Evidence Events
  -> Evidence Root Reconciliation
  -> Evidence Contribution Deltas
  -> Belief Solver
  -> Hypothesis Evolution
  -> BeliefState t+1
  -> Answer Projection or Belief State Projection
```

The target deployment shape is:

```text
Client / CLI / SDK / Benchmark / External Agent
  -> Run Regime Runner
      -> Autonomous Runner
      -> Synchronized Runner
      -> TaskFramer + Initializer
  -> BayesProbe Core
      -> Signal Inbox
      -> Evidence Integration Gate
      -> Belief Solver
      -> Hypothesis Evolution
      -> Projection Builder
  -> Adapters
      -> ModelGateway
      -> ProbeToolGateway
      -> LedgerStore
      -> Dataset / Report IO
```

The most important seam is between **Run Regime Runner** and
**BayesProbe Core**. Runners decide timing, waiting, active probing, passive
intake, and stop conditions. The core owns evidence rules, likelihood judgment,
belief updates, hypothesis evolution, and ledger-visible cycle integration.

## 3. Core Invariants

These invariants are architectural rules. Future work should preserve them even
when adding real model providers, web search, skills, or multi-agent protocols.

### 3.1 Controller/Core Boundary

Controllers and runners may:

- initialize a run;
- allocate or request a cycle id;
- choose whether the cycle is active-only, passive-only, or active-plus-passive;
- collect passive signals;
- plan and execute active probes;
- close a signal collection boundary;
- call `BayesProbeCore.integrate_cycle(...)`;
- decide whether to continue or emit a projection.

Controllers and runners may not:

- convert signals into evidence directly;
- define likelihood rules;
- update posteriors directly;
- evolve hypotheses directly;
- treat another agent's projection as authoritative evidence;
- bypass the Evidence Integration Gate.

### 3.2 Signal/Evidence Separation

`ExternalSignal` is raw intake. It can be active or passive.

`EvidenceEvent` is the assessed interpretation of a signal. It carries target
hypotheses, evidence type, quality dimensions, likelihood bands,
interpretation, and possible discard reason.

Raw signals never update belief directly.

### 3.3 Legal Cycle Shapes

BayesProbe supports three first-class cycle shapes:

- `active_only`: active signals returned by probes.
- `passive_only`: passive signals from human, benchmark, system, or other agent.
- `active_plus_passive`: active probe returns plus passive signals received
  before boundary closure.

All three shapes must close a boundary and pass through the same core path.

For autonomous question runs, non-empty `InitializeRunInput.context` is converted
to one first-cycle passive `ExternalSignal`. It is integrated once, together with
active probe returns when present. If no probe is available, the context still
forms a legal `passive_only` cycle before the runner stops.

The compatibility `context` field is not Task Context. Because it can become an
`ExternalSignal`, ledger record, or provider input, secret-like credential text is
rejected before progress, provider construction, framing, or state materialization.

### 3.4 Projection-As-Signal Rule

A `BeliefStateProjection` received from another human or agent enters as a
passive `ExternalSignal`. It is not imported as a belief update.

When a projection includes both sender judgment and cited source claims, the
Evidence Integration Gate may decompose it:

```text
sender judgment -> EvidenceType.SENDER_JUDGMENT
cited source claim -> EvidenceType.SOURCE_CLAIM
verification need -> ProbeCandidate
```

### 3.5 Schema Failure Rule

Malformed structured model output is not allowed to crash normal benchmark
runs, but it also must not silently influence belief.

The current rule is:

```text
invalid structured judgment
  -> discarded neutral EvidenceEvent
  -> discard_reason = schema_violation: ...
  -> zero quality scores
  -> no BeliefUpdate
```

The repair/retry policy can add an opt-in repair step before this fallback.
The fallback remains belief-neutral.

## 4. Module Map

This section maps the target architecture to the current repository.

### 4.1 BayesProbe Core

Current file: `bayesprobe/core.py`

Primary interface:

```python
BayesProbeCore.integrate_cycle(
    cycle: CycleRecord,
    belief_state: BeliefState,
    probe_set: ProbeSet,
    signals: list[ExternalSignal],
) -> CycleResult
```

Responsibilities:

- validate cycle/probe/belief-state consistency;
- create and close a cycle-local `SignalInbox`;
- validate legal cycle signal shape;
- delegate signal interpretation to `EvidenceIntegrationGate`;
- validate and consume root-owned contribution deltas returned by the gate;
- call `CoverageAwareBeliefSolver` with contribution deltas, never raw Events;
- call `HypothesisEvolutionEngine.evolve(...)`;
- merge ledger references into the next `BeliefState`;
- append cycle, signal, probe, evidence, update, evolution, candidate, and state
  records to the ledger when present.

Design note:

`BayesProbeCore` is intentionally the deep module. Callers learn one main
integration interface and get signal intake, evidence construction, posterior
movement, hypothesis evolution, and audit records behind it.

### 4.2 Signal Inbox

Current file: `bayesprobe/inbox.py`

Responsibilities:

- hold cycle-local signals before boundary closure;
- normalize signal cycle ids;
- reject signals after closure;
- make boundary closure explicit;
- support the core's terminal `open -> closed -> integrated` cycle lifecycle.

The inbox is not a belief state and does not interpret signals.

### 4.3 Evidence Integration Gate

Current file: `bayesprobe/evidence.py`

Responsibilities:

- convert closed-cycle signals into evidence events;
- normalize signal provenance and bind accepted Events to Evidence Roots;
- reconcile each root's current contribution against Evidence Memory;
- emit `EvidenceContributionDelta` and `EpistemicProgress` alongside audit
  Events;
- handle active and passive signals through the same path;
- assign target hypotheses;
- use `ModelGateway` for direct evidence judgment;
- decompose external agent projections into sender judgment and source claim;
- assess signal quality;
- generate verification `ProbeCandidate`s from source claims;
- convert schema validation failure into discarded neutral evidence.

Current limitations:

- quality assessment remains heuristic;
- projection decomposition is cue-based;
- schema repair/retry is opt-in through `EvidenceJudgmentRepairPolicy`.

### 4.4 Belief Solver

Current file: `bayesprobe/belief.py`

Responsibilities:

- consume validated `EvidenceContributionDelta`s rather than raw
  `EvidenceEvent`s;
- apply only the difference between a root's current and previous
  contribution;
- make `NO_CHANGE` an exact belief no-op and allow `REVISE_ROOT` or
  `RETRACT_ROOT` without double counting;
- preserve belief-neutral handling for schema violations;
- preserve normalized categorical mass across active hypotheses for
  `exclusive_exhaustive` frames;
- update targeted `independent` hypotheses in log-odds space without
  cross-normalizing their credences;
- apply complexity and ad-hoc penalties according to the selected relation;
- refresh relation-aware belief, uncertainty, entropy, and top-gap summaries;
- produce auditable `BeliefUpdate`s for direct movement and, for exclusive
  frames, normalization-induced rival movement.

Current limitation:

- exclusive mass and independent credence updates are pragmatic MVP scoring
  rules, not calibrated Bayesian inference.

### 4.5 Hypothesis Evolution

Current file: `bayesprobe/hypothesis_evolution.py`

Responsibilities:

- inspect evidence pressure and belief updates;
- spawn hypotheses from anomaly signals;
- reframe or retire weakened hypotheses;
- preserve explicit independent conflict links while avoiding blanket
  all-to-all rivals for independent anomaly spawns;
- maintain reciprocal all-to-all rivals for exclusive frames;
- emit `HypothesisEvolution` audit records;
- generate probe candidates for evolved hypotheses.

Current limitation:

- evolution policy is deterministic and rule-based; cross-cycle semantic
  evolution remains deferred.

### 4.6 Initialization

Current files:

- `bayesprobe/task_framing.py`
- `bayesprobe/initialization.py`

Responsibilities:

- produce a validated `TaskFrame` through explicit, model, or recorded framers;
- allow at most one sanitized model repair before failing closed;
- initialize a `RunRecord` only after framing succeeds;
- materialize initial hypotheses and `BeliefState` from the `TaskFrame`;
- create seed `ProbeCandidate`s.

Current limitation:

- task-aware `ProbeDesigner` behavior remains deferred; current seed candidates
  are deterministic after framing.

### 4.7 Probe Planning

Current file: `bayesprobe/probe_planner.py`

Responsibilities:

- select a bounded `ProbeSet` from candidate probes;
- rank by expected information value and decision relevance;
- honor max-probe budget and empty-set policy;
- keep probe selection separate from tool execution.

Current limitation:

- candidate scoring is deterministic; future versions can use model-assisted
  ranking through a controlled seam.

### 4.8 Probe Execution and Tool Gateway

Current file: `bayesprobe/probe_executor.py`

Primary seam:

```python
class ProbeToolGateway(Protocol):
    def execute_probe(
        self,
        *,
        probe: ProbeDesign,
        context: ProbeExecutionBrief,
    ) -> list[ExternalSignal]:
        ...
```

Responsibilities:

- execute active probes through a swappable gateway;
- expose only a one-shot, score-free hypothesis brief: no priors, posteriors,
  ranks, accumulated evidence, or current winner enter probe execution;
- normalize returned signals as active external signals;
- enforce that probe execution cannot return passive signals;
- append execution and signal records to the ledger when present.

Current adapters:

- `DeterministicProbeToolGateway`.
- `ModelBackedProbeToolGateway`, which converts the structured `execute_probe`
  result from a `ModelGateway` into an active `ExternalSignal`.
- `TavilyProbeToolGateway`, which uses the model only to plan a query and turns
  each Tavily result URL into a separate `RETRIEVED_SOURCE` signal. Query
  planning is not evidence; retrieval failures produce no reasoning fallback.

`ModelBackedProbeToolGateway` is an internal-deliberation adapter, not a claim of
web search or source verification. Its signals use a conservative quality
baseline (`reliability=0.55`, `independence=0.35`, `verifiability=0.30`) and keep
`source_type=model_probe_gateway` in the trace.

Future adapters:

- production search providers beyond the bounded Tavily adapter;
- document retrieval;
- tool invocation;
- skill execution;
- simulation;
- user-question request that becomes a future passive signal.

### 4.9 Model Gateway

Current file: `bayesprobe/model_gateway.py`

Primary seam:

```python
class ModelGateway(Protocol):
    def complete_structured(self, request: StructuredModelRequest) -> dict[str, Any]:
        ...
```

Current tasks:

- `frame_open_question` and its bounded `repair_task_frame` repair;
- `judge_evidence` and its opt-in `repair_evidence_judgment` repair;
- `execute_probe` for structured `ProbeSignal` generation behind the
  `ProbeToolGateway` seam.

Current adapters:

- `DeterministicModelGateway`;
- `ScriptedModelGateway`;
- `RecordedModelGateway`;
- `OpenAIResponsesModelGateway`;
- `OpenAIChatCompletionsModelGateway`;
- `build_model_gateway(...)` from `ModelGatewayConfig`.

Current validation:

- `evidence_judgment_from_mapping(...)`;
- `ModelGatewayValidationError`.

Current repair support:

- one fail-closed task-frame repair through `TaskFramingRepairPolicy`;
- evidence-judgment repair behind `EvidenceJudgmentRepairPolicy`.

Future extension:

- broader provider registry and provider observability;
- further response schema repair hardening;
- live-provider fixture capture tooling.

### 4.10 Projections

Current file: `bayesprobe/projections.py`

Responsibilities:

- build user-facing `AnswerProjection`;
- build collaboration-facing `BeliefStateProjection`;
- include change-my-mind conditions and structured probe candidates;
- keep output generation downstream from belief state.

Architectural rule:

Projection is an output or exchange object. If it re-enters BayesProbe from an
external participant, it must come back as a passive signal.

### 4.11 Controllers and Runners

Current files:

- `bayesprobe/controllers.py`
- `bayesprobe/runners.py`
- `bayesprobe/question_runner.py`
- `bayesprobe/synchronized_runner.py`

Implemented regimes:

- autonomous loop runner;
- autonomous question runner;
- synchronized round runner;
- lower-level autonomous and synchronized controllers.

Responsibilities:

- shape cycles;
- collect active/passive signals;
- plan and execute active probes;
- call the shared core;
- emit answer or belief-state projections;
- apply continuation and stop conditions.
- publish optional, typed autonomous-question progress observations without
  changing runner control flow or domain state.

The autonomous-question observer is an observation-only seam. Its phase and
cycle observations are consumed by transports such as the WebUI; they do not
select probes, construct evidence, update beliefs, or decide stop conditions.

Current limitations:

- autonomous stop conditions are basic;
- synchronized protocol is fixed-round and local;
- no networked multi-agent transport exists yet.

### 4.12 Ledger

Current file: `bayesprobe/ledger.py`

Responsibilities:

- write auditable JSONL records;
- serialize Pydantic models and plain records;
- preserve run/cycle/evidence/update/evolution/projection traces;
- record canonical probe sets and external signals exactly once through core
  ownership;
- allow initial and terminal run snapshots while preserving immutable event
  history.

Current limitation:

- JSONL is the only storage adapter.

Future adapters:

- SQLite;
- Postgres;
- object-store-backed experiment archive.

### 4.13 Benchmark and Experiment Layer

Current files:

- `bayesprobe/benchmark.py`
- `bayesprobe/benchmark_io.py`
- `bayesprobe/experiment_runner.py`
- `bayesprobe/config.py`
- `bayesprobe/cli.py`

Responsibilities:

- define benchmark samples and signal shapes;
- load datasets from JSON/JSONL;
- run active-only, passive-only, and active-plus-passive samples;
- score final best hypothesis and update direction accuracy;
- report belief-state quality metrics such as discarded evidence, schema
  violations, posterior margin, and total-variation belief revision
  efficiency;
- write reports;
- parse JSON experiment config;
- expose a thin CLI over config-driven experiment runs.

Current limitation:

- datasets are still small methodology fixtures, not a full benchmark suite;
- metrics are useful MVP checks, not a full comparative evaluation suite;
- the legacy fixture harness has no external provider cost/latency accounting;
  the separate capability-evaluation layer below does.

### 4.14 Public SDK

Current file: `bayesprobe/__init__.py`

The package root exports the supported MVP surface for external code:

- `BayesProbeCore`, initialization, autonomous, and synchronized runners;
- belief, signal, hypothesis, probe, run-regime, and run-status schemas;
- probe execution and tool-gateway seams;
- JSONL ledger, benchmark data structures, and benchmark dataset IO;
- experiment config, artifacts, and runner;
- model gateway config and deterministic, recorded, Responses, and
  OpenAI-compatible Chat Completions adapters;
- `ModelTaskFramer`, `RecordedTaskFramer`, `RoutingTaskFramer`,
  `TaskFramingRepairPolicy`, and legacy belief-state migration;
- structured judgment validation and repair contracts.

Architectural rule:

Public exports should stay narrow. Internal modules can evolve, but external
users should configure runs through supported seams instead of reaching into the
core's private internals.

### 4.15 Autonomous WebUI

Current file: `bayesprobe/webui.py`

Responsibilities:

- serve the local autonomous workbench;
- validate local WebUI requests;
- build request-scoped provider gateways;
- enforce a 360-second timeout floor for provider calls initiated by the
  WebUI, while leaving the reusable SDK gateway timeout explicitly
  configurable;
- enforce a 32768-token output floor for official DeepSeek V4 Chat Completions
  requests initiated by the WebUI, leaving generic OpenAI-compatible providers
  and the reusable SDK gateway configurable;
- report output-budget exhaustion separately from connection and credential
  failures without exposing raw provider responses;
- use provider-backed gateways for separate `execute_probe` and
  `judge_evidence` calls;
- run `AutonomousQuestionRunner`;
- adapt runner observations into flushed NDJSON records on
  `POST /api/runs/autonomous/stream` while retaining
  `/api/runs/autonomous` for the synchronous JSON contract;
- serialize the terminal run record, final answer, relation-aware belief state,
  integrated cycle, signal, evidence, root-contribution delta, epistemic
  progress, update, and evolution traces;
- expose run regime/status/stop reason, exclusive posterior mass or independent
  credence, relation-aware top-gap uncertainty, and cycle lifecycle timestamps.

Architectural rule:

The WebUI is an observation and execution surface. It must not convert signals
to evidence, update posterior values, evolve hypotheses, or bypass
`BayesProbeCore`.

Current limitations:

- local-only;
- progress is phase/cycle streaming, not token streaming;
- an HTTP disconnect does not cooperatively cancel an in-flight provider call;
- only autonomous WebUI runs stream in M0.10;
- credentials remain request-scoped and page-memory-only;
- no multi-user auth;
- no provider-side cost/latency telemetry.

### 4.16 Capability Evaluation Layer

Current package: `bayesprobe/evaluation/`

This package is intentionally separate from the fixture-oriented
`BenchmarkHarness`. It implements the HLE text-only multiple-choice capability
pilot without changing BayesProbe core semantics:

```text
gated dataset -> restricted runtime manifest + isolated gold store
  -> DirectFlashArm / BayesProbePythonArm
  -> atomic per-arm terminal results
  -> separate exact-label scorer
  -> restricted details + leak-scanned aggregate report
```

Implemented responsibilities:

- require a full immutable dataset revision and deterministic stratified
  selection;
- canonicalize multiple-choice framing while checking that the public
  initializer creates the same hypotheses;
- keep gold unavailable to both experiment arms and to resume decisions;
- freeze DeepSeek V4 Flash request controls and record every provider attempt;
- run model-generated Python only through a digest-resolved Docker image with
  no network, no host mounts, a read-only filesystem, non-root execution, and
  bounded process, memory, CPU, time, and output resources;
- preserve Python/reasoning results as `ExternalSignal`s that still pass
  through the Evidence Integration Gate;
- report new, revised, retracted, and unchanged Evidence Roots, falsification
  cycles, maximum contribution delta, and epistemic-stagnation termination
  without changing answer scoring;
- run a deterministic 200-task paired schedule with HMAC case paths, atomic
  status transitions, and correctness-blind resume;
- score exact labels once after all tasks are terminal and report Wilson,
  paired bootstrap, exact McNemar, calibration, process, latency, token, and
  estimated-cost metrics, including resource use per correct answer;
- validate a dated USD pricing snapshot with explicit uncached-input,
  cached-input, and output rates per million tokens; publish its hash and rates
  with every estimate, and suppress cost when successful calls lack billable
  token usage;
- recursively reject benchmark content, raw model/Python material, canaries,
  reversible ids, or provider secrets from shareable artifacts;
- expose `bayesprobe eval prepare|run|score|report` as four separate phases.

The engineering workflow is verified on 100 self-authored synthetic cases and
real Docker isolation tests. No formal HLE result is claimed until an operator
accepts gated access, supplies a pinned dataset commit, freezes the pricing
snapshot, and completes the four-phase protocol.

## 5. Implemented Capability Matrix

| Area | Status | Notes |
|---|---:|---|
| BayesProbe paradigm positioning | Strong | Context and v0.2 docs define BayesProbe as a complete paradigm, not wrapper. |
| Core signal-to-belief loop | Strong | `BayesProbeCore.integrate_cycle(...)` is the shared path. |
| Active/passive/mixed cycle shapes | Strong | Implemented in core validation, synchronized runner, and benchmark harness. |
| Signal Inbox and boundary | Strong MVP | Cycle-local closure and terminal `open -> closed -> integrated` timestamps exist; real-time late-signal queueing remains future work. |
| Evidence Integration Gate | Strong MVP | Direct evidence, real projection decomposition, exact target validation, bounded quality overrides, schema violation, and repair paths exist. |
| Evidence Memory / root ownership | Strong MVP | Native memory v3 stores provenance, root bindings, and one current contribution per Evidence Root. Same-root repeats revise or no-op instead of accumulating. |
| Belief update | Strong MVP | Solver consumes contribution deltas only. Exclusive mass remains normalized; independent credences update without cross-normalization; penalties, discarded-evidence neutrality, and relation-aware summaries are implemented. |
| Hypothesis evolution | Good MVP | Anomaly spawn, weakening/reframing/retirement style evolution preserves explicit independent conflicts and reciprocal exclusive rivals; semantic evolution remains deferred. |
| Probe planning | Strong MVP | Task-specific probe design, bounded probe-set ranking, and post-cycle reservation of a genuine top-hypothesis falsifier are implemented. |
| Probe execution/tool seam | Good MVP | Execution receives an immutable score-free brief. Deterministic, model-backed, and bounded Tavily retrieval adapters exist; broader retrieval and tool adapters remain future work. |
| Autonomous question loop | Strong MVP | End-to-end runner returns a terminal run, final integrated cycle, task-aware selection, synthesis, or abstention, answer projection, explicit stop reason, and epistemic-stagnation termination. |
| Synchronized round loop | Strong MVP | Fixed-round runner supports passive-only, active-only, and mixed rounds, reports epistemic progress, and remains externally controlled rather than self-stopping. |
| Ledger/audit | Strong MVP | JSONL audit path has explicit canonical ownership and exactly-once probe-set/signal records. |
| Benchmark harness | Good MVP | Toy and real methodology-path fixtures, suite/report flow, net-direction scoring, and belief-quality metrics exist. |
| Config/CLI/SDK | Strong MVP | JSON experiment config, CLI, public core/runners/tool/framing seams, package-root imports, and external execution regression coverage exist. |
| Autonomous WebUI | Strong MVP | Deterministic/Responses/OpenAI-compatible Chat Completions requests use the shared core; synchronous JSON and autonomous NDJSON streams expose framing, probes, signals, Evidence Events, root deltas, epistemic progress, belief, expansion, and terminal traces. Credentials remain request-scoped and page-memory-only. |
| Model gateway | Strong MVP | Structured seam plus deterministic, scripted, recorded, OpenAI Responses, and OpenAI-compatible Chat Completions adapters exist. Explicit request controls, bounded transport retries, and per-attempt token/latency/error observation are implemented. |
| Structured output robustness | Good MVP | Validation, neutral schema violation, and opt-in repair/retry policy exist. |
| Prompt/version metadata | Good MVP | StructuredModelRequest metadata and EvidenceEvent model_trace are implemented. |
| Multi-agent protocol | Partial | Projection-as-signal semantics exist; transport/protocol schema not complete. |
| Production persistence | Missing | JSONL only. |
| Capability evaluation | Strong MVP | Gold-isolated HLE text-MCQ preparation, Direct/BayesProbe paired arms, Docker Python probes, resumable execution, exact scoring, paired/calibration metrics, leak-safe reports, and the frozen 77-to-30 paradigm checkpoint executor are implemented and synthetic-tested. Formal corrected HLE execution remains pending. |
| Large benchmark suite | Partial | The HLE text-MCQ-100 runner is ready, but no formal gated run or broader multimodal/exact-answer suite has been completed. |

## 6. Open-Question Framing Status

Implemented: autonomous model-reasoning open-question vertical slice with
task-specific probe design, Core-authorized semantic expansion, task-aware
selection, synthesis, and abstention, recorded explanation and exact-answer
fixtures, and dynamic WebUI progress.

The slice retains explicit/model/recorded `TaskFrame` creation before Belief
State materialization, fail-closed structured framing with one bounded repair,
and relation-aware exclusive categorical mass and independent-credence solver
semantics. Models propose task semantics, while `BayesProbeCore` alone admits
Evidence, changes belief, and authorizes hypothesis expansion.

Not implemented: durable multi-provider retrieval or a production-grade tool ecosystem;
coding interventions; public benchmark execution; or probability calibration
claims.

Cross-cycle Evidence Memory v3 is implemented for native runs. It owns
provenance identity, Evidence Root bindings, current root contributions, and
discovery/falsification history. The remaining limitation is empirical: the
frozen 30-case process checkpoint executor is ready but has not yet completed a
provider-backed run, so no corrected HLE accuracy claim follows from the
conformance work. `checkpoint-prepare` freezes the completed/completed 77-case
population before reading gold, hashes the reused Direct results, and writes the
30-case manifest last. `checkpoint-run` refuses to schedule the reused Direct
arm and reruns only corrected BayesProbe; `checkpoint-score` reports cycle-one
and final accuracy plus root, falsification, stagnation, and drift metrics.

## 7. External Seams and Configuration

The architecture should expose variation through a small number of deep seams.

### 7.1 ModelGateway

Use for model-shaped structured decisions:

- evidence judgment;
- open-question framing and its single bounded repair;
- structured model-backed probe execution behind `ProbeToolGateway`;
- evidence-judgment repair when configured;
- future hypothesis evolution assistance;
- future projection writing or compression;
- future prompt-versioned provider calls;
- `ModelInvocationTrace` persists prompt/schema adapter metadata on evidence events.
- `OpenAIResponsesModelGateway` and `OpenAIChatCompletionsModelGateway` provide
  real provider-backed adapters while preserving the same structured output
  validation path.
- `RecordedModelGateway` replays provider-shaped structured judgments for
  deterministic benchmark and artifact tests without network access.

Do not let callers pass arbitrary model outputs into belief update. Model output
must be parsed, validated, and converted into BayesProbe domain objects.

### 7.2 ProbeToolGateway

Use for active external information gathering:

- additional search providers and tool adapters;
- document retrieval;
- code/tool execution;
- skill execution;
- simulation.

Tool output returns as `ExternalSignal`, not `EvidenceEvent`.

The current model-backed adapter is useful for closed-book autonomous reasoning,
but it remains lower-verifiability than future search, retrieval, code, skill,
and simulation adapters. Provider output still passes through the same Evidence
Integration Gate before it can affect belief.

### 7.3 LedgerStore

Use for audit and reproducibility. The current concrete adapter is JSONL.

Future storage adapters should preserve the same conceptual event stream:

```text
run
cycle
probe_set
probe_execution
external_signal
evidence_event
belief_update
hypothesis_evolution
probe_candidate
belief_state
answer_projection / belief_state_projection
benchmark_result
```

### 7.4 Experiment Config

Use JSON experiment config for reproducible benchmark runs. The current config
already supports:

- dataset path;
- report path;
- ledger path;
- max cycles;
- max probes per cycle;
- model gateway config.

Near-term config additions should include:

- evidence judgment repair policy;
- provider model name;
- prompt version;
- seed/run metadata;
- dataset split and sample filters.

### 7.5 Synchronized Collaboration

Use `BeliefStateProjection` as the exchange object and passive `ExternalSignal`
as the intake object.

Future multi-agent integration should not share full internal belief states by
default. It should exchange projections, questions, uncertainties, and requested
signal types.

## 8. Near-Term Completion Roadmap

### Phase 1: Schema Repair / Retry Policy

Status: implemented as MVP.

Goal:

- make structured judgment robust enough for real model adapters without
  changing default deterministic behavior.

Shape:

- add opt-in `EvidenceJudgmentRepairPolicy`;
- default `max_attempts=0`;
- repair task goes through `ModelGateway`;
- valid repaired output becomes normal evidence;
- failed repair falls back to neutral discarded schema violation.

Why this is next:

- it strengthens the model seam before expanding to additional providers and provider observability;
- it protects belief state quality;
- it keeps schema failure visible in the ledger.

### Phase 2: Provider Adapter and Prompt Registry Metadata

Status: OpenAI Responses, OpenAI-compatible Chat Completions, and recorded
fixture adapters are implemented as MVPs. Explicit experiment controls,
sanitized request hashes, bounded transport retries, prompt/schema provenance,
and per-attempt token/latency/error observation are implemented; a broader
provider registry and production telemetry backend remain future work.

Goal:

- add real model provider support without spreading provider details across the
  core.

Shape:

- provider-backed `ModelGateway` adapter;
- provider-side prompt registry metadata and request assembly;
- structured output parser/validator;
- recorded fixtures for reproducible tests.

### Phase 3: Benchmark Expansion

Status: v0.2 methodology fixtures and recorded provider replay remain the core
regression slice. The HLE text-MCQ-100 paired evaluation workflow is now
implemented and synthetic-tested; formal gated execution and wider comparative
baselines remain future work.

Goal:

- move beyond toy samples while preserving dual-objective evaluation.

Shape:

- more active-only and passive-only benchmark cases;
- mixed-cycle smoke cases;
- schema failure and repair cases;
- metrics for final answer utility and belief-state revision quality.

### Phase 4: Multi-Agent Protocol Hardening

Goal:

- support external synchronized collaboration without leaking internal state.

Shape:

- explicit projection exchange schema;
- participant/round metadata;
- passive signal intake from external agents;
- verification probes for cited evidence;
- ledger-visible collaboration traces.

### Phase 5: Persistence and Experiment Packaging

Status: stable artifact directory and model invocation provenance summaries
implemented as v0.1; SQLite persistence, dataset split filters, and full prompt
registry snapshots remain future work.

Goal:

- make experiments shareable and replayable.

Shape:

- stable run artifact directory;
- report + ledger + config + prompt versions + dataset snapshot;
- optional SQLite adapter.

Artifact v0.1 writes a manifest, report, ledger, config snapshot, and dataset
snapshot without changing BayesProbe core control flow.

Model invocation provenance v0.1 summarizes existing ledger `model_trace`
records into `model_invocations.json` and the manifest without changing
BayesProbe core control flow.

## 9. Testing Strategy

Tests should protect BayesProbe's own control flow, not imitate ReAct/ReWOO
behavior.

Core tests should assert:

- all cycle shapes pass through `BayesProbeCore.integrate_cycle(...)`;
- invalid cycle/probe/signal combinations fail early;
- signals are not evidence until the Evidence Integration Gate;
- discarded schema-violation evidence produces no belief update;
- hypothesis evolution emits auditable records;
- projections are derived from belief state and can re-enter only as signals.

Adapter tests should assert:

- model gateway validation behavior;
- scripted/deterministic adapters are reproducible;
- provider-backed adapters use request-scoped credentials and never persist raw
  API keys in artifacts;
- tool gateway returns only active signals;
- config objects propagate into benchmark and experiment runs.
- provider attempts preserve token/error metadata without exposing secrets;
- Docker Python probes have no host fallback and enforce network, filesystem,
  process, timeout, and output boundaries.

Benchmark tests should assert:

- sample loading;
- active-only/passive-only/mixed execution;
- final answer utility;
- update direction accuracy;
- belief-state quality metrics;
- ledger visibility for evidence, updates, and schema failures.

Capability-evaluation tests should assert:

- deterministic HLE eligibility, quotas, selection, and manifest hashing on
  synthetic rows;
- strict runtime/gold separation and one-time exact scoring;
- both arms reach terminal state and resume never uses correctness;
- terminal failures remain in the denominator;
- paired, calibration, process, and provider metrics are reproducible;
- shareable outputs pass recursive leak scans.

## 10. Architectural Non-Goals

The current architecture should not optimize for:

- exact statistical Bayesian inference;
- hidden chain-of-thought storage;
- ReAct/ReWOO compatibility as an internal substrate;
- direct import of another agent's posterior as evidence;
- unrestricted model output influencing belief;
- complex real-time meeting orchestration in the MVP;
- broad provider abstraction before the local model gateway contract is robust.

## 11. Definition of Done for the Kernel

The BayesProbe kernel can be considered substantially complete when:

1. external code can configure model, tool, ledger, benchmark, and run policy
   seams through stable public interfaces;
2. real model adapters can run evidence judgment with validation, repair, and
   reproducible metadata;
3. active-only, passive-only, and mixed synchronized cycles are all benchmarked;
4. multi-agent projections enter as passive signals and produce auditable
   evidence/update traces;
5. final answers are useful on benchmarks while belief-state quality remains
   observable through ledger records;
6. the public SDK can be imported by another project without reaching into
   private internals.

Until then, the project should continue to prioritize core depth over broad
feature surface.

M0.9 status: all six criteria above have an MVP implementation and regression
coverage. This makes the local engineering kernel substantially complete for
manual WebUI, SDK, and benchmark validation; it does not imply production
operations or methodological superiority has been established.
