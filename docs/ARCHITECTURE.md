# BayesProbe Architecture

Date: 2026-07-08
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

This gives the architecture four hard commitments:

1. **Signal before evidence**: raw external information enters as
   `ExternalSignal`.
2. **Evidence through one gate**: signals become `EvidenceEvent`s only through
   the Evidence Integration Gate.
3. **Belief before answer**: posterior changes update `BeliefState`; user-facing
   output is projected from that state.
4. **Hypotheses evolve**: anomaly and pressure can spawn, reframe, weaken, or
   retire hypotheses rather than forcing all signals into the old frame.

## 2. Target Architecture

The target runtime flow is:

```text
Problem / Existing BeliefState
  -> Probe Set Design
  -> Signal Inbox
  -> Signal Collection Boundary
  -> Evidence Integration Gate
  -> Evidence Events
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
- call `solve_updates(...)`;
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
- make boundary closure explicit.

The inbox is not a belief state and does not interpret signals.

### 4.3 Evidence Integration Gate

Current file: `bayesprobe/evidence.py`

Responsibilities:

- convert closed-cycle signals into evidence events;
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
- direct evidence judgment uses deterministic/scripted gateway behavior;
- schema repair/retry is opt-in through `EvidenceJudgmentRepairPolicy`.

### 4.4 Belief Solver

Current file: `bayesprobe/belief.py`

Responsibilities:

- map `EvidenceEvent`s into posterior changes;
- skip discarded evidence events;
- preserve belief-neutral handling for schema violations;
- produce auditable `BeliefUpdate`s.

Current limitation:

- posterior updates are pragmatic MVP scoring rules, not calibrated Bayesian
  inference.

### 4.5 Hypothesis Evolution

Current file: `bayesprobe/hypothesis_evolution.py`

Responsibilities:

- inspect evidence pressure and belief updates;
- spawn hypotheses from anomaly signals;
- reframe or retire weakened hypotheses;
- emit `HypothesisEvolution` audit records;
- generate probe candidates for evolved hypotheses.

Current limitation:

- evolution policy is deterministic and rule-based; it is ready for deeper
  model-assisted evolution later.

### 4.6 Initialization

Current file: `bayesprobe/initialization.py`

Responsibilities:

- initialize a `RunRecord`;
- create initial hypotheses;
- create initial `BeliefState`;
- create seed `ProbeCandidate`s.

Current limitation:

- initialization is still mostly deterministic and template-driven.

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
        context: ProbeExecutionContext,
    ) -> list[ExternalSignal]:
        ...
```

Responsibilities:

- execute active probes through a swappable gateway;
- normalize returned signals as active external signals;
- enforce that probe execution cannot return passive signals;
- append execution and signal records to the ledger when present.

Current adapters:

- `DeterministicProbeToolGateway`.

Future adapters:

- search;
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

- `judge_evidence`.

Current adapters:

- `DeterministicModelGateway`;
- `ScriptedModelGateway`;
- `OpenAIResponsesModelGateway`;
- `build_model_gateway(...)` from `ModelGatewayConfig`.

Current validation:

- `evidence_judgment_from_mapping(...)`;
- `ModelGatewayValidationError`.

Current repair support:

- `repair_evidence_judgment` task behind an opt-in repair policy.

Future extension:

- broader provider registry and provider observability;
- further response schema repair hardening;
- recorded fixture adapter for reproducible experiments.

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

Current limitations:

- autonomous stop conditions are basic;
- synchronized protocol is fixed-round and local;
- no networked multi-agent transport exists yet.

### 4.12 Ledger

Current file: `bayesprobe/ledger.py`

Responsibilities:

- write auditable JSONL records;
- serialize Pydantic models and plain records;
- preserve run/cycle/evidence/update/evolution/projection traces.

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
- write reports;
- parse JSON experiment config;
- expose a thin CLI over config-driven experiment runs.

Current limitation:

- dataset is toy-scale;
- metrics are useful MVP checks, not a full benchmark suite;
- no external provider cost/latency accounting yet.

### 4.14 Public SDK

Current file: `bayesprobe/__init__.py`

The package root exports the supported MVP surface for external code:

- benchmark data structures;
- benchmark dataset IO;
- experiment config and runner;
- model gateway config and adapters;
- structured judgment validation errors.

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
- run `AutonomousQuestionRunner`;
- serialize final answer, belief state, cycle, signal, evidence, update, and
  evolution traces.

Architectural rule:

The WebUI is an observation and execution surface. It must not convert signals
to evidence, update posterior values, evolve hypotheses, or bypass
`BayesProbeCore`.

Current limitations:

- local-only;
- no streaming UI;
- no multi-user auth;
- `openai_chat_completions` protocol is reserved but not implemented.

## 5. Implemented Capability Matrix

| Area | Status | Notes |
|---|---:|---|
| BayesProbe paradigm positioning | Strong | Context and v0.2 docs define BayesProbe as a complete paradigm, not wrapper. |
| Core signal-to-belief loop | Strong | `BayesProbeCore.integrate_cycle(...)` is the shared path. |
| Active/passive/mixed cycle shapes | Strong | Implemented in core validation, synchronized runner, and benchmark harness. |
| Signal Inbox and boundary | Strong MVP | Cycle-local closure exists; real-time late-signal queueing remains future work. |
| Evidence Integration Gate | Good MVP | Direct evidence, projection decomposition, quality heuristics, schema violation path. |
| Belief update | Good MVP | Deterministic update rules and discarded-evidence skip. |
| Hypothesis evolution | Good MVP | Anomaly spawn, weakening/reframing/retirement style evolution exists. |
| Probe planning | Good MVP | Candidate ranking and bounded probe-set design exist. |
| Probe execution/tool seam | Good MVP | `ProbeToolGateway` seam exists; only deterministic adapter implemented. |
| Autonomous question loop | Good MVP | End-to-end question runner exists with stop conditions. |
| Synchronized round loop | Good MVP | Fixed-round runner supports passive-only, active-only, and mixed rounds. |
| Ledger/audit | Good MVP | JSONL audit path exists. |
| Benchmark harness | Good MVP | Toy dataset and suite/report flow exist. |
| Config/CLI/SDK | Good MVP | JSON experiment config, CLI, package exports exist. |
| Autonomous WebUI | MVP | Local deterministic/OpenAI Responses workbench for autonomous runs and trace inspection. |
| Model gateway | Good MVP | Structured seam plus deterministic, scripted, and OpenAI Responses adapters exist. Provider observability remains future work. |
| Structured output robustness | Good MVP | Validation, neutral schema violation, and opt-in repair/retry policy exist. |
| Prompt/version metadata | Good MVP | StructuredModelRequest metadata and EvidenceEvent model_trace are implemented. |
| Multi-agent protocol | Partial | Projection-as-signal semantics exist; transport/protocol schema not complete. |
| Production persistence | Missing | JSONL only. |
| Large benchmark suite | Missing | Current fixture is a tracer bullet. |

## 6. Progress Estimate

Using the final target as:

> configurable, experiment-ready, provider-backed, tool-backed, multi-agent-ready
> BayesProbe agent engineering kernel

the current implementation is approximately **58%-62% complete**.

Using the narrower offline MVP target as:

> deterministic/scripted BayesProbe loop with benchmark and config support

the current implementation is approximately **82%-86% complete**.

The remaining work is mostly depth and robustness rather than direction:

- stronger structured model output handling;
- broader provider registry and provider observability;
- provider adapter prompt-registry metadata;
- richer benchmark datasets and metrics;
- stronger synchronized/multi-agent protocol objects;
- production-grade persistence and experiment trace packaging.

## 7. External Seams and Configuration

The architecture should expose variation through a small number of deep seams.

### 7.1 ModelGateway

Use for model-shaped structured decisions:

- evidence judgment;
- future judgment repair;
- future hypothesis evolution assistance;
- future projection writing or compression;
- future prompt-versioned provider calls.
- `ModelInvocationTrace` persists prompt/schema adapter metadata on evidence events.
- `OpenAIResponsesModelGateway` provides the first real provider-backed adapter
  while preserving the same structured output validation path.

Do not let callers pass arbitrary model outputs into belief update. Model output
must be parsed, validated, and converted into BayesProbe domain objects.

### 7.2 ProbeToolGateway

Use for active external information gathering:

- search;
- document retrieval;
- code/tool execution;
- skill execution;
- simulation.

Tool output returns as `ExternalSignal`, not `EvidenceEvent`.

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

Status: OpenAI Responses adapter implemented as v0.1, and prompt/model
invocation artifact summaries implemented as v0.1; broader provider registry,
prompt registry snapshots, and provider observability remain future work.

Goal:

- add real model provider support without spreading provider details across the
  core.

Shape:

- provider-backed `ModelGateway` adapter;
- provider-side prompt registry metadata and request assembly;
- structured output parser/validator;
- recorded fixtures for reproducible tests.

### Phase 3: Benchmark Expansion

Status: Autonomous WebUI is implemented as the current tracer bullet; the
methodology benchmark expansion remains the next slice.

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
- tool gateway returns only active signals;
- config objects propagate into benchmark and experiment runs.

Benchmark tests should assert:

- sample loading;
- active-only/passive-only/mixed execution;
- final answer utility;
- update direction accuracy;
- ledger visibility for evidence, updates, and schema failures.

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
