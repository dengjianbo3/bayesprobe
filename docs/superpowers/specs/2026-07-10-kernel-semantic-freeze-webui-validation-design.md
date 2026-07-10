# Kernel Semantic Freeze and WebUI Validation Design

Date: 2026-07-10
Status: Implemented and verified

## Context

BayesProbe now has a shared belief-revision core, autonomous and synchronized
runners, provider-backed probe execution and evidence judgment, benchmark
fixtures, experiment artifacts, and a local WebUI. The architecture is aligned
with the v0.2 paradigm, but a read-only audit found semantic inconsistencies
that make the current traces look more complete than the underlying state:

- mutually exclusive rival hypotheses are updated as independent marginals;
- belief-state summaries remain stale after posterior updates;
- completed runs and cycles remain marked running/open;
- synchronized runs can be labeled autonomous;
- planners, executors, and the core duplicate canonical ledger records;
- provider judgments can omit or invent hypothesis likelihoods;
- methodology fixtures name projection and repair behavior without executing
  the corresponding paths.

These issues must be corrected before adding broader tools, persistence, or
benchmark coverage. The target is a manually testable WebUI whose answer,
belief state, and cycle trace all describe the same completed BayesProbe run.

## Goal

Deliver an M0.9 semantic freeze with these observable guarantees:

1. Exclusive rival sets use a normalized posterior distribution.
2. Every returned `BeliefState` has current posterior and uncertainty summaries.
3. Every integrated cycle is terminal and timestamped.
4. Every finished autonomous or fixed-round run is terminal and points at its
   final cycle.
5. Synchronized runs are always labeled synchronized.
6. Canonical domain objects occur exactly once in the JSONL ledger.
7. Provider evidence judgments match the requested hypothesis targets and use
   valid bounded quality overrides.
8. Projection decomposition and schema repair fixtures execute the behavior
   they claim to cover.
9. External Python code can construct and run the supported autonomous and
   synchronized agent interfaces without importing private implementation
   details.
10. The WebUI exposes enough lifecycle and belief metadata for a person to
    verify these guarantees directly.

## Non-Goals

- No ReAct or ReWOO substrate or compatibility layer.
- No search, retrieval, browser, code-execution, or skill adapter in this slice.
- No provider registry expansion.
- No SQLite, Postgres, authentication, or streaming WebUI.
- No networked multi-agent transport.
- No broad rewrite of runners or controllers.
- No claim that methodology effectiveness is proven by this milestone.

## Approaches Considered

### Option A: Surgical Semantic Freeze

Keep `BayesProbeCore.integrate_cycle(...)` as the deep module interface and
concentrate state transition, lifecycle closure, and canonical ledger ownership
there. Make narrow runner and adapter changes around that interface.

This is the selected approach because it provides locality, preserves current
behavioral coverage, and minimizes the regression surface.

### Option B: Orchestration Rewrite

Replace the existing controller and runner families with one new state machine.
This could reduce duplication eventually, but it changes too many interfaces
before the current contracts are frozen.

### Option C: WebUI-Only Corrections

Normalize and relabel data only when serializing WebUI responses. This would
hide rather than fix invalid domain state and would leave SDK and experiment
paths inconsistent.

## Design

### 1. Belief Semantics

The MVP supports one explicit belief family: an exclusive categorical set of
active rival hypotheses. For each evidence event, calculate a score for every
active hypothesis from its current posterior, weighted likelihood ratio,
complexity penalty, and ad-hoc penalty, then normalize scores with softmax.

An event that omits a hypothesis is invalid at the Evidence Integration Gate;
it is not silently treated as no update. Retired hypotheses retain their stored
posterior for audit but are excluded from active normalization. Hypothesis
spawn/evolution must renormalize the active set before the state is returned.

### 2. Atomic Cycle Transition

`BayesProbeCore.integrate_cycle(...)` remains the single transition interface.
It will:

1. validate the open cycle and frozen probe set;
2. close the signal inbox;
3. construct evidence and update/evolve hypotheses;
4. rebuild posterior and uncertainty summaries;
5. return a cycle marked `integrated` with closure/completion timestamps;
6. append canonical cycle-domain records once.

The input objects remain immutable Pydantic models; the core returns new
terminal records.

### 3. Run Lifecycle and Regimes

Runners own continuation and therefore own final run closure. Autonomous and
fixed-round synchronized results return a copied `RunRecord` with:

- the correct regime;
- `status=completed`;
- `current_cycle_id` equal to the last integrated cycle, or `cycle_0` when no
  cycle ran;
- an updated timestamp;
- a machine-readable stop reason in metadata when applicable.

The synchronized runner overrides a new initialization input to synchronized
and rejects an existing run with the wrong regime.

### 4. Exactly-Once Ledger

Ledger ownership is split by record meaning:

- initializer: initial run, belief state, and initial candidates;
- planner: candidate selection diagnostics only;
- executor: `probe_execution` diagnostics only;
- core: canonical cycle, probe set, external signals, evidence events, belief
  updates, evolutions, candidates, and resulting belief state;
- runner: projection and final run record.

The append-only ledger may contain an initial and final `run` snapshot, but a
domain object identified by its ID must occur exactly once per lifecycle state.
Tests assert exact counts and IDs, not only record-type presence.

### 5. Provider Judgment Contract

Generic mapping parsing continues to validate shape-level fields. The Evidence
Integration Gate adds request-aware validation:

- likelihood keys must exactly equal requested target hypotheses;
- quality override keys must be from the six supported quality dimensions;
- values must be finite and between zero and one;
- model-probe quality overrides cannot exceed conservative source-specific
  caps.

Violations enter the existing repair policy. Exhausted repair remains a neutral,
discarded schema-violation event.

### 6. Fixtures and Benchmark Truthfulness

The projection fixture uses `external_agent_projection` and includes a cited
source cue so that sender judgment, source claim, and verification candidate are
all observable. The repair fixture uses a deliberately invalid first recorded
response and a valid repair response, with repair enabled in its experiment
test.

MVP metrics remain engineering checks and are documented as such. Update
direction accuracy will use net initial-to-final movement rather than accepting
any transient matching update. `belief_revision_efficiency` will be renamed or
redefined so it does not reward producing more update records.

### 7. Public Interface and WebUI

The package root exports the supported run inputs, configs, results, and runner
interfaces. No new facade will duplicate the core. Callers inject `ModelGateway`,
`ProbeToolGateway`, and `LedgerStore` through existing seams.

The WebUI response adds run regime/status/current cycle and cycle boundary
status/timestamps. The belief panel shows normalized posterior mass and current
uncertainty. Existing provider fields and the answer/cycle layout stay intact.

## Error Handling

- Contract errors raised before integration remain request/configuration errors.
- Provider schema errors use repair when configured, then become discarded
  neutral evidence rather than crashing the run.
- Provider transport errors remain provider errors and do not produce a fake
  completed run.
- Lifecycle records are appended only after their corresponding transition
  succeeds.

## Testing

Tests are added before implementation for:

- normalized exclusive posteriors across binary and multiple-choice sets;
- penalties and hypothesis evolution normalization;
- current belief summaries;
- integrated cycle timestamps;
- completed autonomous and synchronized runs;
- synchronized regime enforcement;
- exactly-once probe set and signal ledger records;
- target-aware likelihood and quality override validation/repair;
- real projection decomposition and repair fixtures;
- package-root imports;
- WebUI lifecycle and normalized-belief serialization.

After focused tests pass, run the complete pytest suite, start the WebUI, and
verify a deterministic multiple-choice run and an OpenAI-compatible run through
the in-app browser.

## Acceptance Criteria

- All existing and new tests pass.
- A deterministic WebUI MCQ run returns a concrete answer choice, normalized
  posterior mass, a completed run, and an integrated cycle.
- A provider-backed WebUI run follows the same domain path and exposes provider
  errors without mutating belief state.
- No canonical signal or probe set is duplicated in a one-cycle ledger.
- The architecture capability matrix no longer overstates fixture, SDK, or
  lifecycle completeness.

## Verification Record

Completed on 2026-07-10:

- the full suite passed with 365 tests and 2 opt-in live-provider tests skipped;
- a deterministic WebUI A-E run selected D, produced posterior mass `1.000`,
  returned a completed autonomous run, and displayed an integrated cycle;
- desktop and 390 px mobile layouts were checked for global overflow and panel
  overlap;
- OpenAI-compatible success/error behavior remains covered through request
  adapter and WebUI integration tests without persisting a live API key.
