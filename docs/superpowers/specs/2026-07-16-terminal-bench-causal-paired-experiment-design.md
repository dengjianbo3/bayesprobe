# Terminal-Bench Causal-Conformance and Paired Evaluation Design

Date: 2026-07-16

Status: User-approved design, pending implementation plan

Scope: Terminal-Bench adapter correction, engineering qualification, and a
preliminary paired evaluation of BayesProbe against a resource-matched reactive
control

## 1. Purpose

This design returns the Terminal-Bench work to the original BayesProbe research
question:

```text
Belief State -> Probe -> Signal -> Evidence -> Update
```

The objective is not to turn BayesProbe into a general-purpose coding agent.
The objective is to determine whether a faithful implementation of the
BayesProbe paradigm can be evaluated on long-horizon terminal tasks and whether
it changes task outcomes under a controlled comparison.

The work has two sequential stages:

1. qualify the engineering and paradigm mapping on the three frozen
   Terminal-Bench 2.0 regression tasks used previously;
2. if and only if qualification passes, compare BayesProbe with a matched
   reactive control on a new 30-task Terminal-Bench 2.1 holdout.

Stage 0 answers whether BayesProbe is eligible to be evaluated. Stage 1 answers
whether it produces preliminary evidence of an outcome difference. The two
questions must not be conflated.

## 2. Research Questions

### 2.1 Qualification question

Can the BayesProbe Terminal-Bench adapter complete the atomic five-stage loop
against real terminal environments without provider-contract failures,
untraceable state changes, endogenous self-confirmation, or other causal
conformance violations?

### 2.2 Primary effect question

Under the same base model, terminal capabilities, official task timeout, total
token ceiling, and terminal-action ceiling, does BayesProbe change mean official
Terminal-Bench reward relative to a reactive control?

### 2.3 Mechanism question

When BayesProbe succeeds or fails, do its trajectories show discriminative
Probes, request-bound Signals, admissible Evidence, state-compatible belief
revision, and recovery after failed interventions?

The mechanism analysis explains outcome changes. It does not replace official
task reward.

## 3. Why the Previous Result Is Not an Effect Estimate

The previous three-task paired gate produced:

| Arm | Reached verifier | Official reward | Engineering errors |
| --- | ---: | ---: | ---: |
| Reactive/Direct | 3/3 | 2/3 | 0 |
| BayesProbe | 1/3 | 0/1 completed | 2 |

The BayesProbe result contains two distinct failure classes.

First, `break-filter-js-from-html` and `log-summary-date-ranges` terminated
before the first Probe. The model returned an invalid structured Probe design,
the repair request supplied only a generic `invalid` message, and the second
invalid response aborted the trial. Those runs measured a provider/schema
integration failure, not the BayesProbe loop.

Second, `cancel-async-tasks` reached the verifier but used an invalid causal
interpretation. The agent wrote a Semaphore-based implementation and later
treated the resulting code identity as evidence against an unexecuted TaskGroup
alternative. Choosing an intervention changed the environment; the existence
of the chosen implementation was not external evidence that the unchosen
policy was false.

The old comparison therefore cannot adjudicate the paradigm. It combines
provider-contract errors, an unfaithful coding-task hypothesis mapping, and one
official task failure.

## 4. Frozen Paradigm Commitments

The Terminal-Bench adapter must preserve all of the following:

1. Belief State is the primitive epistemic runtime state.
2. Probe is a hypothesis-conditioned inquiry or intervention plan.
3. Every real environment return is a Signal before it can influence belief.
4. LLM interpretation is required to turn a Signal into Evidence.
5. Deterministic policy decides whether the interpreted Evidence is causally
   admissible for the current state and hypothesis frame.
6. Only admitted Evidence can authorize an Update.
7. A mutation acknowledgement is not evidence that the chosen mutation is
   correct.
8. Answer or artifact production is downstream of the loop and cannot write
   directly to posterior belief.
9. The benchmark adapter must reuse the public BayesProbe runner and core. It
   must not implement a second private loop.
10. The official Harbor verifier remains the sole authority for task reward.

## 5. Scope and Non-Goals

### 5.1 In scope

- Terminal-Bench task framing for coding and terminal tasks;
- bounded terminal Probe planning;
- action classification and state lineage;
- request-bound Signal construction;
- causal Evidence admissibility;
- historical trace replay and conformance validation;
- Harbor 2.0 regression qualification;
- Harbor 2.1 paired experiment orchestration;
- resource accounting, analysis, and report generation.

### 5.2 Out of scope

- changing the BayesProbe core domain model;
- building a universal coding agent;
- adding web search, Tavily, or benchmark-solution retrieval;
- optimizing prompts against holdout task reward;
- leaderboard submission in this milestone;
- claiming that one benchmark proves a universal agent paradigm;
- adding counterfactual environment branches or a general causal graph to the
  core.

If a faithful adapter proves impossible without a core change, implementation
must stop and reopen paradigm design. It must not simulate a new core inside
the benchmark package.

## 6. Selected Architecture

BayesProbe core remains unchanged. The correction is isolated under
`benchmarks/terminal_bench`.

### 6.1 Terminal task framing policy

The task framer creates hypotheses about:

- root causes;
- current-state behavior;
- required invariants and constraints;
- falsifiable postconditions;
- explicit intervention-effect claims when a Probe is designed to test a
  causal transition.

Implementation alternatives, patches, and command sequences are policies, not
ranked world claims. They stay in the terminal planner and action history rather
than becoming competing hypotheses solely because they are possible solutions.

### 6.2 Terminal Probe planner

The planner turns one selected BayesProbe Probe into a bounded terminal plan.
A plan is one of:

- `inspect`: read-only actions that characterize the current state;
- `intervene`: at most one mutation followed by a declared verification;
- `verify`: read-only checks against the current state.

A single Probe may contain preparatory inspection, one mutation, and its
verification. It may not contain multiple unrelated mutations whose effects
cannot be attributed separately.

### 6.3 Causal action ledger

Every executed action records:

- run, cycle, Probe, plan, and action identity;
- action role: inspect, intervene, or verify;
- normalized executed request and request fingerprint;
- action index and policy-attempt identity;
- pre-action and post-action environment-state identity;
- return code, timeout status, bounded model-facing output, and full output
  fingerprint;
- whether the action changed the environment;
- the verification target for intervention plans.

### 6.4 Terminal Signal bridge

Every executed action produces one External Signal. The Signal is bound to the
actual request and observed result, not to the model's prior description of
what it intended to run.

A successful file write or patch produces a procedural Signal: the intervention
occurred. It does not by itself support the selected implementation or weaken
unexecuted alternatives.

### 6.5 Causal Evidence guard

The LLM remains responsible for semantic Evidence judgment. The adapter then
applies deterministic admissibility rules:

1. an inspection result may update hypotheses scoped to the inspected state;
2. a mutation acknowledgement is neutral with respect to solution quality;
3. a post-intervention verification may update postconditions scoped to the
   resulting state;
4. a post-intervention outcome may update a pre-intervention causal diagnosis
   only when the Probe declared the intervention and differentiated predicted
   transitions under the target hypotheses before execution;
5. current code identity cannot refute an unexecuted policy;
6. Evidence derived from a stale or incompatible state cannot update the
   current frame;
7. a Signal that cannot be linked to exactly one executed request is discarded;
8. discarded Evidence cannot cause a posterior change.

The guard does not invent Evidence or choose a preferred implementation. It
only prevents an invalid causal route from reaching Update.

### 6.6 Trace conformance validator

The validator checks the complete chain:

```text
state-scoped Belief
-> declared Probe
-> executed action
-> request-bound Signal
-> causally admissible Evidence
-> caused Update
```

It also validates budgets, identities, prompt/schema provenance, environment
lineage, update direction, and the absence of hidden evaluator access.

### 6.7 Reactive control

The control is a tool-using reactive agent, not a one-shot answer generator. It
uses the same base model, terminal action schema, environment bridge, task
instruction, permissions, and resource ceilings. It can inspect, modify, and
verify the environment. It does not maintain explicit BayesProbe hypotheses,
Probe objects, Evidence judgments, or posterior state.

This makes the explicit BayesProbe state machine the principal experimental
variable.

## 7. Structured Output Failure Policy

Probe-design and terminal-plan validation use bounded, targeted repair.

1. Validate the response against the exact semantic schema.
2. Record a response hash, required-key presence, validation stage, and a
   redacted field-level error summary.
3. Send the invalid semantic payload and redacted field-level error to a repair
   request.
4. Permit no more than two repair attempts after the initial response.
5. If all attempts fail, classify the trial as `provider_contract_error` and
   fail closed.

No deterministic Probe template is allowed after provider-contract exhaustion.
No unbounded retry is allowed. Invalid output never becomes a fabricated
Signal.

## 8. Stage 0: Engineering and Paradigm Qualification

Stage 0 uses the original frozen Terminal-Bench 2.0 tasks:

1. `terminal-bench/break-filter-js-from-html`
2. `terminal-bench/cancel-async-tasks`
3. `terminal-bench/log-summary-date-ranges`

The historical task references and image digests are retained. The corrected
code, prompt, and schema hashes receive a new qualification lock.

### 8.1 Historical negative-trace replay

Before any live provider use:

- the two pre-Probe failures must classify as `provider_contract_error`;
- the completed self-confirming trace must classify as
  `causal_conformance_error`;
- a synthetic valid inspect/intervene/verify trace must pass;
- mutations, Signals, Evidence, and Updates with broken identities or state
  lineage must fail.

The historical artifacts are read-only fixtures identified by content hash.

### 8.2 Environment and protocol qualification

- the official Oracle must receive full reward on all three frozen tasks;
- Harbor version, dataset identity, task references, image digests, code SHA,
  adapter tree SHA, model configuration, prompt hashes, schema hashes, and
  budgets must be locked;
- the benchmark adapter must be clean at lock time;
- provider-contract repair must expose field-level diagnostics without
  persisting secrets or private reasoning content.

### 8.3 Live BayesProbe qualification

Run BayesProbe once on each task using the same model and resource policy
intended for Stage 1. Every task must:

- reach the official Harbor verifier;
- contain at least one complete five-stage cycle;
- contain a complete request-to-Update trace for every non-neutral Update;
- remain within all budgets;
- contain no provider, adapter, budget, or causal-conformance error;
- contain no post-intervention self-confirmation or incompatible-state Update.

Official reward is recorded but is not a qualification condition. Three
conformant reward-zero trials pass Stage 0. A reward-one trial with a causal
violation fails Stage 0.

External network, provider 429/5xx, and Docker/Harbor infrastructure failures
permit one fresh-trial retry. Provider-contract, agent, budget, and conformance
failures do not.

## 9. Stage 1 Dataset and Holdout Construction

Stage 1 uses the current official
`terminal-bench/terminal-bench-2-1` dataset. Terminal-Bench 2.1 retains 89 tasks
and corrects problems in 28 Terminal-Bench 2.0 tasks.

### 9.1 Exposure exclusion ledger

Before sampling, create a frozen ledger of tasks or instructions exposed during
development. It must include:

- all three Stage 0 tasks;
- `terminal-bench/build-cython-ext`;
- any task previously run by either adapter;
- any task instruction inspected in a browser, document, log, or code review;
- an instruction-content hash for exposed tasks whose task ID was not known at
  exposure time.

Exposure exclusion is determined before Oracle or agent outcomes are observed.

### 9.2 Deterministic stratified selection

From the remaining 2.1 pool:

1. form strata from official category and difficulty metadata, using an
   explicit `uncategorized` or `unspecified` value when metadata is absent;
2. allocate 30 slots proportionally by largest remainder, while assigning at
   least one slot to each represented category when the number of categories
   does not exceed 30;
3. break allocation ties lexicographically by stratum identifier;
4. within each stratum sort by
   `SHA256("bayesprobe-tb21-v1:" + task_id)`;
5. take the first allocated tasks.

The complete eligible pool, strata, allocation, ranking, and selected tasks are
saved before running an experimental arm.

### 9.3 Oracle qualification and replacement

Run the official Oracle on the selected tasks before either arm begins. If a
task fails due to an Oracle, image, or verifier defect, replace it only with the
next task in the frozen hash order from the same stratum. Record the original
task, replacement, reason, and artifacts.

Once the first agent trial begins, the 30-task set cannot be replaced. A newly
discovered benchmark defect removes that task from the primary analysis and is
reported; it does not authorize a favorable replacement.

## 10. Stage 1 Experimental Matrix

The experiment contains:

```text
30 tasks x 2 arms x 3 repeats = 180 planned trials
```

Each task-repeat combination is one paired block.

### 10.1 Block order

Construct the 90 block identifiers `(task_id, repeat_index)`. Sort them by:

```text
SHA256("bayesprobe-tb21-block-v1:" + task_id + ":" + repeat_index)
```

Within each block, assign arm order from the low bit of:

```text
SHA256("bayesprobe-tb21-arm-v1:" + task_id + ":" + repeat_index)
```

Run the two arms serially and adjacently in independent fresh containers. Run
one paired block at a time. No state, transcript, cache under adapter control,
or artifact is shared between arms.

### 10.2 Model freeze

The primary study uses one frozen OpenAI-compatible model. The intended initial
target is DeepSeek `deepseek-v4-flash` at temperature zero. Before Stage 0, the
manifest must record:

- configured model ID;
- provider base host;
- protocol and SDK versions;
- provider-returned model/system fingerprint when available;
- tokenizer or provider token-accounting semantics;
- maximum output tokens per request.

An unannounced provider identity change pauses the experiment. A second model is
a future replication, not part of the 180-trial primary study.

### 10.3 Resource equality

Both arms receive:

- the official per-task `agent.timeout_sec` without modification;
- a terminal-action ceiling of 24;
- a model-call safety ceiling of 72;
- the same total provider-token ceiling;
- the same command permissions, task network policy, and evaluator isolation;
- the same per-request timeout policy;
- the same provider endpoint and maximum output tokens.

The total provider-token ceiling is frozen after Stage 0 and before holdout
selection is revealed to the model. It is:

```text
max(160000, round_up_to_10000(1.25 * max_stage0_trial_tokens))
```

and is capped at 300000 tokens per trial. `trial_tokens` means provider-reported
input plus output tokens across all calls; cached-token fields are recorded
separately. If reliable token accounting is unavailable, Stage 1 cannot start.

The configured provider request timeout is 360 seconds, bounded by the official
remaining task time. A command timeout is at most 120 seconds and is also
bounded by the official remaining task time. Neither setting extends the
official Harbor agent timeout.

No arm-specific tuning is allowed after Stage 1 begins.

## 11. Error Taxonomy and Retry Policy

### 11.1 Exogenous errors

Exogenous errors include:

- provider 429 or 5xx;
- network transport failure outside the task;
- Docker daemon or Harbor orchestration failure;
- corrupted image pull;
- verifier infrastructure failure that also invalidates Oracle behavior.

The entire paired block is rerun once in fresh containers, even if only one arm
encountered the error. The failed block remains in raw artifacts.

If the paired retry also fails exogenously, that repeat is missing for both
arms. A task with fewer than two valid paired repeats is excluded from the
primary task-level effect estimate and disclosed without replacement.

### 11.2 System outcomes

The following are system outcomes, not retryable infrastructure errors:

- invalid structured output after the bounded repair policy;
- action, token, call, or wall-time budget exhaustion;
- uncaught adapter or agent exception;
- incomplete BayesProbe trace;
- causal-conformance failure;
- prohibited evaluator or solution access.

These receive reward zero in the primary analysis and retain a separate failure
classification. Reward hacking also receives reward zero and triggers the
integrity stop condition.

### 11.3 Benchmark defects

A benchmark defect discovered after Stage 1 begins removes the affected task
from both arms' primary analysis. It is not replaced. The report includes a
sensitivity analysis with the task retained under its raw official rewards.

## 12. Outcomes

### 12.1 Primary outcome

For each task, average official reward across its valid paired repeats within
each arm. The primary effect is:

```text
delta_reward = mean_task(BayesProbe) - mean_task(Reactive)
```

The task is the statistical unit. Repeats are not treated as 90 independent
tasks.

### 12.2 Secondary outcome metrics

- official resolution rate per arm and repeat;
- verifier-reach rate;
- agent/system failure rate by class;
- provider tokens, terminal actions, wall time, and estimated cost;
- reward per 100000 provider tokens;
- reward per 10 terminal actions.

Efficiency metrics are descriptive and do not replace official reward.

### 12.3 BayesProbe mechanism metrics

Across all BayesProbe trajectories, compute:

- Probe-to-request-bound-Signal closure rate;
- proportion of Probes yielding at least one admissible non-neutral Evidence
  judgment;
- mutation-acknowledgement self-confirmation rate;
- stale-state or incompatible-state Evidence rate;
- proportion of posterior Updates linked to admitted Evidence and a valid
  contribution root;
- recovery rate after a failed intervention, defined as a subsequent distinct
  diagnosis or intervention followed by a successful declared verification
  within the same trial;
- structured-contract failure rate;
- causal-conformance failure rate.

The target for self-confirmation, stale-state Update, uncaused posterior change,
and causal-conformance failure is zero.

### 12.4 Manual mechanism audit

Select 18 of the 90 BayesProbe trials by sorting:

```text
SHA256("bayesprobe-tb21-audit-v1:" + task_id + ":" + repeat_index)
```

Two reviewers independently inspect the selected trajectories without official
reward or final verifier output. They code:

- whether the Probe was discriminative relative to its targets;
- whether the Signal came from the declared request;
- whether Evidence interpretation followed the observation;
- whether update direction and state scope were valid;
- whether a failed intervention changed subsequent inquiry.

Disagreements are adjudicated before reward is revealed. Reviewer identity,
rubric version, initial labels, and adjudicated labels are retained.

## 13. Statistical Analysis

### 13.1 Primary estimate

Report the mean paired task-level reward difference in percentage points.

Compute a 95 percent paired bootstrap interval by resampling the included tasks
with replacement 10000 times using the seed string
`bayesprobe-tb21-bootstrap-v1`.

Compute a two-sided task-level sign-flip randomization p-value with 100000
deterministic draws using seed string `bayesprobe-tb21-signflip-v1`. Zero
differences remain zero under sign flips.

### 13.2 Sensitivity analyses

- trial-level paired outcome table by repeat;
- complete-case estimate requiring all three valid paired repeats;
- raw official rewards before retry processing;
- result excluding all system-error trials from both arms, labeled explicitly
  as non-primary;
- category and difficulty breakdowns, descriptive only;
- cost- and action-normalized descriptive outcomes.

No secondary comparison is promoted to primary after results are known.

### 13.3 Interpretation

- If the 95 percent interval is entirely above zero, the study provides
  preliminary support for the effect hypothesis under the frozen model,
  resource policy, and task distribution.
- If the point estimate is positive but the interval crosses zero, the result
  is directional evidence only.
- If the point estimate is near or below zero and the interval crosses zero,
  the study does not support the effect hypothesis.
- If the interval is entirely below zero, the study provides preliminary
  counterevidence.
- Systematic causal-conformance violations prohibit attributing any favorable
  reward difference to the BayesProbe paradigm.

Stage 0 is an eligibility result, not evidence of effectiveness. Stage 1 cannot
establish universal validity.

## 14. Blinding and Monitoring

During Stage 1, operators may inspect:

- process liveness;
- infrastructure error classes;
- token, action, time, and cost budgets;
- artifact completeness;
- conformance status needed for an integrity stop.

They may not inspect cumulative or arm-level official reward. Trial result
files remain immutable, but aggregate reward analysis is not generated until
all planned blocks complete or a non-efficacy stopping rule fires.

The manual mechanism audit is completed before reward unblinding.

## 15. Artifacts and Reproducibility

### 15.1 Experiment manifest

The immutable manifest records:

- experiment ID and schema version;
- dataset and Harbor versions;
- eligible pool, exposure exclusions, strata, task ranks, selected tasks, and
  Oracle replacements;
- task references and image digests;
- block and arm order;
- code, adapter tree, prompt, and schema hashes;
- model and provider identity without credentials;
- all budgets and timeout rules;
- retry, scoring, stopping, and analysis policy;
- analysis script hashes.

API credentials are read only from named environment variables and are never
written to configuration, logs, artifacts, or the manifest.

### 15.2 Trial artifacts

Every trial retains:

- Harbor configuration, result, exception, logs, and official reward;
- ATIF-compatible agent trajectory;
- provider telemetry and resource accounting;
- normalized environment actions;
- artifact manifest and content hashes.

BayesProbe trials additionally retain:

- task admission and frame;
- Belief States, Probe candidates and selected Probe sets;
- request-bound Signals;
- Evidence Events, contribution deltas, and Updates;
- causal state lineage;
- conformance report.

Reactive trials retain the corresponding action/observation trajectory and
provider telemetry.

### 15.3 Final report

The report contains:

- task selection and exclusion flow;
- Stage 0 qualification outcomes;
- all 180 planned trial dispositions;
- primary estimate, interval, and p-value;
- per-task paired results;
- raw and policy-processed errors;
- mechanism metrics and blinded audit results;
- token, action, time, and cost summaries;
- limitations and the exact claim boundary.

## 16. Integrity and Stopping Rules

Pause immediately, without examining cumulative reward, when any of the
following occurs:

- a Stage 0 qualification condition fails;
- a dataset, image, model, code, prompt, schema, or analysis hash differs from
  the manifest;
- a credential or restricted evaluator artifact leaks;
- hidden tests, verifier files, oracle solutions, or public benchmark solutions
  are accessed by an arm;
- the first causal-conformance error appears in Stage 1;
- three consecutive paired blocks encounter the same exogenous failure;
- provider identity or token-accounting behavior changes;
- the pre-registered monetary ceiling would be exceeded.

There is no efficacy-based early stopping.

Any code, prompt, schema, model, budget, or policy change after Stage 1 begins
invalidates the current experiment version. The corrected run receives a new
experiment ID and restarts from the first paired block.

## 17. Security and Benchmark Integrity

Neither arm receives Tavily, unrestricted web search, or any task-solution
retrieval capability. Terminal-Bench treats finding public solutions online as
reward hacking.

The terminal environment follows the official task network policy. A task that
legitimately requires network access may use only the task-authorized resources.
The adapter blocks known evaluator paths, Oracle artifacts, host Docker sockets,
credentials, and benchmark-solution sources from model-facing context and
commands.

Passing trajectories must be exportable in the benchmark's required trajectory
format so that successful runs can be audited for integrity.

## 18. Verification Strategy

Implementation verification proceeds in this order:

1. unit tests for task framing, action roles, request binding, state lineage,
   structured repair, causal Evidence rules, and error classification;
2. historical negative-trace replay;
3. synthetic positive and adversarial conformance fixtures;
4. deterministic Harbor integration test through the public BayesProbe runner;
5. Oracle qualification on the three frozen 2.0 tasks;
6. live Stage 0 provider-backed qualification;
7. 2.1 holdout selection, lock, and Oracle qualification;
8. a manifest-only dry run that expands all 180 planned trial configs without
   provider calls;
9. Stage 1 execution;
10. independent verification of report reproduction from immutable artifacts.

Tests must prove that the old self-confirming trace fails and a state-valid
intervention trace passes. Merely increasing test count is not acceptance.

## 19. Acceptance Criteria

The implementation is ready for Stage 0 when:

- no BayesProbe core file is changed;
- the adapter reuses the public BayesProbe runner and core;
- structured repair emits safe field-level diagnostics and fails closed;
- diagnosis hypotheses and action policies are separated;
- every action produces a request-bound, state-bound Signal;
- the causal guard prevents mutation acknowledgements and stale-state results
  from creating invalid Updates;
- the historical failures classify as specified;
- the synthetic conformant trace passes;
- locks and artifacts contain no secret values;
- the Terminal-Bench adapter test suite and repository regression suite pass.

Stage 1 may start only after all Stage 0 live qualification criteria pass and
the complete Stage 1 manifest is immutable.

## 20. References

- Terminal-Bench 2.1 release and 28-task correction:
  https://www.tbench.ai/news/terminal-bench-2-1
- Official Harbor Terminal-Bench tutorial:
  https://www.harborframework.com/docs/tutorials/running-terminal-bench
- Harbor custom agent integration:
  https://www.harborframework.com/docs/agents
- Terminal-Bench timeout integrity policy:
  https://www.tbench.ai/news/leaderboard-integrity-and-timeouts
- Terminal-Bench reward-hacking and trajectory policy:
  https://www.tbench.ai/news/leaderboard-integrity-update
- Terminal-Bench 2.0 paper:
  https://arxiv.org/abs/2601.11868
