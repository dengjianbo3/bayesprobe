# BayesProbe 02 工程实践指导文档 v0.2 大纲

版本目标：将 v0.1 的 “从 ReWOO-style agent 到 BayesProbe-WOO” 工程叙事，重写为完整 BayesProbe Core 的工程化方案。第一版必须同时支持 Autonomous BayesProbe 和 Synchronized BayesProbe，不把 multi-agent / human-in-the-loop 接口作为事后扩展。

## 封面与定位

- 标题：BayesProbe 工程实践指导文档
- 副标题：BayesProbe Core 与双运行制度的可实现架构
- 版本：v0.2
- 日期：2026-07-07
- 定位：工程实现 / schema 设计 / 控制器边界 / MVP 路线图
- 声明：
  - 本文不再以 ReWOO-style 改造为主线。
  - ReAct/ReWOO 只作为 benchmark baseline 或范式对比对象。

## 1. 工程目标与非目标

### 1.1 工程目标

- 实现 BayesProbe Core：
  - Belief State。
  - Probe Set Design。
  - Signal Inbox。
  - Evidence Integration Gate。
  - Evidence Event Construction。
  - Likelihood Judgment。
  - Belief Update。
  - Hypothesis Evolution。
  - Answer Projection。
- 实现两个一等 Run Regime Controllers：
  - Autonomous Controller。
  - Synchronized Controller。
- 支持三种 cycle signal shape：
  - active-only。
  - passive-only。
  - active-plus-passive。
- 产出可审计 ledger。
- 支持 benchmark harness 和另一个 multi-agent 项目的 synchronized 接入。

### 1.2 非目标

- 第一版不追求严格统计意义上的贝叶斯推断。
- 不要求 LLM posterior 完美校准。
- 不做复杂实时会议调度。
- 不共享完整内部 Belief State 作为默认协作协议。
- 不把 ReAct/ReWOO 作为内部执行框架。

## 2. 总体架构

### 2.1 架构图

```text
Input Problem / Existing Belief State
  ↓
BayesProbe Core
  ├─ Frame Builder
  ├─ Hypothesis Manager
  ├─ Probe Set Designer
  ├─ Signal Inbox
  ├─ Evidence Integration Gate
  ├─ Evidence Event Builder
  ├─ Signal Quality Assessor
  ├─ Likelihood Judge
  ├─ Belief Solver
  ├─ Hypothesis Evolver
  └─ Projection Generator

Run Regime Controllers
  ├─ Autonomous Controller
  └─ Synchronized Controller
```

### 2.2 Controller/Core Boundary

- Controller 负责：
  - 开始/结束 cycle。
  - 等待外部输入。
  - 关闭 Signal Collection Boundary。
  - 判断是否继续。
  - 请求输出 projection。
- Core 负责：
  - signal 是否转为 evidence。
  - quality assessment。
  - likelihood judgment。
  - posterior update。
  - hypothesis evolution。
- Controller 不允许绕过 Evidence Integration Gate。

## 3. 核心数据模型

### 3.1 Run

- run_id。
- regime：autonomous | synchronized。
- status。
- problem。
- current_cycle_id。
- created_at / updated_at。
- budget。
- metadata。

### 3.2 Cycle

- cycle_id。
- run_id。
- round_id nullable。
- cycle_index。
- signal_shape：active_only | passive_only | active_plus_passive。
- boundary_status：open | closed | integrated。
- started_at / boundary_closed_at / completed_at。
- controller_metadata。

### 3.3 Belief State

- belief_state_id。
- run_id。
- cycle_id。
- hypotheses。
- posterior_summary。
- uncertainty_summary。
- active_hypotheses。
- retired_hypotheses。
- ledger_refs。

### 3.4 Hypothesis

- id。
- statement。
- type。
- scope。
- prior。
- posterior。
- status：active | weakened | reframed | split | retired | archived。
- rivals。
- falsifiers。
- predictions。
- complexity_penalty。
- ad_hoc_penalty。
- created_by：initial | spawned | split | reframed。
- why_existing_hypotheses_failed nullable。

### 3.5 Probe Design

- id。
- cycle_id。
- target_hypotheses。
- probe_type。
- inquiry_goal。
- method。
- support_condition。
- weaken_condition。
- reframe_condition。
- expected_information_gain。
- decision_relevance。
- cost_estimate。
- priority。
- status。

### 3.6 Probe Set

- probe_set_id。
- cycle_id。
- probes[]。
- boundary_id。
- selection_reason。
- budget_allocated。
- may_be_empty。

### 3.7 Probe Candidate Pool

- candidate_id。
- source：change_my_mind | uncertainty | anomaly | passive_signal | manual。
- candidate_probe。
- priority_features。
- selected_in_cycle nullable。

### 3.8 External Signal

- id。
- signal_kind：active | passive。
- source_type：tool_result | skill_output | search | user_feedback | expert_comment | external_agent_projection | system_log | benchmark_stream | document | simulation。
- source。
- raw_content。
- generated_by_probe nullable。
- received_at。
- cycle_id。
- inbox_status。
- initial_target_hypotheses nullable。

### 3.9 Signal Inbox

- inbox_id。
- cycle_id。
- signal_ids[]。
- boundary_status。
- accepts_passive_until。
- active_probe_completion_status。

### 3.10 Evidence Event

- id。
- derived_from_signal。
- target_hypotheses。
- evidence_type。
- content。
- reliability。
- independence。
- relevance。
- novelty。
- specificity。
- verifiability。
- likelihoods。
- interpretation。
- discard_reason nullable。

### 3.11 Belief Update

- update_id。
- cycle_id。
- evidence_id。
- hypothesis_id。
- prior。
- posterior。
- direction。
- reason。
- sensitivity。

### 3.12 Hypothesis Evolution

- evolution_id。
- cycle_id。
- operation：spawn | split | merge | reframe | reject | retire | reactivate。
- from_hypothesis。
- to_hypothesis。
- triggered_by。
- reason。
- audit_fields。

### 3.13 Answer Projection

- answer。
- current_best_hypothesis。
- posterior_summary。
- main_uncertainty。
- weakest_assumption。
- main_evidence_events。
- change_my_mind_condition。
- answer_utility_notes。

### 3.14 Belief State Projection

- current_best_hypothesis。
- posterior_or_confidence_interval。
- main_evidence_events。
- main_uncertainties。
- questions_for_others。
- change_my_mind_condition。
- requested_signal_type。
- cited_sources。
- projection_metadata。

## 4. Core 模块接口设计

### 4.1 Frame Builder

- 输入：problem、context、role、task goal。
- 输出：problem_frame、decision_context、success_criteria、hidden_assumptions。
- 验收：不直接回答，必须界定问题边界。

### 4.2 Hypothesis Manager

- 负责 initial hypotheses、rivals、prior、status、lifecycle。
- 支持 new hypothesis entry rule。

### 4.3 Probe Set Designer

- 输入：Belief State、Probe Candidate Pool、budget、controller constraints。
- 输出：bounded Probe Set。
- 允许 empty Probe Set。
- 不直接执行 probe。

### 4.4 Signal Inbox Manager

- 接收 active/passive signals。
- 维护 cycle-local inbox。
- 执行 boundary closure。

### 4.5 Evidence Integration Gate

- 输入：closed Signal Inbox。
- 输出：Evidence Events / discard records / deferred signals。
- 保证 active/passive signals 规则一致。

### 4.6 Evidence Event Builder

- 将 External Signal 解释为 Evidence Event。
- 必须区分 raw claim、source claim、belief projection、direct evidence。

### 4.7 Signal Quality Assessor

- reliability。
- independence。
- relevance。
- novelty。
- specificity。
- verifiability。
- duplicate-source detection。

### 4.8 Likelihood Judge

- 判断 Evidence Event 在不同 hypotheses 下的自然程度。
- 支持 LR 档位。
- 输出 likelihood direction 和 rationale。

### 4.9 Belief Solver

- 执行保守 log-odds / softmax update。
- 记录 ledger。
- 不输出答案。

### 4.10 Hypothesis Evolver

- 处理 anomaly、counterevidence、ad hoc patch。
- 支持 lifecycle 状态转移。
- 生成 Probe Candidate Pool。

### 4.11 Projection Generator

- 生成 Answer Projection。
- 生成 Belief State Projection。
- 强制包含 Change-My-Mind Condition。

## 5. Run Regime Controllers

### 5.1 Autonomous Controller

#### 主流程

```text
initialize Belief State
while not stop_condition:
  open cycle
  select Probe Set
  execute active probes
  collect active signals
  close Signal Collection Boundary
  run BayesProbe Core integration/update/evolution
emit Answer Projection
```

#### Stop conditions

- max_cycles。
- max_cost。
- max_time。
- max_tool_calls。
- stable top hypothesis。
- low entropy reduction。
- no high-value probe candidates。
- change-my-mind outside scope/budget。
- sufficient answer utility。

### 5.2 Synchronized Controller

#### 主流程

```text
open round/cycle
receive passive signals from humans/agents/logs
optionally select Probe Set
execute active probes if any
close Signal Collection Boundary
run BayesProbe Core integration/update/evolution
emit Belief State Projection
```

#### MVP 同步方式

- fixed-round synchronization。
- one-round 或 N-round window。
- round_id。
- participant_id。
- incoming_passive_signals[]。
- deadline 或 manual close。
- produce projection。

### 5.3 Controller 共享不变量

- 不得直接更新 posterior。
- 不得把 signal 当 evidence。
- 不得跳过 Evidence Integration Gate。
- 不得以 regime 定制 evidence 规则。

## 6. Projection 协议

### 6.1 Projection-as-Signal Rule

- 外部 Belief State Projection 进入本系统时是 Passive External Signal。
- 必须经过 Evidence Integration Gate。

### 6.2 Projection Decomposition Rule

- sender judgment 与 cited source 分离。
- cited source 可进入 direct verification probe candidate。

### 6.3 Change-My-Mind Condition Schema

```json
{
  "human_readable_condition": "...",
  "structured_probe_candidates": [
    {
      "target_hypotheses": ["H2"],
      "goal": "test evidence independence",
      "method": "source_tracing",
      "expected_weaken_condition": "sources share same origin",
      "priority_hint": 0.8
    }
  ]
}
```

## 7. 更新算法

### 7.1 LR 档位

- strongly disconfirming。
- moderately disconfirming。
- weakly disconfirming。
- neutral。
- weakly confirming。
- moderately confirming。
- strongly confirming。

### 7.2 更新公式

- 单假设 log-odds。
- 多假设 softmax。
- reliability / independence / relevance / novelty weighting。
- complexity_penalty / ad_hoc_penalty。

### 7.3 保守性规则

- 防伪精确概率。
- 强更新需要高质量信号。
- 低 independence 不重复加权。
- low-quality passive signals 可进入 ledger 但低权重或 discard。

## 8. 存储与可观测性

### 8.1 MVP 存储建议

- SQLite 或 Postgres + JSONB。
- 本地实验可以用 JSONL ledger。

### 8.2 核心表

- runs。
- cycles。
- belief_states。
- hypotheses。
- probe_sets。
- probe_candidates。
- external_signals。
- signal_inboxes。
- evidence_events。
- belief_updates。
- hypothesis_evolutions。
- answer_projections。
- belief_state_projections。

### 8.3 可观测性指标

- token cost。
- latency。
- tool calls。
- probe count。
- passive signal count。
- discarded signal count。
- schema violations。
- posterior entropy。
- entropy reduction。
- strong update count。
- neutral signal drift。
- projection decomposition count。

## 9. 防护规则

- Signal 不得直接进答案。
- Projection 不得直接作为 Evidence。
- Controller 不得绕过 Core。
- 每个高影响 hypothesis 必须有 rival/falsifier。
- 每个 Evidence Event 必须有 quality assessment。
- 每次 belief update 必须 ledger 化。
- Anomaly 先触发 hypothesis evolution。
- 新 hypothesis 必须小但可竞争。
- Retired hypothesis 不物理删除。

## 10. MVP 路线图

### Milestone 1：Core schema 与静态单轮

- Frame / Hypothesis / Belief State。
- External Signal -> Evidence Event。
- Belief Update ledger。

### Milestone 2：Autonomous active-only loop

- Probe Set Design。
- Active External Signal。
- Stop conditions。
- Answer Projection。

### Milestone 3：Synchronized passive-only loop

- Fixed-round synchronization。
- Passive External Signal。
- Belief State Projection。
- Projection-as-Signal Rule。

### Milestone 4：active-plus-passive integration

- Mixed Signal Inbox。
- Signal Collection Boundary。
- Evidence Integration Gate。

### Milestone 5：Hypothesis Evolution

- Anomaly trigger。
- spawn/split/reframe/retire。
- New Hypothesis Entry Rule。

### Milestone 6：Benchmark harness

- active-only。
- passive-only。
- active-plus-passive smoke。
- dual-objective metrics。

## 11. 端到端示例

### 11.1 Autonomous example

- 单 agent 独立研究一个复杂 claim。
- active-only cycles。
- 输出 Answer Projection。

### 11.2 Synchronized example

- 三个 agent 固定轮次交换 Belief State Projections。
- 本 agent 收到外部 projection。
- Projection Decomposition。
- passive-only cycle。
- 输出新的 Belief State Projection。

### 11.3 Mixed example

- 本轮既收到专家反馈，也执行主动 source tracing。
- active-plus-passive cycle。
- Evidence Integration Gate 统一处理。

## 附录

- JSON schema draft。
- State machine draft。
- Error taxonomy。
- 与 v0.1 架构术语对照表。
