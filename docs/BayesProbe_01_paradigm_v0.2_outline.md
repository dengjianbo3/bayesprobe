# BayesProbe 01 范式论述文档 v0.2 大纲

版本目标：将 v0.1 中偏向 “ReWOO-style 改造 / 现有范式升级层” 的叙事，重写为 BayesProbe 作为完整 agent 原子范式的论述。本文档负责回答：BayesProbe 是什么、为什么是新范式、它和 ReAct/ReWOO/ToT/GoT 的哲学差异是什么。

## 封面与定位

- 标题：BayesProbe 范式论述文档
- 副标题：一种基于外部信号的贝叶斯信念修正 agent 原子范式
- 版本：v0.2
- 日期：2026-07-07
- 定位：范式定义 / 哲学基础 / 工程与实验统一基线
- 声明：
  - BayesProbe 是完整 agent 范式，不是 ReAct/ReWOO 的包装、混合或执行层改造。
  - ReAct/ReWOO/ToT/GoT 等作为邻近范式比较对象，不作为 BayesProbe 内部组件。

## 摘要

- BayesProbe 的核心对象是 `Belief State`，不是 action、plan、thought branch 或 evidence slot。
- BayesProbe 的第一原则是 `Epistemic Humility`：agent 不直接拥有答案，而是在不确定环境中维护可修正信念。
- 外部信息首先是 `External Signal`，只有经过 `Evidence Event`、质量评估、似然判断和信念更新后，才能影响答案投影。
- BayesProbe 支持主动与被动两类信号：
  - Active External Signal：由 Probe Design 主动触发。
  - Passive External Signal：由人类、其他 agent、日志、benchmark stream 等被动注入。
- BayesProbe 支持两个一等运行制度：
  - Synchronized BayesProbe。
  - Autonomous BayesProbe。
- 评价采用双目标：Answer Utility + Belief-Revision Quality。

## 1. 引言：为什么需要 BayesProbe

- 复杂问题的困难不只是“产出答案”，而是“在持续进入的外部信号压力下修正信念”。
- 传统 agent 范式常隐含 answer-first 或 action-first 假设。
- BayesProbe 的起点：
  - 复杂问题通常是一组竞争假设。
  - 真正关键的是判断信号如何改变不同假设的可信度。
  - 答案是 Belief State 的压缩投影，而不是原始认知对象。
- 本节应避免说“BayesProbe 是在 ReWOO 上加 belief revision”；应说“BayesProbe 与 ReWOO 等在同一层级比较，但哲学起点不同”。

## 2. 范式比较：现有 agent 范式的核心对象

### 2.1 比较原则

- 比较维度：
  - 核心对象。
  - 原子循环。
  - 隐含哲学假设。
  - BayesProbe 的分歧点。

### 2.2 对比表

- CoT：
  - 核心对象：中间推理链。
  - 隐含假设：线性语言化推理可逼近答案。
- ReAct：
  - 核心对象：action / observation。
  - 隐含假设：通过行动和观察逐步逼近任务成功。
- ReWOO：
  - 核心对象：plan / evidence slot。
  - 隐含假设：问题结构可先规划，外部信息主要用于填充。
- ToT / GoT：
  - 核心对象：thought branch / thought graph。
  - 隐含假设：答案存在于结构化思维搜索空间中。
- Reflexion / Self-Refine：
  - 核心对象：反馈和修订。
  - 隐含假设：语言化反馈可改善后续输出。
- BayesProbe：
  - 核心对象：evolving hypothesis belief state。
  - 隐含假设：认知任务的核心是信号驱动的信念修正。

### 2.3 哲学差异总结

```text
ReAct:    knowing through action and observation
ReWOO:    knowing through planned evidence collection
ToT/GoT:  knowing through structured thought search
BayesProbe: knowing through signal-grounded belief revision
```

## 3. BayesProbe 的认识论基础

### 3.1 Epistemic Humility

- Agent 的原始状态不是答案，而是可修正信念。
- 高置信结论必须能被 counterevidence 改变。
- 所有输出必须包含 `Change-My-Mind Condition`。

### 3.2 贝叶斯式信念修正

- 使用 credence、prior、likelihood、posterior、belief delta。
- 不要求严格频率贝叶斯估计。
- 使用粗粒度 LR 档位、区间和保守更新。

### 3.3 可证伪性与 rival hypotheses

- 高影响假设必须包含：
  - scope。
  - rivals。
  - falsifier。
  - risky prediction。
  - boundary condition。
- 防止不可检验补丁和 ad hoc 保护。

### 3.4 开放假设空间

- BayesProbe 不只是旧假设之间重新分配概率。
- 必须支持：
  - spawn。
  - split。
  - merge。
  - reframe。
  - reject。
  - retire。
- anomaly 应先触发 Hypothesis Evolution，而不是继续围绕旧假设 probe。

## 4. 核心对象定义

### 4.1 Belief State

- 当前假设集合、posterior、scope、rival 关系、不确定性、更新历史。
- 是 BayesProbe 的原始目标状态。

### 4.2 Hypothesis

- 候选命题、解释或决策立场。
- 必须字段：
  - id。
  - statement。
  - type。
  - scope。
  - prior/posterior。
  - status。
  - rivals。
  - falsifiers。
  - predictions。
  - penalties。

### 4.3 Probe Design

- 假设驱动的求证方案。
- 回答：“为了判断这些假设，我需要什么外部信息？”
- 不是工具调用本身，不是 signal，不是 evidence。

### 4.4 Probe Set

- 一轮 cycle 中有限、有边界的一组 Probe Designs。
- 表达 agent 的 `Active Control Signal`。
- 可以为空，尤其在 passive-only synchronized cycle 中。

### 4.5 External Signal

- 外部原始信息注入点。
- 分为：
  - Active External Signal。
  - Passive External Signal。
- signal 不得直接影响 posterior 或 final answer。

### 4.6 Evidence Event

- 对 External Signal 的结构化、质量评估后的解释。
- 必须包含：
  - derived_from_signal。
  - target_hypotheses。
  - evidence_type。
  - reliability。
  - independence。
  - relevance。
  - novelty。
  - likelihoods。

### 4.7 Belief Update

- 信念变化 ledger。
- 记录 prior、posterior、direction、reason、sensitivity。

### 4.8 Hypothesis Evolution

- 假设空间操作。
- 必须可审计，并保留旧假设生命周期。

### 4.9 Answer Projection

- 面向用户任务的答案投影。
- 不是 BayesProbe 的原始目标状态。

### 4.10 Belief State Projection

- 面向人类或其他 agent 的协作投影。
- 必须包含 change-my-mind condition。

## 5. BayesProbe 核心循环

### 5.1 v0.2 推荐循环

```text
Belief State_t
→ Probe Set Design
→ Signal Inbox
→ Signal Collection Boundary
→ Evidence Integration Gate
→ Evidence Event Construction
→ Likelihood Judgment
→ Belief Update
→ Hypothesis Evolution
→ Belief State_t+1
→ Answer Projection / Next Cycle
```

### 5.2 Signal Inbox

- 一轮内收集 active/passive signals。
- passive signals 可以在主动 probe 返回前进入 inbox。

### 5.3 Signal Collection Boundary

- 关闭本轮信号集合。
- 边界后到达的 passive signal 进入下一轮。

### 5.4 Evidence Integration Gate

- 统一处理 active/passive signals。
- 决定：
  - 转为 Evidence Event。
  - discard。
  - defer。
  - request clarification。

### 5.5 合法 cycle shapes

- active-only。
- passive-only。
- active-plus-passive。

## 6. 运行制度

### 6.1 Dual-Regime BayesProbe

- Synchronized 和 Autonomous 都是一等运行制度。
- 二者共享同一个 BayesProbe Core。

### 6.2 Synchronized BayesProbe

- 用于 multi-agent / human-in-the-loop / 固定轮次协作。
- cycle boundary 由外部协作协议定义。
- 支持 passive-only cycle。
- 输出 Belief State Projection。

### 6.3 Autonomous BayesProbe

- 用于单 agent 独立研究、子 agent 执行任务、benchmark。
- cycle boundary 由内部停止条件定义。
- 停止条件：
  - hard limits。
  - posterior stability。
  - entropy reduction。
  - no high-value probe candidates。
  - sufficient answer utility。

## 7. 多 agent 与人类协作语义

### 7.1 Belief State Projection 是交换对象

- 不共享完整内部 Belief State。
- 投影字段：
  - current best hypothesis。
  - posterior/confidence interval。
  - main evidence events。
  - uncertainty。
  - questions for others。
  - change-my-mind condition。
  - requested signal type。

### 7.2 Projection-as-Signal Rule

- 收到其他 agent / 人类 projection 时，作为 Passive External Signal。
- 不能直接作为 Evidence Event。

### 7.3 Projection Decomposition Rule

- 将对方结论和引用来源拆开。
- 对方 posterior 不是本 agent 的 evidence。

## 8. 失败模式与防护原则

- 伪精确概率。
- confirmation bias。
- probe 退化为普通工具调用。
- passive signals 绕过 evidence gate。
- projection 被误当 evidence。
- duplicate evidence。
- authority bias。
- anomaly 被硬塞进旧假设。
- ad hoc hypothesis protection。
- answer-only evaluation。
- belief-state-only evaluation。

## 9. 评价逻辑

### 9.1 Dual-Objective Evaluation

- Answer Utility：
  - final accuracy。
  - task success。
  - decision usefulness。
  - cost-normalized utility。
- Belief-Revision Quality：
  - hypothesis coverage。
  - evidence event quality。
  - likelihood direction。
  - posterior update direction。
  - calibration。
  - counterevidence correction。
  - uncertainty expression。
  - change-my-mind quality。

### 9.2 范式成功标准

- BayesProbe 不是更会说，而是在外部信号压力下更会合理改变信念。
- 最终答案仍必须有实际任务价值。

## 10. 结论

- BayesProbe 的最终定义：

```text
Signal-grounded Bayesian belief revision over evolving hypotheses.
```

- 中文定义：

```text
基于外部信号、在演化假设空间上进行贝叶斯信念修正的 agent 原子范式。
```

- 强调：
  - 完整范式。
  - 非 ReAct/ReWOO wrapper。
  - active/passive signal 同等进入 evidence 判断。
  - dual-regime support。
  - answer projection 与 belief state 的关系。

## 附录

- 核心术语表。
- 与 v0.1 的关键变化表。
- 参考文献。
