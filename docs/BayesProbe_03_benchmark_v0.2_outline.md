# BayesProbe 03 实验与 Benchmark 规划文档 v0.2 大纲

版本目标：将 v0.1 的 evidence-stream belief revision 评测扩展为双运行制度、三种 cycle signal shape、双目标评价的实验规划。本文档负责回答：如何证明 BayesProbe 既有实际答案价值，又有更好的内部信念修正机制。

## 封面与定位

- 标题：BayesProbe 实验与 Benchmark 规划文档
- 副标题：用于验证双运行制度下的外部信号驱动信念修正能力
- 版本：v0.2
- 日期：2026-07-07
- 定位：实验目标 / benchmark 构造 / 指标体系 / 消融设计 / 记录模板
- 声明：
  - 实验不只比较 final accuracy。
  - 实验也不只研究内部 belief state。
  - v0.2 必须覆盖 Autonomous 与 Synchronized 两种运行制度。

## 1. 实验目标

### 1.1 双目标评价

- External Answer Utility：
  - final accuracy。
  - EM/F1。
  - label correctness。
  - decision usefulness。
  - cost-normalized utility。
- Internal Belief-Revision Quality：
  - hypothesis coverage。
  - rival quality。
  - falsifier quality。
  - Evidence Event quality。
  - likelihood direction。
  - posterior update direction。
  - calibration。
  - counterevidence response。
  - uncertainty expression。
  - change-my-mind quality。

### 1.2 v0.2 关键验证对象

- Active External Signal 能否被正确证据化。
- Passive External Signal 能否被同等纳入 Evidence Integration Gate。
- Belief State Projection 是否能作为 Passive External Signal 而非 Evidence。
- Projection Decomposition Rule 是否减少 authority bias 和 duplicate evidence。
- Synchronized BayesProbe 是否能在固定轮次协作中稳定更新 belief state。
- Autonomous BayesProbe 是否能在停止条件下产出可用答案。

## 2. 研究问题与核心假设

### RQ1：BayesProbe 是否提升答案可用性？

- H-BP1：在 claim verification / QA / research QA 上，BayesProbe 的 final accuracy 不低于强基线，并在复杂信号场景下更优。

### RQ2：BayesProbe 是否提升信念修正质量？

- H-BP2：BayesProbe 的 update direction accuracy、calibration、counterevidence correction 优于 Direct/CoT/ReAct/ReWOO/ReProbe。

### RQ3：BayesProbe 是否能正确处理 passive signals？

- H-BP3：在 passive-only cycle 中，BayesProbe 能把人类/agent/log/benchmark stream 转成正确 Evidence Event，并合理更新 posterior。

### RQ4：Projection-as-Signal Rule 是否有贡献？

- H-BP4：把外部 projection 先作为 Passive External Signal，而不是直接作为 Evidence，可降低 authority bias 和重复证据加权。

### RQ5：Synchronized BayesProbe 是否能支持 multi-agent/human 轮次协作？

- H-BP5：在 fixed-round exchange 中，BayesProbe 的 Belief State Projection 更能帮助下一参与者提供有价值 signal。

### RQ6：Hypothesis Evolution 是否有贡献？

- H-BP6：面对 anomaly/counterevidence，开启 hypothesis evolution 的版本比 no-evolution 版本更少出现 stale hypothesis probing 和 ad hoc patch。

## 3. 基线系统

### 3.1 外部答案基线

- Direct Answer。
- CoT。
- ReAct。
- ReWOO。
- ReProbe：Hypothesis -> Probe -> Observation -> Revision，但无显式 posterior。

### 3.2 BayesProbe 系统

- BayesProbe-Autonomous。
- BayesProbe-Synchronized。
- BayesProbe-Batch / BayesProbe-Iterative 仅用于描述运行节奏，不作为依附旧范式的命名。

### 3.3 控制变量

- base model。
- temperature。
- max tokens。
- max tool calls。
- retrieval corpus。
- signal stream order。
- scoring scripts。
- evaluator prompts。
- sample ordering。

## 4. Benchmark 选择

### 4.1 Active-only 主 benchmark

- FEVER-Stream。
- SciFact-Stream。
- PubMedQA-Stream。
- QASPER-Stream。
- HotpotQA supporting facts。
- StrategyQA hidden assumptions。

用途：

- 测试主动 probe / retrieval / tool result 如何进入 Evidence Event。
- 测试 final accuracy 和 belief update direction。

### 4.2 Passive-only benchmark

候选类型：

- Human-feedback simulation。
- External-agent projection exchange。
- System-log signal stream。
- Benchmark-provided evidence stream。

第一版建议覆盖：

- benchmark-provided evidence stream：最容易自动化。
- external-agent projection exchange：最贴近另一个 multi-agent 项目。

### 4.3 Active-plus-passive smoke tests

- 小样本混合场景：
  - 本轮主动检索返回 signal。
  - 同时收到专家反馈或另一个 agent projection。
  - Signal Inbox 统一进入 Evidence Integration Gate。

### 4.4 Later-stage benchmarks

- TruthfulQA / FreshQA / SimpleQA。
- GAIA。
- BrowseComp。
- AgentBench。
- WebArena。

这些适合更成熟的工具与 browsing 场景，不作为第一版核心。

## 5. Evidence-Stream 与 Signal-Stream 构造

### 5.1 从 Evidence-Stream 到 Signal-Stream

v0.1 的 evidence stream 应重命名或扩展为 signal stream，因为输入系统的是 raw external signal，而不是已经判断完的 evidence。

样本格式：

```json
{
  "sample_id": "...",
  "question_or_claim": "...",
  "hypotheses": ["H1", "H2", "H3"],
  "initial_context": "...",
  "cycle_signal_stream": [
    {
      "cycle": 1,
      "signal_shape": "passive_only",
      "signals": [
        {
          "signal_kind": "passive",
          "source_type": "external_agent_projection",
          "raw_content": "...",
          "gold_evidence_event": "...",
          "gold_update_effect": {"H1": "weak_weaken"}
        }
      ]
    }
  ],
  "final_gold": "H2"
}
```

### 5.2 Active External Signal 构造

- 检索结果。
- 工具返回。
- 文档片段。
- 模拟实验结果。
- skill output。

### 5.3 Passive External Signal 构造

- 人类专家反馈模板。
- 其他 agent 的 Belief State Projection。
- 系统日志片段。
- benchmark evidence stream。
- 用户纠错。

### 5.4 Projection Decomposition 数据

构造包含结论和引用来源的 projection：

```text
Agent X believes H2 because source A says ...
```

评估系统是否拆分为：

- sender belief signal。
- claimed source signal。
- direct verification probe candidate。

## 6. 具体 Benchmark 设计

### 6.1 FEVER-Signal

- H_supported。
- H_refuted。
- H_NEI。
- active-only：主动检索 evidence。
- passive-only：直接注入 claim-related passages / external projection。
- 指标：
  - final label。
  - evidence event target/type。
  - update direction。
  - neutral drift。

### 6.2 SciFact-Signal

- H_supports。
- H_refutes。
- H_insufficient。
- H_needs_narrower_scope。
- 重点测试 scope narrowing 和 reframe。

### 6.3 PubMedQA-Signal

- H_yes。
- H_no。
- H_maybe。
- 逐步输入 background/method/result/conclusion signals。
- 测试 maybe humility 和 calibration。

### 6.4 QASPER-Signal

- 按 paper sections 输入 signal。
- 测试 supporting evidence traceability。
- 可用于 active-plus-passive smoke。

### 6.5 Multi-agent Projection Benchmark

- 多个 synthetic agents 输出 Belief State Projections。
- 包含：
  - correct projection。
  - overconfident projection。
  - projection with duplicated source。
  - projection with cited source requiring verification。
- 测试：
  - Projection-as-Signal Rule。
  - Projection Decomposition Rule。
  - authority bias resistance。

### 6.6 System Log Signal Benchmark

- 输入系统日志、错误片段、性能异常。
- 假设为不同 root cause。
- 测试 passive-only cycle 和 anomaly-triggered hypothesis evolution。

## 7. Noisy External Signal Benchmark

### 7.1 噪声类型

- irrelevant signal。
- repeated signal。
- low-reliability signal。
- misleading signal。
- partial evidence。
- counterevidence。
- scope-narrowing evidence。
- anomaly signal。
- authority-biased projection。
- duplicated-source projection。

### 7.2 期望行为

- irrelevant -> posterior stable。
- repeated -> independence 降权。
- low-reliability -> reliability 降权。
- misleading -> 不强更新。
- counterevidence -> 被反驳 H 明显下降。
- anomaly -> Hypothesis Evolution Trigger。
- authority-biased projection -> 不直接变 evidence。
- duplicated-source projection -> source tracing / 降权。

## 8. 指标体系

### 8.1 Answer Utility Metrics

- accuracy。
- EM/F1。
- label correctness。
- abstention correctness。
- cost-normalized accuracy。
- answer usefulness human rating。

### 8.2 Evidence Event Metrics

- target accuracy。
- type accuracy。
- relevance accuracy。
- reliability ranking。
- supporting fact F1。
- discard precision。

### 8.3 Belief Update Metrics

- update direction accuracy。
- strong update precision。
- counterevidence correction。
- neutral signal stability。
- posterior entropy reduction。
- Brier score。
- ECE。
- high-confidence wrong rate。

### 8.4 Signal Handling Metrics

- active signal conversion quality。
- passive signal conversion quality。
- projection-as-signal compliance。
- projection decomposition accuracy。
- duplicate evidence detection。
- authority bias resistance。

### 8.5 Probe Metrics

- Probe Candidate Pool quality。
- Probe Set expected value。
- entropy reduction per probe。
- cost per useful signal。
- change-my-mind candidate usefulness。

### 8.6 Hypothesis Evolution Metrics

- valid spawn/split/reframe rate。
- stale hypothesis persistence。
- ad hoc patch rate。
- anomaly response accuracy。
- retired hypothesis audit quality。

### 8.7 Synchronized Metrics

- projection completeness。
- change-my-mind condition quality。
- requested signal usefulness。
- cross-round belief improvement。
- passive-only cycle update quality。

## 9. 实验协议

### 9.1 通用协议

- 固定 base model 和 temperature。
- 固定样本顺序。
- 固定 signal stream order。
- 记录所有 token/tool/latency 成本。
- 记录 schema violation。
- 每个 run 保存完整 ledger。

### 9.2 Autonomous 实验协议

- 设置 max_cycles / budget / stop conditions。
- 主要覆盖 active-only。
- 可加入 active-plus-passive smoke。

### 9.3 Synchronized 实验协议

- 使用 fixed-round synchronization。
- 每轮输入 passive signals。
- 允许 empty Probe Set。
- 输出 Belief State Projection。
- 下一轮把其他 projection 作为 Passive External Signal。

### 9.4 人评协议

- 抽样审核：
  - hypothesis coverage。
  - evidence event quality。
  - projection quality。
  - change-my-mind usefulness。
  - hypothesis evolution validity。

### 9.5 显著性检验

- bootstrap confidence interval。
- paired significance test。
- per-sample delta analysis。

## 10. 消融实验

### 10.1 Core 消融

- BP-no-rivals。
- BP-no-falsifier。
- BP-no-quality。
- BP-no-LR。
- BP-no-evolution。
- BP-qualitative-only。

### 10.2 Signal 消融

- BP-active-only。
- BP-no-passive。
- BP-no-inbox-boundary。
- BP-passive-direct-evidence：错误对照，把 passive signal 直接当 evidence。
- BP-no-projection-decomposition。

### 10.3 Controller 消融

- BP-autonomous-only。
- BP-synchronized-without-fixed-boundary。
- BP-controller-updates-belief：错误对照，测试 controller 绕过 core 的风险。

### 10.4 Change-My-Mind 消融

- BP-no-CMMC。
- BP-CMMC-text-only。
- BP-CMMC-no-candidate-pool。

## 11. 实验记录模板

### 11.1 Run Card

- run_id。
- date。
- model。
- system_variant。
- regime：autonomous | synchronized。
- dataset。
- split。
- sample_count。
- temperature。
- max_tokens。
- max_cycles。
- max_tool_calls。
- retrieval_corpus。
- synchronization_mode。
- notes。

### 11.2 Cycle Record

- cycle_id。
- round_id。
- signal_shape。
- probe_set。
- passive_signals。
- active_signals。
- boundary_closed_at。
- evidence_events。
- belief_updates。
- hypothesis_evolutions。
- projection。

### 11.3 Sample Record

- sample_id。
- question_or_claim。
- gold_label。
- hypotheses。
- signal_stream。
- final_answer。
- final_belief_state_summary。
- final_correct。
- evidence_f1。
- update_direction_accuracy。
- neutral_signal_drift。
- counterevidence_response。
- projection_decomposition_accuracy。
- token_cost。
- tool_calls。
- error_tags。

### 11.4 Error Taxonomy

- HYP_MISSING。
- RIVAL_WEAK。
- FALSIFIER_MISSING。
- PROBE_LOW_VALUE。
- SIGNAL_MISCLASSIFIED。
- PASSIVE_SIGNAL_BYPASS。
- PROJECTION_AS_EVIDENCE。
- PROJECTION_NOT_DECOMPOSED。
- QUALITY_OVERTRUST。
- DUPLICATE_EVIDENCE。
- LR_WRONG_DIRECTION。
- OVERCONFIDENT_WRONG。
- NO_UPDATE_ON_COUNTEREVIDENCE。
- NO_HYPOTHESIS_EVOLUTION。
- AD_HOC_PATCH。
- BAD_CHANGE_MY_MIND_CONDITION。

## 12. 结论判定标准

### 弱证明

- active-only benchmark 上 update direction accuracy 高于 ReWOO/ReAct。
- final accuracy 不低于强基线。
- passive-only smoke 能正确通过 Evidence Integration Gate。

### 中证明

- 多个 signal benchmark 上 calibration、counterevidence correction、neutral stability、evidence quality 均优于基线。
- passive-only 与 active-only 都有稳定结果。
- 消融显示 LR、quality weighting、rivals、passive handling、projection decomposition 有贡献。

### 强证明

- 在 synchronized multi-agent / human-feedback 场景中，BayesProbe 的 Belief State Projection 能显著提升跨轮信念修正质量。
- 在开放式复杂任务中，人评显示 BayesProbe 在假设覆盖、反证吸收、不确定性表达、change-my-mind quality、决策帮助度上优于 ReAct/ReWOO。
- cost-normalized utility 可接受。

## 13. 第一阶段实验实施顺序

1. 构造 active-only FEVER/PubMedQA signal stream。
2. 构造 passive-only benchmark-provided signal stream。
3. 构造 external-agent projection passive-only 小集。
4. 跑 BayesProbe-Autonomous active-only。
5. 跑 BayesProbe-Synchronized passive-only。
6. 跑 active-plus-passive smoke。
7. 做核心消融。
8. 做 projection 消融。
9. 写第一轮 error analysis。

## 附录

- Benchmark JSON schema draft。
- Scoring scripts draft。
- Human evaluation rubric。
- 与 v0.1 指标对照表。
