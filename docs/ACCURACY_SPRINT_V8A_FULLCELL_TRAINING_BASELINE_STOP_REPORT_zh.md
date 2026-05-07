# v8A full-cell training baseline stop report

Generated: 2026-05-07

## 一句话结论

这轮没有通过训练升级门。

当前 clean full-cell 数据已经通过最终数据审查，并且可以跑 development-only 训练诊断；主模型在 ordinary view 和严格 count-balanced 小子集上都能把 H/M 分开到 `1.0`。但是这还不能作为“模型学到了晶体差异”的证据，因为：

- ordinary view 的 total-count-only control 太强，H/M min recall 达到 `0.7292`，超过 `<0.60` ceiling；
- count-balanced view 的严格匹配支持严重不足，只有 train `24` 对、validation `18` 对、stress-holdout `12` 对，远低于 `300/200/200` 准入线；
- count-balanced 小子集虽然主模型也能到 `1.0`，但 shuffled-label null p95 达到 `0.7222`，样本太少导致 null 不稳，不能用作主证据。

所以正确决策是 `stop_or_rework_fullcell_clean_development_baseline`。不能进入 stability replication、advanced model probe、开发矩阵大跑、shadow/final 或产品准确率叙事。

## 本轮实际做了什么

新增最终数据审查脚本：

- `analysis/audit_v8a_fullcell_training_data_final.py`

它专门审查 admitted full-cell view：

- 输入：`results/accuracy_v3/v8a_fullcell_residualization_protocol_rework_v1_source_scaled_no_residualization_event_to_feature/`
- admission：`results/accuracy_v3/v8a_fullcell_residualization_protocol_rework_v1_source_scaled_no_residualization_admission/`
- 输出：`results/accuracy_v3/v8a_fullcell_training_data_final_audit/`

最终数据审查通过：

- decision `training_unlocked_with_count_controls_required`
- `gate_passed=true`
- `training_unlocked=true`
- `count_controls_required=true`
- max split-level standardized count gap `1.5366`
- top main-feature absolute correlation with total count `0.6634`

新增 full-cell baseline 训练脚本：

- `analysis/train_v8a_fullcell_clean_development_model.py`

它同时跑两条线：

- ordinary admitted view；
- count-balanced matched view，默认 `fixed_bin_width_0p003`，fallback 检查 `fixed_bin_width_0p005`。

输出位置：

- `results/accuracy_v3/v8a_fullcell_clean_development_model/`

## 训练门结果

整体 gate：

- decision `stop_or_rework_fullcell_clean_development_baseline`
- `gate_passed=false`
- ordinary gate `false`
- count-balanced gate `false`
- count-balance strategy `fixed_bin_width_0p003`
- fallback `fixed_bin_width_0p005` 也没有达到支持线

ordinary view：

- selected main model `LogisticEventMain`
- validation H/M min recall `1.0000`
- stress-holdout H/M min recall `1.0000`
- worst thickness / pose / seed-block / count-bin recall 都是 `1.0000`
- validation ECE `0.0006`
- stress-holdout ECE `0.0006`
- total-count-only H/M min recall `0.7292`
- shuffled-label null p95 `0.5804`
- stop reasons：
  - `total_count_only_below_ceiling`
  - `shuffled_label_null_p95_below_ceiling`

count-balanced view：

- selected main model `LogisticEventMain`
- validation H/M min recall `1.0000`
- stress-holdout H/M min recall `1.0000`
- total-count-only H/M min recall `0.4444`
- lineage-only H/M min recall `0.5000`
- shuffled-label null p95 `0.7222`
- shuffled-label single-seed max `0.7778`
- real minus null margin `0.2778`
- stop reasons：
  - `shuffled_label_null_p95_below_ceiling`
  - `shuffled_label_single_seed_below_ceiling`
  - `real_minus_null_margin`

## 为什么这次不能继续往后跑

大白话说：现在不是“模型分不开”，而是“模型太容易分开”。

ordinary view 里，主模型确实 100% 分开 H/M，但 total-count-only control 也能分得很高。这说明当前数据里 H/M 之间仍有一个很强的总计数差异。这个差异可能是真实物理效应的一部分，也可能是采样/源强/厚度响应带来的捷径；在审稿标准下，不能把它直接写成晶体衍射差异。

严格 count-balanced view 把总计数压下去了，total-count-only control 也低了；但剩下的样本太少。validation 只有 `18` 对，stress-holdout 只有 `12` 对，这种规模下 shuffled-label null 会非常不稳定，所以它不能承担主证据。

最关键的诊断是 paired H/M 自身的 total count 差异：当前已 admitted 的 clean pairs 平均差约 `0.013`，而我们要求 `0.003` 或 `0.005` 的严格匹配时，绝大多数 pair 都进不来：

- train：`432` 对里只有 `24` 对能进 `0.003` bin；
- validation：`288` 对里只有 `18` 对能进 `0.003` bin；
- stress-holdout：`288` 对里只有 `12` 对能进 `0.003` bin。

这说明问题不在训练脚本，而在下一轮 clean matrix 采样目标：它没有从源头把 H/M 的 total-count support 做到足够可匹配。

## 下一阶段长期计划

下一步不继续调模型，也不启动 advanced model。

要先做 `v8A_fullcell_count_matched_matrix_prereg`：

1. 从源头重新设计 H/M-only development matrix。
2. 每个最细 nuisance cell 仍保持 full-cell null support：至少两对 H/M。
3. 新增 count-target matching 约束，让 H/M pair 的 `control_total_count_norm` 预期差更小。
4. 预期目标不是事后 `fixed_bin_width_0p003` 勉强抽样，而是在 matrix 设计阶段就保证：
   - train `>=300` count-matched pairs；
   - validation `>=200` count-matched pairs；
   - stress-holdout `>=200` count-matched pairs。
5. 如果模拟后 count gap 仍然过大，优先调采样/源强/归一化策略，不进入训练。
6. 只有新矩阵同时满足 clean admission 和 count-balanced support，才重新跑 baseline training gate。

训练恢复顺序保持不变：

1. final data audit；
2. ordinary baseline；
3. strict count-balanced baseline；
4. stability replication；
5. advanced model feature-sufficiency probe；
6. H/M-only development large matrix prereg。

任一阶段失败都停，不进入下一层。

## 仍然不能说什么

不能说：

- H/M development training 已通过；
- 当前数据能支持 crystal-difference model claim；
- advanced model 可以开始验证上限；
- shadow/final 可以打开；
- full ten-material v8A matrix 可以开跑；
- 产品准确率、硬件验证、manuscript-grade powder XRD claim 成立。

可以说：

- clean full-cell data admission 已通过；
- ordinary main signal 极强；
- 但 total-count control 和 count-balanced support 仍阻止训练证据升级；
- 下一阶段必须重做 count-matched clean matrix prereg，而不是继续在当前数据上调模型。

## Key artifacts

- Final data audit script: `analysis/audit_v8a_fullcell_training_data_final.py`
- Full-cell baseline script: `analysis/train_v8a_fullcell_clean_development_model.py`
- Final data audit gate: `results/accuracy_v3/v8a_fullcell_training_data_final_audit/v8a_fullcell_training_data_final_audit_gate.json`
- Training gate: `results/accuracy_v3/v8a_fullcell_clean_development_model/v8a_fullcell_clean_development_model_gate.json`
- Count-balance support: `results/accuracy_v3/v8a_fullcell_clean_development_model/v8a_fullcell_count_balance_support.csv`
