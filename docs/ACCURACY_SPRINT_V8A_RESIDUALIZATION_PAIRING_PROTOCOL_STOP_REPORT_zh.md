# v8A Residualization And Pairing Protocol Stop Report

## 一句话结论

Phase 2 已完成，并找到了比“删峰”更具体的根因方向：当前 crystal-clean residualization 会放大 Logistic 的 threshold-free null direction。pre-residual source-scaled view 的 max oriented AUC p95 是 `0.5996`，residualized clean view 升到 `0.6236`；其中 Logistic validation 的 delta 达到 `0.0617`，超过 `0.03` stop line。同时，高阶 paired orientation diagnostic 出现 max abs sum `3.0`，说明当前配对假标签协议在 thickness/pose/count-bin 高阶交互上仍有不平衡。

因此下一步不是进入训练，也不是直接做 physics representation v2，而是先修 residualization 与 paired orientation protocol。

## 本轮做了什么

新增脚本：

- `analysis/audit_v8a_residualization_pairing_protocol.py`

它比较两个视图：

1. pre-residual source-scaled view：
   - `results/accuracy_v3/v8a_clean_hm_nullrep_event_to_feature`

2. residualized clean view：
   - `results/accuracy_v3/v8a_clean_hm_nullrep_crystal_clean_design_cell_event_to_feature`

比较方式：

- 按 `sample_id` 对齐；
- 使用 manifest 中记录的 `residualization.features[*].source_column` 映射；
- source features 只做 train-fit z-score，再 apply 到 validation/stress；
- 重放 paired-clean null pseudo-label；
- 比较 oriented AUC p95、score variance ratio、高阶 orientation imbalance。

这一步仍然不训练真实模型，不跑 Geant4，不读 existing XRT cube，不碰 shadow/final。

## 关键结果

gate：

- decision：`residualization_artifact_suspected`
- gate passed：`false`
- max residualized oriented AUC p95：`0.6236`
- max source oriented AUC p95：`0.5996`
- max AUC delta residual-source：`0.0617`
- score std ratio max/min：`1.3459 / 1.1629`
- higher-order orientation max abs sum：`3.0`

stop reasons：

- `residualization_increases_null_auc_p95`
- `residualized_threshold_free_null_direction_remains_high`
- `higher_order_orientation_imbalance_detected`

## 最关键的对比

| mode | model | split | source AUC p95 | residualized AUC p95 | delta |
| --- | --- | --- | ---: | ---: | ---: |
| paired nuisance | Logistic | validation | 0.5579 | 0.6196 | +0.0617 |
| paired nuisance | Logistic | stress-holdout | 0.5708 | 0.6236 | +0.0528 |
| seed-block random | Logistic | validation | 0.5612 | 0.6175 | +0.0563 |
| seed-block random | Logistic | stress-holdout | 0.5677 | 0.6108 | +0.0431 |

ExtraTrees 的变化小得多，有些 split 还下降。这进一步支持：问题主要是 residualization 后形成了 Logistic 可利用的线性 null direction。

## 方法学解释

这轮结果说明三件事：

1. null direction 在 source features 里已经有一点，但 residualization 明显放大了 Logistic tail；
2. current train-fit residualization 可能把 nuisance 结构扣掉以后，留下了少数全局 peak-family 方向；
3. paired orientation 只做到一阶平衡还不够，高阶 thickness/pose/count-bin 交互仍可能给 null 模型留下结构。

这也解释了为什么 tail rebuild v1 没有成功：如果 residualization 与 orientation protocol 本身在放大 null direction，简单删峰或求 mean/sum 只能改变方向，不一定消除根因。

## 下一步

下一阶段应命名为：

`v8A_residualization_protocol_rework_v1`

必须先做三个候选协议，而不是直接训练：

1. `source_scaled_no_residualization`
   - 不做 nuisance residualization；
   - 只做 train-fit scaling；
   - 检查 shortcut gate 是否仍干净；
   - 如果 shortcut 干净且 null AUC 低于 residualized view，说明 residualization 是主要问题。

2. `within_pair_contrast`
   - 在 clean matched pair 内做 H/M-agnostic pair centering 或 rank contrast；
   - 不使用 material label 决定方向；
   - 目的是消除全局 intensity/scale 方向，而不是扣出新线性方向。

3. `higher_order_balanced_null`
   - 重新设计 paired orientation；
   - 必须在 seed_block × thickness × pose × count_bin 或更高阶 cell 内严格 0-sum；
   - 如果 cell support 不足，则不能用该 null mode 放行。

每个候选都必须重新跑：

- shortcut gate；
- threshold-free null gate；
- paired-clean null gate；
- tail-family gate；
- admission gate。

## 训练状态

训练继续锁住：

- `training_unlocked=false`
- 不允许 Logistic / ExtraTrees real-label training
- 不允许高级模型
- 不允许 shadow/final
- 不允许 full ten-material matrix
- 不允许产品准确率、硬件验证、manuscript-grade powder XRD claim

## 大白话

这次终于抓到一个更像“真正原因”的东西：我们之前为了清掉 thickness、pose、count 这些干扰，做了 residualization。它本意是清洗，但现在看，它可能把数据压成了一条更容易被 Logistic 抓住的线性方向。也就是说，我们在清洗旁门的时候，可能又造出了一个更隐蔽的旁门。

另外，假标签翻转虽然在 seed-block 层面平衡了，但在 thickness/pose/count-bin 的高阶组合里还不够平衡。这会让假标签审计里残留一点结构。

所以下一步不要训练，也不要继续盲目删峰。要先修两个底层协议：一个是 residualization 怎么做，一个是假标签 paired orientation 怎么做。只有这两个干净了，数据才有资格重新进 admission。
