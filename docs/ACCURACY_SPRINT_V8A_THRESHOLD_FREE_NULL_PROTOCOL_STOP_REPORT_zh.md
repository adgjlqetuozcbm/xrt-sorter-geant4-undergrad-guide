# v8A Threshold-Free Null Protocol Stop Report

## 一句话结论

Phase 1 已完成：当前 paired-clean null 失败不是单纯的 threshold selection artifact。即使不用 fixed/selected recall 阈值，而改看 threshold-free 的 AUC / rank separation，null 仍然明显偏离 chance。原始 clean view 的 primary oriented AUC p95 是 `0.6236`，best tail rebuild view (`leave_out_peak_hematite`) 甚至升到 `0.6599`，都高于 `0.58` ceiling。因此训练继续锁住，下一步应进入 residualization / pairing protocol audit 和 physics-preserving representation v2。

## 本轮做了什么

新增脚本：

- `analysis/audit_v8a_threshold_free_null_protocol.py`

它重放 paired-clean null pseudo-label protocol，但不只看 recall threshold，而是输出：

- oriented AUC；
- rank separation / rank overlap；
- signed score gap；
- pooled score gap；
- fixed vs selected threshold inflation；
- by model / split / shuffle mode 的 null summary。

这一步仍然是 development-only，只读现有 clean outputs，不跑 Geant4，不训练真实模型，不读 existing XRT cube，不碰 shadow/final。

## 审计对象

审计了两个关键视图：

1. 当前 clean null-support view：
   - `results/accuracy_v3/v8a_clean_hm_nullrep_crystal_clean_design_cell_event_to_feature`

2. tail rebuild v1 里最接近的候选：
   - `results/accuracy_v3/v8a_tail_rebuild_v1_leave_out_peak_hematite`

## 结果 1：当前 clean view

输出目录：

- `results/accuracy_v3/v8a_clean_hm_nullrep_crystal_clean_design_cell_threshold_free_null/`

gate 结果：

- decision：`threshold_free_null_direction_found`
- threshold-free gate passed：`false`
- primary oriented AUC p95：`0.6236`
- all-mode oriented AUC p95：`0.6236`
- primary oriented AUC max：`0.6697`
- primary rank overlap p05：`0.7527`
- primary positive threshold inflation p95：`0.0000`
- primary fixed recall p95：`0.5569`
- primary selected recall p95：`0.5569`

stop reasons：

- `primary_threshold_free_auc_p95_exceeded_ceiling`
- `all_modes_threshold_free_auc_p95_exceeded_ceiling`
- `primary_rank_overlap_below_minimum`
- `all_modes_rank_overlap_below_minimum`

解释：这里 selected threshold 没有放大 tail，因为 threshold inflation p95 是 `0.0000`。问题在 threshold-free AUC 本身已经偏高。

## 结果 2：leave_out_peak_hematite

输出目录：

- `results/accuracy_v3/v8a_tail_rebuild_v1_leave_out_peak_hematite_threshold_free_null/`

gate 结果：

- decision：`threshold_free_null_direction_found`
- threshold-free gate passed：`false`
- primary oriented AUC p95：`0.6599`
- all-mode oriented AUC p95：`0.6599`
- primary oriented AUC max：`0.6919`
- primary rank overlap p05：`0.6802`
- primary positive threshold inflation p95：`0.0559`
- primary fixed recall p95：`0.5767`
- primary selected recall p95：`0.5767`

stop reasons：

- `primary_threshold_free_auc_p95_exceeded_ceiling`
- `all_modes_threshold_free_auc_p95_exceeded_ceiling`
- `primary_threshold_free_auc_single_seed_max_exceeded_ceiling`
- `all_modes_threshold_free_auc_single_seed_max_exceeded_ceiling`
- `primary_rank_overlap_below_minimum`
- `all_modes_rank_overlap_below_minimum`
- `primary_threshold_selection_inflation_detected`
- `all_modes_threshold_selection_inflation_detected`

解释：这个候选虽然去掉了 hematite individual peaks，但 threshold-free AUC 更高，说明简单删 peak family 不是修复，而可能让剩余表示形成更强的整体方向。

## 方法学判断

这一步把两种可能区分开了：

- 如果 AUC 很接近 `0.5`，但 recall p95 仍高，说明主要是 threshold protocol artifact；
- 现在 AUC p95 本身超过 `0.58`，说明 representation 中确实存在 threshold-free null direction。

因此不能通过以下方式放行：

- 降低或调整 recall threshold；
- 只改 selected threshold protocol；
- 直接训练；
- 用高级模型试图证明信号；
- 用 `leave_out_peak_hematite` 当“修好了”的候选。

## 下一步

进入 Phase 2：

`v8A_residualization_and_pairing_protocol_audit`

重点检查：

- residualization 前后是否把 nuisance/材料差异压成少数 global score direction；
- train-fit / validation-apply scaling 是否制造 split-specific score variance；
- paired orientation 在 thickness/pose/count-bin 的高阶交互里是否仍有不平衡；
- Logistic score direction 是否稳定对应某些 residualized peak-family 组合；
- `leave_out_peak_hematite` 的恶化是否来自剩余特征 score variance 增大。

只有 Phase 2 解释清楚，才能进入 physics-preserving representation v2。仍然不训练。

## 大白话

这次我们问的是：“是不是假标签失败只是因为阈值选法不公平？”答案是否定的。

如果只是阈值问题，那么不看阈值、只看排序能力时，假标签模型应该接近乱猜。但现在 AUC 仍然偏高，说明模型在假标签下确实能从当前表示里抓到一点排序方向。删掉 hematite peaks 后还更糟，说明简单删特征不是药。

所以现在不能开训。下一步要查 residualization 和配对打乱协议是不是把一些差异压成了隐蔽方向，然后再设计更物理、更受约束的 H/M peak-pair 或 q-window contrast。
