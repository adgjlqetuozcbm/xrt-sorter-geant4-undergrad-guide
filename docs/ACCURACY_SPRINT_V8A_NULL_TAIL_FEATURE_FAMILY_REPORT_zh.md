# v8A Null-Tail Feature-Family Anatomy Report

## 一句话结论

这一步确认了：我们不是在寻找一种“把真实数据也打乱”的办法，而是在寻找一种让**假标签审计足够有效**、同时让真实 H/M 晶体差异仍然保留的干净表示。新的 tail feature-family anatomy 显示，null-tail 不是单个 seed、split、mode 崩掉，也不是某一个单独特征列作怪；它更像是 Logistic 在 individual peak features 上抓到一条弱的线性方向，其中 `peak_hematite` family 占比过高。

因此下一步可以进入 **feature-family rebuild prereg**，但不能粗暴删光峰特征，也不能为了过 null gate 把真实晶体差异洗没。

## 本轮新增审计

新增脚本：

- `analysis/audit_v8a_paired_null_tail_feature_family.py`

输入仍然只来自 development-only clean null-support replication：

- feature view：`results/accuracy_v3/v8a_clean_hm_nullrep_crystal_clean_design_cell_event_to_feature/`
- paired-clean null rows：`results/accuracy_v3/v8a_clean_hm_nullrep_crystal_clean_design_cell_paired_null/`
- tail anatomy：`results/accuracy_v3/v8a_clean_hm_nullrep_crystal_clean_design_cell_null_tail_anatomy/`

输出：

- `results/accuracy_v3/v8a_clean_hm_nullrep_crystal_clean_design_cell_null_tail_feature_family/v8a_paired_null_tail_feature_family_gate.json`
- `results/accuracy_v3/v8a_clean_hm_nullrep_crystal_clean_design_cell_null_tail_feature_family/v8a_paired_null_tail_feature_family_report.md`
- family / feature / row-level contribution CSVs

这一步不训练真实模型，不解锁 admission，不触碰 shadow/final，不读 existing XRT cube。

## 关键结果

gate 结论：

- decision：`feature_family_rebuild_prereg_needed`
- gate passed：`false`
- tail rows probed：`44`
- top family abs weight share：`0.4823`
- top feature abs weight share：`0.1151`
- top feature direction consistency：`0.6875`
- stop reasons：
  - `paired_clean_null_tail_still_above_ceiling`
  - `null_tail_single_feature_family_dominates`

最重要的 family 结果：

- Logistic / `peak_hematite`：family share `0.4823`
- Logistic / `peak_magnetite`：family share `0.3856`
- Logistic / `ratio_balance`：family share `0.0541`
- Logistic window families 均远低于 peak families

最重要的单特征结果：

- top single feature：`diffraction_crystal_clean_peak_hematite_24p1_norm`
- top single feature share：`0.1151`

这说明问题不是“一个坏特征列删除就好”，而是 individual peak family 的自由度/线性组合在 null-tail 中仍然过强。

## 如何解释

目前证据支持下面这个判断：

1. 源头 clean sampling 路线仍然是对的。visible shortcut gate 继续通过，max non-material balanced accuracy 只有 `0.4080`。
2. 支持量扩大有效，但没有完全解决问题。paired-clean null p95 已从上一轮 `0.5972` 降到 `0.5569/0.5628`，接近但仍未低于 `0.55`。
3. tail 不是 seed/split/mode 单点塌缩。tail rows 是 `44/960`，validation 和 stress-holdout 各 `22`，top seed share 只有 `0.0909`。
4. tail 更偏向 Logistic，所以重点不是树模型过拟合，而是低维线性方向还可能抓到假标签尾部。
5. top family 是 `peak_hematite`，但 top single feature 不支配，所以不能简单删一个峰。

## 不允许做什么

这份报告不允许：

- 开始真实 H/M training；
- 跑高级模型；
- 降低 null threshold；
- 宣称当前 clean data 已经可用于训练；
- 直接大开发矩阵；
- shadow/final；
- full ten-material matrix；
- 产品准确率、硬件验证、ordinary XRT solves H/M、manuscript-grade powder XRD claim。

## 下一步 prereg 方向

下一阶段应该新建 `v8A_tail_rebuild_v1_prereg`，只做 feature representation rebuild，不新增 Geant4 大跑。

候选 view 应该至少包括：

- `peak_family_balanced`：把 16 个 individual peak features 压缩成更少的 family-level aggregate / contrast features，保留 H/M 物理差异但减少自由度。
- `window_ratio_only`：只保留 unique window、all-peak window 和 ratio/balance features，作为低自由度 sanity view。
- `leave_out_peak_hematite`：诊断 `peak_hematite` family 是否确实驱动 null-tail；不能直接当最终 view。
- `leave_out_individual_peaks`：去掉 individual peak features，只保留 window/ratio aggregates，作为最保守下界。

准入顺序必须是：

1. 先跑 paired-clean null gate；
2. 再跑 null-tail feature-family gate；
3. 再跑 shortcut/admission；
4. 只有 admission 写出 `training_unlocked=true`，才允许 development-only baseline training；
5. 如果 reduced view 过了 null 但把真实 H/M signal 洗没，则不能升级，必须回到物理特征设计。

## 大白话

你刚才那句话基本对，但要加一个边界：我们不是要把真实数据弄得越乱越好，而是要让“假标签考试”足够乱。真实数据里该保留的晶体差异必须保留，否则后面就算过了 null gate，也没有分选价值。

这次查到的更具体问题是：模型在假标签里偶尔能靠一组 hematite peak 相关特征拼出一条弱线性方向。它不是一个坏 seed，也不是一个坏 split，也不是某个单独列坏了。所以下一步不是继续盲目扩样，也不是把所有峰都删掉，而是做一个预注册的低自由度 feature rebuild：把 individual peaks 压缩成更稳的 family/窗口/ratio 表示，然后重新过 null 和 admission。
