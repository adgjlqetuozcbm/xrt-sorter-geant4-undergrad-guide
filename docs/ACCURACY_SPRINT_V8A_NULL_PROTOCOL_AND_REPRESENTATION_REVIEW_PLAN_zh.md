# v8A Null-Protocol And Representation Review Plan

## 一句话结论

现在的问题不是“再找一种更花哨的打乱方式”，也不是“删掉某个峰就能训练”。问题已经收敛到更深的一层：当前 clean sampling 已经把明显的 source/origin/stress/count 捷径压下去了，但 paired-clean null 仍有尾巴，说明特征表示和 null protocol 之间还存在弱的可利用方向。下一阶段目标是把这条弱方向解释清楚、修干净，直到 admission 明确写出 `training_unlocked=true`，才允许数据进入模型训练。

## 问题为什么会出现

这不是一个单点错误，而是几个早期决策叠加后逐层暴露出来的结果。

第一层原因是旧数据生成视图过于复杂。早期把 medium、extension、default、stress、source-on/off 等来源拼到同一个学习视图里，材料、来源、应力路径、seed block、count structure 之间很容易发生纠缠。模型不需要真正理解 H/M 晶体差异，也可能靠这些旁路结构拿分。

第二层原因是 absolute count 和窗口强度太容易成为捷径。我们后面发现 total-count-only / overlap-only control 能抬高，说明原始主特征里有部分信息接近总量、强度或窗口大小的替代变量。count-robust 和 crystal-clean 处理解决了很多显性捷径，但也把问题推到更细的 feature-family 层面。

第三层原因是早期 shuffled-label 审计协议不够严格。普通 row shuffle、threshold selection、小样本 support 都可能让 null 结果看起来异常。后来 paired-clean null 把 orientation、seed-block、nuisance cell 都控制住，才把问题从“明显审计缺陷”推进到“真实残余尾巴”。

第四层原因是当前 diffraction sidecar 表示仍然有过多相关的 individual peak 自由度。20 个主特征里，许多 hematite/magnetite peaks 不是完全独立观测，而是同一物理过程在不同窗口里的相关投影。Logistic 在假标签下仍能偶尔沿着 `peak_hematite` / `peak_magnetite` family 拼出弱线性方向。tail rebuild v1 证明简单 mean/sum、删峰或只保留窗口比例都不足以解决。

第五层原因是我们一直在非常接近 gate 的区域工作。当前 p95 从 `0.5972` 降到 `0.5569/0.5628`，再到 best rebuild `0.5767`，说明问题不是巨大泄漏，而是尾部稳定性。越接近阈值，越不能靠主观判断放行；必须让 protocol、表示和 support 同时干净。

## 为什么审计了这么多次才找到这里

之前不是完全没找到原因，而是每一轮只清掉了上一层更明显的问题。

- total-count audit 找到的是总量捷径；
- root-cause audit 找到的是 stress/origin/source 结构；
- crystal-clean 找到的是旧数据无法靠事后清洗完全补救；
- paired-clean null 找到的是普通 shuffle 协议不足；
- null-support replication 找到的是 support 不足贡献了一部分尾巴；
- null-tail anatomy 找到尾巴不集中在单个 seed/split/mode；
- feature-family anatomy 找到 tail 偏向 Logistic 的 `peak_hematite` family；
- tail rebuild v1 证明简单低自由度压缩不能让数据进训练。

所以现在不是“原因完全没找到”，而是已经把大问题剥到更小、更难的一层：特征表示和 null protocol 的交互。

## 下一阶段目标

下一阶段命名为：

`v8A_null_protocol_and_representation_review`

目标只有一个：让 clean development 数据达到模型训练准入，而不是追求漂亮结果。

数据进入训练前必须同时满足：

- source/origin/stress/count/thickness/pose 等非材料 shortcut 仍低于 ceiling；
- paired-clean null primary 和 all-mode p95 `<= 0.55`；
- paired-clean null single-seed max `<= 0.65`；
- null-tail feature-family gate 不再显示单个 family 支配；
- threshold-free null 指标也干净；
- admission gate 写出 `training_unlocked=true`；
- 真实 H/M 晶体信号没有被清洗到不可用。

## Phase 1: Threshold-Free Null Protocol Review

目的：判断当前 null 失败是否被 recall threshold / selected threshold 放大。

要新增或扩展的指标：

- fixed threshold recall；
- selected threshold recall；
- AUROC / oriented AUC null distribution；
- signed score mean gap；
- score distribution overlap；
- calibration-free rank separation；
- by split / mode / seed block 的 null score tail。

通过标准：

- null AUC p95 接近 chance，建议 `<= 0.58`；
- fixed 与 selected threshold 不应系统性扩大 null tail；
- 如果 selected threshold 是主要原因，修 gate protocol，而不是修物理特征；
- 如果 threshold-free 指标也高，说明 representation 仍有可学 null direction。

止损：

- 如果 threshold-free null 很干净而 recall gate 不干净，优先修 gate protocol；
- 如果 threshold-free null 也不干净，进入 Phase 2/3，不允许训练。

## Phase 2: Residualization And Pairing Protocol Audit

目的：确认 train-only residualization 和 paired-clean pseudo-label orientation 没有制造新的不稳定方向。

必须检查：

- residualization 前后每个 feature family 的 real-label 与 null-label score direction；
- train fit / validation apply 是否引入 split-specific scale artifact；
- paired orientation 是否在 thickness/pose/count-bin 更高阶交互里仍有偏差；
- seed block 之间是否存在 score variance asymmetry；
- `leave_out_peak_hematite` 改善是否来自真实去 shortcut，还是只是降低 score 方差。

通过标准：

- residualized features 不能把 nuisance 结构压成少数全局 peak-family 方向；
- orientation diagnostic 不只看一阶 cell，也要看高阶交互；
- null-tail 不能由 residualization scale 或 split variance 解释。

止损：

- 如果 residualization 是根因，先设计 non-residualized / rank-normalized / within-pair contrast view；
- 如果 pairing protocol 是根因，先修 null protocol，不做 feature rebuild。

## Phase 3: Physics-Preserving Representation v2 Prereg

目的：不要再机械删峰，而是设计更符合物理的低维表示。

候选表示：

- `peak_pair_contrast_v2`：只使用 preregistered H/M 近邻峰对 contrast，不让模型自由组合一整组 peaks；
- `q_neighborhood_merged_v2`：把相近 q-window 合并，减少同源峰重复自由度；
- `shared_window_contrast_v2`：只比较 H/M 共享或近邻窗口中的相对形状；
- `rank_within_pair_v2`：在 matched nuisance pair 内做 rank/contrast，降低全局强度方向；
- `family_contrast_minimal_v2`：保留少数有物理解释的 family-level contrast，不使用 individual peaks。

硬约束：

- 不使用 material/source_id/sample_id/seed/thickness/pose/split/origin/path/row_index/count_target_bin；
- scaling/residualization 只能 train fit，再 frozen apply；
- 不新增 stress/source-off 混入训练候选；
- 不碰 shadow/final；
- 不把真实晶体差异洗没。

## Phase 4: Candidate Gate Loop

每个 v2 candidate 必须按同一顺序过门：

1. schema/integrity gate；
2. shortcut gate；
3. threshold-free null gate；
4. paired-clean null gate；
5. null-tail feature-family gate；
6. admission gate。

通过标准：

- max non-material BA `<= 0.75`，建议目标 `<= 0.55`；
- paired-clean primary/all-mode p95 `<= 0.55`；
- paired-clean single-seed max `<= 0.65`；
- null AUC p95 `<= 0.58`；
- top family abs contribution share `< 0.45`；
- top feature share `< 0.25`；
- no lineage-like main feature；
- admission `training_unlocked=true`。

止损：

- 如果连续 3 个 preregistered representation candidates 都失败，停止旧数据 feature washing；
- 如果最优 p95 仍 `> 0.60`，说明不是接近过线问题，回到物理采样/统计量；
- 如果 null 过了但 real H/M signal preservation 失败，不能训练，回到物理表示设计。

## Phase 5: Signal Preservation Check

只有 null/admission 接近过线或通过时，才做 signal preservation。

目标不是训练模型，而是确认清洗没有把晶体差异洗死。

允许指标：

- univariate H/M feature separation；
- train-only fitted simple score applied to validation/stress，但只作为 admission 后训练前 sanity；
- by thickness / pose / count-bin 的 signal consistency；
- real-label score direction 是否符合 peak provenance 预期。

通过标准：

- 至少一组物理可解释特征在 validation 和 stress-holdout 中方向一致；
- signal 不能只存在于一个 thickness、pose 或 count-bin；
- 如果 signal 只来自被审计判定危险的 family，需要重新设计。

## Phase 6: Training Admission

只有以下条件全部满足，数据才可以进模型训练：

- candidate view schema gate passed；
- shortcut gate passed；
- threshold-free null gate passed；
- paired-clean null gate passed；
- tail-family gate passed；
- admission gate writes `training_unlocked=true`；
- signal preservation check passed；
- no shadow/final；
- no full ten-material matrix；
- no existing XRT cube reads；
- no leakage-like main features；
- claim boundary written in report。

首轮训练只允许 development-only baseline：

- Logistic；
- ExtraTrees；
- calibration check；
- threshold sweep；
- by thickness / pose / seed-block recall；
- total-count/source-origin/shuffled-label controls。

不允许：

- CNN / Transformer 作为绕过 clean admission 的捷径；
- shadow/final；
- product accuracy；
- hardware validation；
- manuscript-grade powder XRD claim。

## Phase 7: 及时止损

如果 Phase 1-4 后仍无法得到 clean admission，有三个出口：

1. `null_protocol_artifact_found`：修 gate protocol，重跑审计；
2. `representation_insufficient`：当前 sidecar 表示不够，回到物理特征设计；
3. `new_clean_matrix_needed`：旧 clean run 的统计量不够，preregister 新 clean matrix，不在旧数据上继续硬洗。

明确禁止无限循环。每轮必须输出：

- decision；
- pass/fail gates；
- stop reasons；
- 下一步是否继续、回退、还是止损。

## 大白话总结

现在最关键的反思是：我们不能为了让假标签测试过关，把真实晶体差异也洗掉。我们也不能因为 p95 只差一点就放行。之前的问题像一层层洋葱：先是来源混杂，再是 count 捷径，再是 stress/origin，再是普通 shuffle 不严，现在剩下的是更隐蔽的“特征表示 + null protocol”尾巴。

下一步要做的不是开训，而是先问清楚：假标签尾巴到底是阈值规则造成的，还是 residualization / peak family 表示造成的。如果是规则问题，修规则；如果是表示问题，设计更物理的 peak-pair、q-neighborhood、shared-window contrast。只有这些 gate 全部干净，并且 admission 明确写出 `training_unlocked=true`，数据才能进模型训练。
