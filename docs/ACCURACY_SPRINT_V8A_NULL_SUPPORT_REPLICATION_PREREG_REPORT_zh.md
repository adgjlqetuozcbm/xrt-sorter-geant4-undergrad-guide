# v8A Null-Support Replication Prereg Report

## 一句话结论

这一阶段不是为了把真实数据打乱，也不是为了训练模型，而是为了增加 paired-clean null 审计的统计支撑量。真实 H/M 数据仍然保持物理晶体差异、source-on/default、单 origin/source family、厚度/姿态/count-bin/seed-block 成对平衡；只有假标签审计时才打乱材料标签对应关系。

## 为什么要做这一轮

上一轮 clean H/M development run 已经比旧数据干净很多：

- source-on/default only；
- H/M 在 strict nuisance cell 内成对；
- non-material shortcut gate 通过，max balanced accuracy `0.3646`；
- paired-clean null 的 effective shuffle fraction 是 `0.50-0.50`；
- train seed-block orientation max abs sum 是 `0.0`。

但是 paired-clean null 仍然没有通过：

- primary/all-mode fixed/selected p95：`0.5972`；
- ceiling：`0.55`；
- single-seed max：`0.6250`，没有超过 `0.65`。

这说明当前问题已经不是明显“没打乱干净”，而是 null 分布尾部仍偏高。最合理的下一步不是继续换模型或把数据洗坏，而是扩大 validation/stress-holdout 的 matched-pair 支撑量，判断这个尾部偏高是小样本/支撑不足导致，还是仍存在更深的结构性 shortcut。

## 新矩阵设计

新增配置：

- `analysis/configs/v8a_clean_hm_null_support_replication_matrix_config.json`

新增 profile：

- `source_models/config/material_sorting_matrix/v8a_hm_clean_nullrep_cif_lit/`

核心约束：

- 只含 `Hematite` 和 `Magnetite`；
- 只含 `source_mode=on` 和 `stress_label=default`；
- 不含 source-off；
- 不含 stress variants；
- 不含 shadow/final；
- 不含 full ten-material matrix；
- 不读 existing XRT cube；
- 不训练模型；
- 保留 H/M 真实晶体峰差异；
- 打乱只发生在 null audit，不写入真实训练候选 view。

strict nuisance cell 仍然是：

`split × clean_matrix_origin × source_family × thickness_mm × pose_index × count_target_bin × seed_block`

## 支撑量变化

上一轮 clean matrix：

- train strict pairs：`108`
- validation strict pairs：`72`
- stress-holdout strict pairs：`72`
- total rows：`504`

这一轮 null-support replication：

- train strict pairs：`216`
- validation strict pairs：`144`
- stress-holdout strict pairs：`144`
- total rows：`1008`

这不是开发矩阵大跑。它只是把 clean null 审计的支撑量翻倍，目的是让 p95 判断更稳。

工程约束：profile 与 seed-block 名称刻意保持较短，避免 phase-space/config 路径在 Windows/WSL/UNC 访问下触发文件名过长问题。这不改变采样设计，只降低生成 artifacts 时的路径风险。

## 准入顺序

本阶段只允许到 preflight：

1. 生成 matrix/config/phase-space artifacts。
2. 跑 clean matrix preflight。
3. 如果 preflight 不过，停止并修设计。
4. 如果 preflight 过，只解锁 development-only Geant4 replication run。
5. Geant4 replication run 完成后，必须重新跑 event-to-feature、crystal-clean view、shortcut gate、paired-clean null gate、admission gate。
6. 只有 admission 写出 `training_unlocked=true`，才允许恢复 development-only baseline training。

## Claim Boundary

本阶段可以说：

- 我们设计了一个更大支撑量的 clean null replication matrix；
- 它用于检验 paired-clean null p95 是否受小样本尾部影响；
- 它保持真实 H/M 晶体差异不被洗掉。

本阶段不能说：

- 已经证明 H/M 可以可靠分选；
- 已经可以训练模型；
- ordinary XRT solves H/M；
- 产品准确率已经验证；
- 硬件验证已经完成；
- shadow/final 或 full ten-material matrix 已经通过；
- powder XRD simulation 达到可投稿级别。

## 大白话

我们现在不是在把数据搅成一锅粥。真实数据要保持“赤铁矿就是赤铁矿、磁铁矿就是磁铁矿”，峰位、峰强、厚度、姿态、seed-block 都要清清楚楚。

我们要打乱的是“假标签考试”里的答案钥匙。也就是说，模型如果面对假标签还能考得很高，就说明它可能在偷看页码、批次、seed、count-bin 之类的旁门；如果假标签考不高，而真标签考得高，才说明它更可能真的学到了晶体差异。

上一轮假标签已经打乱得比较干净，但样本量还小，尾部 p95 还是偏高。所以下一步是把同样干净的出题方式复制得更多一点，让审计更稳，而不是为了过关去把真实物理信号洗掉。
