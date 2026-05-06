# v8A Clean H/M Matrix Prereg Report

## 一句话结论

旧的 v2/clean views 只能继续当诊断对照，不能作为恢复训练的主证据。下一轮必须从源头重新采样：第一版 clean matrix 只保留 `source-on + default`，并让 Hematite/Magnetite 在每个 nuisance cell 内天然成对。

本阶段已经完成 clean matrix prereg、生成了 development-only matrix/phase-space/config artifacts，并通过 preflight；没有运行 Geant4，没有训练模型，也没有触碰 shadow/final。

## 为什么旧路线停止

之前的错误不是“模型不够高级”，而是数据视图里存在模型可能学习的旁门结构：

- medium 与 extension 属于不同 origin，后拼成训练视图后，材料、stress/source family、origin 结构可能纠缠。
- default、stress、source-off 混在同一个候选学习视图里，导致主特征能预测非材料 stress/source 结构。
- 事后 count matching 的支持不足，严格匹配时 train/validation/stress-holdout 对数太少。
- crystal-clean view 虽然压低了可见 shortcut，但 shuffled-label/null 仍然偏高，说明隐藏结构没有被完全清掉。
- near-identity shuffle 曾经暴露出 null protocol 本身也需要更严审计。

因此，旧数据只允许用于诊断和对照，不允许用于恢复训练、Phase 4、开发矩阵大跑或任何产品/硬件/论文级 claim。

## 新 clean matrix 设计

源头设计文件：

- `analysis/configs/v8a_clean_hm_development_matrix_config.json`
- `analysis/generate_v8a_clean_hm_development_matrix.py`
- `analysis/audit_v8a_clean_hm_matrix_preflight.py`

生成 profile：

- `source_models/config/material_sorting_matrix/v8a_hm_clean_default_development_cif_literature/`

核心约束：

- 只含 `Hematite` 与 `Magnetite`。
- 只含 `source_mode=on` 与 `stress_label=default`。
- 第一轮只含单一 `clean_matrix_origin` 与单一 `source_family`。
- 不生成 shadow/final，不生成 full ten-material matrix。
- 不读取 existing XRT cube。
- 不运行 Geant4。
- 不解锁训练。

H/M 成对平衡的 strict nuisance cell 是：

`split × clean_matrix_origin × source_family × thickness_mm × pose_index × count_target_bin × seed_block`

目标和实际 strict pair support：

- train: target `>=100`，actual `108`
- validation: target `>=50`，actual `72`
- stress_holdout: target `>=50`，actual `72`

## Preflight 结果

preflight gate：

- `results/accuracy_v3/v8a_clean_hm_development_matrix_preflight/v8a_clean_hm_matrix_preflight_gate.json`

结论：

- `gate_passed=true`
- `decision=clean_matrix_preflight_passed_ready_for_development_run_prereg_only`
- `development_run_prereg_unlocked=true`
- `training_unlocked=false`

这只表示可以进入下一步 development-only Geant4 clean run 的准备阶段。它不是模型训练许可。

## 下一阶段准入顺序

下一步只能按这个顺序走：

1. 运行 clean H/M development-only Geant4 matrix。
2. event-to-feature 只输出 sidecar diffraction features 与 lineage audit fields。
3. 跑 schema/integrity gate。
4. 跑 non-material shortcut gate，确认主特征不能预测 source/origin/thickness/pose/count-bin/seed-block。
5. 跑 row-level、pair-aware、seed-block-aware null gates。
6. 只有 clean admission 写出 `training_unlocked=true`，才允许 Logistic/ExtraTrees baseline。
7. 高级模型仍只能作为 feature sufficiency probe，不能绕过 null/shortcut gate。

## Claim Boundary

本阶段可以说：

- clean H/M development matrix preregistration passed preflight。
- source-first sampling design solved the previous strict-pair support blocker at the matrix-design level。

本阶段不能说：

- 模型已经能可靠分选 H/M。
- ordinary XRT solves H/M。
- 产品准确率已经验证。
- 硬件验证已经完成。
- shadow/final 或 full ten-material matrix 已经通过。
- powder XRD simulation 已达到可投稿级别。

## 大白话总结

之前的问题像是题库里偷偷带了页码、水印、装订方式，模型可能不是在学赤铁矿和磁铁矿的晶体差异，而是在学“这行数据来自哪一批、哪种 stress、哪个 source/off 结构”。这次我们没有继续训练，也没有继续洗旧数据，而是重新设计了一套更干净的出题方式：每一个厚度、姿态、count bin、seed block 里都同时放一对赤铁矿和磁铁矿，训练/验证/holdout 的 seed block 也完全隔开。

现在的状态是：新试卷的排版规则通过检查了，可以准备让 Geant4 真正生成答案卷；但还不能开始训练。等模拟结果出来后，还要再证明模型学不到页码、水印、seed、姿态、厚度这些旁门，才允许重新检验分选成果。
