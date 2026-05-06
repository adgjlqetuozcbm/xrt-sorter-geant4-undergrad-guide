# v8A Clean H/M Development Run Stop Report

## 一句话结论

clean H/M development-only Geant4 matrix 已经跑完，数据生成层面是干净完整的：`504/504` 行成功，H/M 平衡，厚度平衡，source-on/default only。

但是不能训练。原因不是显性的 source/origin/stress shortcut，而是 shuffled-label/null gate 仍然失败：假标签在当前 clean-design-cell view 上仍能达到过高 H/M min recall。

## 本阶段做了什么

运行 profile：

- `v8a_hm_clean_default_development_cif_literature`

运行结果：

- rows completed: `504/504`
- failed: `0`
- Hematite rows: `252`
- Magnetite rows: `252`
- thickness rows: `3/10/30/60 mm` 各 `126`
- source mode: `custom_diffraction_on` only
- stress label: `default` only

输出状态文件：

- `results/material_sorting/run_status_v8a_hm_clean_default_development_cif_literature.csv`

event-to-feature 输出：

- `results/accuracy_v3/v8a_clean_hm_development_event_to_feature/`
- samples: `504`
- sidecar rows: `89565`
- schema/control gate: `gate_passed=true`
- tiny training gate: `false`

这里的 tiny training gate 被锁住是正确的，因为 clean baseline 第一轮没有 source-off；source-off 是后续独立 robustness/control gate，不是这一轮训练许可。

## Clean View 与 Shortcut/Null 结果

clean-design-cell view：

- output: `results/accuracy_v3/v8a_clean_hm_development_crystal_clean_design_cell_event_to_feature/`
- matched pairs: train `108`，validation `72`，stress-holdout `72`
- `gate_passed=true`
- `training_unlocked=false`

non-material shortcut audit：

- output: `results/accuracy_v3/v8a_clean_hm_development_crystal_clean_design_cell_feature_shortcut/`
- `gate_passed=true`
- max non-material balanced accuracy: `0.3646`

这说明我们这次确实把显性的 source/origin/stress/thickness/pose/count-bin/seed-block shortcut 压下来了。

shuffled-label/null audit：

- output: `results/accuracy_v3/v8a_clean_hm_development_crystal_clean_design_cell_null_behavior/`
- `gate_passed=false`
- fixed-threshold null max H/M min recall: `0.7361`
- selected-threshold null max H/M min recall: `0.7361`
- fixed-threshold null p95: `0.6389`
- selected-threshold null p95: `0.6417`
- ExtraTrees fixed max: `0.7361`
- Logistic fixed max: `0.7083`
- within-strata fixed max: `0.0`

admission gate：

- output: `results/accuracy_v3/v8a_clean_hm_development_crystal_clean_design_cell_admission/`
- `gate_passed=false`
- `training_unlocked=false`
- stop reason: null gate failed

## 方法学判断

这个结果比旧数据前进一步，但还没有到可以训练的标准。

进步在于：

- clean matrix 源头平衡成功；
- Geant4 全量运行成功；
- event schema/integrity 通过；
- clean-design-cell strict pair support 充足；
- 非材料 shortcut gate 通过，max non-material BA 只有 `0.3646`。

失败在于：

- shuffled-label/null p95 仍高于 ceiling `0.55`；
- Logistic 和 ExtraTrees 都能在假标签下跑出过高 recall，所以不能简单归因于 tree overfit；
- within-strata shuffle 当前无效/过窄，说明现有 strict cell 下每层可打乱自由度不足，null protocol 还需要一个更适合 paired clean design 的版本。

## 下一步该怎么做

不要训练。下一阶段应做 paired-clean null protocol rework：

1. 建立 pair-level null audit：
   - 在 `clean_pair_id` 层打乱 H/M 标签；
   - 保持 split、seed_block、thickness、pose、count_target_bin 不变；
   - 报告有效 shuffle fraction，不接受 near-identity shuffle。

2. 建立 seed-block holdout null audit：
   - train label shuffle 只发生在 train seed blocks；
   - validation/stress-holdout 保持真实标签；
   - 检查模型是否仍能从 train 假标签泛化到真实 validation/holdout。

3. 重新定义 clean admission：
   - source-on/default clean baseline 不需要 source-off 才通过 schema；
   - 训练许可必须来自 shortcut + paired-null + seed-block-null 同时通过。

4. 只有新 null gate 通过，才允许 Logistic/ExtraTrees baseline。

## Claim Boundary

本阶段可以说：

- clean H/M development-only Geant4 matrix completed。
- clean source-first design removed visible non-material shortcut predictors。
- data are closer to auditable development evidence than the old mixed-origin views。

本阶段不能说：

- 可以训练了。
- H/M 分选成果已经恢复检验。
- 模型学到的是晶体差异。
- product accuracy、hardware validation、shadow/final、full ten-material matrix 或 manuscript-grade powder XRD 已经成立。

## 大白话总结

这次我们真的把新试卷跑出来了，而且试卷本身没有再露出明显的页码、水印、批次暗号。也就是说，之前最显眼的隐藏结构基本压住了。

但还有一个硬问题：我们把答案随机打乱以后，模型居然还能考得偏高。这说明现在还不能相信它一定是在学赤铁矿和磁铁矿的晶体差异。更准确地说，显性的脏东西少了，但 null 检查还没干净，或者我们的 null 打乱方式还不适合这种一对一配对的新数据。

所以这一步的正确结论是：跑完了，但不能训练。下一步不是上模型，而是重做“成对数据专用”的假标签审计，确认假答案真的学不起来以后，才重新开始分选成果检验。
