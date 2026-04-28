# Accuracy v3 方法学笔记

## 方法定位

Accuracy v3 把十材料分选视为严格泛化问题，而不是一次性训练集拟合问题。模型选择只允许使用 train/validation 数据；final test 在方案冻结后运行一次。已看过的 final test seeds 记录为 burned seeds，不能再参与调参。

## 分层分类

v3 使用三层结构：

1. 吸收组分类：区分 low absorption 与 high absorption。
2. 组内分类：分别在低吸收组和高吸收组内部训练材料分类器。
3. H/M 专家：对 Hematite/Magnetite 这对高混淆铁氧化物训练二分类专家，并只重新分配这对材料的概率质量。

这种结构保留十材料输出，但把最难的相似材料对交给局部专家处理。验收仍按全局十材料 Top-1、macro-F1 和最低单类召回计算。

## 特征增强

v3 在原有 calibrated transmission、attenuation、thickness-normalized attenuation、spectral shape、scatter/direct 和 dictionary-distance 特征基础上，增加以下准确率导向特征：

- event-level 能量沉积分位数和非零比例。
- hit-level energy 分位数、IQR、entropy。
- direct/scatter hit fraction 与 direct/scatter energy mean。
- theta/r 的中位数和高分位数。
- detector-response smoothed counts，使用 `2 keV` 和 `5 keV` 高斯展宽近似能量分辨率。

这些特征不使用材料名、化学式、密度或 group label 作为模型输入。group label 只用于构建层级路径，不进入最终特征列。

## 记录与复现

每次严格审计输出：

- `validation_model_selection.csv`
- `final_test_summary.csv`
- `per_class_recall_final_test.csv`
- `final_test_decisions.csv`
- `split_audit.csv`
- `experiment_registry.csv`
- `failure_analysis.csv`
- `strict_generalization_manifest.json`

其中 `experiment_registry.csv` 负责记录实验假设、seeds、模型、指标、失败原因和下一步动作；`failure_analysis.csv` 负责记录每个材料的召回、主要混淆和物理假设。

首轮 v3 诊断输出位于 `results/accuracy_v3/`。该目录使用已烧掉的 `sr2` final seeds，因此只能作为管线验证和负结果台账，不能作为新的泛化 claim。其主要结论是：现有 `sr2` 数据上的 H/M 最低召回仍低于 `0.70`，下一步必须跑新的 H/M-focused `accuracy_v3_hm` 数据。

v4 增加 development-only 审计模式，用于 H/M-focused 数据阶段。该模式只写 validation 指标、failure analysis 和 H/M pairwise audit，不输出 final-test claim。`v3_hm_smoke` 已验证该路径可运行；`v3_hm_dev1` 是下一阶段主要数据矩阵。

## H/M Pairwise Audit

`hm_pairwise_audit.csv` 单独训练 Hematite/Magnetite 二分类器，并记录：

- H/M pairwise Top-1、macro-F1、Hematite recall、Magnetite recall、H/M min recall。
- Magnetite-positive ROC AUC。
- 主要混淆方向。
- top feature、top source 和 top feature family。

这个表用于判断问题是模型层级不足，还是输入物理信号仍不足。如果 pairwise audit 也不能稳定超过 `0.70` recall，不能进入十材料 full matrix。

## Claim 边界

只有以下条件同时满足，才允许声称十材料自动分选达标：

- Top-1 accuracy >= `0.85`
- macro-F1 >= `0.80`
- min class recall >= `0.70`
- 每类 final-test support >= `30`
- train/validation/test seeds 完全不重叠
- final test seeds 不在 burned seed 列表中

如果 Top-1 和 macro-F1 达标但 min class recall 不达标，只能称为诊断或候选检索系统，不能称为自动十材料分类成功。
