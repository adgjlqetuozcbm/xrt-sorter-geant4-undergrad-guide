# Geant4/XRT 材料分选共享说明

更新时间：2026-05-13

本文档用于说明本仓库和本地成果压缩包之间的对应关系。项目原始结果目录不建议移动或重命名，因为分析脚本和复现实验会按固定路径读取。

## 共享方式

本项目分两条路径共享：

1. GitHub 仓库：保存代码、配置、文档、峰表、可复现实验入口和小型结果摘要。
2. 本地压缩包：保存较大的事件特征表、关键模型输出和“程序-结果”对应关系说明，适合直接发给协作者。

普通 GitHub 仓库不适合保存数百 MB 的 sidecar/event feature CSV 或完整大结果目录。因此 `.gitignore` 已加入清理规则，避免把大型生成物误提交到 GitHub。

## 当前结论边界

| 层级 | 结论 | 主要路径 |
| --- | --- | --- |
| 主结果 | v6 的 10 材料扰动增强实验通过数据、模型和 shortcut 审计；validation/stress 的 H/M min recall 和 macro-F1 均为 `1.0`。 | `results/accuracy_v3/v8a_multiclass_context_v6_physical_robust_model` |
| 扩展结果 | v8 的 20 材料 relaxed ensemble 可作为论文/报告扩展结果；stress top-1 和 macro-F1 为 `0.9958`，stress H/M min recall 为 `0.9167`。 | `results/accuracy_v3/v8a_multiclass_context_v8_vote_log3_hist1_relaxed_gate` |
| 限制 | 严格 Hematite-Ilmenite hard-negative robustness 仍未完全解决；强 held-out stress 下仍有 Hematite 被判为 Ilmenite 的样本。 | `results/accuracy_v3/v8a_multiclass_context_v8_hard_negative_robust_hard_negative_diagnostics` |

## GitHub 中应重点查看

- `README.md`：项目总入口。
- `docs/G4_XRT_MATERIAL_SORTING_RESULTS_INDEX_zh.md`：材料分选成果索引。
- `docs/material_sorting_results_manifest/`：机器可读和表格化结果清单。
- `docs/MATERIAL_SORTING_SHARE_GUIDE_zh.md`：本文档，说明共享和打包边界。
- `analysis/`：实验生成、审计、训练和诊断脚本。
- `analysis/configs/`：v1-v8 关键实验配置。
- `source_models/config/diffraction_peak_tables/`：10 材料和 20 材料峰表 manifest。

## 压缩包中应重点查看

压缩包根目录包含：

- `README_SHARE_PACKAGE_zh.md`：压缩包内入口说明。
- `docs/`：成果索引和 manifest。
- `analysis/`：与 v6/v7/v8 相关的关键脚本。
- `source_models/config/diffraction_peak_tables/`：峰表 manifest。
- `source_models/config/material_sorting_matrix_selected/`：选定实验的输入矩阵 manifest 和矩阵 CSV。
- `results_selected/`：报告和衍生研究最常用的结果文件。

## 程序和结果对应关系

| 目标 | 主要脚本 | 主要结果 |
| --- | --- | --- |
| v6 10 材料物理扰动鲁棒性 | `analysis/generate_v8a_multiclass_context_v6_physical_robust_matrix.py`, `analysis/train_v8a_multiclass_context_v6_physical_robust.py`, `analysis/audit_v8a_multiclass_context_v6_skeptical_shortcut.py` | `results/accuracy_v3/v8a_multiclass_context_v6_physical_robust_model`, `results/accuracy_v3/v8a_multiclass_context_v6_skeptical_shortcut_audit` |
| v7 20 材料扩展侦察 | `analysis/generate_v8a_multiclass_context_v7_twenty_material_scalability_scout_matrix.py`, `analysis/train_v8a_multiclass_context_v7_twenty_material_scalability_scout.py`, `analysis/audit_v8a_multiclass_context_v7_hard_negative_diagnostics.py` | `results/accuracy_v3/v8a_multiclass_context_v7_twenty_material_scalability_scout_model`, `results/accuracy_v3/v8a_multiclass_context_v7_twenty_material_scalability_scout_hard_negative_diagnostics` |
| v8 hard-negative 鲁棒性 | `analysis/generate_v8a_multiclass_context_v8_hard_negative_robust_matrix.py`, `analysis/train_v8a_multiclass_context_v8_hard_negative_robust.py`, `analysis/audit_v8a_multiclass_context_v8_hard_negative_robust_diagnostics.py` | `results/accuracy_v3/v8a_multiclass_context_v8_hard_negative_robust_model`, `results/accuracy_v3/v8a_multiclass_context_v8_hard_negative_robust_hard_negative_diagnostics` |
| v8 relaxed 20 材料扩展门控 | `analysis/probe_v8_reporting_model_sweep.py`, `analysis/probe_v8_ensemble_relaxed.py`, `analysis/train_v8a_multiclass_context_v8_vote_log3_hist1_relaxed_gate.py`, `analysis/audit_v8a_multiclass_context_v8_reporting_grade_extension_gate.py` | `results/accuracy_v3/v8a_multiclass_context_v8_vote_log3_hist1_relaxed_gate`, `results/accuracy_v3/v8a_multiclass_context_v8_reporting_grade_extension_gate` |

## 不应直接上传 GitHub 的内容

- `results/accuracy_v3/*/*sidecar_long.csv`
- `results/accuracy_v3/*/*feature_table*.csv`
- `results/accuracy_v3/*/*.npz`
- `results/accuracy_v3/*/*.pdparams`
- 大型 `event_to_feature/` 全目录
- 大型打包文件和临时共享包

这些内容保留在本地原始项目目录和独立压缩包中。
