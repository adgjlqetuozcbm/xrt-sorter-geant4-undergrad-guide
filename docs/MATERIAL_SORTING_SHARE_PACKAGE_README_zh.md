# 材料分选成果共享压缩包说明

本压缩包是 Geant4/XRT 材料级智能分选研究的“结果共享包”，用于发给协作者快速理解当前主要成果和继续做衍生研究。它不是完整仓库备份；完整代码和文档应优先查看 GitHub 仓库。

## 使用顺序

1. 先读 `docs/G4_XRT_MATERIAL_SORTING_RESULTS_INDEX_zh.md`，了解 v6/v7/v8 的成果层级。
2. 再读 `docs/MATERIAL_SORTING_SHARE_GUIDE_zh.md`，理解 GitHub 和压缩包的分工。
3. 需要机器读取路径时，查看 `docs/material_sorting_results_manifest/material_sorting_results_manifest.json`。
4. 需要直接做表格整理时，查看 `docs/material_sorting_results_manifest/material_sorting_results_inventory.csv`。

## 核心结论

| 层级 | 当前结论 | 对应目录 |
| --- | --- | --- |
| 主结果 | v6 的 10 材料扰动增强实验通过 gate，validation/stress 的 H/M min recall 和 macro-F1 均为 `1.0`。 | `results/accuracy_v3/v8a_multiclass_context_v6_physical_robust_model` |
| 扩展结果 | v8 relaxed ensemble 可作为 20 材料扩展结果；stress top-1 和 macro-F1 为 `0.9958`，stress H/M min recall 为 `0.9167`。 | `results/accuracy_v3/v8a_multiclass_context_v8_vote_log3_hist1_relaxed_gate` |
| 限制 | 严格 Hematite-Ilmenite hard-negative robustness 尚未完全解决，仍应作为未来工作。 | `results/accuracy_v3/v8a_multiclass_context_v8_hard_negative_robust_hard_negative_diagnostics` |

## 目录内容

```text
docs/
  G4_XRT_MATERIAL_SORTING_RESULTS_INDEX_zh.md
  MATERIAL_SORTING_SHARE_GUIDE_zh.md
  material_sorting_results_manifest/
analysis/
  configs/
  selected v6/v7/v8 generation, audit, training and probe scripts
source_models/config/
  diffraction_peak_tables/
  material_sorting_matrix/
    selected matrix_manifest.json and material_sorting_matrix.csv files
results/
  material_sorting/
    selected run_status CSV files
  accuracy_v3/
    selected v6/v7/v8 feature, model, audit and diagnostic outputs
```

## 程序和结果对应

| 研究动作 | 程序入口 | 结果入口 |
| --- | --- | --- |
| v6 10 材料物理扰动鲁棒性 | `analysis/generate_v8a_multiclass_context_v6_physical_robust_matrix.py`, `analysis/train_v8a_multiclass_context_v6_physical_robust.py` | `results/accuracy_v3/v8a_multiclass_context_v6_physical_robust_model` |
| v6 shortcut 审计 | `analysis/audit_v8a_multiclass_context_v6_skeptical_shortcut.py` | `results/accuracy_v3/v8a_multiclass_context_v6_skeptical_shortcut_audit` |
| v7 20 材料扩展侦察 | `analysis/generate_v8a_multiclass_context_v7_twenty_material_scalability_scout_matrix.py`, `analysis/train_v8a_multiclass_context_v7_twenty_material_scalability_scout.py` | `results/accuracy_v3/v8a_multiclass_context_v7_twenty_material_scalability_scout_model` |
| v8 hard-negative 鲁棒性 | `analysis/generate_v8a_multiclass_context_v8_hard_negative_robust_matrix.py`, `analysis/train_v8a_multiclass_context_v8_hard_negative_robust.py` | `results/accuracy_v3/v8a_multiclass_context_v8_hard_negative_robust_model` |
| v8 hard-negative 诊断 | `analysis/audit_v8a_multiclass_context_v8_hard_negative_robust_diagnostics.py` | `results/accuracy_v3/v8a_multiclass_context_v8_hard_negative_robust_hard_negative_diagnostics` |
| v8 relaxed ensemble | `analysis/probe_v8_reporting_model_sweep.py`, `analysis/probe_v8_ensemble_relaxed.py`, `analysis/train_v8a_multiclass_context_v8_vote_log3_hist1_relaxed_gate.py` | `results/accuracy_v3/v8a_multiclass_context_v8_vote_log3_hist1_relaxed_gate` |

## 重要边界

- 所有结果都是 synthetic Geant4 development evidence，不是硬件实测结果。
- v6 是最适合作为主结论的 10 材料鲁棒结果。
- v8 relaxed ensemble 是可写入报告的 20 材料扩展结果，但必须写清 hard-negative caveat。
- Hematite-Ilmenite 严格 held-out stress 仍是未完全解决的问题。
- 压缩包中的大 CSV 用于复核和衍生研究，不建议直接提交到普通 GitHub 仓库。
