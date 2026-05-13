# 材料分选成果清单（报告/衍生研究入口）

生成时间：2026-05-13T06:44:42+00:00

| id | 类型 | 阶段 | 状态 | 主要指标/用途 | 路径 |
| --- | --- | --- | --- | --- | --- |
| v6_main_10_material_robust | report_main_result | v6 | passed | validation/stress H/M min recall = 1.0; macro-F1 = 1.0 | `results/accuracy_v3/v8a_multiclass_context_v6_physical_robust_model` |
| v6_shortcut_audit | report_supporting_evidence | v6 | passed | controls weak; exact cross-split feature hash overlap absent | `results/accuracy_v3/v8a_multiclass_context_v6_skeptical_shortcut_audit` |
| v8_relaxed_20_material_extension | report_extension_result | v8 | passed_with_caveat | stress top-1 = 0.9958; stress macro-F1 = 0.9958; stress H/M min recall = 0.9167; worst-profile H/M = 0.75 | `results/accuracy_v3/v8a_multiclass_context_v8_vote_log3_hist1_relaxed_gate` |
| v8_reporting_grade_gate | report_extension_supporting_evidence | v8 | passed_with_caveat | reporting gate passed; strict hard-negative gate not passed | `results/accuracy_v3/v8a_multiclass_context_v8_reporting_grade_extension_gate` |
| v8_strict_hard_negative | limitation_and_future_work | v8 | failed_strict_gate | strict stress worst-profile H/M = 0.5 in original v8 main gate; relaxed ensemble improves to 0.75 | `results/accuracy_v3/v8a_multiclass_context_v8_hard_negative_robust_model` |
| v8_hard_negative_diagnostics | diagnostic_evidence | v8 | completed | Hematite->Ilmenite errors and nearest-centroid collapse under stress profiles | `results/accuracy_v3/v8a_multiclass_context_v8_hard_negative_robust_hard_negative_diagnostics` |
| v8_event_features | reusable_dataset | v8 | available | 2160 samples; 651557 sidecar rows | `results/accuracy_v3/v8a_multiclass_context_v8_hard_negative_robust_event_to_feature` |
| v6_event_features | reusable_dataset | v6 | available | 1400 samples; 409983 sidecar rows | `results/accuracy_v3/v8a_multiclass_context_v6_physical_robust_event_to_feature` |
| input_matrices | reproducibility | v4-v8 | available | v4-v8 matrix profiles preserved | `source_models/config/material_sorting_matrix` |
| peak_manifests | reproducibility | v8a | available | 10-material and 20-material manifests available | `source_models/config/diffraction_peak_tables` |

## 使用建议

- 写报告优先使用 `v6_main_10_material_robust`、`v6_shortcut_audit`、`v8_relaxed_20_material_extension`、`v8_reporting_grade_gate`。
- 继续研究优先使用 `v8_event_features`、`v8_hard_negative_diagnostics`、`peak_manifests`。
- 复现实验优先使用 `input_matrices` 和 `results/material_sorting/run_status_*.csv`。
