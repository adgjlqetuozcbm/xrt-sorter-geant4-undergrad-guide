# v8A v7 twenty-material scalability scout model report

Generated: 2026-05-11T16:39:11+00:00

- Decision: `stop_v7_twenty_material_scalability_scout_gate`
- Gate passed: `false`
- Worst validation profile H/M min recall: `1.0000`
- Worst stress profile H/M min recall: `0.0000`
- Worst validation profile macro-F1: `1.0000`
- Worst stress profile macro-F1: `0.8697`
- Total-count-only worst profile H/M: `0.0000`
- Lineage-only worst profile H/M: `0.0000`

## Best Main Model

| eval_split | physical_perturbation_profile | method | top1_accuracy | macro_f1 | min_class_recall | hematite_recall | magnetite_recall | hm_min_recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stress_holdout | __overall__ | ExtraTreesMulticlassMain | 0.9750 | 0.9748 | 0.7500 | 0.7500 | 1.0000 | 0.7500 |
| stress_holdout | calibration_shift_stress_neg | LogisticMulticlassMain | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| stress_holdout | combined_stress_high | ExtraTreesMulticlassMain | 0.8875 | 0.8697 | 0.0000 | 0.0000 | 1.0000 | 0.0000 |
| stress_holdout | nominal | LogisticMulticlassMain | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| stress_holdout | resolution_blur_stress_high | ExtraTreesMulticlassMain | 0.9875 | 0.9873 | 0.7500 | 0.7500 | 1.0000 | 0.7500 |
| stress_holdout | source_energy_scale_stress_low | LogisticMulticlassMain | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| validation | __overall__ | ExtraTreesMulticlassMain | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| validation | calibration_shift_validation_mid | LogisticMulticlassMain | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| validation | combined_validation_mid | ExtraTreesMulticlassMain | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| validation | nominal | LogisticMulticlassMain | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| validation | resolution_blur_validation_mid | ExtraTreesMulticlassMain | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| validation | source_energy_scale_validation_mid | LogisticMulticlassMain | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Control Models

| eval_split | physical_perturbation_profile | method | top1_accuracy | macro_f1 | hm_min_recall |
| --- | --- | --- | --- | --- | --- |
| stress_holdout | __overall__ | ExtraTreesLineageOnly | 0.0500 | 0.0048 | 0.0000 |
| stress_holdout | __overall__ | ExtraTreesTotalCountOnly | 0.1425 | 0.1287 | 0.0000 |
| stress_holdout | calibration_shift_stress_neg | ExtraTreesLineageOnly | 0.0500 | 0.0048 | 0.0000 |
| stress_holdout | calibration_shift_stress_neg | ExtraTreesTotalCountOnly | 0.1625 | 0.1538 | 0.0000 |
| stress_holdout | combined_stress_high | ExtraTreesLineageOnly | 0.0500 | 0.0048 | 0.0000 |
| stress_holdout | combined_stress_high | ExtraTreesTotalCountOnly | 0.0500 | 0.0051 | 0.0000 |
| stress_holdout | nominal | ExtraTreesLineageOnly | 0.0500 | 0.0048 | 0.0000 |
| stress_holdout | nominal | ExtraTreesTotalCountOnly | 0.2625 | 0.2639 | 0.0000 |
| stress_holdout | resolution_blur_stress_high | ExtraTreesLineageOnly | 0.0500 | 0.0048 | 0.0000 |
| stress_holdout | resolution_blur_stress_high | ExtraTreesTotalCountOnly | 0.0875 | 0.0569 | 0.0000 |
| stress_holdout | source_energy_scale_stress_low | ExtraTreesLineageOnly | 0.0500 | 0.0048 | 0.0000 |
| stress_holdout | source_energy_scale_stress_low | ExtraTreesTotalCountOnly | 0.1500 | 0.1767 | 0.0000 |
| validation | __overall__ | ExtraTreesLineageOnly | 0.0500 | 0.0048 | 0.0000 |
| validation | __overall__ | ExtraTreesTotalCountOnly | 0.2100 | 0.2079 | 0.0000 |
| validation | calibration_shift_validation_mid | ExtraTreesLineageOnly | 0.0500 | 0.0048 | 0.0000 |
| validation | calibration_shift_validation_mid | ExtraTreesTotalCountOnly | 0.2500 | 0.2430 | 0.0000 |
| validation | combined_validation_mid | ExtraTreesLineageOnly | 0.0500 | 0.0048 | 0.0000 |
| validation | combined_validation_mid | ExtraTreesTotalCountOnly | 0.1125 | 0.1068 | 0.0000 |
| validation | nominal | ExtraTreesLineageOnly | 0.0500 | 0.0048 | 0.0000 |
| validation | nominal | ExtraTreesTotalCountOnly | 0.2125 | 0.1763 | 0.0000 |
| validation | resolution_blur_validation_mid | ExtraTreesLineageOnly | 0.0500 | 0.0048 | 0.0000 |
| validation | resolution_blur_validation_mid | ExtraTreesTotalCountOnly | 0.2375 | 0.2394 | 0.0000 |
| validation | source_energy_scale_validation_mid | ExtraTreesLineageOnly | 0.0500 | 0.0048 | 0.0000 |
| validation | source_energy_scale_validation_mid | ExtraTreesTotalCountOnly | 0.2375 | 0.2287 | 0.0000 |

## Shuffled Label Null

| eval_split | physical_perturbation_profile | method | hm_min_recall_p95 | macro_f1_p95 |
| --- | --- | --- | --- | --- |
| stress_holdout | calibration_shift_stress_neg | LogisticMulticlassMain | 0.0000 | 0.0792 |
| stress_holdout | combined_stress_high | LogisticMulticlassMain | 0.0000 | 0.0552 |
| stress_holdout | nominal | LogisticMulticlassMain | 0.0000 | 0.0711 |
| stress_holdout | resolution_blur_stress_high | LogisticMulticlassMain | 0.0000 | 0.1034 |
| stress_holdout | source_energy_scale_stress_low | LogisticMulticlassMain | 0.2500 | 0.0779 |
| validation | calibration_shift_validation_mid | LogisticMulticlassMain | 0.0000 | 0.0664 |
| validation | combined_validation_mid | LogisticMulticlassMain | 0.0000 | 0.0842 |
| validation | nominal | LogisticMulticlassMain | 0.1375 | 0.0802 |
| validation | resolution_blur_validation_mid | LogisticMulticlassMain | 0.0000 | 0.0493 |
| validation | source_energy_scale_validation_mid | LogisticMulticlassMain | 0.1375 | 0.0514 |

## Stop Reasons

- stress_worst_profile_hm_min_recall_below_threshold
- stress_overall_hm_min_recall_below_threshold
- real_minus_shuffled_worst_profile_hm_margin_too_small
