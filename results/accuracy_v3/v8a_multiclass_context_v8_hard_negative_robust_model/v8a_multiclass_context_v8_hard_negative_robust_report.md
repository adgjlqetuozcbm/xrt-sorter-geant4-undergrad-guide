# v8A v8 twenty-material hard-negative robustness model report

Generated: 2026-05-12T18:22:12+00:00

- Decision: `stop_v8_hard_negative_robust_gate`
- Gate passed: `false`
- Worst validation profile H/M min recall: `1.0000`
- Worst stress profile H/M min recall: `0.5000`
- Worst validation profile macro-F1: `0.9733`
- Worst stress profile macro-F1: `0.9733`
- Total-count-only worst profile H/M: `0.5000`
- Lineage-only worst profile H/M: `0.0000`

## Best Main Model

| eval_split | physical_perturbation_profile | method | top1_accuracy | macro_f1 | min_class_recall | hematite_recall | magnetite_recall | hm_min_recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stress_holdout | __overall__ | LogisticMulticlassMain | 0.9917 | 0.9915 | 0.8333 | 0.8333 | 1.0000 | 0.8333 |
| stress_holdout | calibration_shift_stress_neg | LogisticMulticlassMain | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| stress_holdout | combined_stress_high | LogisticMulticlassMain | 0.9875 | 0.9873 | 0.7500 | 0.7500 | 1.0000 | 0.7500 |
| stress_holdout | hard_negative_stress_alt | LogisticMulticlassMain | 0.9750 | 0.9733 | 0.5000 | 0.5000 | 1.0000 | 0.5000 |
| stress_holdout | nominal | LogisticMulticlassMain | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| stress_holdout | resolution_blur_stress_high | ExtraTreesMulticlassMain | 0.9875 | 0.9873 | 0.7500 | 1.0000 | 1.0000 | 1.0000 |
| stress_holdout | source_energy_scale_stress_low | LogisticMulticlassMain | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| validation | __overall__ | LogisticMulticlassMain | 0.9875 | 0.9873 | 0.7500 | 1.0000 | 1.0000 | 1.0000 |
| validation | calibration_shift_validation_mid | ExtraTreesMulticlassMain | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| validation | combined_validation_mid | ExtraTreesMulticlassMain | 0.9750 | 0.9733 | 0.5000 | 1.0000 | 1.0000 | 1.0000 |
| validation | hard_negative_validation_bridge | LogisticMulticlassMain | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| validation | nominal | LogisticMulticlassMain | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| validation | resolution_blur_validation_mid | ExtraTreesMulticlassMain | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| validation | source_energy_scale_validation_mid | LogisticMulticlassMain | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Control Models

| eval_split | physical_perturbation_profile | method | top1_accuracy | macro_f1 | hm_min_recall |
| --- | --- | --- | --- | --- | --- |
| stress_holdout | __overall__ | ExtraTreesLineageOnly | 0.0500 | 0.0048 | 0.0000 |
| stress_holdout | __overall__ | ExtraTreesTotalCountOnly | 0.1562 | 0.1485 | 0.1250 |
| stress_holdout | calibration_shift_stress_neg | ExtraTreesLineageOnly | 0.0500 | 0.0048 | 0.0000 |
| stress_holdout | calibration_shift_stress_neg | ExtraTreesTotalCountOnly | 0.2000 | 0.2012 | 0.0000 |
| stress_holdout | combined_stress_high | ExtraTreesLineageOnly | 0.0500 | 0.0048 | 0.0000 |
| stress_holdout | combined_stress_high | ExtraTreesTotalCountOnly | 0.1750 | 0.1367 | 0.5000 |
| stress_holdout | hard_negative_stress_alt | ExtraTreesLineageOnly | 0.0500 | 0.0048 | 0.0000 |
| stress_holdout | hard_negative_stress_alt | ExtraTreesTotalCountOnly | 0.0625 | 0.0117 | 0.0000 |
| stress_holdout | nominal | ExtraTreesLineageOnly | 0.0500 | 0.0048 | 0.0000 |
| stress_holdout | nominal | ExtraTreesTotalCountOnly | 0.1875 | 0.1779 | 0.0000 |
| stress_holdout | resolution_blur_stress_high | ExtraTreesLineageOnly | 0.0500 | 0.0048 | 0.0000 |
| stress_holdout | resolution_blur_stress_high | ExtraTreesTotalCountOnly | 0.1750 | 0.1656 | 0.0000 |
| stress_holdout | source_energy_scale_stress_low | ExtraTreesLineageOnly | 0.0500 | 0.0048 | 0.0000 |
| stress_holdout | source_energy_scale_stress_low | ExtraTreesTotalCountOnly | 0.1375 | 0.1420 | 0.0000 |
| validation | __overall__ | ExtraTreesLineageOnly | 0.0500 | 0.0048 | 0.0000 |
| validation | __overall__ | ExtraTreesTotalCountOnly | 0.1979 | 0.1929 | 0.0417 |
| validation | calibration_shift_validation_mid | ExtraTreesLineageOnly | 0.0500 | 0.0048 | 0.0000 |
| validation | calibration_shift_validation_mid | ExtraTreesTotalCountOnly | 0.2500 | 0.2573 | 0.0000 |
| validation | combined_validation_mid | ExtraTreesLineageOnly | 0.0500 | 0.0048 | 0.0000 |
| validation | combined_validation_mid | ExtraTreesTotalCountOnly | 0.2000 | 0.1729 | 0.0000 |
| validation | hard_negative_validation_bridge | ExtraTreesLineageOnly | 0.0500 | 0.0048 | 0.0000 |
| validation | hard_negative_validation_bridge | ExtraTreesTotalCountOnly | 0.2250 | 0.2015 | 0.0000 |
| validation | nominal | ExtraTreesLineageOnly | 0.0500 | 0.0048 | 0.0000 |
| validation | nominal | ExtraTreesTotalCountOnly | 0.1750 | 0.1526 | 0.0000 |
| validation | resolution_blur_validation_mid | ExtraTreesLineageOnly | 0.0500 | 0.0048 | 0.0000 |
| validation | resolution_blur_validation_mid | ExtraTreesTotalCountOnly | 0.1750 | 0.1755 | 0.0000 |
| validation | source_energy_scale_validation_mid | ExtraTreesLineageOnly | 0.0500 | 0.0048 | 0.0000 |
| validation | source_energy_scale_validation_mid | ExtraTreesTotalCountOnly | 0.1625 | 0.1592 | 0.0000 |

## Shuffled Label Null

| eval_split | physical_perturbation_profile | method | hm_min_recall_p95 | macro_f1_p95 |
| --- | --- | --- | --- | --- |
| stress_holdout | calibration_shift_stress_neg | LogisticMulticlassMain | 0.0000 | 0.0886 |
| stress_holdout | combined_stress_high | LogisticMulticlassMain | 0.0000 | 0.0804 |
| stress_holdout | hard_negative_stress_alt | LogisticMulticlassMain | 0.0000 | 0.0559 |
| stress_holdout | nominal | LogisticMulticlassMain | 0.0000 | 0.0909 |
| stress_holdout | resolution_blur_stress_high | LogisticMulticlassMain | 0.0000 | 0.0598 |
| stress_holdout | source_energy_scale_stress_low | LogisticMulticlassMain | 0.1375 | 0.0873 |
| validation | calibration_shift_validation_mid | LogisticMulticlassMain | 0.0000 | 0.0759 |
| validation | combined_validation_mid | LogisticMulticlassMain | 0.0000 | 0.0746 |
| validation | hard_negative_validation_bridge | LogisticMulticlassMain | 0.0000 | 0.0809 |
| validation | nominal | LogisticMulticlassMain | 0.0000 | 0.0693 |
| validation | resolution_blur_validation_mid | LogisticMulticlassMain | 0.0000 | 0.0572 |
| validation | source_energy_scale_validation_mid | LogisticMulticlassMain | 0.0000 | 0.0758 |

## Stop Reasons

- stress_worst_profile_hm_min_recall_below_threshold
- stress_overall_hm_min_recall_below_threshold
- stress_hematite_to_ilmenite_main_errors_above_threshold
