# v8A v6 physical robustness model report

Generated: 2026-05-10T18:16:37+00:00

- Decision: `v6_physical_robust_gate_passed_ready_for_scalability_scout`
- Gate passed: `true`
- Worst validation profile H/M min recall: `1.0000`
- Worst stress profile H/M min recall: `1.0000`
- Total-count-only worst profile H/M: `0.2500`
- Lineage-only worst profile H/M: `0.0000`

## Best Main Model

| eval_split | physical_perturbation_profile | method | top1_accuracy | macro_f1 | min_class_recall | hematite_recall | magnetite_recall | hm_min_recall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| stress_holdout | __overall__ | ExtraTreesMulticlassMain | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| stress_holdout | calibration_shift_stress_neg | LogisticMulticlassMain | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| stress_holdout | combined_stress_high | ExtraTreesMulticlassMain | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| stress_holdout | nominal | LogisticMulticlassMain | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| stress_holdout | resolution_blur_stress_high | LogisticMulticlassMain | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| stress_holdout | source_energy_scale_stress_low | LogisticMulticlassMain | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| validation | __overall__ | LogisticMulticlassMain | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| validation | calibration_shift_validation_mid | LogisticMulticlassMain | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| validation | combined_validation_mid | LogisticMulticlassMain | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| validation | nominal | LogisticMulticlassMain | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| validation | resolution_blur_validation_mid | LogisticMulticlassMain | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| validation | source_energy_scale_validation_mid | LogisticMulticlassMain | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Control Models

| eval_split | physical_perturbation_profile | method | top1_accuracy | macro_f1 | hm_min_recall |
| --- | --- | --- | --- | --- | --- |
| stress_holdout | __overall__ | ExtraTreesLineageOnly | 0.1000 | 0.0182 | 0.0000 |
| stress_holdout | __overall__ | ExtraTreesTotalCountOnly | 0.1525 | 0.1438 | 0.0250 |
| stress_holdout | calibration_shift_stress_neg | ExtraTreesLineageOnly | 0.1000 | 0.0182 | 0.0000 |
| stress_holdout | calibration_shift_stress_neg | ExtraTreesTotalCountOnly | 0.1500 | 0.1554 | 0.0000 |
| stress_holdout | combined_stress_high | ExtraTreesLineageOnly | 0.1000 | 0.0182 | 0.0000 |
| stress_holdout | combined_stress_high | ExtraTreesTotalCountOnly | 0.1000 | 0.0182 | 0.0000 |
| stress_holdout | nominal | ExtraTreesLineageOnly | 0.1000 | 0.0182 | 0.0000 |
| stress_holdout | nominal | ExtraTreesTotalCountOnly | 0.1875 | 0.1727 | 0.1250 |
| stress_holdout | resolution_blur_stress_high | ExtraTreesLineageOnly | 0.1000 | 0.0182 | 0.0000 |
| stress_holdout | resolution_blur_stress_high | ExtraTreesTotalCountOnly | 0.1500 | 0.1010 | 0.0000 |
| stress_holdout | source_energy_scale_stress_low | ExtraTreesLineageOnly | 0.1000 | 0.0182 | 0.0000 |
| stress_holdout | source_energy_scale_stress_low | ExtraTreesTotalCountOnly | 0.1750 | 0.1694 | 0.0000 |
| validation | __overall__ | ExtraTreesLineageOnly | 0.1000 | 0.0182 | 0.0000 |
| validation | __overall__ | ExtraTreesTotalCountOnly | 0.2150 | 0.2147 | 0.1250 |
| validation | calibration_shift_validation_mid | ExtraTreesLineageOnly | 0.1000 | 0.0182 | 0.0000 |
| validation | calibration_shift_validation_mid | ExtraTreesTotalCountOnly | 0.1875 | 0.1821 | 0.2500 |
| validation | combined_validation_mid | ExtraTreesLineageOnly | 0.1000 | 0.0182 | 0.0000 |
| validation | combined_validation_mid | ExtraTreesTotalCountOnly | 0.2625 | 0.2329 | 0.0000 |
| validation | nominal | ExtraTreesLineageOnly | 0.1000 | 0.0182 | 0.0000 |
| validation | nominal | ExtraTreesTotalCountOnly | 0.2875 | 0.2746 | 0.1250 |
| validation | resolution_blur_validation_mid | ExtraTreesLineageOnly | 0.1000 | 0.0182 | 0.0000 |
| validation | resolution_blur_validation_mid | ExtraTreesTotalCountOnly | 0.1750 | 0.1831 | 0.1250 |
| validation | source_energy_scale_validation_mid | ExtraTreesLineageOnly | 0.1000 | 0.0182 | 0.0000 |
| validation | source_energy_scale_validation_mid | ExtraTreesTotalCountOnly | 0.1625 | 0.1643 | 0.0000 |

## Shuffled Label Null

| eval_split | physical_perturbation_profile | method | hm_min_recall_p95 | macro_f1_p95 |
| --- | --- | --- | --- | --- |
| stress_holdout | calibration_shift_stress_neg | LogisticMulticlassMain | 0.0687 | 0.1430 |
| stress_holdout | combined_stress_high | LogisticMulticlassMain | 0.1250 | 0.1294 |
| stress_holdout | nominal | LogisticMulticlassMain | 0.0687 | 0.1437 |
| stress_holdout | resolution_blur_stress_high | LogisticMulticlassMain | 0.0000 | 0.1314 |
| stress_holdout | source_energy_scale_stress_low | LogisticMulticlassMain | 0.0687 | 0.1114 |
| validation | calibration_shift_validation_mid | LogisticMulticlassMain | 0.1250 | 0.1498 |
| validation | combined_validation_mid | LogisticMulticlassMain | 0.0000 | 0.1109 |
| validation | nominal | LogisticMulticlassMain | 0.0687 | 0.1430 |
| validation | resolution_blur_validation_mid | LogisticMulticlassMain | 0.0687 | 0.1349 |
| validation | source_energy_scale_validation_mid | LogisticMulticlassMain | 0.1937 | 0.1462 |

## Stop Reasons

- None.
