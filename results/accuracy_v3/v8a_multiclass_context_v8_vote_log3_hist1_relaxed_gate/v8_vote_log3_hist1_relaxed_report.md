# v8 Vote(Logistic x3 + HistGB x1) relaxed extension gate

Generated: 2026-05-12T19:36:30+00:00

- Decision: `v8_vote_log3_hist1_relaxed_extension_passed`
- Gate passed: `true`
- Stress overall top-1 / macro-F1 / H-M: `0.9958` / `0.9958` / `0.9167`
- Stress worst-profile H-M / macro-F1: `0.7500` / `0.9873`
- H->Ilmenite unique stress samples: `2`

## Profiles

| eval_split | physical_perturbation_profile | top1_accuracy | macro_f1 | hematite_recall | magnetite_recall | hm_min_recall |
| --- | --- | --- | --- | --- | --- | --- |
| stress_holdout | __overall__ | 0.9958 | 0.9958 | 0.9167 | 1.0000 | 0.9167 |
| stress_holdout | calibration_shift_stress_neg | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| stress_holdout | combined_stress_high | 0.9875 | 0.9873 | 0.7500 | 1.0000 | 0.7500 |
| stress_holdout | hard_negative_stress_alt | 0.9875 | 0.9873 | 0.7500 | 1.0000 | 0.7500 |
| stress_holdout | nominal | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| stress_holdout | resolution_blur_stress_high | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| stress_holdout | source_energy_scale_stress_low | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| validation | __overall__ | 0.9854 | 0.9853 | 0.9583 | 1.0000 | 0.9583 |
| validation | calibration_shift_validation_mid | 0.9875 | 0.9873 | 1.0000 | 1.0000 | 1.0000 |
| validation | combined_validation_mid | 0.9625 | 0.9564 | 1.0000 | 1.0000 | 1.0000 |
| validation | hard_negative_validation_bridge | 0.9875 | 0.9873 | 0.7500 | 1.0000 | 0.7500 |
| validation | nominal | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| validation | resolution_blur_validation_mid | 0.9750 | 0.9733 | 1.0000 | 1.0000 | 1.0000 |
| validation | source_energy_scale_validation_mid | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
