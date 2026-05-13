# v8 twenty-material reporting-grade extension gate

Generated: 2026-05-12T19:02:11+00:00

- Decision: `v8_twenty_material_reporting_extension_passed_with_caveats`
- Gate passed: `true`
- Claim scope: development-only 20-material reporting extension; not a strict hard-negative robustness claim, not hardware validation, and not product accuracy
- Strict hard-negative gate passed: `false`

## Key Metrics

- Validation overall top-1 / macro-F1 / H-M min recall: `0.9875` / `0.9873` / `1.0000`
- Stress overall top-1 / macro-F1 / H-M min recall: `0.9917` / `0.9915` / `0.8333`
- Stress worst-profile macro-F1: `0.9733`
- Strict stress worst-profile H-M min recall, report-only: `0.5000`

## Control Checks

- Total-count-only worst-profile H-M: `0.5000`
- Lineage-only worst-profile H-M: `0.0000`
- Shuffled-label H-M p95: `0.1375`
- Exact cross-split feature hash overlap: `0`

## Stress Profiles

| physical_perturbation_profile | method | top1_accuracy | macro_f1 | hematite_recall | magnetite_recall | hm_min_recall |
| --- | --- | --- | --- | --- | --- | --- |
| calibration_shift_stress_neg | LogisticMulticlassMain | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| combined_stress_high | LogisticMulticlassMain | 0.9875 | 0.9873 | 0.7500 | 1.0000 | 0.7500 |
| hard_negative_stress_alt | LogisticMulticlassMain | 0.9750 | 0.9733 | 0.5000 | 1.0000 | 0.5000 |
| nominal | LogisticMulticlassMain | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| resolution_blur_stress_high | ExtraTreesMulticlassMain | 0.9875 | 0.9873 | 1.0000 | 1.0000 | 1.0000 |
| source_energy_scale_stress_low | LogisticMulticlassMain | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |

## Caveats

- strict hard-negative robustness gate still fails; this reporting gate must not be described as strong hard-negative success
- worst stress-profile H/M recall is below the strict robustness threshold and should be reported as a limitation
- Hematite-to-Ilmenite errors remain in held-out stress profiles: 6 unique samples

## Stop Reasons

- None.
