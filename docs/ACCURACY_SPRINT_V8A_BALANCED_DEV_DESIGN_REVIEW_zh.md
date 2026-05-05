# Accuracy Sprint v8A: balanced development design review

Date: 2026-05-05

## 1. Review decision

The completed 90-row v8A event-feature tiny gate supports continuing the diffraction-aware sidecar route. It does not support a full v8A matrix, shadow/final validation, product accuracy, hardware validation, or manuscript-grade powder-XRD claims.

The next implementation unit is therefore a stricter development review:

- upgrade H/M peak provenance from project-scan anchors to a development reference candidate;
- audit the candidate manifest;
- run a stricter event-feature stress gate on the existing development features;
- permit medium development matrix preregistration only if both gates pass.

## 2. New source-controlled artifacts

Peak provenance candidate:

- `source_models/config/diffraction_peak_tables/hm_powder_peaks_cif_or_literature_v8a_manifest.json`

Peak provenance audit:

- `analysis/audit_v8a_peak_provenance.py`

Stress gate config:

- `analysis/configs/v8a_event_feature_stress_gate_config.json`

Stress gate script:

- `analysis/v8a_event_feature_stress_gate.py`

## 3. Commands

Peak provenance audit:

```bash
/home/dyd/geant4-projects/xrt_sorter/.venv/bin/python \
  analysis/audit_v8a_peak_provenance.py \
  --project-root . \
  --manifest source_models/config/diffraction_peak_tables/hm_powder_peaks_cif_or_literature_v8a_manifest.json \
  --output-dir results/accuracy_v3/v8a_peak_provenance_audit \
  --overwrite
```

Stress gate:

```bash
/home/dyd/geant4-projects/xrt_sorter/.venv/bin/python \
  analysis/v8a_event_feature_stress_gate.py \
  --project-root . \
  --config analysis/configs/v8a_event_feature_stress_gate_config.json \
  --overwrite
```

## 4. Stricter gate thresholds

The stress gate uses stricter thresholds than the 90-row tiny gate:

| Gate item | Threshold |
| --- | --- |
| main H/M min recall | `>= 0.95` |
| worst-thickness H/M min recall | `>= 0.90` |
| total-count-only H/M min recall | `< 0.60` |
| overlap-only H/M min recall | `< 0.60` |
| shuffled-label H/M min recall | `< 0.55` |
| source-off H/M min recall | `< 0.60` |
| main minus source-off margin | `>= 0.35` |
| leave-one-thickness H/M min recall | `>= 0.90` |
| validation seed-group H/M min recall | `>= 0.90` |

## 5. Stress scenarios

The stress gate consumes only development outputs from `results/accuracy_v3/v8a_event_to_feature_smoke/`. The current rerun uses the successor peak manifest for event-to-feature re-windowing, but the already-generated Geant4 phase-space/source rows still record `hm_powder_peaks_project_scan_v8a` as their source peak table. This lineage difference is acceptable for the present development review and must be preserved in the gate output; it blocks any medium matrix until new source rows are generated directly from the successor manifest.

It trains each model on the unchanged baseline development train rows and applies each stress variant only to the validation rows. The q-jitter cases are feature-level proxies over already aggregated event features; they do not re-window the raw long table.

It applies:

- identity baseline;
- small/medium/strong feature-proxy peak perturbation;
- low/high relative intensity perturbation;
- detector-resolution-style feature smoothing;
- background noise injection;
- overlap-window suppression audit.

The main model may use only `diffraction_*` features. It must not use material, source id, sample id, path, seed, thickness, pose, split, or row index fields.

## 6. Stop rules

Stop before any medium development matrix if:

- peak provenance audit fails;
- stress gate fails;
- source peak table lineage is not regenerated from the successor manifest before medium matrix execution;
- overlap-only or source-off controls exceed the stricter ceilings;
- leave-one-thickness or validation seed-group performance falls below threshold;
- any main feature name suggests lineage leakage;
- any input reports shadow/final use or existing XRT cube reads.

## 7. Claim boundary

Passing this review means only:

- development-only diffraction-aware sidecar robustness is sufficient to preregister a medium H/M development matrix.

It does not mean:

- ordinary XRT solves H/M;
- product accuracy is known;
- hardware is validated;
- the simulation is publishable powder XRD;
- shadow/final may be opened.

## 8. Current gate result

The peak provenance audit passed:

- output: `results/accuracy_v3/v8a_peak_provenance_audit/`
- decision: `proceed_to_v8a_event_feature_stress_gate`
- peak count: `16`
- external reference count: `2`
- max q error: `0.000021`
- max d error: `0.000077`

The event-to-feature rerun uses the successor manifest for analysis windows but records old source peak-table lineage:

- analysis peak table: `hm_powder_peaks_cif_or_literature_v8a`
- source peak table ids: `hm_powder_peaks_project_scan_v8a`
- consequence: current stress gate may be used as development review evidence only; medium matrix generation must create source rows from the successor manifest.

The stricter event-feature stress gate passed after rerunning the event-to-feature pipeline with the successor manifest:

- output: `results/accuracy_v3/v8a_event_feature_stress_gate/`
- decision: `proceed_to_medium_development_matrix_preregistration_requires_successor_source_regeneration`
- worst main H/M min recall: `1.0`
- worst-thickness H/M min recall: `1.0`
- total-count-only H/M min recall: `0.4167`
- overlap-only H/M min recall: `0.5000`
- shuffled-label H/M min recall: `0.3333`
- source-off H/M min recall: `0.5`
- leave-one-thickness H/M min recall: `1.0`
- validation seed-group H/M min recall: `1.0`

The earlier project-scan re-window failed the stricter overlap-only ceiling (`0.6667`), but the successor manifest re-window reduced overlap-only H/M min recall below the `<0.60` ceiling. This is a useful improvement, not a product metric.

Remaining lineage condition:

- the 90-row Geant4 source rows still came from `hm_powder_peaks_project_scan_v8a`;
- medium matrix execution must regenerate source rows directly from `hm_powder_peaks_cif_or_literature_v8a`.

## 9. Medium development preregistration

The medium development matrix preregistration package is:

- `analysis/configs/v8a_medium_development_matrix_config.json`
- `analysis/generate_v8a_medium_development_matrix.py`
- `analysis/audit_v8a_medium_development_prereg.py`

Current preregistration result:

- profile: `v8a_hm_medium_development_cif_literature`
- matrix rows: `864`
- train rows: `432`
- validation rows: `216`
- stress-holdout rows: `216`
- peak table: `hm_powder_peaks_cif_or_literature_v8a`
- decision: `medium_development_matrix_preregistered_not_run`
- training unlocked: `false`

The next executable step is to run the medium development matrix only. Development model training may start only after the medium matrix completes and then passes event-to-feature schema, stress, and leakage audits on its own outputs.
