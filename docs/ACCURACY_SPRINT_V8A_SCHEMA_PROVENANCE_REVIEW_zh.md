# Accuracy Sprint v8A: Geant4 diffraction output schema and provenance review

Date: 2026-05-05

## 1. Review decision

The v8A minimal Geant4 boundary smoke passed only one gate: a custom/table diffraction-like phase-space source can enter the real Geant4 source, geometry, and detector-output path, and source-off leakage is suppressed. This is enough to start a schema and provenance review. It is not enough to start a full v8A matrix, open shadow/final data, or claim Hematite/Magnetite (H/M) accuracy.

The next implementation unit is therefore `v8A_geant4_diffraction_output_schema_provenance_review`. Its output is a source-controlled design contract, not a new simulation result.

## 2. Evidence boundary

Completed evidence:

- Ordinary XRT development evidence remains negative for H/M: v7B H/M min recall `0.6083`, v7B2 formal H/M Pilot `0.5078`, Phase 1 compact sidecar best policy remains baseline/no sidecar at `0.6083`.
- v8A synthetic, transport-like, custom-source, leakage-off, and small sidecar training smokes passed as development-only observability gates.
- Minimal Geant4 boundary smoke completed `12` rows with source-on high-angle primary median `0.615375`, source-on minimum `0.60825`, leakage-off median `0.001`, leakage-off maximum `0.00125`, and decision `proceed_to_geant4_diffraction_output_schema_review`.

Current unsupported claims:

- H/M sorting is not solved.
- The current branch is not validated powder XRD.
- The current branch is not hardware validation.
- The current branch is not full ten-material v8A evidence.
- Geant4 Rayleigh scattering and `G4XrayReflection` are not being treated as powder diffraction.

## 3. Output schema contract

The machine-readable contract is:

- `analysis/configs/v8a_diffraction_output_schema_contract.json`

The contract defines the minimum traceable path:

1. Source/config metadata
2. Phase-space photon rows
3. Geant4 event rows
4. Geant4 hit rows
5. Sidecar long table
6. Sidecar feature table
7. Gate/control manifest

Required physical axis:

- Use `q_a_inv` or `d_a` for diffraction feature alignment.
- Do not fuse multi-energy rows with fixed cross-energy `2theta`.
- If `two_theta_deg` is retained, it is a derived per-source diagnostic, not the canonical cross-energy feature axis.

Current Geant4 hit inputs are limited to:

- `event_id`
- `detector_id`
- `x_mm`
- `y_mm`
- `z_mm`
- `photon_energy_keV`
- `is_primary`
- `theta_deg`
- `is_direct_primary`
- `is_scattered_primary`

The first event-to-feature pipeline must not invent unrecorded hit fields. If it needs additional quantities such as process name, track parent, step length, local detector coordinates, or source photon id, those must be added to the C++ output in a separate preregistered implementation phase before being used.

## 4. Sidecar long-table schema

Each sidecar long row should represent one binned detector response for one sample, source, detector sector, pose, thickness, and q/d bin.

Required columns:

| Column | Meaning |
| --- | --- |
| `sample_id` | Stable sample identifier; must include split/material/seed/thickness/pose/source lineage. |
| `split` | Development split only, normally `train` or `validation`. |
| `material` | Ground-truth development label; H/M only until this review passes. |
| `random_seed` | Seed used for source/sample generation. |
| `thickness_mm` | Slab thickness used in config and manifest. |
| `pose_index` | Pose/orientation perturbation index; `0` if not varied. |
| `source_id` | Source/config identifier. |
| `source_mode` | `custom_diffraction_on`, `custom_diffraction_off`, or future explicit mode. |
| `source_energy_kev` | Source photon energy used to compute wavelength. |
| `source_wavelength_a` | Derived wavelength in Angstrom. |
| `peak_table_id` | Provenance table id, never an implicit hardcoded constant. |
| `bin_axis` | `q_a_inv` or `d_a`. |
| `q_bin_center_a_inv` | Canonical q-bin center when `bin_axis=q_a_inv`. |
| `d_bin_center_a` | Derived d-spacing for the same bin. |
| `detector_sector` | Stable detector sector label derived from detector id/geometry. |
| `detector_id_source` | Original Geant4 `detector_id` source column or aggregation rule. |
| `hit_count` | Count of hits contributing to the bin. |
| `primary_hit_count` | Primary-hit count contributing to the bin. |
| `sidecar_intensity_raw` | Raw bin response before normalization. |
| `sidecar_intensity_norm` | Baseline-corrected/normalized bin response. |
| `background_level_effective` | Estimated background for the sample/sector/bin family. |
| `throughput` | Detector/source throughput factor or `not_modelled`. |
| `detector_resolution_deg` | Detector angular resolution assumption. |
| `angular_bin_width_deg` | Angular bin width used for aggregation. |
| `absorption_factor` | Thickness absorption/self-absorption factor or `not_modelled`. |

## 5. Feature-table schema

Feature rows must keep controls separate from diffraction features.

Required groups:

- `diffraction_peak_*`: peak-window intensities, ratios, and q/d-local summaries from non-overlap and overlap windows.
- `control_total_count_*`: total count and hit-rate controls.
- `control_thickness_*`: explicit thickness fields used only for audit/control models.
- `control_overlap_only_*`: features restricted to overlapping H/M peak windows.
- `lineage_*`: identifiers needed for split, seed, source, pose, and provenance audit.

No gate can pass unless the main model outperforms the control-only and overlap-only models under the preregistered thresholds.

## 6. Peak-table provenance

The current v8A peak anchors are approximate project-scan anchors from `analysis/v8a_diffraction_observability.py`. They are useful only for development smokes.

The current manifest is:

- `source_models/config/diffraction_peak_tables/hm_powder_peaks_project_scan_v8a_manifest.json`

Before any full matrix, the project must replace or upgrade this manifest with one of:

- CIF-derived peak table with documented code path.
- Literature/Rietveld table with citation, phase, wavelength, and intensity normalization.
- Measured reference table from the intended instrument concept.

Minimum provenance fields:

- `peak_table_id`
- `material`
- `phase_name`
- `chemical_formula`
- `reference_type`
- `reference_citation`
- `reference_url_or_doi`
- `wavelength_a`
- `source_energy_kev_if_applicable`
- `two_theta_deg`
- `q_a_inv`
- `d_a`
- `relative_intensity`
- `intensity_normalization`
- `known_limitations`

## 7. Confounders and controls

The next implementation package must explicitly track these confounders:

- Mixtures and impurities: excluded from the first schema gate unless encoded in manifest as excluded.
- Grain size and peak broadening: must be parameterized before larger matrix expansion.
- Texture/preferred orientation: must be perturbed or documented as excluded.
- Detector resolution and angular bin width: required manifest fields.
- Fluorescence/background: required stress/control fields, not silently ignored.
- Throughput and absorption: must be either modelled or explicitly marked `not_modelled`.
- Source-off leakage: mandatory.
- Total-count-only, overlap-only, shuffled-label, and thickness/pose stress controls: mandatory.

## 8. Gate for the next implementation package

Proceed from this review to a tiny H/M event-to-feature pipeline only if:

- The schema contract is present and passes JSON validation.
- Each planned feature can be traced to a recorded source/config/event/hit/manifest field.
- The peak-table manifest is present and explicitly labels the current anchors as development-only.
- The implementation plan includes leakage-off, total-count-only, overlap-only, shuffled-label, and thickness/pose stress controls.
- No shadow/final data is referenced.

Proceed from the tiny H/M event-to-feature pipeline to a larger development matrix only if:

- Source-on signal remains clearly above leakage-off at the new schema boundary.
- Leakage-off H/M separability remains near zero.
- Main H/M min recall is at least `0.80`.
- Worst-thickness H/M min recall is at least `0.78`.
- Total-count-only and overlap-only H/M min recall remain below `0.75`.
- Shuffled-label H/M min recall remains below `0.65`.
- The team can state the claim as diffraction-aware sidecar observability, not ordinary-XRT H/M sorting.

Stop or revise if:

- Features depend on hidden label leakage from source construction.
- Source-off rows become separable.
- Fixed cross-energy `2theta` is used as the canonical axis.
- The only successful model uses total counts, thickness, or overlapping peaks.
- Peak provenance remains undocumented.

## 9. Phase-space/table source versus C++ custom process

Default next step: stay with phase-space/table source while the event-to-feature schema is being hardened. This is lower risk and keeps the source of diffraction information explicit.

Promote to C++ custom diffraction process only if:

- The sidecar schema passes the tiny H/M event-to-feature gate.
- A reviewer can audit the peak-table provenance.
- The team needs event-local transport interactions that phase-space/table source cannot represent.
- The added C++ process will record enough lineage to avoid hidden leakage.

Do not promote to C++ only to make the method sound more physical. The promotion must add traceable modelling value.

## 10. Immediate implementation checklist

1. Validate `analysis/configs/v8a_diffraction_output_schema_contract.json`.
2. Validate `source_models/config/diffraction_peak_tables/hm_powder_peaks_project_scan_v8a_manifest.json`.
3. Add the event-to-feature script only after the schema contract is stable.
4. Run only a tiny development gate after source-on/off, control, and manifest checks are implemented.
5. Keep all generated results untracked unless explicitly packaged.

## 11. Tiny event-to-feature implementation

The tiny event-to-feature implementation is:

- `analysis/v8a_event_to_feature_pipeline.py`

It reads only completed `v8a_custom_diffraction_g4_smoke` boundary-smoke rows and does not start Geant4. It converts recorded `*_events.csv`, `*_hits.csv`, and metadata into:

- `v8a_event_sidecar_long.csv`
- `v8a_event_sidecar_features.csv`
- `v8a_event_feature_manifest.json`
- `v8a_event_control_audit.csv`
- `v8a_event_schema_gate.json`
- `v8a_event_schema_gate_report.md`

Default command:

```bash
python3 analysis/v8a_event_to_feature_pipeline.py \
  --project-root . \
  --profile v8a_custom_diffraction_g4_smoke \
  --schema-contract analysis/configs/v8a_diffraction_output_schema_contract.json \
  --peak-manifest source_models/config/diffraction_peak_tables/hm_powder_peaks_project_scan_v8a_manifest.json \
  --output-dir results/accuracy_v3/v8a_event_to_feature_smoke \
  --overwrite
```

Current tiny result interpretation:

- The schema/control gate may pass if source-on signal is above source-off and lineage/contract checks are clean.
- The training gate must remain blocked when only the original 12 boundary-smoke rows are available, because those rows are train-only and do not provide balanced source-on/off H/M validation support.
- This is intentional. It prevents the boundary-smoke conversion from becoming an accuracy claim.
