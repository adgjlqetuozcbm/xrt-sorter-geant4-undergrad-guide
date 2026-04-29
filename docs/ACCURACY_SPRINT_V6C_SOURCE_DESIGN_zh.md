# Accuracy Sprint v6c: H/M source-design recovery note

Date: 2026-04-29

## 1. Positioning

v6c is a physical-information sprint for the Hematite/Magnetite blocker. It is not a GPU model-search sprint.

The current v6/v6b evidence shows that model and feature variants on the same transmission-only basis keep H/M validation recall in the failing range. v6c therefore changes the virtual measurement design before any shadow or final evaluation:

- materials: Hematite, Magnetite
- energies: 50, 70, 90, 120, 150, 200 keV
- thicknesses: 5, 10, 15, 20, 30 mm
- source variants: normal_narrow, normal_wide, oblique_10deg
- train seeds: 2101-2112
- validation seeds: 2201-2206
- shadow seeds: 2301-2306, reserved until the development gate passes

## 2. Implemented changes

Geant4/config changes:

- `source_variant`, `detector_layout`, `incidence_angle_deg`, beam size, and side-detector geometry are now written into generated configs and metadata.
- `detector_layout=transmission_plus_side_scatter` adds a side detector at positive Y while keeping the transmission detector.
- `hits.csv` now includes `detector_id` and `x_mm`; old hits without these columns remain compatible in the Python aggregator.
- Transmission spatial features continue to use Y/Z. Side-scatter radius now uses X/Z, because the side detector plane is fixed in Y.

Matrix/runner changes:

- `analysis/generate_material_sorting_matrix.py` supports the `v6c_hm_source_design` profile plus source-variant and thickness overrides.
- `analysis/run_material_sorting_matrix.py` maps `v6c_*` profiles to the v6c macro by default.
- `analysis/configs/run_material_sorting_v6c.mac` uses 10000 events.
- `analysis/configs/run_material_sorting_v6c_smoke.mac` uses 2000 events for schema smoke checks.

Feature/audit changes:

- `analysis/material_sorting_v2.py` adds side-scatter rates, side energy bins, scatter/transmission ratios, transmission spatial moments, and dual-energy features.
- `analysis/hm_v6c_development_audit.py` writes thickness-aware and thickness-blind H/M metrics, model selection, feature/source ablations, split audit, registry, decisions, and manifest.
- `analysis/accuracy_v6c_gate.py` enforces the v6c development gate.

## 3. Verification completed

Build and smoke checks completed on 2026-04-29:

- C++ build passed with `cmake --build build -j2`.
- Python compile passed for the changed analysis scripts.
- Smoke matrix generated at `source_models/config/material_sorting_matrix/v6c_hm_source_design_smoke/`.
- Smoke runner completed `12/12` rows with `failed=0`.
- Smoke metadata includes `source_variant`, `detector_layout`, `incidence_angle_deg`, beam size, and side-detector geometry.
- Smoke hits include both `transmission` and `side_scatter` detector IDs.
- Smoke audit completed with Windows bundled Python and `--photon-budget 2000`.
- Smoke gate ran and failed as expected because support is only 1 per class; this is a schema/pipeline check, not a performance claim.

Full development matrix was generated:

- matrix: `source_models/config/material_sorting_matrix/v6c_hm_source_design/material_sorting_matrix.csv`
- total rows: 4752
- material rows: 4320
- calibration rows: 432
- current status: `completed=0 failed=0 pending=4752`

## 4. Gate rule

Development gate:

- thickness-aware H/M min recall >= 0.80
- thickness-blind H/M min recall >= 0.75
- H/M pairwise min recall >= 0.75
- validation support per class >= 120
- runner failures = 0

If the development gate fails, do not run shadow.

If the development gate passes, run the reserved shadow seeds once only. Do not tune on shadow. Final seeds remain sealed.

## 5. Next commands

Run the full development batch from the repo root after sourcing Geant4:

```bash
source /home/dyd/geant4-install/bin/geant4.sh
python3 analysis/run_material_sorting_matrix.py --profile v6c_hm_source_design
python3 analysis/run_material_sorting_matrix.py --profile v6c_hm_source_design --status-only
```

Run the development audit with a Python environment that has pandas/numpy/scikit-learn. The WSL system `python3` currently does not have these packages. The Codex bundled Windows Python worked when called with a long UNC path prefix.

```powershell
$py = 'C:\Users\m1516\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$root = '\\?\UNC\wsl.localhost\Ubuntu-22.04\home\dyd\geant4-projects\xrt_sorter\release\xrt_sorter_public_undergrad_repo_20260426'
& $py "$root\analysis\hm_v6c_development_audit.py" `
  --project-root $root `
  --raw-dir build/material_sorting_runs/v6c_hm_source_design `
  --output-dir results/accuracy_v3/v6c_hm_source_design
& $py "$root\analysis\accuracy_v6c_gate.py" `
  --project-root $root `
  --audit-dir results/accuracy_v3/v6c_hm_source_design `
  --status-profile v6c_hm_source_design `
  --output-json results/accuracy_v3/v6c_hm_source_design/gate_v6c.json
```

## 6. Interpretation discipline

As a paper-facing result, v6c is still preregistered development work. Passing smoke only means the upgraded virtual measurement pipeline runs. It does not support an H/M accuracy claim.

If v6c fails the gate, the correct conclusion is that this本科级 virtual XRT modality is still insufficient for robust H/M separation under the current simplified assumptions. The next paper-safe move would be a limitation/negative-result interpretation or a new preregistered modality, not wider model search.
