# Accuracy Sprint v7B2: H/M physics-matrix Pilot preregistration

Date: 2026-05-03

## 1. Scientific position

v7B2 is triggered by the failed v7B R2/R3 decision gate. v7B already improved coarse ten-material development sorting, but Hematite/Magnetite remained the hard scientific blocker:

- v7B R1 selected `ExtraTrees`, H/M min recall `0.6083`.
- v7B R2/R3 selected `R2ExtraTreesTransmissionHighEnergy`, H/M min recall `0.6083`.
- R2 improvement over R1 was `0.0`.
- H/M top-3 containment was high, but view selection and pairwise experts did not split the pair.

Therefore v7B2 is not another model-shopping round on the same validation evidence. It is a preregistered physical-observation redesign for H/M separability.

This remains a development-only phase. Shadow/final seeds must not be generated, exported, trained, or used for any decision.

## 2. Locked Pilot matrix

Profile: `v7b2_hm_physics_dev`

- materials: `Hematite, Magnetite`
- thickness: `3, 5, 8, 10, 15, 20, 30, 40 mm`
- energies: `50, 70, 90, 120, 150, 200 keV`
- source variants: `normal_narrow, normal_wide, oblique_20deg, oblique_30deg, oblique_40deg`
- train seeds: `5101-5104`
- validation seeds: `5201-5202`
- reserved shadow seeds: `5301-5302`, recorded only; not generated or used in v7B2 Pilot
- events/run: `40000`
- measurement aggregation: `photon_budget=5000`, giving 8 complete samples/run

Expected full Pilot rows:

- material rows: `2 x 8 x 30 x 6 = 2880`
- calibration rows: `30 x 6 = 180`
- total rows: `3060`

## 3. New code surfaces

- `analysis/generate_material_sorting_matrix.py`
  - adds profile `v7b2_hm_physics_dev`
  - adds `oblique_30deg` and `oblique_40deg`
  - supports explicit source-id selection for v7B2 full reruns
- `analysis/configs/run_material_sorting_v7b2.mac`
  - runs `40000` events per row
- `analysis/run_material_sorting_matrix.py`
  - maps `v7b2_*` profiles to the v7B2 macro
- `analysis/export_measurement_cube_v7b2.py`
  - exports H/M-only v7B2 measurement cubes
  - refuses shadow/final leakage unless explicitly overridden
  - exports ten-material full v7B2 only with `--full-materials` after a passing Pilot gate, and restricts sources to the Pilot-selected source list
- `analysis/export_physics_features_v7b2.py`
  - writes physical-response feature tables
  - includes path-length normalized attenuation, energy response slope, angle sensitivity, thickness response, scatter/direct reliability, and H/M contrast tables
- `analysis/v7b2_pilot_signal_gate.py`
  - runs the preregistered Pilot signal gate
  - writes validation decisions, view/source ranking, gate JSON, and limitation note if the stop rule is hit
- `analysis/train_v7b2.py`
  - full v7B2 training wrapper
  - refuses to run unless the Pilot gate passes, except tiny smoke wiring checks with `--allow-without-pilot`

## 4. Pilot signal gate

The Pilot gate passes only if all preregistered checks hold:

- H/M min recall `>=0.70`
- improvement over v7B baseline `>=0.08`
- shadow/final used `false`

Decision rules:

- `hm_min_recall >= 0.70` and improvement `>=0.08`: proceed to full v7B2.
- `hm_min_recall < 0.68`: stop physical expansion and write a limitation note.
- `0.68 <= hm_min_recall < 0.70`: allow only one small v7C feasibility sidecar.

The Pilot model is a fixed `ExtraTreesClassifier` over the exported physical-response features. It is a signal test, not a publishable final model.

## 5. Full v7B2 rule

Full v7B2 can run only after the Pilot gate passes.

The source set must be selected from `v7b2_pilot_source_ranking.csv`, capped at 18 source ids. The full matrix may expand back to the ten v7B materials only after that source cap is applied.

Full gate:

- H/M min recall `>=0.80`
- H/M pairwise min recall `>=0.78`
- ten-material macro-F1 `>=0.84`
- validation support/class `>=120`
- runner failures `0`
- runner pending `0`
- shadow/final used `false`

If full v7B2 fails, do not run shadow/final. Record the failure and return to physics/design analysis.

## 6. Commands

Generate Pilot smoke matrix:

```bash
python3 analysis/generate_material_sorting_matrix.py \
  --profile v7b2_hm_physics_dev \
  --profile-alias v7b2_hm_physics_dev_smoke \
  --material-list Hematite,Magnetite \
  --energy-list-kev 50,200 \
  --thickness-list 3 \
  --source-variant-list normal_narrow,oblique_30deg,oblique_40deg \
  --seed-list 5101,5201 \
  --events-per-run 40000
```

Generate full Pilot matrix:

```bash
python3 analysis/generate_material_sorting_matrix.py --profile v7b2_hm_physics_dev
```

Run matrix:

```bash
python3 analysis/run_material_sorting_matrix.py --profile v7b2_hm_physics_dev --status-only
```

Export Pilot cube:

```powershell
$py = 'C:\Users\m1516\AppData\Local\Programs\Python\Python39\python.exe'
$root = '\\?\UNC\wsl.localhost\Ubuntu-22.04\home\dyd\geant4-projects\xrt_sorter\release\xrt_sorter_public_undergrad_repo_20260426'
& $py "$root\analysis\export_measurement_cube_v7b2.py" `
  --project-root $root `
  --raw-dir build/material_sorting_runs/v7b2_hm_physics_dev `
  --output-dir results/accuracy_v3/v7b2_hm_physics_dev
```

Export physical features and run Pilot gate:

```powershell
& $py "$root\analysis\export_physics_features_v7b2.py" `
  --project-root $root `
  --cube-dir results/accuracy_v3/v7b2_hm_physics_dev

& $py "$root\analysis\v7b2_pilot_signal_gate.py" `
  --project-root $root `
  --feature-dir results/accuracy_v3/v7b2_hm_physics_dev
```

Full v7B2 generation after Pilot pass:

```bash
python3 analysis/generate_material_sorting_matrix.py \
  --profile v7b2_hm_physics_dev \
  --profile-alias v7b2_full_dev \
  --material-list all \
  --selected-source-ids <comma-separated source ids from v7b2_pilot_source_ranking.csv> \
  --seed-list 5101,5102,5103,5104,5201,5202 \
  --events-per-run 40000
```

Full v7B2 cube export after Pilot pass:

```powershell
& $py "$root\analysis\export_measurement_cube_v7b2.py" `
  --project-root $root `
  --raw-dir build/material_sorting_runs/v7b2_full_dev `
  --output-dir results/accuracy_v3/v7b2_full_dev `
  --full-materials `
  --pilot-gate results/accuracy_v3/v7b2_hm_physics_dev/v7b2_pilot_gate.json
```

Full training after Pilot pass:

```powershell
& $py "$root\analysis\train_v7b2.py" `
  --project-root $root `
  --cube-dir results/accuracy_v3/v7b2_full_dev `
  --pilot-gate results/accuracy_v3/v7b2_hm_physics_dev/v7b2_pilot_gate.json
```

## 7. Claim discipline

Do not claim robust H/M separation until development gates and sealed evaluations pass.

Do not claim that traditional classifiers failed globally. The correct manuscript narrative is that traditional classifiers worked for coarse sorting and exposed a fine-grained H/M boundary in the current virtual XRT observation design.

Do not claim XRT cannot separate H/M in general. Failed Pilot evidence means only that this simulated physical matrix did not create enough H/M contrast.
