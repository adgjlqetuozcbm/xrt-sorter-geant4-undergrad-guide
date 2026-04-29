# Accuracy Sprint v7: measurement-cube foundation sorter roadmap

Date: 2026-04-30

## 1. Paper-facing narrative

The original route followed the advisor-recommended simple and traditional classification path. That route remains scientifically useful: the traditional baselines are interpretable, inexpensive, and appropriate for coarse virtual sorting or materials with clearly separated XRT responses.

The v6/v6b/v6c evidence shows a narrower limitation rather than a global failure. Hematite and Magnetite remain mutually confused after multiple source, feature, and model iterations. The observed H/M development min recall plateau stayed around `0.6333-0.65`, including v6c after high-energy source variants and side-scatter features.

v7 is therefore a problem-driven escalation. The project does not abandon the original plan; it uses the traditional route as a baseline, identifies a fine-grained oxide bottleneck, and upgrades the representation/modeling stack for physically similar materials.

## 2. Staged roadmap

### v7A: reuse v6c development data

Purpose: test whether richer input representation can break the H/M plateau without rerunning Geant4.

- Input: existing v6c development `hits/metadata`.
- Representation: measurement cube with energy/source/detector/spatial/channel axes.
- Models: flattened cube + ExtraTrees/XGBoost/PCA/hard-negative variants.
- Gate: H/M min recall `>=0.75`, pairwise H/M min recall `>=0.72`, both H/M class recalls `>=0.72`, validation support/class `>=60`.
- Repeat rule: if the gate fails, repeat hard-negative training up to 3 rounds. If the third round still fails, stop v7A and move to v7B rather than tuning indefinitely.

### v7B: new hard-negative Geant4 matrix

Purpose: create new physical evidence for hard pairs if v7A cannot extract enough signal from v6c.

- Increase photon budget, seeds, incidence angles, and scatter detector geometry.
- Keep H/M as the priority pair.
- Add the most confusable ten-material pairs.
- Gate: H/M min recall `>=0.80`, initial ten-material macro-F1 `>=0.80`.

### v7C: high-capacity representation learning

Purpose: use models that can exploit the full measurement cube.

- Configure PyTorch.
- Train multi-view transformer, contrastive encoder, and prototype retrieval.
- Keep measurement-cube input; do not regress to only scalar tabular features.
- Gate: H/M min recall `>=0.85`, hard-negative pairwise min recall `>=0.80`.

### v7D: ten-to-hundreds architecture

Purpose: make the method scalable beyond the current ten-material set.

- Build hierarchical classifier: coarse material family -> family expert -> top-K reranker.
- Add open-set and uncertainty handling so low-confidence samples can be flagged for repeat measurement or review.
- Gate: ten-material locked development macro-F1 `>=0.88`, key hard-pair min recall `>=0.82`.

### v7E: shadow/final and manuscript bundle

Purpose: convert the development work into claim-safe evidence.

- Run shadow only after development gates pass.
- If shadow fails, record the failure and do not tune on the same shadow.
- Final seeds remain sealed until the last evaluation.
- Paper outputs: traditional baseline, H/M plateau evidence, v7 escalation rationale, final model results, limitations.

## 3. v7A implementation

New scripts:

- `analysis/export_measurement_cube_v7a.py`
- `analysis/train_hm_v7a.py`

Smoke export:

```powershell
$py = 'C:\Users\m1516\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$root = '\\?\UNC\wsl.localhost\Ubuntu-22.04\home\dyd\geant4-projects\xrt_sorter\release\xrt_sorter_public_undergrad_repo_20260426'
& $py "$root\analysis\export_measurement_cube_v7a.py" `
  --project-root $root `
  --raw-dir build/material_sorting_runs/v6c_hm_source_design `
  --output-dir results/accuracy_v3/v7a_hm_measurement_cube_smoke `
  --source-ids mono_50kev_normal_narrow,mono_70kev_normal_narrow `
  --seeds 2101,2201 `
  --thicknesses 5 `
  --write-feature-csv
& $py "$root\analysis\train_hm_v7a.py" `
  --project-root $root `
  --cube-dir results/accuracy_v3/v7a_hm_measurement_cube_smoke `
  --methods ExtraTrees `
  --repeat-rounds 1
```

Full v7A development export/training:

```powershell
$py = 'C:\Users\m1516\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$root = '\\?\UNC\wsl.localhost\Ubuntu-22.04\home\dyd\geant4-projects\xrt_sorter\release\xrt_sorter_public_undergrad_repo_20260426'
& $py "$root\analysis\export_measurement_cube_v7a.py" `
  --project-root $root `
  --raw-dir build/material_sorting_runs/v6c_hm_source_design `
  --output-dir results/accuracy_v3/v7a_hm_measurement_cube
& $py "$root\analysis\train_hm_v7a.py" `
  --project-root $root `
  --cube-dir results/accuracy_v3/v7a_hm_measurement_cube
```

## 4. Output discipline

Every v7 stage should write:

- gate JSON
- failure analysis
- per-class recall
- pairwise hard-negative audit
- view/source ablation where applicable
- Chinese phase report
- memory update checkpoint

Do not use shadow or final seeds in v7A. Do not make a paper-facing accuracy claim from development-only results.
