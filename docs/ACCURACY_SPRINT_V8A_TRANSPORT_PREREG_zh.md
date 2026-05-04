# Accuracy Sprint v8A: transport/detector sidecar preregistration

日期: 2026-05-04

## 1. Scientific position

本轮目标是 `v8A_transport_preregistration`: 把已经通过的 synthetic powder-peak observability 推进到一个可仿真、可审稿、可停损的 transport/detector 设计。它不是正式 H/M accuracy claim, 也不是 full hard-negative material matrix。

核心立场保持不变:

- diffraction sidecar 的物理来源必须是 tabulated powder pattern, CIF-derived reference table, 或 custom diffraction photon generator/process。
- Geant4 在下一阶段只承担 transport, absorption, background, detector response 和 geometry bookkeeping。
- 不得把 standard Geant4 Rayleigh scattering 写成 powder XRD。
- 不得把 `G4XrayReflection` 写成 bulk Hematite/Magnetite powder diffraction。
- ordinary XRT attenuation feature stacking 已经作为开发证据失败, 不能替代 diffraction sidecar。

v8A synthetic observability default/stress gate 均为 `proceed_to_v8a_transport_preregistration`, 但这只说明 tabulated powder-peak sidecar 值得继续, 不说明真实 transport、硬件 throughput 或 full sorter 已经成立。

## 2. Detector and geometry concept

下一阶段先做最小 detector-sidecar, 不直接修改 C++ Geant4 主程序:

- source: 窄能谱或准单色 beam, manifest 记录 `source_energy_kev`, `source_wavelength_a`, `source_bandwidth_fraction`。
- sample: H/M development-only synthetic powder pattern, 记录 `material`, `split`, `seed`, `thickness_mm`, `pose_index`。
- detector: sample 后方或侧向 angular detector bins, 以 `detector_sector` 表示不同角扇区。
- feature axis: 默认使用 `q_a_inv` 或 d-spacing。若有多波长/多 source energy, 禁止跨波长复用固定 `2theta` bins。
- output: `q_or_d_bin x detector_sector x pose x thickness` sidecar features, 同时保留 sample-level controls 作为 guardrail。
- manifest 必须记录 detector resolution, angular bin width, background model, throughput model, absorption model, peak table id, and whether any existing XRT cube was read。

最小 schema 字段:

- `material`
- `split`
- `random_seed`
- `thickness_mm`
- `pose_index`
- `source_id`
- `source_energy_kev`
- `source_wavelength_a`
- `peak_table_id`
- `bin_axis`
- `q_bin_center_a_inv`
- `d_bin_center_a`
- `detector_sector`
- `background_level_effective`
- `detector_resolution_deg`
- `angular_bin_width_deg`
- `throughput`
- `absorption_factor`
- `sidecar_intensity_norm`

## 3. Smoke prototype

Implementation:

- `analysis/v8a_transport_sidecar_smoke.py`

Default output:

- `results/accuracy_v3/v8a_transport_sidecar_smoke/`

Stress output:

- `results/accuracy_v3/v8a_transport_sidecar_smoke_stress/`

Expected outputs:

- `v8a_transport_sidecar_long.csv`
- `v8a_transport_sidecar_features.csv`
- `v8a_transport_observability_metrics.csv`
- `v8a_transport_model_selection.csv`
- `v8a_transport_validation_decisions.csv`
- `v8a_transport_sidecar_manifest.json`
- `v8a_transport_gate.json`
- `v8a_transport_gate_report.md`

Default command:

```bash
/home/dyd/geant4-projects/xrt_sorter/.venv/bin/python \
  analysis/v8a_transport_sidecar_smoke.py \
  --project-root /home/dyd/geant4-projects/xrt_sorter/release/xrt_sorter_public_undergrad_repo_20260426
```

Stress command:

```bash
/home/dyd/geant4-projects/xrt_sorter/.venv/bin/python \
  analysis/v8a_transport_sidecar_smoke.py \
  --project-root /home/dyd/geant4-projects/xrt_sorter/release/xrt_sorter_public_undergrad_repo_20260426 \
  --output-dir results/accuracy_v3/v8a_transport_sidecar_smoke_stress \
  --detector-resolution-deg 0.22 \
  --angular-bin-width-deg 0.30 \
  --source-bandwidth-fraction 0.008 \
  --intrinsic-q-sigma 0.055 \
  --q-calibration-jitter 0.030 \
  --orientation-sigma 0.95 \
  --background-level 0.12 \
  --background-slope-sigma 0.45 \
  --counts-scale 600 \
  --read-noise-sigma 0.009 \
  --absorption-strength 0.025
```

The default command is already a detector-realistic baseline rather than an ideal noiseless pattern. The stress command further widens angular bins, broadens q peaks, increases background/noise, lowers count scale, and strengthens absorption. The smoke is transport-like only: it simulates angular resolution, q-bin aggregation, thickness absorption, background, count noise, sector throughput, and sector aggregation. It does not read v7B/v7B2 cubes, shadow, or final artifacts. It is a preregistered gate before real Geant4/custom diffraction integration.

## 4. Gate

Proceed only if all checks pass:

- physical observability: best single non-overlap peak/ratio feature has oriented AUC `>=0.95` or absolute d-prime `>=3.0`。
- main H/M model: H/M min recall `>=0.80`。
- pairwise H/M min recall: `>=0.80`。
- worst-thickness H/M min recall: `>=0.78`。
- control-only H/M min recall `<0.75`。
- total-count-only H/M min recall `<0.75` and must not approach the main model。
- shuffled-label H/M min recall `<0.65`。
- overlap-only H/M min recall `<0.75`。
- manifest confirms `development_only=true`, `shadow_or_final_used=false`, `reads_existing_xrt_cubes=false`, `bin_axis=q_a_inv` or d-spacing。

Decision rules:

- Pass: `proceed_to_geant4_custom_diffraction_integration_design`。
- Gray-zone: `gray_zone_strengthen_detector_background_stress` when H/M is near threshold or depends too much on optimistic detector assumptions。
- Stop: `stop_transport_sidecar_write_limitation` when main H/M min recall `<0.75`, controls are too strong, or realistic detector bins remove unique peak signal。

## 5. No-go and claim discipline

Do not run:

- shadow/final。
- full v7B2。
- full v8A hard-negative material matrix。
- GPU/deep model as the primary next path。
- ordinary XRT feature stacking as a replacement for diffraction sidecar。

If smoke passes, the next step is a real integration design for custom diffraction/table sidecar plus Geant4 transport. If smoke is gray-zone, strengthen background, detector resolution, throughput, and overlap stress. If smoke stops, write v8A as a limitation and method discussion instead of implementing a larger matrix.

## 6. Integration smoke and small training gate

After the transport-like smoke passes, the next package is still Python-only and development-only. It introduces an explicit custom diffraction/table source switch so we can prove the sidecar signal disappears when the diffraction source is off.

Integration smoke commands:

```bash
/home/dyd/geant4-projects/xrt_sorter/.venv/bin/python \
  analysis/v8a_custom_diffraction_integration_smoke.py \
  --project-root /home/dyd/geant4-projects/xrt_sorter/release/xrt_sorter_public_undergrad_repo_20260426

/home/dyd/geant4-projects/xrt_sorter/.venv/bin/python \
  analysis/v8a_custom_diffraction_integration_smoke.py \
  --project-root /home/dyd/geant4-projects/xrt_sorter/release/xrt_sorter_public_undergrad_repo_20260426 \
  --output-dir results/accuracy_v3/v8a_custom_diffraction_integration_smoke_stress \
  --detector-resolution-deg 0.22 \
  --angular-bin-width-deg 0.30 \
  --source-bandwidth-fraction 0.008 \
  --intrinsic-q-sigma 0.055 \
  --q-calibration-jitter 0.030 \
  --orientation-sigma 0.95 \
  --background-level 0.12 \
  --background-slope-sigma 0.45 \
  --counts-scale 600 \
  --read-noise-sigma 0.009 \
  --absorption-strength 0.025

/home/dyd/geant4-projects/xrt_sorter/.venv/bin/python \
  analysis/v8a_custom_diffraction_integration_smoke.py \
  --project-root /home/dyd/geant4-projects/xrt_sorter/release/xrt_sorter_public_undergrad_repo_20260426 \
  --output-dir results/accuracy_v3/v8a_custom_diffraction_integration_smoke_leakage_off \
  --diffraction-source off
```

Small sidecar training gate:

```bash
/home/dyd/geant4-projects/xrt_sorter/.venv/bin/python \
  analysis/train_v8a_sidecar_smoke.py \
  --project-root /home/dyd/geant4-projects/xrt_sorter/release/xrt_sorter_public_undergrad_repo_20260426
```

The training script may run only after the source-on integration gate passes and the source-off leakage control passes. The leakage-off best H/M min recall must remain `<0.75` and at least `0.20` below the source-on main model.

## 7. Environment

Fixed Python:

- `/home/dyd/geant4-projects/xrt_sorter/.venv/bin/python`

Required checks:

```bash
/home/dyd/geant4-projects/xrt_sorter/.venv/bin/python -m py_compile analysis/v8a_transport_sidecar_smoke.py
git diff --check
```

Generated results are development artifacts and should stay untracked unless explicitly selected for evidence packaging.

## 8. Minimal Geant4 boundary smoke

After the Python-only integration/training smoke, the next step is a minimal
Geant4 boundary smoke, not a full matrix. The purpose is to check that a
custom/table diffraction source can enter the real Geant4 source, geometry, and
detector-output path while preserving a source-on/source-off leakage control.

Source generation:

```bash
/home/dyd/geant4-projects/xrt_sorter/.venv/bin/python \
  analysis/generate_v8a_diffraction_g4_smoke_matrix.py \
  --project-root /home/dyd/geant4-projects/xrt_sorter/release/xrt_sorter_public_undergrad_repo_20260426
```

Minimal runner:

```bash
cmake --build build -j2
LD_LIBRARY_PATH=/home/dyd/geant4-install/lib:${LD_LIBRARY_PATH} \
  /home/dyd/geant4-projects/xrt_sorter/.venv/bin/python \
  analysis/run_material_sorting_matrix.py \
  --profile v8a_custom_diffraction_g4_smoke \
  --limit 12 \
  --rerun-existing
```

Boundary audit:

```bash
/home/dyd/geant4-projects/xrt_sorter/.venv/bin/python \
  analysis/audit_v8a_diffraction_g4_smoke.py \
  --project-root /home/dyd/geant4-projects/xrt_sorter/release/xrt_sorter_public_undergrad_repo_20260426 \
  --min-completed-rows 12
```

This gate only checks that source-on rows produce high-angle primary detector
signal through the real Geant4 path, while source-off leakage rows do not. It is
not H/M accuracy evidence, not hardware validation, and not a reason to open
shadow/final.
