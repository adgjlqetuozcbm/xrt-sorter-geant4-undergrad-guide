# 文件地图

这份文件说明仓库里每个主要目录在证据链中的位置。读代码时建议先按“输入配置 -> Geant4 输出 -> Python 分析 -> 结果证据 -> 文档表达”的顺序理解，而不是只按文件夹字母顺序浏览。

## 1. 总览

```text
include/              Geant4 C++ 头文件
src/                  Geant4 C++ 实现
source_models/        材料目录、源项、能谱和批量配置
analysis/             Python 分析脚本和 Geant4 run 宏
results/              公开验证证据包
figures/              论文和 README 使用的图
docs/                 队友入门、运行说明、文件地图、术语表
paper/                论文式正文、复现说明、参考文献
```

## 2. 输入配置层

`source_models/materials/material_catalog.csv` 是当前公开验证的材料索引。它定义材料名、化学式、密度、类别、吸收组标签、配置文件、事件文件和证据状态。Python 脚本会读取这个目录，因此它是“哪些材料进入公开验证”的事实来源。

`source_models/config/undergrad_batch/*.txt` 是每种材料的 Geant4 配置。配置文件指定材料、样本厚度、源项、探测器位置和输出前缀。当前公开验证使用十种单一材料、10 mm slab 和 W 靶 120 kV 能谱。

`source_models/spectra/w_target_120kV_1mmAl.csv` 是 X 射线源项能谱。Geant4 程序按该能谱采样 photon energy，而不是使用单一固定能量。

## 3. Geant4 仿真层

`include/` 和 `src/` 包含 C++ 仿真程序。你可以按下面的角色理解核心文件：

| 组件 | 作用 |
| --- | --- |
| `DetectorConstruction` | 定义世界体、矿物样本、探测器几何和材料 |
| `PrimaryGeneratorAction` | 按配置文件生成 gamma 源项 |
| `SteppingAction` | 记录进入探测器的 gamma 命中、能量、位置、偏转角和 primary 标记 |
| `EventAction` | 每个事件结束时汇总探测器响应 |
| `RunAction` | 管理 CSV、metadata 和 run 级统计输出 |

Geant4 运行后会在 `build/` 下生成 `*_events.csv`、`*_hits.csv` 和 metadata。`build/` 是本机生成目录，不提交 Git；仓库提交的是可复现配置、脚本和紧凑结果证据。

## 4. Python 分析层

`analysis/configs/run_research.mac` 控制 Geant4 run 的宏命令，当前核心命令为 `/run/beamOn 5000`。

`analysis/classify_absorption_groups.py` 是公开验证的核心分析脚本。它完成材料目录读取、事件 CSV 检查、事件聚合、样本级特征计算、训练/测试拆分、阈值法、Logistic Regression、accuracy、confusion matrix 和 manifest 输出。

`analysis/generate_material_sorting_matrix.py`、`analysis/run_material_sorting_matrix.py` 和 `analysis/material_sorting.py` 是材料级分选诊断线。它们用于生成 270-run pilot/full 矩阵、分批运行 Geant4，并评价十材料标签预测、top-3 候选、复核门控和开放集表现。当前结果是未通过验收的诊断证据，不是材料级成功结论。

脚本当前使用的主要特征是：

| 特征 | 来源 | 含义 |
| --- | --- | --- |
| `primary_transmission_rate` | `primary_gamma_entries` / event 数 | 主 gamma 透射率 |
| `mean_detector_edep_keV` | `detector_edep_keV` 平均值 | 平均探测器能量沉积 |
| `detector_gamma_rate` | `detector_gamma_entries` / event 数 | 探测器 gamma 命中率 |

## 5. 结果证据层

`results/undergrad_validation/` 是当前公开结果的核心证据包。

| 文件 | 作用 |
| --- | --- |
| `event_row_summary.csv` | 检查每种材料事件行数、重复事件和尾部丢弃事件 |
| `absorption_group_virtual_samples.csv` | 每 100 个 event 聚合后的 500 个虚拟样本 |
| `train_test_split_samples.csv` | 每个虚拟样本属于训练集还是测试集 |
| `feature_group_summary.csv` | 低/高吸收组在训练/测试中的特征分布 |
| `material_feature_summary.csv` | 每种材料的特征统计 |
| `absorption_group_classification_summary.csv` | 三种方法的测试集 accuracy |
| `absorption_group_confusion_*.csv` | 混淆矩阵 |
| `test_predictions.csv` | 每个测试样本的预测结果 |
| `validation_manifest.json` | 材料、样本政策、软件版本、结论边界 |

如果论文或 README 中的数字和这些文件不一致，以 `results/undergrad_validation/` 为准。

`results/material_sorting/` 是材料级诊断证据包。当前主方法 top-1 accuracy 为 `0.464`，top-3 accuracy 为 `0.876`，所有材料级验收条件均未通过；因此它用于说明下一阶段问题，而不是替代 `results/undergrad_validation/` 的二分类主结论。

## 6. 文档表达层

`README.md` 是仓库入口，负责用最短路径说明项目是什么、如何读、结果是什么、边界是什么。

`docs/TEAM_GUIDE_zh.md` 写给没有参与过项目的队友，重点解释数据从哪里来、如何清洗、如何构造特征、模型是什么、accuracy 怎么来、哪些话不能说。

`docs/RUN_LOCALLY_zh.md` 写给要复现的人，重点说明环境、构建、十材料运行、Python 分析和结果检查。

`docs/GLOSSARY_BY_FIRST_APPEARANCE.md` 解释术语，避免组员因为 Geant4、XRT、event、primary gamma、Logistic Regression 等词卡住。
`docs/MATERIAL_LEVEL_SORTING_DIAGNOSTIC_zh.md` 记录十材料物种级诊断为什么未通过，以及后续应如何继续。

`docs/MATERIAL_SORTING_V2_PROTOCOL_zh.md` 记录十材料 v2 的 air-path 校准、物化指纹、字典底座和 seed 留出评估协议。

`docs/public_explainer_zh.md` 是最短的通俗讲解，适合发给只想快速知道项目做了什么的人。

`docs/FINAL_ELEMENTARY_REVIEW_zh.md` 是最终边界复核，适合在答辩或交接前检查哪些表述可以说、哪些表述不能说。

`paper/main_thesis_HIT_revised_zh.md` 是论文式正文，适合导师审阅和答辩准备。它比 README 更正式，比队友指南更像完整论文。

## 7. 新增材料时要改哪些地方

新增材料不是只改一个 CSV。最小流程是：

1. 在 C++ 中确认或新增 Geant4 材料定义。
2. 在 `source_models/config/undergrad_batch/` 新建材料配置。
3. 在 `source_models/materials/material_catalog.csv` 添加材料行。
4. 运行 Geant4 生成该材料事件 CSV。
5. 运行 `analysis/classify_absorption_groups.py` 重新生成证据包。
6. 同步更新 README、论文、队友指南和运行说明中的材料范围与结果数字。

只有新的证据包生成并检查通过后，才能更新对该材料的结论。
## 8. v8A diffraction schema/provenance review

The v8A diffraction-aware branch has a pre-matrix review layer:

- `docs/ACCURACY_SPRINT_V8A_SCHEMA_PROVENANCE_REVIEW_zh.md`
- `analysis/configs/v8a_diffraction_output_schema_contract.json`
- `source_models/config/diffraction_peak_tables/hm_powder_peaks_project_scan_v8a_manifest.json`

These files define the Geant4 event/hit/source to `q/d` sidecar contract, the
development-only H/M peak-table boundary, and the leakage/control/no-go checks
that must pass before any larger v8A matrix.

The tiny implementation that applies this contract to the completed boundary
smoke is:

- `analysis/v8a_event_to_feature_pipeline.py`

Its generated output defaults to `results/accuracy_v3/v8a_event_to_feature_smoke/`
and remains an untracked development evidence directory unless explicitly
packaged later.

After the completed 90-row v8A boundary-smoke matrix, the tiny
training/control gate is:

- `analysis/train_v8a_event_feature_smoke.py`

Its generated output defaults to
`results/accuracy_v3/v8a_event_training_smoke/` and remains an untracked
development evidence directory unless explicitly packaged later. The current
passing decision is development-only and only permits v8A balanced development
design review.

The parallel peak-provenance review plan is:

- `docs/ACCURACY_SPRINT_V8A_PEAK_PROVENANCE_UPGRADE_PLAN_zh.md`

The stricter balanced development design review is:

- `docs/ACCURACY_SPRINT_V8A_BALANCED_DEV_DESIGN_REVIEW_zh.md`

Its source-controlled audit/gate files are:

- `source_models/config/diffraction_peak_tables/hm_powder_peaks_cif_or_literature_v8a_manifest.json`
- `analysis/audit_v8a_peak_provenance.py`
- `analysis/configs/v8a_event_feature_stress_gate_config.json`
- `analysis/v8a_event_feature_stress_gate.py`
- `analysis/diagnose_v8a_overlap_control.py`
- `analysis/configs/v8a_medium_development_matrix_config.json`
- `analysis/generate_v8a_medium_development_matrix.py`
- `analysis/audit_v8a_medium_development_prereg.py`

The stress gate uses fixed baseline training with stressed validation features.

After the medium development matrix is executed, the Phase 4 development-only
model training/calibration gate is:

- `analysis/train_v8a_medium_development_model.py`

Its generated output defaults to
`results/accuracy_v3/v8a_medium_development_model/`. It performs Logistic and
ExtraTrees main-model checks, threshold sweeps, calibration bins, and
by-thickness/by-pose/by-stress-label recall summaries while keeping shadow/final
sealed.

The total-count confounder diagnostic for a stopped Phase 4 gate is:

- `analysis/diagnose_v8a_total_count_control.py`

Its generated output defaults to
`results/accuracy_v3/v8a_medium_total_count_control_diagnostic/`.

The count-matched total-count rework gate is:

- `analysis/train_v8a_medium_count_matched_rework.py`

Its generated output defaults to
`results/accuracy_v3/v8a_medium_count_matched_rework/`. It evaluates a
development-only total-count-matched subset and residualized sensitivity checks.

The follow-up count-balance sensitivity audit is:

- `analysis/audit_v8a_count_balance_sensitivity.py`

Its generated output defaults to
`results/accuracy_v3/v8a_count_balance_sensitivity/`.

The count-overlap extension preregistration package is:

- `analysis/configs/v8a_count_overlap_extension_config.json`
- `analysis/generate_v8a_count_overlap_extension_matrix.py`
- `analysis/audit_v8a_count_overlap_extension_prereg.py`
- `analysis/combine_v8a_medium_count_overlap_features.py`
- `analysis/configs/v8a_medium_plus_count_overlap_stress_gate_config.json`
- `analysis/train_v8a_count_balanced_retest.py`

Its generated profile is
`source_models/config/material_sorting_matrix/v8a_hm_count_overlap_extension_cif_literature/`.
It is development-only, source-on only, and does not unlock training by itself.
After the extension rows are run, the combiner creates a controlled
medium-plus-extension feature directory that keeps the existing medium
source-off rows as leakage controls before any renewed stress/count gates run.
If count-balance sensitivity identifies a supported strategy, the retest script
runs the selected strict count-balanced development gate while keeping
shadow/final sealed.

The current count-overlap/count-balanced stage report is:

- `docs/ACCURACY_SPRINT_V8A_COUNT_BALANCED_STAGE_REPORT_zh.md`

It records the accepted development-only positive result, the ordinary Phase 4
total-count blocker, the no-go boundary for shadow/final and full v8A matrix,
and the next feature-rework plan.

The follow-up count-robust v1 rework and development-matrix admission report is:

- `docs/ACCURACY_SPRINT_V8A_COUNT_ROBUST_REWORK_REPORT_zh.md`

Its source-controlled scripts/config are:

- `analysis/diagnose_v8a_total_count_anatomy.py`
- `analysis/build_v8a_count_robust_features.py`
- `analysis/audit_v8a_count_robust_stability.py`
- `analysis/audit_v8a_development_matrix_admission.py`
- `analysis/configs/v8a_count_robust_v1_stress_gate_config.json`

The v1 result keeps the H/M sidecar signal visible but does not unlock large
development matrix preregistration because the ordinary Phase 4 total-count
control and count-balanced shuffled-label sanity checks still block promotion.

The follow-up count-robust v2 stop report is:

- `docs/ACCURACY_SPRINT_V8A_COUNT_ROBUST_V2_STOP_REPORT_zh.md`

Its source-controlled scripts/config are:

- `analysis/build_v8a_count_robust_v2_features.py`
- `analysis/build_v8a_count_stratified_ordinary_view.py`
- `analysis/audit_v8a_multiseed_shuffled_label.py`
- `analysis/audit_v8a_count_robust_v2_candidates.py`
- `analysis/configs/v8a_count_robust_v2_proportion_only_count_stratified_stress_gate_config.json`
- `analysis/configs/v8a_count_robust_v2_residualized_no_absolute_windows_count_stratified_stress_gate_config.json`

The v2 result is a clean stop before Phase 4: count-stratified support and
main H/M signal remain strong, but stress and multi-seed shuffled-label sanity
fail. Therefore Phase 4, count-balanced retest, stability replication, and
development matrix large-run preregistration remain locked.

The follow-up v8A root-cause decision report is:

- `docs/ACCURACY_SPRINT_V8A_ROOT_CAUSE_DECISION_REPORT_zh.md`

Its source-controlled diagnostic scripts are:

- `analysis/diagnose_v8a_shuffled_label_null_behavior.py`
- `analysis/audit_v8a_feature_shortcut_structure.py`
- `analysis/audit_v8a_stress_null_path.py`
- `analysis/probe_v8a_feature_sufficiency_models.py`
- `analysis/decide_v8a_root_cause.py`

The current root-cause decision is `sampling_or_origin_shortcut_found`: v2
main features can predict non-material `stress_label` structure, so advanced
model probes are blocked until null/stress/origin shortcuts are cleaned.
