# Geant4/XRT 材料级智能分选成果索引

更新时间：2026-05-13

本文档是当前 Geant4/XRT 材料级智能分选研究的成果入口。原始结果目录不建议移动，因为已有脚本按固定路径读取；后续衍生研究应优先通过本文档定位数据、模型结果和诊断证据。

## 总入口

- WSL 主仓库：`/home/dyd/geant4-projects/xrt_sorter/release/xrt_sorter_public_undergrad_repo_20260426`
- Windows 镜像/中转路径：`C:\home\dyd\geant4-projects\xrt_sorter\release\xrt_sorter_public_undergrad_repo_20260426`
- 项目记忆：`C:\Users\m1516\Documents\New project 7\.codex-memory\projects\g4-xrt-material-sorting`

## 顶层目录结构

```text
xrt_sorter_public_undergrad_repo_20260426/
├─ docs/
│  ├─ G4_XRT_MATERIAL_SORTING_RESULTS_INDEX_zh.md   # 本索引
│  ├─ material_sorting_results_manifest/            # 机器可读成果清单
│  └─ ACCURACY_*/MATERIAL_*/HM_* 等历史中文记录
├─ analysis/
│  ├─ configs/                                      # v4-v8 实验配置
│  ├─ generate_v8a_multiclass_context_v*_matrix.py  # 矩阵生成脚本
│  ├─ audit_v8a_multiclass_context_v*_*.py          # 数据审查/捷径审查/诊断脚本
│  ├─ train_v8a_multiclass_context_v*_*.py          # 模型训练脚本
│  └─ probe_v8_*.py                                 # v8 模型探索 probe
├─ source_models/config/
│  ├─ material_sorting_matrix/                      # 每一轮 Geant4 输入矩阵
│  └─ diffraction_peak_tables/                      # 10/20 材料峰表 manifest
└─ results/
   ├─ material_sorting/                             # Geant4 运行状态 CSV
   └─ accuracy_v3/                                  # 特征、审查、模型、诊断结果
```

## 机器可读成果清单

为了后续写报告、做图、继续实验时不用从 Markdown 手工复制路径，已生成一个轻量 manifest：

```text
docs/material_sorting_results_manifest/
├─ material_sorting_results_manifest.json      # 结构化成果清单，适合脚本读取
├─ material_sorting_results_inventory.csv      # 表格版，适合 Excel/论文表格整理
└─ material_sorting_results_inventory.md       # Markdown 快速浏览版
```

其中每一项包含：

- `id`：成果编号，例如 `v6_main_10_material_robust`
- `tier`：用途层级，例如 `report_main_result`、`report_extension_result`、`reusable_dataset`
- `stage`：v4-v8 阶段
- `status`：passed、passed_with_caveat、failed_strict_gate、available 等
- `primary_metric`：主要指标或用途
- `path`：成果目录
- `key_files`：核心文件
- `reuse_for`：建议复用场景

## 报告优先使用的成果

### 1. 主结果：10 材料 H/M 稳健分选 v6

定位：这是当前最干净、最适合作为主结论的结果。

- profile：`v8a_multiclass_context_v6_physical_robust`
- Geant4 状态：`results/material_sorting/run_status_v8a_multiclass_context_v6_physical_robust.csv`
- 特征数据：`results/accuracy_v3/v8a_multiclass_context_v6_physical_robust_event_to_feature`
- 最终数据审查：`results/accuracy_v3/v8a_multiclass_context_v6_physical_robust_training_data_final_audit`
- 模型结果：`results/accuracy_v3/v8a_multiclass_context_v6_physical_robust_model`
- 捷径审查：`results/accuracy_v3/v8a_multiclass_context_v6_skeptical_shortcut_audit`

关键文件：

- `v8a_multiclass_context_v6_physical_robust_gate.json`
- `v8a_multiclass_context_v6_physical_robust_summary.csv`
- `v8a_multiclass_context_v6_physical_robust_best_main.csv`
- `v8a_multiclass_context_v6_physical_robust_confusions.csv`
- `v8a_multiclass_context_v6_skeptical_shortcut_gate.json`
- `v8a_multiclass_context_v6_skeptical_shortcut_report.md`

可写入报告的结论：

- 10 材料扰动增强训练通过数据、模型和捷径审查。
- validation/stress H/M min recall 均为 `1.0`。
- validation/stress macro-F1 均为 `1.0`。
- total-count、lineage、shuffled-label 等控制模型较弱，未发现简单捷径解释。

### 2. 扩展结果：20 材料 relaxed ensemble v8

定位：这是论文中的“扩展验证/锦上添花”结果，不应描述成严格 hard-negative 已完全解决。

- 原始 v8 profile：`v8a_multiclass_context_v8_hard_negative_robust`
- 原始特征数据：`results/accuracy_v3/v8a_multiclass_context_v8_hard_negative_robust_event_to_feature`
- relaxed ensemble 结果：`results/accuracy_v3/v8a_multiclass_context_v8_vote_log3_hist1_relaxed_gate`
- reporting-grade gate：`results/accuracy_v3/v8a_multiclass_context_v8_reporting_grade_extension_gate`

关键文件：

- `v8_vote_log3_hist1_relaxed_gate.json`
- `v8_vote_log3_hist1_relaxed_summary.csv`
- `v8_vote_log3_hist1_relaxed_decisions.csv`
- `v8_vote_log3_hist1_relaxed_report.md`
- `v8a_multiclass_context_v8_reporting_grade_extension_gate.json`
- `v8a_multiclass_context_v8_reporting_grade_extension_report.md`

可写入报告的结论：

- relaxed ensemble `VoteLogistic3HistGB1Main` 通过扩展门禁。
- validation overall top-1 `0.9854`，macro-F1 `0.9853`，H/M min recall `0.9583`。
- stress overall top-1 `0.9958`，macro-F1 `0.9958`，H/M min recall `0.9167`。
- stress worst-profile H/M min recall `0.75`。
- Hematite->Ilmenite stress 独立样本错误数降至 `2`。

推荐写法：

> 20 材料扩展实验显示模型在更复杂材料空间中仍具有较高整体分选能力，但 Hematite 与 Ilmenite 在强 hard-negative 条件下仍存在少量混淆。

### 3. 局限性证据：20 材料 strict hard-negative v7/v8

定位：这些是“为什么不能声称完全解决”的关键证据，也可作为论文讨论部分亮点。

v7：

- 模型结果：`results/accuracy_v3/v8a_multiclass_context_v7_twenty_material_scalability_scout_model`
- hard-negative 诊断：`results/accuracy_v3/v8a_multiclass_context_v7_twenty_material_scalability_scout_hard_negative_diagnostics`
- 捷径审查：`results/accuracy_v3/v8a_multiclass_context_v7_twenty_material_scalability_scout_skeptical_shortcut_audit`

v8：

- 原 strict 模型结果：`results/accuracy_v3/v8a_multiclass_context_v8_hard_negative_robust_model`
- hard-negative 诊断：`results/accuracy_v3/v8a_multiclass_context_v8_hard_negative_robust_hard_negative_diagnostics`
- 捷径审查：`results/accuracy_v3/v8a_multiclass_context_v8_hard_negative_robust_skeptical_shortcut_audit`
- 模型 sweep：`results/accuracy_v3/v8a_multiclass_context_v8_model_sweep_reporting_probe`
- HistGB probe：`results/accuracy_v3/v8a_multiclass_context_v8_histgb_relaxed_hard_negative_model`
- ensemble probe：`results/accuracy_v3/v8a_multiclass_context_v8_ensemble_relaxed_probe`

关键文件：

- `v8a_multiclass_context_v8_hard_negative_robust_gate.json`
- `v8a_multiclass_context_v8_hard_negative_robust_skeptical_shortcut_gate.json`
- `v8a_multiclass_context_v8_hard_negative_robust_diagnostics_gate.json`
- `v8_model_sweep_ranking.csv`
- `v8_ensemble_relaxed_ranking.csv`

可写入报告的结论：

- v7 暴露出 Hematite/Ilmenite hard-negative。
- v8 targeted hard-negative training 缩小了问题，但 strict gate 仍未通过。
- 错误不由 total-count、lineage 或跨 split 完全重复特征解释。
- 后续应从峰形、峰强比、局部窗口特征增强入手，而不是只堆更多样本。

## 可复用数据入口

### Geant4 输入矩阵

```text
source_models/config/material_sorting_matrix/
├─ v8a_multiclass_context_v4_count_overlap_50kev_t60_t120/
├─ v8a_multiclass_context_v5_physical_perturbation_scout/
├─ v8a_multiclass_context_v6_physical_robust/
├─ v8a_multiclass_context_v7_twenty_material_scalability_scout/
└─ v8a_multiclass_context_v8_hard_negative_robust/
```

这些目录用于复现实验设计和重跑 Geant4。优先复用 v6 和 v8：

- v6：10 材料主结果。
- v8：20 材料扩展和 hard-negative 后续研究基础。

### Geant4 运行状态

```text
results/material_sorting/
├─ run_status_v8a_multiclass_context_v4_count_overlap_50kev_t60_t120.csv
├─ run_status_v8a_multiclass_context_v5_physical_perturbation_scout.csv
├─ run_status_v8a_multiclass_context_v6_physical_robust.csv
├─ run_status_v8a_multiclass_context_v7_twenty_material_scalability_scout.csv
└─ run_status_v8a_multiclass_context_v8_hard_negative_robust.csv
```

这些 CSV 用于证明 Geant4 是否跑完、是否有失败行。

### 特征数据

主要特征目录都在 `results/accuracy_v3/` 下，命名后缀一般是 `_event_to_feature`。

重点：

- `v8a_multiclass_context_v6_physical_robust_event_to_feature`
- `v8a_multiclass_context_v7_twenty_material_scalability_scout_event_to_feature`
- `v8a_multiclass_context_v8_hard_negative_robust_event_to_feature`

典型文件：

- `v8a_event_sidecar_features.csv`：宽表特征，模型训练最常用。
- long/sidecar CSV：保存峰/窗口级展开信息，适合做特征工程和诊断。

## 脚本入口

### 配置

```text
analysis/configs/
├─ v8a_multiclass_context_v4_count_overlap_config.json
├─ v8a_multiclass_context_v5_physical_perturbation_scout_config.json
├─ v8a_multiclass_context_v6_physical_robust_config.json
├─ v8a_multiclass_context_v7_twenty_material_scalability_scout_config.json
└─ v8a_multiclass_context_v8_hard_negative_robust_config.json
```

### 矩阵生成

```text
analysis/generate_v8a_multiclass_context_v4_count_overlap_matrix.py
analysis/generate_v8a_multiclass_context_v5_physical_perturbation_scout_matrix.py
analysis/generate_v8a_multiclass_context_v6_physical_robust_matrix.py
analysis/generate_v8a_multiclass_context_v7_twenty_material_scalability_scout_matrix.py
analysis/generate_v8a_multiclass_context_v8_hard_negative_robust_matrix.py
```

### 审查和训练

重点脚本：

- `analysis/train_v8a_multiclass_context_v6_physical_robust.py`
- `analysis/audit_v8a_multiclass_context_v6_skeptical_shortcut.py`
- `analysis/train_v8a_multiclass_context_v7_twenty_material_scalability_scout.py`
- `analysis/audit_v8a_multiclass_context_v7_hard_negative_diagnostics.py`
- `analysis/train_v8a_multiclass_context_v8_hard_negative_robust.py`
- `analysis/audit_v8a_multiclass_context_v8_hard_negative_robust_diagnostics.py`
- `analysis/audit_v8a_multiclass_context_v8_reporting_grade_extension_gate.py`
- `analysis/train_v8a_multiclass_context_v8_vote_log3_hist1_relaxed_gate.py`

### 探索性 probe

这些用于模型选择和后续研究，不建议作为主报告唯一依据：

- `analysis/probe_v8_reporting_model_sweep.py`
- `analysis/probe_v8_ensemble_relaxed.py`
- `analysis/train_v8a_multiclass_context_v8_histgb_relaxed_hard_negative.py`

## 峰表和材料定义相关

峰表 manifest：

```text
source_models/config/diffraction_peak_tables/
├─ hm_powder_peaks_cif_or_literature_v8a_manifest.json
├─ hm_powder_peaks_project_scan_v8a_manifest.json
├─ ten_material_powder_peaks_cif_or_literature_v8a_manifest.json
└─ twenty_material_powder_peaks_rruff_v8a_manifest.json
```

重点：

- 10 材料研究用 `ten_material_powder_peaks_cif_or_literature_v8a_manifest.json`。
- 20 材料扩展用 `twenty_material_powder_peaks_rruff_v8a_manifest.json`。

## 后续衍生研究建议

### 如果要写报告

优先读取：

1. v6 model gate/report/summary。
2. v6 skeptical shortcut gate/report。
3. v8 relaxed ensemble gate/report/summary。
4. v8 strict hard-negative diagnostics gate/report。

### 如果要继续做模型研究

优先读取：

1. `v8a_multiclass_context_v8_hard_negative_robust_event_to_feature/v8a_event_sidecar_features.csv`
2. `v8_model_sweep_ranking.csv`
3. `v8_ensemble_relaxed_ranking.csv`
4. `v8a_multiclass_context_v8_hard_negative_robust_hard_negative_diagnostics/*`

### 如果要做 v9

不要从零开始。建议复制 v8 配置和生成器，新增：

- Hematite/Ilmenite/Rutile/Goethite/Siderite 局部峰形特征。
- 峰强比值特征。
- hard-negative ablation 输出。
- 保留 count-only、lineage-only、shuffled-label、exact-hash、H/M off-diagonal 审查。

## 推荐的论文结果组织

```text
第一层：系统实现
  Geant4 仿真、XRT 探测器输出、峰表/材料定义、特征提取流水线

第二层：主实验
  v6 10 材料扰动增强稳健分选

第三层：扩展实验
  v8 20 材料 relaxed ensemble 扩展验证

第四层：局限性与未来工作
  v7/v8 Hematite-Ilmenite hard-negative 诊断
```
