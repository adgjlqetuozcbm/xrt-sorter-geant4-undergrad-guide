# G4 Accuracy Sprint v5 恢复记录

## 恢复结论

本轮从压缩中断的会话 `019dd468-8431-7960-ba81-09ef7aca8b9b` 恢复。直接 transcript 和 durable memory 均显示：上一轮在用户提出 v5 实施计划后，只完成了记忆读取、脚本检查和计划更新，没有生成 v5 矩阵、没有运行 v5 仿真、没有提交 v5 代码。

因此当前状态不是 v5 跑到一半，而是 v5 刚开始实施。

## v4 负结果基线

`v3_hm_dev1` 是进入 v5 前的严格负结果基线。

| 项目 | 结果 |
| --- | --- |
| runner status | `819/819` completed, `0` failed |
| materials | Hematite, Magnetite, Pyrite, Chalcopyrite |
| sources | `30/40/50/70/90/110/120/150/200 keV` |
| train seeds | `1201-1205` |
| validation seeds | `1301/1302` |
| selected method | `HistGradientBoosting` |
| validation Top-1 / macro-F1 / min recall | `0.8125` / `0.8122` / `0.5833` |
| Hematite / Magnetite recall | `0.5833` / `0.6667` |
| H/M pairwise min recall | `0.3333` |

导师视角结论：v4 提升了四材料总体指标，但没有解决 Hematite/Magnetite 物理可分性。`pairwise` 二分类仍低，说明不能靠十材料全局模型或简单 hierarchy 包装来声称达标。

## v5 本轮实施内容

本轮先完成 v5 的代码与矩阵落地，不打开 final test。

代码变更：

- `analysis/generate_material_sorting_matrix.py`
  - 新增 `--energy-list-kev`，允许直接传入任意 mono energy 列表。
  - 保留 `--profile-alias`，可用 `accuracy_v3_hm` 生成 `v5_hm_lowwide`。
- `analysis/material_sorting_v2.py`
  - 修正 mono source id 生成逻辑，支持整数和小数 keV 的稳定命名。
- `analysis/run_material_sorting_matrix.py`
  - 将 `v5_*` profile alias 自动映射到 `selected_rebuild` macro。
- `analysis/strict_generalization_audit.py`
  - 模型选择排序改为固定优先级：`hm_min_recall -> min_class_recall -> macro_f1 -> top1_accuracy`。
  - 新增 H/M-specific 候选：
    - `HMAllEnergyExpertExtraTrees`
    - `HMLowEnergyExpertExtraTrees`
    - `HMThicknessGroupedExpertExtraTrees`
    - `HMPairwiseVotingExpertExtraTrees`
  - 保留原有 `ExtraTrees`, `HistGradientBoosting`, `HierarchicalExtraTrees`, `HMExpertHierarchicalExtraTrees`, H/M recall-weighted ExtraTrees, `HighGroupRecallExtraTrees`, `XGBoostGPU`。

## v5 H/M low-wide 矩阵

已生成：

`source_models/config/material_sorting_matrix/v5_hm_lowwide/material_sorting_matrix.csv`

矩阵定义：

| 项目 | 设置 |
| --- | --- |
| materials | Hematite, Magnetite, Pyrite, Chalcopyrite |
| energies | `15/20/25/30/35/40/50/70/90/110/120/150/200 keV` |
| thickness | `5/10/20 mm` |
| train seeds | `1501-1520` |
| validation seeds | `1601-1610` |
| shadow-validation seeds | `1701-1710` |
| material runs | `6240` |
| calibration runs | `520` |
| total runs | `6760` |

Runner 预检：

```text
profile=v5_hm_lowwide role=all
selected_rows=6760 completed=0 failed=0 pending=6760
```

## v5 模型 smoke

为避免直接把未经运行验证的候选带入 6760-run 矩阵，本轮用已完成的 `v3_hm_smoke` 数据跑了小型 development-only audit：

`results/accuracy_v3/v5_model_smoke/`

该 smoke 不作为成功依据，只验证 v5 audit 可以完整产出 ledger。结果：

| 项目 | 结果 |
| --- | --- |
| selected method | `XGBoostGPU` |
| validation Top-1 / macro-F1 / min recall | `0.8333` / `0.8322` / `0.6667` |
| Hematite / Magnetite recall | `0.6667` / `0.8333` |
| H/M pairwise min recall | `0.3333` |
| claim-safe | `False` |

问题指正：smoke 中 H/M pairwise 仍未达 `0.70`。这再次提醒我们，v5 的成功条件不是模型表面变复杂，而是 low-wide 数据是否真正增加 H/M 可分辨物理信息。

## 下一阶段 gate

1. 运行 `v5_hm_lowwide` 全矩阵，并确保 runner status 为 `completed=6760`, `failed=0`, `pending=0`。
2. 先用 train `1501-1520` 到 validation `1601-1610` 运行 development-only audit。
3. 若 validation H/M min recall 未达 `0.70`，停止，不使用 shadow。
4. 若 validation 达标，再用同一冻结候选对 shadow-validation `1701-1710` 复核。shadow 不能被反复调成训练集。
5. 只有 validation 与 shadow 都达标，且 H/M pairwise min recall 达 `0.70`，才允许生成十材料 `v5_full_trainval`。
6. 十材料 validation 达到 Top-1 `>=0.85`, macro-F1 `>=0.80`, min recall `>=0.70` 后，才能冻结方案并只运行一次 `v5_full_final_locked`。

Final seeds `2001-2010` 仍未打开。已 burned final seeds `303/505/707/808/909/1001/1102` 禁止调参使用。
