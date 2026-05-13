# Geant4 XRT 矿物分选仿真本科项目：吸收组基线与材料级诊断

这是一个面向本科项目展示、答辩准备和组内学习的 **X 射线透射（XRT）矿物分选仿真系统**。项目用 Geant4 搭建 X 射线源、矿物样本和探测器，用 C++ 输出事件数据，再用 Python 完成数据质量检查、样本聚合、特征提取、训练/测试拆分和基础分类验证。

一句话概括：本仓库展示了一条从物理仿真到可复查机器学习结果的完整链路，而不是一个已经可部署的工业 XRT 产品。

![系统流程](figures/elementary_system_flow.png)

## 项目亮点

- **物理仿真链路完整**：包含 X 射线源、矿物材料、样本几何、探测器响应、事件输出和命中输出。
- **数据来源可追踪**：材料目录、批量配置、事件 CSV、虚拟样本、训练/测试拆分和结果表都保存在仓库中。
- **特征含义可解释**：核心特征包括主 gamma 透射率、平均探测器能量沉积和 gamma 命中率，均能回到 XRT 吸收差异的物理直觉。
- **模型选择克制**：使用阈值法 baseline 和 `StandardScaler + LogisticRegression`，避免用复杂模型掩盖数据链路。
- **结果边界清楚**：当前结论只属于固定仿真条件下的低/高吸收组二分类验证，不代表真实设备、复杂矿流或所有矿物覆盖。
- **适合组内交接**：提供论文式正文、队友入门指南、运行说明、文件地图和术语表。

## 先看哪里

如果你第一次打开这个仓库，建议按下面顺序阅读：

| 你是谁 / 你想做什么 | 推荐入口 |
| --- | --- |
| 完全没参与过项目的组员 | `docs/TEAM_GUIDE_zh.md` |
| 只想看通俗解释 | `docs/public_explainer_zh.md` |
| 想看正式论文式表述 | `paper/main_thesis_HIT_revised_zh.md` |
| 想在本机跑起来 | `docs/RUN_LOCALLY_zh.md` |
| 想知道每个文件干什么 | `docs/FILE_MAP_zh.md` |
| 看不懂专业词 | `docs/GLOSSARY_BY_FIRST_APPEARANCE.md` |
| 想核对结果数字 | `results/undergrad_validation/validation_manifest.json` |
| 想看十材料物种级分选是否成立 | `docs/MATERIAL_LEVEL_SORTING_DIAGNOSTIC_zh.md` |
| 想继续推进十材料 v2 物化指纹/字典路线 | `docs/MATERIAL_SORTING_V2_PROTOCOL_zh.md` |
| 想看最新严格泛化复核 | `docs/STRICT_GENERALIZATION_REVIEW_zh.md` |
| 想看最终边界复核 | `docs/FINAL_ELEMENTARY_REVIEW_zh.md` |
| 想看最新材料级分选成果、v6/v8 结果和共享说明 | `docs/G4_XRT_MATERIAL_SORTING_RESULTS_INDEX_zh.md`, `docs/MATERIAL_SORTING_SHARE_GUIDE_zh.md` |

## 数据链路

```mermaid
flowchart LR
    A["材料目录与配置文件"] --> B["Geant4 C++ 仿真"]
    B --> C["events.csv / hits.csv / metadata"]
    C --> D["Python 质量检查"]
    D --> E["100 events -> 1 个虚拟样本"]
    E --> F["特征工程"]
    F --> G["训练/测试拆分"]
    G --> H["阈值法与 Logistic Regression"]
    H --> I["结果证据包与论文"]
```

公开证据包使用十种单一材料：Quartz、Calcite、Orthoclase、Albite、Dolomite、Pyrite、Hematite、Magnetite、Chalcopyrite 和 Galena。每种材料运行 5000 个仿真事件，每 100 个事件聚合为 1 个虚拟样本，因此每种材料形成 50 个虚拟样本，总计 500 个虚拟样本。训练集和测试集按每种材料 25/25 切分，最终各有 250 个样本。

## 当前验证结果

| 方法 | 特征 | 测试样本 | 正确样本 | accuracy |
| --- | --- | ---: | ---: | ---: |
| 阈值法 | `primary_transmission_rate` | 250 | 246 | 0.9840 |
| Logistic Regression | `primary_transmission_rate` | 250 | 248 | 0.9920 |
| Logistic Regression | 三个物理相关特征 | 250 | 249 | 0.9960 |

![基础分类精度](figures/elementary_absorption_accuracy.png)

这里的 `0.9960` 表示三特征 Logistic Regression 在 250 个测试虚拟样本中正确 249 个。它不是训练集准确率，也不是所有矿物、真实设备或工业场景的普适准确率。

## 材料级分选诊断

为了回应“十种材料是否已经能被逐一识别”的问题，仓库新增了材料级分选诊断脚本和结果。当前诊断结论是负面的：在旧的 10 mm、120 kV spectrum 公开数据上，十材料直接分类的主方法 `Calibrated Extra Trees` 只有 `0.464` top-1 accuracy、`0.876` top-3 accuracy、`0.4486` macro-F1，复核率为 `0.848`，所有预设验收条件均未通过。

这说明当前 `0.9960` 只能作为低/高吸收组二分类结果引用，不能改写成十材料物种识别准确率。材料级实验骨架已经具备：`random_seed` 配置、pilot/full 矩阵、批量运行脚本、top-3 候选、置信度复核和开放集检查；但完整 270-run pilot/full 矩阵尚未完成，因此材料级分选目前只能作为下一阶段方向。详见 `docs/MATERIAL_LEVEL_SORTING_DIAGNOSTIC_zh.md`。

仓库现在还新增了 v2 协议底座：`air_path` 空束校准、279-run 矩阵（270 个材料 run + 9 个校准 run）、可恢复 runner、校准后的多能谱物化指纹、材料字典距离特征和严格 seed 留出评估脚本 `analysis/material_sorting_v2.py`。v2 full matrix 已经完成，但正式结果仍是负结果：Top-1 `0.3813`、macro-F1 `0.3153`、最低单类召回 `0.0`。

后续 selected rebuild、GPU XGBoost 候选、`sr2` 独立新 seed final test 和 `es2` 扩展能量扫描也已经完成复核。`sr2` 在每类 final test support `30` 的条件下达到 Top-1 `0.88`、macro-F1 `0.8789`，但 Hematite 召回 `0.5`、Magnetite 召回 `0.6333`，仍未通过最低单类召回门槛。详见 `docs/MATERIAL_SORTING_V2_PROTOCOL_zh.md` 和 `docs/STRICT_GENERALIZATION_REVIEW_zh.md`。

Accuracy Sprint v3 已加入分层分类、Hematite/Magnetite 专家候选、更细 hit/event 谱特征、detector-response smoothed counts，以及论文证据台账。首轮 v3 诊断输出在 `results/accuracy_v3/`，仍使用已烧掉的 `sr2` final seeds，因此不能作为新 claim；它的结论是 H/M 最低召回仍低于 `0.70`，下一步应生成新的 `accuracy_v3_hm` 小矩阵，而不是继续在 `sr2` 上调参。详见 `docs/ACCURACY_SPRINT_V3_LAB_NOTEBOOK_zh.md` 和 `docs/ACCURACY_V3_METHODS_NOTES_zh.md`。

Accuracy Sprint v4 已完成 `v3_hm_smoke` 连通性验证和 `v3_hm_dev1` 开发矩阵：`v3_hm_dev1` 的 `819/819` 个 H/M-focused Geant4 run 全部成功，development-only validation Top-1 为 `0.8125`，macro-F1 为 `0.8122`，但 Hematite/Magnetite recall 仍只有 `0.5833`/`0.6667`，未达到 `0.70` 门槛。因此当前不能进入十材料 full matrix 或 locked final test；下一步应继续 H/M-focused 低能细扫和特征诊断。

## 快速运行

如果你的环境已经安装 Geant4、CMake、C++17 编译器、Python、pandas 和 scikit-learn，可以从仓库根目录运行：

```bash
pip install pandas scikit-learn
cmake -S . -B build
cmake --build build
cd build

for material in quartz calcite orthoclase albite dolomite pyrite hematite magnetite chalcopyrite galena; do
  XRT_EXPERIMENT_CONFIG=../source_models/config/undergrad_batch/${material}.txt \
    ./xrt_sorter ../analysis/configs/run_research.mac
done

cd ..
python analysis/classify_absorption_groups.py
```

运行成功后，重点检查：

- `results/undergrad_validation/event_row_summary.csv`：每种材料是否都有 5000 个事件。
- `results/undergrad_validation/train_test_split_samples.csv`：训练/测试拆分是否为 250/250。
- `results/undergrad_validation/absorption_group_classification_summary.csv`：三种方法的测试集结果。
- `results/undergrad_validation/validation_manifest.json`：样本政策、软件版本和结论边界。

更详细的依赖、VS Code/CMake 配置和排错见 `docs/RUN_LOCALLY_zh.md`。

## 目录结构

```text
include/              Geant4 C++ 头文件
src/                  Geant4 C++ 实现
source_models/        材料目录、源项、能谱和实验配置
analysis/             Python 分析脚本和运行宏
results/              本科级验证证据包
figures/              README 和论文可引用图
docs/                 组员入门、术语、文件地图和运行说明
paper/                论文式正文、复现说明和参考文献
```

## 项目边界

本仓库可以说明：我们完成了 Geant4 XRT 仿真系统、事件级数据输出、样本级特征构造、明确训练/测试拆分和仿真数据上的粗粒度吸收组分类验证。

本仓库不能说明：真实 XRT 设备已经验证、复杂矿流已经覆盖、所有矿物都能识别、模型可直接控制工业分选设备，或当前结果已经达到产品部署条件。

如果后续导师要求更高标准，下一阶段应增加固定随机种子、多种子重复、不同厚度和几何、混合材料、独立 run-level 测试以及真实设备或真实样品对照。

## 权利声明

本仓库用于项目组学习、运行、答辩展示和组内协作。代码、文档、图表和项目构思均保留权利。未经项目负责人和指导教师许可，不得复制为个人项目、转发为独立成果、商用、二次发布或冒名使用。详见 `LICENSE.md` 和 `NOTICE.md`。
