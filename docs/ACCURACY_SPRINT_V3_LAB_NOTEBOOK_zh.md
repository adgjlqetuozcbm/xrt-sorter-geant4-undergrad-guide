# Accuracy Sprint v3 实验台账

这份台账记录十材料分选从负结果到改进路线的完整过程。它的目的不是只保存成功指标，而是把每一次失败、混淆、训练假设、下一步选择都保留下来，供后续论文的 Methods、Ablation、Limitations 和 Discussion 使用。

## 记录原则

- 每次实验必须记录数据划分、是否使用 burned test seeds、模型、特征、能量、厚度、photon budget 和关键指标。
- 负结果不能删除。失败路线是论文中证明方案演进和避免 cherry-picking 的证据。
- train/validation 可以反复迭代；final test 只能在方案冻结后打开一次。
- 如果 final test 不达标，该 final test 立即烧掉，不能继续拿它调参。
- 当前准确率优先，材料理化性质输出后置。

## 已知基线

| 阶段 | 证据 | 结论 |
| --- | --- | --- |
| v2 full matrix | Top-1 `0.3813`，macro-F1 `0.3153`，min recall `0.0` | 十材料自动分选失败，不能包装成成功 |
| selected rebuild old test | Top-1 约 `0.90`，但每类 support 只有 `6`，min recall `0.5` | 样本太少，不 claim-safe |
| `sr2` locked final test | Top-1 `0.88`，macro-F1 `0.8789`，min recall `0.5` | 总体指标可看，但 Hematite/Magnetite 未达标 |
| `es2` energy scan | 最好候选 validation min recall 约 `0.6` | 单纯换三/四个能量还不够 |

## Accuracy v3 假设

当前失败集中在 Hematite 与 Magnetite。两者都属于铁氧化物，密度和当前 XRT transmission 特征相近，所以单一十分类器容易把它们互相混淆。v3 采用分层路线：先判断吸收组，再做组内分类，最后对 Hematite/Magnetite 使用专家二分类器重新分配这对材料的概率质量。

这一做法不是为了“硬凑结果”，而是把材料体系拆成相似子空间。未来扩展到更多材料时，也可以先按物理相似性或谱相似性聚类，再在组内训练专家。

## 每轮实验记录模板

| 字段 | 内容 |
| --- | --- |
| experiment_id | 唯一实验名，例如 `accuracy_v3_sr2_diagnostic` |
| hypothesis | 本轮为什么可能提高准确率 |
| feature_change | 增加或删除了哪些特征 |
| model_change | 模型或层级结构变化 |
| source_thickness_change | 能量、source、厚度变化 |
| seeds | train/validation/test seeds |
| burned_seed_policy | 是否含 burned test seeds，若含则只能作为诊断 |
| validation_metrics | Top-1、macro-F1、min recall、H/M recall |
| final_metrics | 仅 locked final test 使用 |
| failure_analysis | 失败类别、混淆对象、可能物理原因 |
| next_action | 继续训练、换特征、换能量、换模型或停止路线 |

## 当前 v3 实施内容

- 在样本聚合中增加更细的 event/hit 统计，包括能量沉积分位数、非零比例、hit energy 分位数、direct/scatter hit 能量、角度和径向分布。
- 增加 detector-response smoothed counts，用简单高斯能量分辨率模拟把离散 hit energy 展宽到响应谱 bin。
- 在严格审计中加入 `HMExpertHierarchicalExtraTrees`：吸收组模型、组内模型、Hematite/Magnetite 专家模型三层合成。
- 审计脚本输出 `experiment_registry.csv` 和 `failure_analysis.csv`，把每轮实验写成论文可追溯记录。

## 2026-04-28 v3 sr2 诊断

这轮使用已烧掉的 `sr2` final seeds，只能作为诊断 smoke，不能作为新 claim。其作用是验证 v3 特征、H/M 专家模型和论文台账是否能完整运行。

| 项 | 结果 |
| --- | --- |
| protocol | `accuracy_v3_sr2_diagnostic_burned_test` |
| raw dirs | `selected_rebuild + sr2` |
| train seeds | `101/202/303/404/505` |
| validation seed | `606` |
| diagnostic test seeds | `707/808/909/1001/1102`，已烧掉 |
| selected method | `HematiteMagnetiteRecallExtraTrees` |
| validation Top-1 / macro-F1 / min recall | `0.9500` / `0.9497` / `0.6667` |
| validation Hematite / Magnetite recall | `0.6667` / `0.8333` |
| diagnostic test Top-1 / macro-F1 / min recall | `0.8833` / `0.8826` / `0.5333` |
| diagnostic test Hematite / Magnetite recall | `0.5667` / `0.5333` |

结论：v3 特征和 H/M recall-weighted 候选略有改善，但仍没有解决 H/M separability。`HematitePriorityExtraTrees`、`HMStrongRecallExtraTrees` 和 `HMExpertHierarchicalExtraTrees` 在 validation 上没有超过 `0.6667` 的 H/M 最低召回。下一步不能继续在 `sr2` 上调参，应生成新的 `accuracy_v3_hm` H/M-focused 数据，优先验证新能量组合和响应谱特征是否真的提供额外信息。

## 下一步

1. 先在已烧掉的 `sr2` 数据上做诊断 smoke，验证 v3 管线能运行并正确记录失败。
2. 生成并运行 `accuracy_v3_hm` 小矩阵，先只覆盖 Hematite、Magnetite、Pyrite、Chalcopyrite，使用新 seeds 和 `30/40/50/70/90/110/120/150/200 keV`。
3. 如果 H/M-focused validation 的 H/M recall 仍低于 `0.70`，继续只在 train/validation 上改特征、能量或模型。
4. 只有多个 validation seeds 稳定达标后，才生成并运行新的 unseen final test。

## 2026-04-28 v4 H/M smoke

这轮是 `accuracy_v3_hm` 的第一阶段连通性检查。它不使用 final test，只评估 train seed `1201` 到 validation seed `1301`，目标是验证 H/M-focused 配置、runner、development-only 审计和 `hm_pairwise_audit.csv` 能完整运行。

| 项 | 结果 |
| --- | --- |
| profile | `v3_hm_smoke` |
| materials | Hematite、Magnetite、Pyrite、Chalcopyrite |
| sources | `40/70/110/200 keV` |
| thickness | `5/10/20 mm` |
| seeds | train `1201`，validation `1301` |
| matrix | 96 material runs + 8 calibration runs |
| run status | `104/104` completed，`0` failed |
| selected method | `HierarchicalExtraTrees` |
| validation Top-1 / macro-F1 / min recall | `0.7083` / `0.7045` / `0.5000` |
| validation Hematite / Magnetite recall | `0.5000` / `0.6667` |
| H/M pairwise Top-1 / min recall / AUC | `0.5833` / `0.3333` / `0.6111` |

结论：全链路已经打通，但 `v3_hm_smoke` 明显未达标。失败集中在 Hematite 被误判为 Magnetite，pairwise audit 的 top features 主要来自 `mono_200kev` 的 detector-response、attenuation 和 spectral-shape 特征。这说明新增数据路线是可运行的，但 2 seeds + 4 energies 不足以支持 H/M separability。

下一步已生成 `v3_hm_dev1` 配置：4 materials x 3 thicknesses x 9 sources x 7 seeds = 756 material runs，加 63 calibration runs，总计 819 runs。`v3_hm_dev1` 应分批运行，先跑 calibration，再按材料 run；运行完成后用 development-only audit 评估 train seeds `1201-1205` 到 validation seeds `1301/1302`。
