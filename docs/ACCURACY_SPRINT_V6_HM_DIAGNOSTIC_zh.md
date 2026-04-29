# G4 Accuracy Sprint v6 H/M 诊断结果

## 范围

本诊断只使用 development 数据：

- train seeds: `1501-1520`
- validation seeds: `1601-1610`
- materials: `Hematite`, `Magnetite`
- shadow seeds `1701-1710`: 未使用
- final seeds: 未使用

输出目录：

`results/accuracy_v3/v5_hm_lowwide/hm_diagnostic/`

## 关键输出

| 文件 | 用途 |
| --- | --- |
| `hm_single_source_pairwise_models.csv` | 单能量、跨厚度 H/M pairwise 诊断 |
| `hm_single_source_thickness_pairwise_models.csv` | 单能量、固定厚度 H/M pairwise 诊断 |
| `hm_fused_pairwise_by_thickness.csv` | 全能量融合后按厚度诊断 |
| `hm_feature_separability.csv` | 按 train 排序、用 validation 复核的单特征可分性 |
| `hm_validation_error_by_thickness.csv` | 当前 selected model 的 validation H/M 错误按厚度切片 |
| `hm_diagnostic_manifest.json` | manifest 和 seed 记录 |

## 主要发现

### 1. 低能点没有解决 H/M

`15/20/25/30 keV` 的单能量 pairwise 模型在 validation 上退化为预测同一类：

- H/M min recall: `0.0`
- feature count: `6`

这说明本轮加入的低能点主要增加了矩阵宽度，但没有提供稳定的 H/M 判别信息。

### 2. 最有用信号集中在高能厚样本，但强度不足

单能量固定厚度里，最好切片为：

| source | thickness | model | H recall | M recall | H/M min recall | ROC AUC M |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| `mono_200kev` | `20 mm` | ExtraTrees | `0.70` | `0.70` | `0.70` | `0.7737` |

按单特征看，最强 validation AUC 也集中在 `mono_200kev, 20 mm` 的高能响应：

- `Rsig050_e_120_inf`: validation AUC `0.7675`
- `T_e_120_inf`: validation AUC `0.7613`
- `A_e_120_inf`: validation AUC `0.7613`
- `A_per_mm_e_120_inf`: validation AUC `0.7613`

这是真实可用方向，但还没有达到 v6 H/M gate 的 `>=0.80` validation H/M min recall。

### 3. 全能量融合没有自动超过高能局部信号

全能量融合 H/M pairwise：

| group | model | H recall | M recall | H/M min recall | ROC AUC M |
| --- | --- | ---: | ---: | ---: | ---: |
| all thickness | ExtraTrees | `0.75` | `0.60` | `0.60` | `0.7246` |
| fixed `5 mm` | ExtraTrees | `0.80` | `0.65` | `0.65` | `0.7675` |
| fixed `20 mm` | ExtraTrees | `0.80` | `0.65` | `0.65` | `0.7587` |

这说明已有特征融合会稀释部分高能局部信号，不能靠更长训练直接解决。

### 4. 当前 selected model 的 H/M 错误不是单一厚度造成

当前 `HistGradientBoosting` selected model 的 validation H/M 错误：

| material | thickness | correct | confused |
| --- | ---: | ---: | ---: |
| Hematite | `5 mm` | `15/20` | `5/20` |
| Hematite | `10 mm` | `13/20` | `7/20` |
| Hematite | `20 mm` | `12/20` | `8/20` |
| Magnetite | `5 mm` | `14/20` | `6/20` |
| Magnetite | `10 mm` | `11/20` | `9/20` |
| Magnetite | `20 mm` | `14/20` | `6/20` |

最弱点是 `20 mm Hematite` 和 `10 mm Magnetite`，但三个厚度都有混淆。

## 阶段结论

v6 失败不是 runner、support 或 GPU 算力问题，而是 H/M 物理信息和特征表达不足。当前数据里存在一条有希望的线索：高能端，尤其 `200 keV` 的 `>120 keV` 响应，对 H/M 有中等可分性；但现有模型没有把它转化成稳定的 gate 通过。

## v6b 高能差分特征试验

基于上述诊断，新增了受控开关：

`--enable-hm-differential-features`

该开关只增加少量预注册的 H/M 高能尾部差分特征，输出到独立目录：

`results/accuracy_v3/v6b_hm_feature_dev/`

v6b development gate 结果：

| 指标 | v6 门槛 | 观测值 | 结论 |
| --- | ---: | ---: | --- |
| Top-1 | `0.88` | `0.8292` | fail |
| macro-F1 | `0.84` | `0.8291` | fail |
| min class recall | `0.75` | `0.6333` | fail |
| H/M min recall | `0.80` | `0.6333` | fail |
| H/M pairwise min recall | `0.75` | `0.5333` | fail |
| min class support | `40` | `60` | pass |

Selected method: `HematiteMagnetiteRecallExtraTrees`.

v6b 没有改善 H/M gate，反而低于 v6 baseline 的 `0.65` H/M min recall。因此该特征集只能作为负结果保留，不能进入 shadow validation。

## 下一阶段建议

不要启动 GPU 长网格，也不要使用 shadow/final。下一阶段应先做一个更小、更物理化的 `v6c_hm_source_design`：

1. 收缩低能无效点：`15/20/25/30 keV` 暂不再作为 H/M 主要判别信息。
2. 优先新增或加密高能尾部信息：围绕 `150/200 keV`、`>120 keV` 响应、厚样本路径做物理设计，而不是继续扩大模型。
3. 若仍只在 validation 上达到 `0.65-0.70`，停止模型侧扩展，转向新增仿真物理维度，例如角度/散射几何、探测器位置分布或更高 photon statistics。
4. 只有 development H/M min recall `>=0.80` 且 pairwise min recall `>=0.75`，才允许对 shadow `1701-1710` 做一次确认。

Nature 编辑角度的边界：当前不能声称十材料严格泛化准确率提高；只能声称 H/M bottleneck 已定位到高能尾部响应不足和融合特征表达不足。
