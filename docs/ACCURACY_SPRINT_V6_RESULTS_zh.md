# G4 Accuracy Sprint v6 阶段结果

## v5_hm_lowwide runner

`v5_hm_lowwide` 已完整跑完。

| 项目 | 结果 |
| --- | --- |
| selected rows | `6760` |
| completed | `6760` |
| failed | `0` |
| pending | `0` |
| completed at | `2026-04-29T04:53:39+08:00` |

## H/M development gate

Audit 输出：

`results/accuracy_v3/v5_hm_lowwide/`

Gate report：

`results/accuracy_v3/v5_hm_lowwide/v6_gate_report.json`

| 指标 | v6 门槛 | 观测值 | 结论 |
| --- | ---: | ---: | --- |
| Top-1 | `0.88` | `0.8292` | fail |
| macro-F1 | `0.84` | `0.8292` | fail |
| min class recall | `0.75` | `0.6500` | fail |
| H/M min recall | `0.80` | `0.6500` | fail |
| H/M pairwise min recall | `0.75` | `0.5667` | fail |
| min class support | `40` | `60` | pass |
| runner failed zero | `true` | `true` | pass |

Selected method: `HistGradientBoosting`.

Per-class recall:

| material | support | recall |
| --- | ---: | ---: |
| Pyrite | `60` | `1.0000` |
| Hematite | `60` | `0.6667` |
| Magnetite | `60` | `0.6500` |
| Chalcopyrite | `60` | `1.0000` |

H/M pairwise audit:

| 指标 | 结果 |
| --- | ---: |
| pairwise Top-1 | `0.6250` |
| pairwise macro-F1 | `0.6237` |
| Hematite recall | `0.6833` |
| Magnetite recall | `0.5667` |
| H/M min recall | `0.5667` |
| ROC AUC Magnetite | `0.7078` |

## 结论

v6 H/M development gate 未通过，因此不得启动预注册 GPU grid search，也不得进入 shadow gate 或十材料阶段。

这不是样本数不足问题：validation 每类 support 已达到 `60`。失败仍集中在 Hematite/Magnetite 互相混淆：

- Magnetite：`21/60` 被误判为 Hematite。
- Hematite：`20/60` 被误判为 Magnetite。

低能扩展 `15/20/25/30/35 keV` 加上原有高能点提升了数据覆盖，但仍不足以让现有 transmission / spectral-shape / detector-response 特征稳定区分两种铁氧化物。

下一步应先改变物理信息或特征，而不是用 GPU 长训练硬刷 validation。优先方向：

1. H/M-only 物理诊断：按 energy 和 thickness 分组输出 pairwise recall，定位哪些能量/厚度提供真实区分信息。
2. 增加差分特征：跨能量 attenuation slope、low/high energy ratio、thickness response curvature、H/M pairwise margin features。
3. 若仍低于 gate，考虑新增仿真物理维度，例如角度/散射几何或更高统计量，而不是打开 final。
