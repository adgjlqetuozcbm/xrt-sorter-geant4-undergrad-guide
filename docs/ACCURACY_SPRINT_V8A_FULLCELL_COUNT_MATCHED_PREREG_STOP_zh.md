# v8A full-cell count-matched 训练前置 stop/prereg

Generated: 2026-05-08

## 一句话结论

当前 admitted full-cell view 仍然不能提交为模型训练证据。它已经通过 clean admission，但在训练前置控制里暴露出两个问题：

- `0.020` 和 `0.015` count-matched 窗口的 matched-pair support 不够；
- 支持量够的宽窗口 `0.040/0.050` 仍然挡不住 total-count-only 和 shuffled-label/null 控制。

因此本阶段的正确动作不是继续调模型，而是预注册一个新的 source-side full-cell count-matched matrix。该矩阵只允许进入生成与 preflight，不解锁训练、不碰 shadow/final、不扩展到 full ten-material。

## 当前 view 为什么不能提交训练

当前输入仍是：

- `results/accuracy_v3/v8a_fullcell_residualization_protocol_rework_v1_source_scaled_no_residualization_event_to_feature/`

它的 final data audit 结论是：

- decision `training_unlocked_with_count_controls_required`
- `training_unlocked=true`
- `count_controls_required=true`

这只表示可以做 development-only 训练诊断，不表示可以把训练结果当成干净证据提交。

full-cell baseline gate 已经 stop：

- decision `stop_or_rework_fullcell_clean_development_baseline`
- ordinary gate `false`
- count-balanced gate `false`
- ordinary total-count-only H/M min recall `0.7292`
- ordinary shuffled-label null p95 `0.5804`
- count-balanced total-count-only H/M min recall `0.7946`
- count-balanced shuffled-label null p95 `0.5540`

这些控制没有过，所以不能用当前数据启动稳定性复制、advanced probe 或任何 shadow/final 计划。

## Count support 复核

对当前 admitted view 做 count-balanced support sweep：

| window | train pairs | validation pairs | stress pairs | support |
| --- | ---: | ---: | ---: | --- |
| `0.003` | 24 | 18 | 12 | fail |
| `0.005` | 48 | 37 | 27 | fail |
| `0.010` | 97 | 67 | 66 | fail |
| `0.015` | 146 | 98 | 105 | fail |
| `0.020` | 194 | 132 | 123 | fail |
| `0.040` | 326 | 224 | 224 | pass |
| `0.050` | 419 | 280 | 277 | pass |

预注册的训练支持目标是：

- train `>=300`
- validation `>=200`
- stress-holdout `>=200`

所以 `0.020` 主窗口和 `0.015` strict sensitivity 都不够。`0.040/0.050` 虽然够，但之前 gate 已经证明它们仍保留很强 count shortcut，不能用来升级证据。

## 新矩阵 prereg

新增 source-side config：

- `analysis/configs/v8a_clean_hm_fullcell_countmatched_matrix_config.json`

核心设计：

- profile `v8a_hm_clean_fullcell_countmatched_prereg_cif_lit`
- H/M only
- source-on default only
- thickness `[3, 10, 30, 60]`
- pose `[0, 1, 2]`
- count target bins `3600 / 4000 / 4400`
- 每个最细 full nuisance cell 从 `2` 对 H/M 增加到 `5` 对 H/M
- expected rows `5040`
- expected strict pairs train/validation/stress `1080 / 720 / 720`

这个设计的意图不是“靠更多数据刷结果”，而是让 `0.020` 和 `0.015` 窗口有足够 matched-pair support，避免再退回到 `0.040/0.050` 这种已知 confounded 的宽窗口。

## Gate 语义

新增 audit：

- `analysis/audit_v8a_fullcell_count_matched_prereg.py`

输出：

- `results/accuracy_v3/v8a_fullcell_count_matched_training_prereg/v8a_fullcell_count_matched_training_prereg_gate.json`
- `results/accuracy_v3/v8a_fullcell_count_matched_training_prereg/v8a_fullcell_count_matched_training_prereg_report.md`
- `results/accuracy_v3/v8a_fullcell_count_matched_training_prereg/v8a_fullcell_count_matched_support_sweep.csv`

预期 decision：

- `current_view_not_training_submittable_source_matrix_preregistered`

这个 decision 的意思是：

- 当前 view 不能提交训练；
- 新 source-side matrix prereg 通过；
- 下一步只允许生成矩阵和跑 preflight；
- 训练仍然锁住。

## 下一步顺序

只允许按以下顺序继续：

1. 运行 prereg audit。
2. 生成 `v8a_hm_clean_fullcell_countmatched_prereg_cif_lit` matrix。
3. 跑 matrix preflight，确认 rows、pairs、split/seed 隔离、source-on/default、no shadow/final。
4. preflight 通过后才允许 development-only Geant4。
5. 新 Geant4 输出完成后再跑 event-to-feature。
6. 再跑 final data audit。
7. 再跑 full-cell ordinary/count-balanced training gate，且必须同时满足 count-only、lineage-only、shuffled-label/null 控制。

## 明确禁止

- 不从当前 admitted view 继续训练晋级。
- 不把 `0.040/0.050` support-pass 写成训练证据。
- 不做 shadow/final。
- 不做 full ten-material matrix。
- 不做产品准确率、硬件验证、ordinary XRT 解决 H/M、manuscript-grade powder XRD claim。
- 不用 advanced model 绕过 failed count/null controls。

## Claim boundary

可以说：

- 当前 full-cell clean view 是 admitted diagnostic input；
- count-aware training evidence 仍被阻断；
- 新的 5-replicate source-side full-cell matrix 已被预注册用于解决 `0.020/0.015` matched-pair support 不足。

不能说：

- H/M 已经训练成功；
- 当前数据已经可以提交模型训练证据；
- count shortcut 已经解决；
- shadow/final 或产品级性能已解锁。
