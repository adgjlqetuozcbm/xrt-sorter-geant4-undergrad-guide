# v8A Tail Rebuild v1 Stop Report

## 一句话结论

我们做了一个受控的低自由度 feature rebuild，不新增 Geant4，不训练真实模型，只用现有 clean null-support replication 输出构造 4 个候选视图，并逐个跑 shortcut、paired-clean null、null-tail feature-family、admission。结果是：所有候选都保持 visible shortcut 干净，但没有任何一个通过 paired-clean null 和 admission。因此训练继续锁住。

这说明问题不能简单理解成“把数据再打乱一点”或“删掉某个峰就好”。我们现在面对的是更深的 representation/null-protocol 问题：真实晶体差异必须保留，但当前低自由度压缩仍然不能让假标签尾部稳定压到 `0.55` 以下。

## 本轮做了什么

新增 prereg/config：

- `analysis/configs/v8a_tail_rebuild_v1_config.json`

新增脚本：

- `analysis/build_v8a_tail_rebuild_features.py`
- `analysis/summarize_v8a_tail_rebuild_v1.py`

候选视图：

- `peak_family_balanced`
- `window_ratio_only`
- `leave_out_peak_hematite`
- `leave_out_individual_peaks`

每个候选都跑了：

- feature view builder；
- non-material shortcut gate；
- paired-clean null gate；
- null-tail feature-family gate；
- crystal-clean admission gate。

## 候选结果

| candidate | features | shortcut max BA | paired-null primary p95 | all-mode p95 | primary max | admission |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `peak_family_balanced` | 9 | 0.3924 | 0.6670 | 0.6670 | 0.6875 | stopped |
| `window_ratio_only` | 5 | 0.3924 | 0.7240 | 0.7240 | 0.8333 | stopped |
| `leave_out_peak_hematite` | 12 | 0.4080 | 0.5767 | 0.5767 | 0.6250 | stopped |
| `leave_out_individual_peaks` | 9 | 0.3924 | 0.6670 | 0.6670 | 0.6875 | stopped |

汇总 gate：

- decision：`stop_tail_rebuild_v1_before_training`
- passed candidates：`0`
- training unlocked：`false`
- stop reasons：
  - `no_tail_rebuild_candidate_passed_full_clean_admission`
  - `all_tail_rebuild_candidates_failed_paired_clean_null`
  - `training_remains_locked`

## 如何解释

这轮给出了三个有价值的负结果。

第一，visible shortcut 不是当前主要问题。四个候选的 non-material shortcut gate 都通过，max BA 在 `0.3924-0.4080`。也就是说模型没有明显通过 source/origin/stress/thickness/pose 这些审计变量作弊。

第二，简单降低 individual peak 自由度不够。`peak_family_balanced` 和 `leave_out_individual_peaks` 都把 20 个主特征压到 9 个，但 null p95 反而升高到 `0.6670`。这说明低自由度不等于干净，压缩方式如果保留了某种整体方向，假标签仍可能抓住。

第三，删除 hematite individual peaks 有改善但没过线。`leave_out_peak_hematite` 把 primary p95 降到 `0.5767`，比其他 rebuild views 好，但仍高于 `0.55`，所以不能训练。它只能说明 hematite peak family 是一个重要方向，不说明删除它就是最终方案。

## 下一步应该怎么走

下一步不应该继续盲目删特征，也不应该扩到大开发矩阵。建议进入 `v8A_null_protocol_and_representation_review`：

1. 检查 paired-clean null protocol 是否仍然给了验证集真实标签方向过多选择空间，尤其是 fixed threshold 和 selected threshold 的一致性；
2. 对 `leave_out_peak_hematite` 做更细的 threshold-free 指标，例如 AUC / signed score distribution，而不是只看 recall threshold；
3. 重新审查 residualization 是否把真实材料差异压成了过强的少数线性方向；
4. 设计更物理的低维特征：不是简单 mean/sum，而是 preregistered peak-pair contrast、q-neighborhood merge、H/M shared-window contrast；
5. 如果这些仍失败，及时止损，转向重新生成更高统计量、更稳定物理观测的 clean matrix，而不是继续在旧表示上硬洗。

## 大白话

这次我们没有乱跑，也没有训练。我们只是问了一个很具体的问题：既然假标签尾巴集中在 peak family，那把这些 peak 特征压缩、删减、换成窗口/比例，会不会让假标签考试过关？

答案是：不会，至少这一版不会。

好消息是，脏的 source/origin/stress 结构没有回来。坏消息是，假标签还是能从这些低维表示里抓到一点方向，尤其说明“简单压缩特征”不是万能药。现在最稳的判断是：方向仍然没跑偏，但要从“删特征”升级到“重新设计物理表示和 null protocol”。在 admission 真正写出 `training_unlocked=true` 前，训练和高级模型都继续不许开。
