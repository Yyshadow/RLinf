# LWD / pi0.5 阶段性实验结果

更新时间：2026-07-07

本文档记录当前 RoboTwin `beat_block_hammer` 上已经完成的阶段性结果，主要包含两部分：

1. pi0.5 SFT actor 的闭环评估结果。
2. LWD critic 的训练指标和离线曲线诊断结果。

## 1. 当前主要产物

### pi0.5 actor

当前主要使用的 pi0.5 SFT checkpoint：

```text
/data/wam_codebase/RLinf/checkpoints/rlinf_pi05_sft_10000/global_step_10000
```

对应权重：

```text
/data/wam_codebase/RLinf/checkpoints/rlinf_pi05_sft_10000/global_step_10000/actor/model_state_dict/full_weights.pt
```

训练配置目标：

- 使用 50 条 `beat_block_hammer` 成功 demo 做 pi0.5 full finetuning。
- 训练 10000 step。
- 当前云端配置每 500 step 保存一次 checkpoint，便于机器重启后自动 resume。
- 当前评估主要看 `global_step_10000`。

### LWD critic

当前主要使用的 critic checkpoint：

```text
/data/wam_codebase/RLinf/checkpoints/robotwin_lwd_critic_train_8a100/checkpoints/global_step_8000
```

对应权重：

```text
/data/wam_codebase/RLinf/checkpoints/robotwin_lwd_critic_train_8a100/checkpoints/global_step_8000/actor/model_state_dict/full_weights.pt
```

保存节点：

```text
global_step_2000
global_step_4000
global_step_6000
global_step_8000
```

## 2. pi0.5 actor 闭环评估结果

### 2.1 `global_step_500` 早期结果

早期 `global_step_500` 的闭环效果较差：

| Eval 目录 | 轨迹数 | 成功率 |
| --- | ---: | ---: |
| `outputs/eval_pi05_hammer50_step500_quick` | 4 | 0/4 |
| `outputs/eval_pi05_hammer50_step500_train4_noise05` | 4 | 0/4 |

结论：`global_step_500` 基本还没有形成稳定闭环能力，后续改为从 base package 重新训练 10000 step，而不是继续沿用这个 500-step run。

### 2.2 `global_step_10000` 小规模 eval

`global_step_10000` 相比 step500 有明显提升，但仍不稳定。

| Eval 目录 | 轨迹数 | 成功率 | 备注 |
| --- | ---: | ---: | --- |
| `outputs/eval_pi05_hammer50_step10000_metrics_eval4_clean` | 4 | 0/4 | eval seeds 小批量失败 |
| `outputs/eval_pi05_hammer50_step10000_metrics_train4` | 4 | 2/4 | train seeds 上能完成部分闭环 |

结论：模型不是完全没有学会任务，而是对初始状态、末端对准和扰动仍然敏感。训练 seed 上出现成功，说明已经学到一部分 hammer 行为；eval seed 上不稳定，说明泛化和闭环鲁棒性还不足。

### 2.3 不同 `action_exec_horizon` 的 30 次 eval

使用 `global_step_10000` 权重，对同一任务测试不同执行步长：

| 执行方式 | Eval 目录 | 轨迹数 | 成功数 | 成功率 |
| --- | --- | ---: | ---: | ---: |
| `action_exec_horizon=20` | `outputs/eval_pi05_hammer50_gs10000_exec20_n30` | 30 | 2 | 6.67% |
| `action_exec_horizon=30` | `outputs/eval_pi05_hammer50_gs10000_exec30_n30` | 30 | 11 | 36.67% |
| `action_exec_horizon=50` | `outputs/eval_pi05_hammer50_gs10000_exec50_n30` | 30 | 2 | 6.67% |

解释：

- 模型仍然一次预测 50-step action chunk。
- `action_exec_horizon=30` 表示只执行前 30 个 step，然后重新观测和重新规划。
- `action_exec_horizon=50` 是完整执行整个 chunk，重规划频率最低。
- `action_exec_horizon=20` 重规划更频繁，但当前效果也不好，可能是动作 chunk 前段尚未充分完成抓取/挥锤动作，过早重规划反而破坏时序。

阶段性结论：

```text
当前 pi0.5 hammer actor 最合理的 eval 设置是 action_exec_horizon=30。
```

这说明当前模型受益于 receding-horizon 执行，但不能过短，也不能完整执行 50 步。

### 2.4 `exec_horizon=30` 视频诊断批次

为了具体观察成功和失败行为，又使用 `action_exec_horizon=30` 跑了一批带内部相机录像的 eval：

```text
/data/wam_codebase/RLinf/outputs/eval_pi05_hammer50_gs10000_exec30_video_pick20
```

结果：

| 轨迹数 | 成功数 | 失败数 | 成功率 |
| ---: | ---: | ---: | ---: |
| 20 | 5 | 15 | 25% |

已经从中整理出 5 个成功视频和 5 个失败视频：

成功视频：

```text
/data/wam_codebase/RLinf/outputs/eval_pi05_hammer50_gs10000_exec30_selected_videos/success
```

失败视频：

```text
/data/wam_codebase/RLinf/outputs/eval_pi05_hammer50_gs10000_exec30_selected_videos/failure
```

对应关系说明：

```text
/data/wam_codebase/RLinf/outputs/eval_pi05_hammer50_gs10000_exec30_selected_videos/README.txt
```

5 个成功视频：

| 文件 | seed | env_id |
| --- | ---: | ---: |
| `success_01_seed_100112514_env_1.mp4` | 100112514 | 1 |
| `success_02_seed_100175033_env_2.mp4` | 100175033 | 2 |
| `success_03_seed_100100065_env_0.mp4` | 100100065 | 0 |
| `success_04_seed_100162521_env_0.mp4` | 100162521 | 0 |
| `success_05_seed_100162537_env_1.mp4` | 100162537 | 1 |

5 个失败视频：

| 文件 | seed | env_id |
| --- | ---: | ---: |
| `failure_01_seed_100137506_env_0.mp4` | 100137506 | 0 |
| `failure_02_seed_100187555_env_3.mp4` | 100187555 | 3 |
| `failure_03_seed_100150010_env_4.mp4` | 100150010 | 4 |
| `failure_04_seed_100125012_env_1.mp4` | 100125012 | 1 |
| `failure_05_seed_100162556_env_2.mp4` | 100162556 | 2 |

当前观察倾向：

- 成功率还不够高，但模型已经能完成一部分抓取、接近和敲击过程。
- 失败不一定是完全不会做，很多更可能是 near-miss，例如抓取、挥锤或末端对准差一点。
- 需要继续逐条看失败视频，把失败分成“抓锤失败 / 接近但没敲准 / 时序问题 / 末端姿态问题”等类别。

## 3. LWD critic 训练结果

### 3.1 训练指标

critic 的主要 TensorBoard 目录：

```text
/data/server/tensorboard_7.4
```

训练 8000 step 后，关键指标如下：

| 指标 | step 0 | step 7999 | 趋势 |
| --- | ---: | ---: | --- |
| `train/q_loss` | 0.2506 | 0.0318 | 明显下降 |
| `train/value_loss` | 5.7347 | 3.7984 | 下降 |
| `train/q_head_gap` | 0.1071 | 0.0104 | 明显收敛 |
| `train/grad_norm` | 87.57 | 8.86 | 明显变稳 |
| `train/q_min_mean` | 0.0074 | 0.4487 | 上升到合理区间 |
| `train/target_q_mean` | 0.5090 | 0.4453 | 保持在中间价值范围 |

解释：

- `q_loss` 下降说明 action-conditioned Q 分支在拟合 bootstrap target。
- `value_loss` 下降说明 distributional value head 在学习 target distribution。
- `q_head_gap` 下降说明 double-Q 两个 head 的差距在缩小，训练更稳定。
- `grad_norm` 从很高降下来，说明训练早期的大梯度逐渐稳定。

### 3.2 critic eval 指标

step 7999 的关键 eval 指标：

| Eval 数据 | `q_min_mean` | `target_q_mean` | 备注 |
| --- | ---: | ---: | --- |
| success | 0.6157 | 0.7351 | 成功轨迹价值最高 |
| failed | 0.3665 | 0.2896 | 失败轨迹较低 |
| nearmiss | 0.3387 | 0.2705 | near-miss 也较低 |

success 内部更细指标：

| 指标 | 数值 |
| --- | ---: |
| `eval/beat_block_success/success/q_min_mean` | 0.7580 |
| `eval/beat_block_success/other/q_min_mean` | 0.5523 |

阶段性结论：

```text
critic 已经能把 success 和 failed/nearmiss 拉开。
```

这比只看训练 loss 更重要，因为它说明 critic 不是单纯拟合数值，而是在不同类型 episode 上产生了可解释的价值差异。

## 4. LWD critic 离线曲线诊断

离线曲线输出目录：

```text
/data/wam_codebase/RLinf/outputs/lwd_critic_value_curves
```

主要文件：

```text
robotwin_lwd_critic_episode_values.pdf
robotwin_lwd_critic_episode_values.png
robotwin_lwd_critic_episode_values.csv
robotwin_lwd_critic_episode_values_summary.json
```

对应视频：

```text
outputs/lwd_critic_value_curves/videos/success/
outputs/lwd_critic_value_curves/videos/failed/
outputs/lwd_critic_value_curves/videos/nearmiss/
```

summary 中记录的第一帧和最后一帧价值：

| 类型 | V 首帧 | V 末帧 | Q 首帧 | Q 末帧 |
| --- | ---: | ---: | ---: | ---: |
| success | 0.536 | 0.932 | 0.601 | 0.942 |
| failed | 0.464 | 0.062 | 0.501 | 0.049 |
| nearmiss | 0.530 | 0.044 | 0.593 | -0.008 |

解释：

- success episode 后期 V/Q 明显升高，符合“接近任务完成”的趋势。
- failed 和 nearmiss 后期 V/Q 明显降低，说明 critic 能识别出后续成功概率变低。
- nearmiss 的末尾 Q 甚至低于 failed，可能是因为该轨迹后续 action chunk 对完成任务更不利。

需要注意：

```text
LWD critic 不是 collision detector，也不是逐帧失败分类器。
```

它学到的是“当前状态和动作 chunk 往后是否更可能完成任务”的进展信号。因此价值不一定在某个 collision 或失误帧突然断崖式下降，而可能表现为一段趋势上的平台化或逐渐下降。

## 5. 当前阶段性判断

### pi0.5 actor

当前 pi0.5 actor 已经学到部分任务行为，但闭环成功率还不稳定：

- `exec_horizon=30` 下 30 次 eval 达到 36.67%。
- 带视频的 20 次 eval 达到 25%。
- 说明模型不是完全不可用，但距离稳定策略还有差距。

更可能的问题：

1. 50 条 success demo 的覆盖范围偏小。
2. 末端姿态和敲击对准误差对成功率影响很大。
3. 完整执行 50 步 chunk 太长，容易积累误差。
4. 过短执行 20 步也不好，可能破坏动作时序。

当前推荐 eval 设置：

```yaml
env:
  eval:
    action_exec_horizon: 30
```

### LWD critic

critic 的阶段性结果更积极：

- 训练 loss 正常下降。
- eval 上 success 的 Q 明显高于 failed/nearmiss。
- 离线曲线中 success 后期上升，failed/nearmiss 后期下降。

当前判断：

```text
critic 已经具备一定区分能力，可以作为后续策略优化或数据筛选的候选信号。
```

但仍需要注意：

- 当前只是离线诊断和 supervised/TD critic 训练闭环。
- 还没有真正接入 actor 更新。
- 后续要验证 critic 指导策略更新时是否能提升真实闭环 success rate。

## 6. 下一步建议

1. 逐条观看 `exec_horizon=30` 的 5 个成功和 5 个失败视频，人工标注失败类型。
2. 如果失败主要是 near-miss，可以优先补充更多末端对准和敲击角度多样的数据。
3. 如果失败主要是抓锤失败，需要补充更多抓取阶段覆盖。
4. 继续保留 `action_exec_horizon=30` 作为当前主 eval 设置。
5. 用 critic 对成功、失败、near-miss 轨迹做更大规模离线打分，确认排序是否稳定。
6. critic 排序稳定后，再考虑把 critic 接入后续 actor 更新。
