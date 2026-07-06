# pi0.5 Hammer50 SFT 调试记录

## 2026-07-06：pi0.5 SFT expert-only 的 view-inplace 报错

### 现象

云端运行 `robotwin_sft_openpi_pi05_hammer50_cloud.yaml` 时，环境、离线 tokenizer、OpenPI import、RLinf import 都已经通过，训练在第一步 forward/backward 前后失败。

关键日志形态：

```text
pi05 sft imports ok
Forcing gradient checkpointing to be enabled for Gemma expert model
RuntimeError: Output 0 of ViewBackward0 is a view and its base or another view of its base has been modified inplace.
```

报错栈进入了 OpenPI 的 Gemma expert：

```text
FSDPVlaSftWorker.run_training
-> OpenPi0ForRLActionPrediction.sft_forward
-> PI0Pytorch.forward
-> paligemma_with_expert.forward
-> openpi/models_pytorch/gemma_pytorch.py
```

所以这次不是数据路径、norm_stats、tokenizer、离线缓存的问题，而是 pi0.5 SFT 的模型前向路径触发了 PyTorch autograd 的 view/inplace 限制。

### 根因判断

当前失败配置里 `actor.model.openpi.train_expert_only: true`，也就是只训练 action expert、冻结 PaliGemma/VLM。

RLinf 现有配置中已经记录过同类问题：

```yaml
# model/pi0_5 defaults train_expert_only to True, which freezes
# PaliGemma and breaks FSDP+gradient-checkpointing with a
# view-inplace error. SFT here trains the full model.
train_expert_only: False
```

对应文件：

```text
examples/sft/config/realworld_sft_openpi_dual_franka_tcp_rot6d.yaml
```

也就是说，这不是 hammer50 数据特有问题，而是当前 RLinf + OpenPI pi0.5 + FSDP SFT 路径下的已知组合风险：

```text
pi0.5 SFT
+ train_expert_only=True
+ FSDP
+ OpenPI 内部 Gemma expert gradient checkpointing
=> view-inplace autograd error
```

即使 RLinf 配置里写了：

```yaml
actor.fsdp_config.gradient_checkpointing: false
```

日志里仍然出现：

```text
Forcing gradient checkpointing to be enabled for Gemma expert model
```

说明 OpenPI 内部仍会在 Gemma expert 侧强制启用相关 checkpoint 路径，最终与 FSDP/冻结参数组合产生冲突。

### 为什么 PPO/GRPO 默认可以 expert-only

PPO/GRPO 的 OpenPI 配置多数继承：

```text
examples/embodiment/config/model/pi0_5.yaml
```

其中默认：

```yaml
train_expert_only: True
```

这表示 RL 后训练通常冻结 VLM，只训练 action expert；如果 `add_value_head=True`，还会训练 value head。

它和 SFT 的关键区别是 forward 路径不同：

```text
SFT:
  sft_forward()
  -> super().forward(observation, actions)
  -> OpenPI 原生 PI0Pytorch.forward

PPO/GRPO:
  default_forward()
  -> get_log_prob_value()
  -> 对 rollout 保存的 denoise chain 重算 logprob/value
```

所以 PPO/GRPO 默认 expert-only 不等于 SFT expert-only 一定可跑。当前出错的是 SFT 的 `super().forward()` 路径。

### 当前处理方案

为了先把 hammer50 的 pi0.5 SFT 云端训练链路跑通，当前采用 full finetuning：

```yaml
actor.model.openpi.train_expert_only: false
```

最初用于验证链路的 500-step 配置比较保守：

```yaml
actor.micro_batch_size: 2
actor.global_batch_size: 16
actor.optim.lr: 7.5e-6
actor.optim.min_lr: 7.5e-7
runner.max_steps: 500
runner.save_interval: 100
```

smoke 作业进一步覆盖为更小配置：

```yaml
actor.micro_batch_size: 1
actor.global_batch_size: 8
runner.max_steps: 10
runner.save_interval: 10
```

推荐运行顺序：

1. 先跑 `examples/sft/hope/robotwin_pi05_hammer50_smoke_8a100.hope`，确认 10 step 能完整 forward/backward/save。
2. smoke 通过后，再跑 `examples/sft/hope/robotwin_pi05_hammer50_train_8a100.hope`。
3. 如果 full finetune 显存稳定，正式训练使用下面的 10k overfit 配置；如果 OOM，再把 `micro_batch_size` 从 4 降到 2。

### 后续如果必须只训 expert

只训 expert 不是理论上不可行，但当前不能只靠 YAML 稳定解决。需要代码级处理 OpenPI 内部强制启用 Gemma expert gradient checkpointing 与 FSDP/view 的冲突，例如：

- 禁止 expert-only SFT 时 Gemma expert 进入强制 checkpoint 路径；
- 或在 OpenPI Gemma expert 的 view/inplace 敏感路径上做更细的 out-of-place/clone 处理；
- 或改用不触发该 SFT `super().forward()` 的训练路径。

在没有完成代码级修复前，hammer50 SFT 的稳定方案是 full finetuning。

## 2026-07-06：hammer50 50 条数据的 10k overfit 方案

step500 checkpoint 已经能保存并 eval，但闭环效果不理想：普通 eval seed 失败，
训练集 seed 上的快速闭环检查也没有成功。这说明问题不只是泛化，至少当前 500
step 还没有把 50 条成功数据拟合到足够好。

参考 RLinf 自带的 RobotWin pi0.5 SFT 配置：

```text
examples/sft/config/robotwin_sft_openpi_pi05.yaml
```

官方风格的关键参数是：

```yaml
actor.micro_batch_size: 32
actor.global_batch_size: 64
actor.optim.lr: 2.5e-5
actor.optim.weight_decay: 1.0e-10
actor.optim.lr_warmup_steps: 1000
actor.model.openpi.train_expert_only: false
```

hammer50 只有 50 条成功 demo，目标先不是泛化，而是确认训练链路能否把训练集
overfit 到闭环可用。因此当前正式云端配置改为：

```yaml
runner.max_steps: 10000
runner.save_interval: 1000
runner.logger.experiment_name: pi05_hammer50_overfit50_10k_v1

actor.micro_batch_size: 4
actor.global_batch_size: 64
actor.model.openpi.train_expert_only: false
actor.model.openpi.noise_level: 0.5

actor.optim.lr: 2.5e-5
actor.optim.min_lr: 2.5e-6
actor.optim.weight_decay: 1.0e-10
actor.optim.lr_warmup_steps: 500
actor.optim.total_training_steps: ${runner.max_steps}
```

这里不从 `global_step_500` 继续 resume。原因是旧 500-step run 用的是更小学习率和
短 cosine schedule，末尾学习率已经衰减很多；继续 resume 会继承旧 optimizer 和
scheduler 状态，不适合新的 10k overfit 实验。应该从
`pi05_base_hammer50` base package 重新启动一个新实验。

后续验证建议：

1. 每 1000 step 保存一次 checkpoint。
2. 优先看 `train/loss` 是否继续降到更低，至少先接近 `0.00x` 量级。
3. 每个 checkpoint 用训练集 seeds 做闭环 eval，`total_num_envs` 建议先不超过 4，
   因为 RoboTwin 多相机环境曾在 10 env 下出现 `cannot create buffer`。
4. 对最好的 checkpoint 再开启 `record_chunk_frames=true` 生成完整视频，确认失败
   是模型动作问题还是评估/录像误判。

## 2026-07-06：RoboTwin pi0.5 eval 完整视频生成逻辑

### 现象

最开始用 `evaluations/robotwin/robotwin_beat_block_hammer_openpi_pi05_eval.yaml`
做 step500 checkpoint 可视化时，视频可以生成，但只有很少几帧：

```text
200 env steps
num_action_chunks = 50
=> 只有约 4 个 chunk 末尾观测 + reset/final 相关帧
=> mp4 大约 5-6 帧，无法判断中间抓锤子、移动、敲击过程
```

这个不是 mp4 编码坏了，也不是 TensorBoard 或 `ffmpeg` 问题，而是
RoboTwin eval 的 step 接口返回粒度太粗。

### 旧逻辑为什么视频很短

RLinf 的视频保存由 `RecordVideo` wrapper 负责：

```text
rlinf/workers/env/env_worker.py
  if env_cfg.video_cfg.save_video:
      env = RecordVideo(env, env_cfg.video_cfg)

rlinf/envs/wrappers/record_video.py
  RecordVideo.chunk_step()
  -> env.chunk_step()
  -> record_video_in_result(result)
  -> 从 obs_list 里取图并写 mp4
```

也就是说，`RecordVideo` 能保存多少帧，取决于 `RoboTwinEnv.chunk_step()`
返回的 `obs_list` 里有多少个观测。

原来的 `RoboTwinEnv.chunk_step()` 是一次性把完整 action chunk 交给
RoboTwin：

```text
chunk_actions: [B, 50, 14]
self.venv.step(chunk_actions)
```

然后只调用一次 `_extract_obs_image(raw_obs)`，所以 `obs_list` 里只有
一个 chunk 末尾观测。对于 `max_episode_steps=200`、`chunk=50` 的 eval，
整条 episode 只有 4 次 `chunk_step()`，因此视频天然只有几帧。

注意：只改 YAML 里的 `render_freq: 1` 不足以解决这个问题。RLinf 当前
视频不是直接读取 RoboTwin 内部 render 日志，而是读取 env step 返回的
observation；只要 `chunk_step()` 只返回 chunk 末尾观测，`RecordVideo`
就没有中间帧可以写。

### 当前解决方案

在 `rlinf/envs/robotwin/robotwin_env.py` 中新增了一个视频诊断开关：

```yaml
env:
  eval:
    video_cfg:
      save_video: true
      record_chunk_frames: true
      fps: 10
```

对应配置已经写在：

```text
evaluations/robotwin/robotwin_beat_block_hammer_openpi_pi05_eval.yaml
```

核心代码逻辑是：

```text
RoboTwinEnv.chunk_step(chunk_actions)
  if video_cfg.record_chunk_frames:
      _chunk_step_with_recorded_frames(chunk_actions)
  else:
      保持原来的整 chunk venv.step(chunk_actions) 路径
```

打开 `record_chunk_frames` 后，模型侧仍然一次输出完整 action chunk：

```text
rollout.model.num_action_chunks = 50
rollout.model.openpi.action_chunk = 50
policy output: [B, 50, 14]
```

但环境侧为了录像，把这个 chunk 拆成 50 次单步执行：

```text
for step_idx in range(50):
    self.venv.step(chunk_actions[:, step_idx : step_idx + 1, :])
    obs_list.append(extracted_obs)
    infos_list.append(infos)
    chunk_rewards[:, step_idx] = step_reward
    chunk_terminations[:, step_idx] = terminations
    chunk_truncations[:, step_idx] = truncations
```

这样 `RecordVideo.record_video_in_result()` 收到的 `obs_list` 长度就是
50，而不是 1，因此可以把 chunk 内部每一步都写进 mp4。

### 这个方案解决了什么

1. 解决了 `chunk=50` eval 视频只有 5-6 帧的问题。
2. 保留了 pi0.5 policy 一次输出 50 个 action 的真实推理方式。
3. 不再需要把 `num_action_chunks` 临时改成 1；改成 1 虽然能录完整视频，
   但会变成每一步重新规划，不能代表正式 `chunk=50` 行为。
4. 不需要修改外部 RoboTwin 仓库，只在 RLinf 的 env wrapper 侧补齐中间帧。
5. `record_chunk_frames` 默认不改变旧路径；关闭该开关时仍走原来的整 chunk
   `venv.step(chunk_actions)` 逻辑，适合大规模指标评估或追求速度时使用。

### 已验证结果

烟测 50 step：

```text
log_path:
/data/wam_codebase/RLinf/outputs/eval_pi05_hammer50_step500_record_chunk_frames_smoke

video:
/data/wam_codebase/RLinf/outputs/eval_pi05_hammer50_step500_record_chunk_frames_smoke/video/eval/seed_0/0.mp4

ffprobe:
avg_frame_rate=10/1
duration=5.200000
nb_read_frames=52
```

完整 200 step：

```text
log_path:
/data/wam_codebase/RLinf/outputs/eval_pi05_hammer50_step500_video_full_chunk50

video:
/data/wam_codebase/RLinf/outputs/eval_pi05_hammer50_step500_video_full_chunk50/video/eval/seed_0/0.mp4

contact sheet:
/data/wam_codebase/RLinf/outputs/eval_pi05_hammer50_step500_video_full_chunk50/sample_frames/contact_sheet.jpg

ffprobe:
avg_frame_rate=10/1
duration=20.200000
nb_read_frames=202
```

完整 200 step 本次指标：

```text
eval/episode_len: 200
eval/success_once: 0
eval/success_at_end: 0
eval/return: 0
eval/reward: 0
eval/num_trajectories: 1
```

这说明视频生成链路已经正常，当前这条 seed 的失败是模型行为问题，不是视频
保存问题。抽帧中可以看到锤子和红块没有形成成功敲击结果。

### 推荐使用方式

生成单条完整诊断视频：

```bash
PATH=/data/wam_codebase/RLinf/.venv-openpi/bin:$PATH \
ROBOTWIN_PATH=/data/wam_codebase/RoboTwin_RLinf \
ROBOTWIN_ASSETS_PATH=/data/wam_codebase/RoboTwin_RLinf \
PYTHONPATH=/data/wam_codebase/RLinf:/data/wam_codebase/RoboTwin_RLinf \
HYDRA_FULL_ERROR=1 MPLCONFIGDIR=/tmp XDG_CACHE_HOME=/tmp CUDA_VISIBLE_DEVICES=0 \
bash evaluations/run_eval.sh robotwin robotwin_beat_block_hammer_openpi_pi05_eval \
  env.eval.total_num_envs=1 \
  runner.logger.log_path=/data/wam_codebase/RLinf/outputs/eval_pi05_hammer50_step500_video_full_chunk50
```

如果只做批量指标评估、不需要完整视频，可以覆盖关闭逐帧记录以节省时间：

```bash
bash evaluations/run_eval.sh robotwin robotwin_beat_block_hammer_openpi_pi05_eval \
  env.eval.video_cfg.record_chunk_frames=false \
  env.eval.total_num_envs=4 \
  runner.logger.log_path=/data/wam_codebase/RLinf/outputs/eval_pi05_hammer50_step500_metrics
```
