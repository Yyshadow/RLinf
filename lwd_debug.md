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
runner.save_interval: 500
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

1. 每 500 step 保存一次 checkpoint。
2. 优先看 `train/loss` 是否继续降到更低，至少先接近 `0.00x` 量级。
3. 每个 checkpoint 用训练集 seeds 做闭环 eval，`total_num_envs` 建议先不超过 4，
   因为 RoboTwin 多相机环境曾在 10 env 下出现 `cannot create buffer`。
4. 对最好的 checkpoint 开启 `record_internal_camera=true` 生成完整视频，确认失败
   是模型动作问题还是评估/录像误判。这个录像路径不拆 action chunk。

## 2026-07-07：RoboTwin pi0.5 eval 内部相机视频逻辑

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

### 为什么外层 RecordVideo 会很短

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

正式方案改为 RoboTwin 内部相机录制，不再拆 action chunk。RLinf 只把
`video_cfg` 传入 RoboTwin，正式执行路径仍然保持：

```text
OpenPI output: [B, 50, 14]
RoboTwinEnv.chunk_step()
  -> self.venv.step(chunk_actions)
  -> RoboTwin gen_sparse_reward_data(chunk_actions)
  -> TOPP 对完整 50-step chunk 规划
  -> 内部 scene.step() 执行
```

新增配置：

```yaml
env:
  eval:
    video_cfg:
      save_video: true
      record_internal_camera: true
      camera: head_camera
      fps: 25
      write_every_n_sim_steps: 10
      video_base_dir: ${runner.logger.log_path}/video/internal
```

核心代码改动：

```text
rlinf/envs/robotwin/robotwin_env.py
  _init_env()
  -> task_config["video_cfg"] = cfg.video_cfg

rlinf/workers/env/env_worker.py
  record_internal_camera=true 时跳过外层 RecordVideo wrapper

/data/wam_codebase/RoboTwin_RLinf/robotwin/envs/vector_env.py
  VectorEnv 读取 video_cfg
  -> eval_video_log / eval_video_base_dir / fps / stride / camera

/data/wam_codebase/RoboTwin_RLinf/envs/_base_task.py
  gen_sparse_reward_data()
  -> 原有 scene.step()
  -> 原有 _update_render()
  -> 每隔 write_every_n_sim_steps 只读取 head_camera RGB 并写入 ffmpeg
```

这个方案只增加相机读取和视频写入，不新增 `scene.step()`，不改变 TOPP 输入，
不改变 joint target，不改变 reward / success 计算。

### 为什么移除旧拆 chunk 录像开关

之前的临时方案是在 RLinf wrapper 里把 `[B, 50, 14]` 拆成 50 次
`[B, 1, 14]` 执行。它能让外层 `RecordVideo` 得到更多帧，但会改变
RoboTwin 的执行语义：

```text
完整 chunk:
  TOPP 对 50 个路点整体规划

拆 chunk:
  TOPP 每次只看 1 个路点，重复局部规划 50 次
```

因此旧拆 chunk 录像开关和对应函数已移除。
正式视频现在只能走内部相机录制，避免把诊断视频误当成正式执行效果。

### 这个方案解决了什么

1. 解决了 `chunk=50` eval 视频只有 5-6 帧的问题。
2. 保留 pi0.5 policy 一次输出 50 个 action 的真实推理方式。
3. 保留 RoboTwin/TOPP 对完整 chunk 的原始执行方式。
4. 录像不会改变仿真步进、动作执行、reward 或 success 判断。
5. 多 env 录像目录按 `task/seed/env_id` 隔离，避免并行 eval 覆盖视频。

### 推荐使用方式

生成单条正式语义视频：

```bash
PATH=/data/wam_codebase/RLinf/.venv-openpi/bin:$PATH \
ROBOTWIN_PATH=/data/wam_codebase/RoboTwin_RLinf \
ROBOTWIN_ASSETS_PATH=/data/wam_codebase/RoboTwin_RLinf \
PYTHONPATH=/data/wam_codebase/RLinf:/data/wam_codebase/RoboTwin_RLinf \
HYDRA_FULL_ERROR=1 MPLCONFIGDIR=/tmp XDG_CACHE_HOME=/tmp CUDA_VISIBLE_DEVICES=0 \
bash evaluations/run_eval.sh robotwin robotwin_beat_block_hammer_openpi_pi05_eval \
  env.eval.total_num_envs=1 \
  runner.logger.log_path=/data/wam_codebase/RLinf/outputs/eval_pi05_hammer50_internal_video
```

如果只做批量指标评估、不需要视频，可以关掉内部相机录制以节省时间：

```bash
bash evaluations/run_eval.sh robotwin robotwin_beat_block_hammer_openpi_pi05_eval \
  env.eval.video_cfg.save_video=false \
  env.eval.video_cfg.record_internal_camera=false \
  env.eval.total_num_envs=4 \
  runner.logger.log_path=/data/wam_codebase/RLinf/outputs/eval_pi05_hammer50_metrics
```

## 2026-07-07：视频与数据闭环文档最终清理

本次进一步把临时方案和重复文档收紧：

1. 移除旧拆 chunk 录像配置和对应代码路径。RoboTwin eval 现在只有内部相机录制
   这一条正式视频路径，不再保留会改变 TOPP 执行语义的调试开关。
2. `RoboTwinEnv.chunk_step()` 永远把完整 `[B, 50, 14]` chunk 交给
   `venv.step(chunk_actions)`，避免视频诊断和正式 success rate 使用不同执行方式。
3. 内部相机录制开启时，`EnvWorker` 会跳过 RLinf 外层 `RecordVideo` wrapper，
   避免同时生成一份只有 chunk 末尾帧的旧式视频。
4. 删除 `RoboTwinEnv` 中没有任何指标产出的 `fail_once` 缓存，以及未使用且引用
   未定义 `self.horizon` 的 `sample_action_space()`。
5. 将单独的数据闭环说明合并进 `lwd.md` 的 “Robotwin pi0.5 / LWD 数据闭环”
   章节，并删除原来的独立文档，避免两份说明后续不一致。

## 2026-07-07：pi0.5 eval 支持 receding-horizon 执行

本次给 RoboTwin / OpenPI eval 增加了 `env.eval.action_exec_horizon`：

```yaml
env:
  eval:
    max_steps_per_rollout_epoch: 200
    action_exec_horizon: 20

rollout:
  model:
    num_action_chunks: 50
```

语义是：模型仍然一次预测 50 步 action chunk，但环境只执行前 20 步，
然后重新观测并重新预测下一段 50 步。这样可以测试 “短执行视野 + 高频重规划”
是否比一次完整执行 50 步更适合 hammer 任务。

这次同步改了两侧循环：

1. `EnvWorker` 按 `ceil(max_steps_per_rollout_epoch / action_exec_horizon)` 计算
   eval 交互轮数，并在送入环境前截断 action chunk。
2. HuggingFace rollout worker 使用同样的交互轮数，保证 env 需要多少轮 action，
   rollout 就预测多少轮 action，避免评估中途等待。
3. `robotwin_beat_block_hammer_openpi_pi05_eval.yaml` 默认设置
   `action_exec_horizon: 20`。如果不设置该字段，则直接沿用旧逻辑：
   `max_steps_per_rollout_epoch // num_action_chunks` 个完整 chunk，旧配置保持
   完整 chunk 执行语义。

这个改动只影响 eval，不改变 SFT 训练的 action chunk 长度，也不改变内部相机
录制逻辑。

## 2026-07-07：云端 hope 脚本化和自动 resume

### 背景

云端训练有时会因为机器异常而被平台重启。原来的 hope 文件把 conda 环境、
路径、离线 cache、import 检查和训练命令全部写在 `worker.script` 里，而且
`runner.resume_dir` 默认是 `null`。如果平台重启后重新执行同一个 hope，训练会
从头开始，浪费已经完成的 step。

### 当前改法

把云端训练逻辑从 hope 文件下沉到两个 repo 内脚本：

```text
examples/sft/scripts/train_pi05_hammer50_cloud.sh
examples/lwd/scripts/train_lwd_critic_cloud.sh
```

四个 hope 文件现在只负责申请资源、指定 docker、打开 failover，并调用脚本：

```text
examples/sft/hope/robotwin_pi05_hammer50_smoke_8a100.hope
examples/sft/hope/robotwin_pi05_hammer50_train_8a100.hope
examples/lwd/hope/robotwin_lwd_critic_smoke_8a100.hope
examples/lwd/hope/robotwin_lwd_critic_train_8a100.hope
```

train 模式会自动扫描最新完整 checkpoint，并把它作为
`runner.resume_dir=<global_step_dir>` 传给 RLinf。smoke 模式默认不 resume，
避免调试 smoke 时接到旧的 smoke checkpoint。

### 完整 checkpoint 判断

pi0.5 SFT 需要：

```text
actor/dcp_checkpoint/.metadata
actor/model_state_dict/full_weights.pt
```

LWD critic 额外需要：

```text
actor/target_model.pt
```

这样可以避免机器刚好在保存 checkpoint 时挂掉，留下一个目录存在但文件不完整的
`global_step_*`，导致下次启动加载坏 checkpoint。

### 强制从头开始

如果要重新开一个干净实验，可以在提交前设置：

```bash
export RLINF_FORCE_RESTART=1
```

设置后脚本会跳过自动 resume。否则 train 模式会优先从最新完整 checkpoint 继续。

### 本次解决的问题

1. hope 文件不再维护几十行 shell，降低路径、cache、Hydra 参数改错的概率。
2. 云端自动重启后，同一个 hope 会自动从最新完整 checkpoint 接着训练。
3. pi0.5 和 LWD critic 的 smoke/train 共用同一套环境初始化和离线检查逻辑。
4. 打开 `afo.app.support.engine.failover = true`，平台重启和 RLinf resume 配合使用。
5. 正式 pi0.5 保存间隔改为 500 step，LWD critic 保存间隔改为 1000 step，
   减少机器异常时损失的训练进度。

### 2026-07-07 补充：`CONDA_ENV=smoke` 云端变量冲突

第一次提交拆分后的 smoke hope 时，stderr 里只有：

```text
EnvironmentNameNotFound: Could not find conda environment: smoke
```

原因不是缺少名为 `smoke` 的 conda 环境，而是 cloud 脚本内部变量原来叫
`CONDA_ENV`，这个名字在云端运行环境里可能已经被平台或提交参数污染成了
`smoke`。脚本执行到 `conda activate "${CONDA_ENV}"` 时就变成了
`conda activate smoke`。

已修复为：

```bash
RLINF_CONDA_ENV_NAME="${RLINF_CONDA_ENV_NAME:-${RLINF_CONDA_ENV:-rlinf_lwd}}"
conda activate "${RLINF_CONDA_ENV_NAME}"
```

后续如果要覆盖 conda 环境，使用 `RLINF_CONDA_ENV_NAME` 或兼容旧写法
`RLINF_CONDA_ENV`；不要在 hope 或云端环境里依赖裸的 `CONDA_ENV`。
