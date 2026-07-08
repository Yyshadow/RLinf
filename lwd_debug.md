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

第一版修复尝试改用了 `RLINF_CONDA_ENV_NAME`，但云端仍可能通过兼容变量继续污染
环境名。最终修法是只按固定环境路径激活，不再按环境名激活：

```bash
ENV_PREFIX="${CLOUD_ROOT}/Miniforge/envs/rlinf_lwd"
source "${CLOUD_ROOT}/Miniforge/bin/activate" "${ENV_PREFIX}"
python examples/lwd/train_lwd_critic.py ...
```

后续 cloud 脚本不再依赖 `CONDA_ENV`、`RLINF_CONDA_ENV` 或
`RLINF_CONDA_ENV_NAME`。如果 stderr 里还出现 `EnvironmentNameNotFound: smoke`，
就说明云端运行的不是当前脚本版本，需要检查是否已经 `git push`、云端是否已经
`git pull`，以及新增的 `examples/*/scripts/*.sh` 是否真的进入了仓库。

进一步收敛后，hope 文件也不再在 `worker.script` 尾部传 `smoke` 或 `train`
参数。当前入口是：

```text
train:
  worker.script = bash .../train_lwd_critic_cloud.sh
smoke:
  worker.script = bash .../smoke_lwd_critic_cloud.sh
```

这样云端平台的命令解析层不会再看到裸的 `smoke` 参数。`smoke` 只在 wrapper
脚本内部作为 `RLINF_RUN_MODE=smoke` 使用，用来选择短步数配置。

## 2026-07-07：第一版 LWD QAM policy extraction 接入

### 背景

此前代码已经有三块基础能力：

```text
pi0.5 SFT actor
LWD distributional V + chunk Q critic
RoboTwin eval / 视频诊断
```

但还缺少 LWD 论文里真正用 critic 更新 VLA 的策略提取部分。直接最大化
`Q(s, a_chunk)` 会把梯度穿过完整 denoise 过程，计算贵且容易不稳定；只做
advantage-weighted BC 又不适合 OpenPI 这种 flow action generator。因此这次接入
第一版 QAM。

### 当前实现

新增文件：

```text
rlinf/algorithms/lwd/qam.py
rlinf/data/datasets/lwd/qam_dataset.py
rlinf/workers/sft/fsdp_lwd_qam_worker.py
examples/lwd/train_lwd_qam.py
examples/lwd/config/robotwin_beat_block_hammer_lwd_qam_openpi_pi05.yaml
examples/lwd/config/robotwin_beat_block_hammer_lwd_qam_openpi_pi05_smoke.yaml
examples/lwd/scripts/train_lwd_qam_cloud.sh
examples/lwd/scripts/smoke_lwd_qam_cloud.sh
examples/lwd/hope/robotwin_lwd_qam_openpi_pi05_train_8a100.hope
examples/lwd/hope/robotwin_lwd_qam_openpi_pi05_smoke_8a100.hope
```

训练时显式区分三类模型：

```text
actor.model.model_path
  当前要训练的 OpenPI/pi0.5 actor 初始化权重

algorithm.reference_model_path
  固定 reference actor，通常和 SFT checkpoint 一致

critic.model.model_path
  固定 LWD critic checkpoint，通常指向 .../checkpoints/global_step_xxx/actor
```

这样做是为了避免把 base package、SFT checkpoint、critic checkpoint 混成一个
`model_path`，导致 OpenPI loader、critic loader 和 resume 语义互相污染。

### QAM 训练逻辑

每个 batch：

```text
1. LWDQAMDataCollator 同时产出 critic_observation 和 policy_inputs；
2. reference actor 从 Gaussian noise 做 flow_ode rollout；
3. LWD critic 在 reference endpoint 上计算 Q(s, a)；
4. 对 endpoint 求 action gradient；
5. 用 reference flow transition 的 VJP 反向解 adjoint；
6. 当前 actor 用 ForwardType.NFT 在同一批中间 x_t/t 上预测 velocity；
7. 优化 QAM local regression loss + 小权重 anchor loss + 小权重 BC flow loss。
```

第一版固定 critic，不和 actor 一起更新。这样 smoke 和早期实验更容易定位问题：
如果 QAM 后闭环变差，优先排查 policy extraction 和 critic gradient，而不是
critic/value 同时漂移。

### 需要重点看的指标

```text
train/qam_loss
train/anchor_loss
train/bc_loss
train/q_mean
train/action_grad_norm
train/adjoint_norm
train/qam_delta_norm
train/qam_delta_clip_frac
train/grad_norm
```

判断方式：

```text
qam_loss 不应快速 NaN；
action_grad_norm 不能长期为 0，否则 critic 对 action 没有有效梯度；
qam_delta_clip_frac 如果长期接近 1，说明 Q 引导过强或 lambda/clip 太激进；
q_mean 上升只能作为参考，最终仍要看 RoboTwin success rate 和视频。
```

### 当前边界

这一版不是完整在线 LWD：

```text
没有在 QAM step 中继续训练 critic/value；
没有接 QAM-FQL；
没有接 edit policy；
没有 online replay 和 autonomous rollout 混合采样；
reference actor 固定，不做周期性更新。
```

下一步建议先跑 smoke，确认三模型加载、QAM forward/backward/save 都通过；再跑
短训 500-2000 step，并用 `action_exec_horizon=20/30/50` 做闭环 eval 对比 SFT
checkpoint。

## 2026-07-08：QAM 第一性原理修正和当前待办

### 已修正的问题

1. QAM 更新方向按 OpenPI `flow_ode` 重新对齐

   OpenPI 的 deterministic flow ODE 等价于：

   ```text
   x_next = x - dt * v
   ```

   因此 critic 希望最终 action 往高 Q 方向移动时，velocity 的修正方向和 action
   endpoint 的移动方向相反。当前 `qam_vector_field_loss()` 使用
   `v_beta - v_theta` 构造 residual，已经把这个时间方向修正进去。新增的单元测试
   会检查：在简单一维 Q 梯度下，正确方向的 velocity loss 更低，并且 ODE step
   确实把 endpoint 推向更高 Q 的方向。

2. critic action gradient 支持 `mean|min`

   当前配置新增：

   ```yaml
   algorithm:
     qam_critic_grad_mode: mean
   ```

   `mean` 使用所有 Q heads 的均值求 action gradient，默认采用它，因为更平滑，也
   对齐 QAM 源码里 ensemble mean 的做法。`min` 保留为 clipped double-Q 风格的
   保守 ablation。训练日志会同时记录 `q_mean`、`q_min` 和 `q_head_gap`，用于判断
   多个 Q heads 是否分歧过大。

3. action clamp 契约显式化

   当前配置新增：

   ```yaml
   algorithm:
     qam_clip_action_for_critic: false
   ```

   QAM 源码里 clamp 是可选工程开关，LWD 论文公式本身没有强制要求查询 critic 前
   clamp action。默认不 clamp，可以避免 endpoint 超出 `[-1, 1]` 后由于 clamp
   饱和导致 action gradient 变成 0。若后续想做保守实验，可以打开该开关。训练日志
   会记录 `endpoint_min/max`、`endpoint_saturation_frac` 和
   `critic_action_min/max`，用于判断 critic 实际看到的 action 范围。

4. QAM dataloader 可以跨 epoch 继续训练

   之前 QAM worker 直接 `next(data_iter)`，如果 `runner.max_steps` 大于 dataloader
   一个 epoch 的长度，训练会在 `StopIteration` 处中断。现在 worker 内部封装
   `_next_batch()`，一个 epoch 取完后会更新 sampler epoch 并重新创建 iterator，
   因此可以跑完整的多 epoch QAM 训练。

### 当前仍建议补的诊断

这些不是 QAM 本体必须条件，所以本次先不改 critic 算法，只作为后续 TODO：

```text
value_target_clamp_frac
  统计 C51 projection 里有多少 scalar target 被 clip 到 value support 边界。
  如果长期很高，说明 support 范围或 reward/return 尺度需要重新检查。

value_entropy
  统计 V 分布是否过早塌成尖峰，帮助判断 distributional value 是否还在表达不确定性。

success/nearmiss/failed ranking
  离线比较成功、near-miss、失败轨迹的 Q/V 排序，确认 critic 是否真的具备区分能力。
```

QAM 下一轮实验建议先固定：

```text
qam_critic_grad_mode: mean
qam_clip_action_for_critic: false
```

跑通 smoke 后，再用同一个 critic 分别做 `mean` vs `min`、`clip=false` vs
`clip=true` 的小规模 ablation。最终判断仍以 RoboTwin success rate 和完整视频为准，
不要只看 `q_mean` 是否上升。

## 2026-07-08：QAM 云端训练入口路径固化

### 背景

本轮要验证的是 frozen LWD critic 是否能通过 QAM 改进已经 SFT 过的 pi0.5 actor。
云端实际可用 checkpoint 是：

```text
SFT actor:
/mnt/dolphinfs/hdd_pool/docker/user/hadoop-uavcvml/yangyi122/checkpoints/rlinf_pi05_sft_10000/pi05_hammer50_overfit50_10k_v1/checkpoints/global_step_10000

LWD critic:
/mnt/dolphinfs/hdd_pool/docker/user/hadoop-uavcvml/yangyi122/checkpoints/rlinf_lwd/robotwin_lwd_critic_train_8a100/checkpoints/global_step_8000/actor
```

之前 QAM cloud 脚本里的 actor 默认路径仍指向旧的
`checkpoints/rlinf_pi05_sft/...`，容易导致云端提交后加载不到正确 SFT actor。

### 本次修改

1. `examples/lwd/scripts/train_lwd_qam_cloud.sh`

   默认 `RLINF_QAM_ACTOR_MODEL_PATH` 改为新的
   `checkpoints/rlinf_pi05_sft_10000/pi05_hammer50_overfit50_10k_v1/checkpoints/global_step_10000`。
   `RLINF_QAM_REFERENCE_MODEL_PATH` 仍默认等于 actor path，也就是 QAM 从 SFT actor
   初始化，并用同一个 SFT actor 作为固定 reference flow。

2. `examples/lwd/config/robotwin_beat_block_hammer_lwd_qam_openpi_pi05.yaml`

   同步更新 actor 和 reference 的默认路径，保证不用 cloud wrapper 直接启动
   Hydra config 时也不会回到旧 checkpoint。

3. cloud 脚本启动前新增最小路径检查

   检查项包括：

   ```text
   actor/actor/model_state_dict/full_weights.pt
   reference/actor/model_state_dict/full_weights.pt
   critic/model_state_dict/full_weights.pt
   LWD replay pi05_norm_stats.json
   OpenPI norm_stats.json
   big_vision/paligemma_tokenizer.model
   ```

   这些检查只用于尽早暴露路径错误，不改变训练算法。

### 当前验证路线

先跑：

```text
hope run examples/lwd/hope/robotwin_lwd_qam_openpi_pi05_smoke_8a100.hope
```

smoke 通过后再跑：

```text
hope run examples/lwd/hope/robotwin_lwd_qam_openpi_pi05_train_8a100.hope
```

第一轮仍保持：

```text
qam_critic_grad_mode: mean
qam_clip_action_for_critic: false
train_expert_only: false
```

也就是先验证 LWD offline 风格的全量 actor QAM。训练输出保存到
`RLINF_QAM_LOG_ROOT/robotwin_beat_block_hammer_lwd_qam_openpi_pi05/checkpoints`，
默认每 500 step 保存一次，用于后续和 SFT baseline 做闭环 eval 对比。

## 2026-07-08：QAM smoke actor.model 缺少 is_lora

### 现象

QAM smoke 已经通过了路径检查，日志里可以看到：

```text
lwd qam imports ok
QAM actor/reference/critic 路径均解析到预期 checkpoint
```

但 worker 初始化 actor 时失败：

```text
omegaconf.errors.ConfigAttributeError: Key 'is_lora' is not in struct
full_key: actor.model.is_lora
```

### 原因

`rlinf.models.get_model()` 是所有模型共用入口。模型实例化完成后，它会读取
`cfg.is_lora` 判断是否接 LoRA。critic 的 config 已经有 `is_lora: false`，但
QAM 新增的 OpenPI actor config 漏了这个 RLinf 公共字段，所以 Hydra struct 模式下
访问 `actor.model.is_lora` 会直接报错。

### 修复

在 `examples/lwd/config/robotwin_beat_block_hammer_lwd_qam_openpi_pi05.yaml`
的 `actor.model` 下补充：

```yaml
is_lora: false
```

这个字段不改变 QAM 算法，也不会启用 LoRA，只是显式告诉 RLinf 模型工厂当前 actor
走普通全量参数路径。修复后重新提交 smoke 即可继续验证后续模型加载和
forward/backward。

### 进一步根治

后续 `stdout.20260708161034` 里打印出的 Hydra 配置仍然没有
`actor.model.is_lora`，说明云端实际运行的配置还没有包含本地这次 YAML 修复。
为了避免任何旧 config 或新增 config 再触发同类问题，模型工厂入口也改成：

```python
cfg.get("is_lora", False)
```

也就是：只有显式写了 `is_lora: true` 才启用 LoRA；没有写时默认全量模型路径。
这和 RLinf 的模型配置语义更一致，也能避免 Hydra struct 模式下访问缺失字段报错。

## 2026-07-08：QAM smoke 后续隐患排查

### 已确认不是问题的部分

1. hope 和 conda 激活

   `stdout.20260708144145` 里已经打印出：

   ```text
   Using python: .../Miniforge/envs/rlinf_lwd/bin/python
   lwd qam imports ok
   ```

   说明当前 smoke 已经进入云端 Python 环境和 RLinf import 阶段。当前报错不是
   `worker.script` 后面是否带 `smoke`、也不是 conda 激活失败。

2. actor/reference/critic 路径形态

   OpenPI loader 支持：

   ```text
   global_step_xxx/actor/model_state_dict/full_weights.pt
   model_state_dict/full_weights.pt
   ```

   因此 SFT actor 使用 `global_step_10000`，critic 使用 `global_step_8000/actor`
   是符合当前 loader 契约的。cloud script 也会提前检查这些文件，避免路径写错后
   进入很晚才失败的模型构建阶段。

3. QAM policy input 和 critic input 的空间

   QAM dataset 给 OpenPI 的 policy view 是原始 `images/state/prompt`，会走
   `AlohaInputs -> Normalize -> model_transforms`；给 critic 的 action chunk 则是
   pi0.5 归一化 action。两者都使用同一套 pi05 norm stats，因此当前没有发现
   action 空间或 state 空间错位。

### 本次补充的代码检查

1. batch 并行契约

   QAM worker 现在显式检查：

   ```text
   actor.global_batch_size % (actor.micro_batch_size * world_size) == 0
   ```

   当前 smoke/train 配置是 `global_batch_size=8, micro_batch_size=1, world_size=8`，
   所以每步正好一个 micro batch。这个检查是为了避免以后改 GPU 数或 batch size
   时，`gradient_accumulation` 变成错误值，导致训练循环后面才出现难读的报错。

2. QAM 数学契约

   `algorithm.qam_num_denoise_steps` 必须至少为 2，因为当前实现会跳过 OpenPI flow
   的端点 `t=1`，只在内部 denoise state 上做 QAM local regression。`lambda_q`
   也必须大于 0，否则 `adjoint = -grad_Q / lambda_q` 没有数学意义。

### 仍需关注的真实风险

1. 显存风险

   QAM worker 会同时驻留三套模型：

   ```text
   current actor: 需要训练和保存 optimizer
   reference actor: 冻结，但需要保留 x_t -> x_next 的伴随梯度图
   frozen critic: 需要对 action 输入求 ∇a Q
   ```

   当前 FSDP 配置是 `sharding_strategy: no_shard`，所以每张 GPU 都会放完整模型。
   8x80G 大概率可以做 smoke，但正式 `qam_num_denoise_steps=10` 会比普通 SFT 更吃
   显存。如果下一次日志变成 CUDA OOM，优先把正式训练的
   `algorithm.qam_num_denoise_steps` 从 10 降到 4 或 6 做验证，而不是先改算法方向。

2. 训练有效性风险

   当前 QAM 默认使用：

   ```yaml
   qam_critic_grad_mode: mean
   qam_clip_action_for_critic: false
   anchor_weight: 0.05
   bc_weight: 0.1
   ```

   这是为了先验证 frozen critic 是否能给 actor 提供有用方向。是否真正提升策略，
   仍然要看后续 RoboTwin eval success rate 和视频，而不能只看 `q_mean` 是否上升。
