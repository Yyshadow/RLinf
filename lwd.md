# LWD-style Critic 接入说明

本文档说明当前新增的类 LWD critic 代码边界、数据格式和后续使用方式。

## 目标

本次实现的是 LWD critic 的基础闭环，不是 ReCap 的子模块。它面向后续 pi0.5 / Flow / QGF / BPG 实验，核心形式是：

```text
V(s): distributional state value
Q(s, a_chunk): action-conditioned chunk critic
```

其中 `V(s)` 参考现有 ReCap value model 的 SigLIP/Gemma3 编码与 categorical value head；`Q(s, a_chunk)` 是本次新增的 action chunk critic。

## 新增代码位置

```text
rlinf/models/embodiment/lwd_critic/
  lwd_critic_model.py   # LWD-style critic model: distributional V + double Q
  lwd_loss.py           # DIVL 风格 value loss、chunk TD Q loss、EMA target update
  __init__.py           # get_model 入口

rlinf/data/datasets/lwd/
  chunk_dataset.py      # 从 LeRobot-Aloha 数据构造 chunk transition
  __init__.py

examples/lwd/
  train_lwd_critic.py
  config/robotwin_lwd_critic.yaml
```

同时在 `rlinf/config.py` 和 `rlinf/models/__init__.py` 注册了新的：

```yaml
model_type: lwd_critic
```

## 模型结构

`LWDCriticModel` 继承现有 `ValueCriticModel`，复用其 observation 编码链路：

```text
多视角图像 + task prompt
  -> SigLIP2 vision encoder
  -> Gemma3 language/VLM backbone
  -> critic expert readout token
  -> state feature z_t
```

在此基础上新增：

```text
Distributional V head:
  z_t -> value logits over atoms -> value mean / quantile

ActionChunkEncoder:
  a_t:t+H -> temporal attention pooling -> action feature

Double Q head:
  concat(z_t, action_feature) -> q1, q2
```

输出包括：

```text
value_logits
value_probs
value_mean
value_quantile
q_values
q_min
```

## 数据格式

`LWDChunkDataset` 直接读取当前 LeRobot-Aloha 数据，例如：

```text
datasets/robotwin_aloha/beat_block_hammer_30ep
datasets/robotwin_aloha/beat_block_hammer_failed_20ep
datasets/robotwin_aloha/beat_block_hammer_nearmiss_20ep
```

每个样本构造成：

```text
observation
next_observation
action_chunk
reward_chunk
done
success
episode_id
frame_idx
source
prompt
```

这里的 `action_chunk` 使用数据集中已有的 action 表达。对 RoboTwin-Aloha 数据来说，关节部分应继续和 pi0.5 SFT 数据保持一致，避免 critic 和 policy 的 action 空间错位。

## Loss 逻辑

`lwd_loss.py` 当前实现了类 DIVL 的 critic 训练基本项：

```text
reward_sum = sum_i gamma^i r_{t+i}
target_q = reward_sum + gamma^H * (1 - done) * Quantile(V_target(s_{t+H}))
L_Q = MSE(Q1(s_t, a_chunk), target_q) + MSE(Q2(s_t, a_chunk), target_q)
```

distributional value 的监督来自 target critic：

```text
target_v = min(Q_target1(s_t, a_chunk), Q_target2(s_t, a_chunk))
L_V = CE(project_to_atoms(target_v), V_logits(s_t))
```

target model 使用 EMA 更新：

```text
target <- (1 - tau) * target + tau * online
```

## 单卡 smoke 训练

配置文件：

```text
examples/lwd/config/robotwin_lwd_critic.yaml
```

运行前需要把以下路径改成真实模型路径：

```yaml
model:
  siglip_path: /path/to/siglip2-so400m-patch14-224
  gemma3_path: /path/to/gemma-3-270m
  tokenizer_path: /path/to/gemma-3-270m
```

启动命令示例：

```bash
cd /data/wam_codebase/RLinf
source .venv-openpi/bin/activate

export PYTHONNOUSERSITE=1
export HF_HOME=/data/wam_codebase/RLinf/.cache/hf
export HF_DATASETS_CACHE=/data/wam_codebase/RLinf/.cache/hf_datasets

python examples/lwd/train_lwd_critic.py \
  --config-name robotwin_lwd_critic
```

## 当前实现边界

当前版本完成的是 LWD critic 的基础可运行骨架：

1. 独立 `lwd_critic` 模型模块；
2. LeRobot-Aloha chunk transition 数据读取；
3. distributional V + action-conditioned double Q；
4. chunk TD target、distributional value loss、EMA target critic；
5. 单卡 smoke 训练入口。

还没有接入 actor 更新、QAM、BPG 或在线 rollout 后训。建议下一步先验证 critic 质量：

```text
成功轨迹 value/Q 更高；
失败轨迹 value/Q 更低；
near-miss 在失误附近 value/Q 下跌；
同一状态下 expert action chunk 的 Q 高于扰动 action chunk。
```

critic 曲线稳定后，再接后续的 Flow/QGF/BPG policy improvement。
