# QGF Flow Policy 接入说明

这次新增的是 RLinf 原生 embodied model：`qgf_flow_policy`。它不是替换旧的 `flow_policy`，而是为 QGF 这条路线单独放一个模型族，避免把 Robotwin 动作变换、IQL value、Q-guidance 这些逻辑塞进已有 baseline 里。

## 目录和文件

```text
rlinf/models/embodiment/qgf_flow_policy/
  qgf_flow_policy.py   # RLinf policy 入口：ForwardType、rollout、SAC/SFT/IQL 接口
  obs_encoder.py       # state 或 image+state 编码，沿用 RLinf ResNetEncoder
  flow_actor.py        # flow-matching actor，学习 normalized action
  action_transform.py  # model action <-> env action 的转换
  q_critic.py          # Q(s,a) ensemble，供 SAC 和后续 Q-guidance 使用
  value_model.py       # IQL-style V(s)，后续 IQL worker 使用
```

为什么不是像 `flow_policy/flow_policy.py` 那样一个文件：旧 flow baseline 是一个较小的视觉 flow policy；QGF 这条线同时涉及 action transform、flow BC、Q critic、IQL value、后续 test-time Q-guidance。拆开后每个文件职责清楚，后面接 Robotwin、LIBERO、ManiSkill 时更不容易把动作语义写乱。

## 工具脚本放在哪里

`build_qgf_stats.py` 放在：

```text
toolkits/robotwin/build_qgf_stats.py
```

不是放在 `scripts/robotwin/build_qgf_stats.py`，原因是当前 RLinf 仓库已经有很多离线处理/诊断工具放在 `toolkits/*`，但没有统一的根目录 `scripts/robotwin`。所以 Robotwin/QGF 的 stats 和检查工具都放在：

```text
toolkits/robotwin/
  build_qgf_stats.py   # 从 replay/demo 里统计 state mean/std、delta scale
  check_qgf_replay.py  # 检查 replay 里的 model_action/env_action/range/gripper
```

示例：

```bash
python toolkits/robotwin/check_qgf_replay.py \
  --input /data/wam_codebase/RLinf/results/qgf_transform_smoke/rank_0 \
  --action-dim 14
```

## Robotwin 的 gripper 是什么

Robotwin 里 gripper 不是离散类别。它是连续归一化值：

```text
0 = close
1 = open
```

代码位置：

```text
/data/wam_codebase/RoboTwin_RLinf/envs/robot/robot.py
```

关键逻辑是：

```python
set_gripper(gripper_val)  # gripper_val in [0,1]
is_open:  gripper_val > 0.8
is_close: gripper_val < 0.2
```

所以任务判断里会出现 open/close，但那只是阈值判断，不代表动作空间是离散的。

## Robotwin 动作语义

RLinf 的 Robotwin wrapper 当前把 action 原样传给 RoboTwin，不会额外归一化或解码。

Robotwin `gen_sparse_reward_data(..., action_type="qpos")` 期望的是绝对 qpos 目标：

```text
[left_arm_6_qpos, left_gripper, right_arm_6_qpos, right_gripper]
```

也就是 14 维：

```text
0..5   left arm absolute qpos
6      left gripper in [0,1]
7..12  right arm absolute qpos
13     right gripper in [0,1]
```

QGF 这版不直接让模型学习绝对 qpos。模型学习的是 normalized action：

```text
model_action in [-1,1]
```

对 Robotwin 执行时再解码：

```text
arm env qpos = current_qpos + model_action * robotwin_delta_scale
gripper env  = (model_gripper + 1) / 2
```

这样做的原因是：flow policy 和 Q critic 在固定范围 `[-1,1]` 的动作空间里更稳定；Robotwin 环境仍然收到它需要的绝对 qpos/gripper。

配置里已经明写：

```yaml
action_space: robotwin_delta_qpos
robotwin_delta_scale: [0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 1.0,
                       0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 1.0]
robotwin_gripper_indices: [6, 13]
```

## Replay 里存什么 action

这点非常重要。

rollout 执行环境时：

```text
rollout_result.actions = env_action
```

replay / critic / BC 学习时：

```text
forward_inputs["action"] = model_action
trajectory.actions       = model_action
```

额外诊断字段：

```text
forward_inputs["model_action"] = model_action
forward_inputs["env_action"]   = env_action flattened
```

所以 SAC critic 看到的是 normalized model action，而不是 Robotwin 绝对 qpos。这个和 `/data/vlarl/qgf` 的假设一致：QGF 原版默认 action 已经是适合学习的连续 Box，通常裁剪到 `[-1,1]`。

## `/data/vlarl/qgf` 有没有 delta qpos

没有 Robotwin 专用的 delta qpos 实现。

`/data/vlarl/qgf` 的核心假设是：环境或 wrapper 已经把 action 处理成连续 action space，训练时直接用：

```text
batch["actions"]
```

并且常见流程会把 action clip 到：

```text
[-1 + eps, 1 - eps]
```

它的 flow BC 是：

```text
x0 ~ Normal(0, I)
x1 = dataset action
x_t = (1 - t) * x0 + t * x1
velocity target = x1 - x0
```

QGF 训练 actor 是 flow matching BC；critic/value 是 IQL；推理时才用 Q-gradient 引导 flow 采样。Robotwin 的 delta qpos 是我们为了接这个仿真平台新增的 action transform 层，不是 QGF 原仓库自带的东西。

## RLinf 其它 policy 的动作语义

RLinf 里“模型名”不直接决定动作是关节、末端、增量还是绝对。真正语义由三部分一起决定：

```text
policy 输出格式 + env/action wrapper + 具体环境 step 期望
```

大致可以这样理解：

```text
mlp_policy / cnn_policy
  通常输出连续 Box action。是否是关节/末端取决于 env。SAC 时一般 tanh 到 [-1,1]。

flow_policy
  RLinf 原有 flow baseline。输出连续 action chunk，动作语义同样取决于 env。

openpi / openvla / gr00t / starvla
  常有自己的 dataconfig/action transform。模型输出和环境动作可能不是同一个坐标系。
  比如 gripper 可能从 [0,1] 转成 {-1,1}，或者只取前若干有效维度。

realworld dual-franka / gim-arm
  环境侧更明确地区分 joint、tcp/end-effector、absolute、delta 等控制接口。

robotwin 当前 wrapper
  环境实际接收 qpos action：绝对关节目标 + 连续 gripper [0,1]。
```

因此 QGF 对 Robotwin 的选择是：

```text
模型空间：normalized delta qpos + normalized gripper [-1,1]
环境空间：absolute qpos + gripper [0,1]
```

## 已实现到什么程度

已经实现并通过静态/局部检查：

```text
ForwardType.SAC       # SAC actor 采样 normalized action
ForwardType.SAC_Q     # Q(s,a) critic，a 是 model_action
ForwardType.SFT       # flow-matching BC loss
ForwardType.DEFAULT   # rollout/eval 重算接口
ForwardType.IQL_*     # 模型侧接口预留
Robotwin action transform
state/image preprocessing
replay 诊断字段 model_action/env_action
Robotwin replay 检查脚本
```

已做的轻量验证：

```text
python -m py_compile 通过
robotwin_delta_qpos encode/decode 单测通过
QGFFlowPolicy state-only 前向 smoke 通过
沙箱外 Robotwin 最短 rollout + replay + SAC update 通过
```

真实 Robotwin smoke 保存目录：

```text
/data/wam_codebase/RLinf/results/qgf_robotwin_smoke/rank_0
```

`check_qgf_replay.py` 检查结果显示：

```text
actions/model_action: min=-0.99384 max=0.90654
env_action:          Robotwin qpos/gripper 执行动作
robotwin gripper:   min=0.60459 max=0.93692 in_[0,1]=True
```

这说明当前链路里 critic/BC 看到的是 normalized model action，而 Robotwin 环境执行的是解码后的 qpos/gripper action。

## 目前还不是完整 QGF 的部分

现在挂在 RLinf 的 `embodied_sac` worker 下，可以跑在线闭环和 critic 更新，但这不是严格的 QGF 算法。

严格 QGF 应该是：

```text
1. 先用 demonstration / replay 做 flow BC
2. 用 IQL 学 Q(s,a) 和 V(s)
3. 推理时在 flow denoising 每一步用 Q-gradient 或 Q-ranking 引导采样
```

还需要补：

```text
rlinf/workers/actor/fsdp_qgf_iql_policy_worker.py
```

或在现有 worker 中新增 QGF/IQL 分支：

```text
critic loss: Q(s,a) -> r + gamma * V(s')
value loss:  expectile_loss(Q(s,a) - V(s))
actor loss:  flow BC，可选 advantage-weighted flow BC
inference:   Q-guided flow sampling
```

另外，当前 SAC 路径需要 log_prob，但 flow matching actor 没有精确 likelihood；现在的 log_prob 只是让现有 SAC worker 能跑通的近似 bookkeeping。真正做 QGF 时，不应该把这个当作严格 flow policy likelihood。

## 下一步建议

1. 先用现在的 Robotwin QGF 配置跑一段短 rollout，保存 replay。
2. 用 `check_qgf_replay.py` 检查：
   - `actions/model_action` 是否在 `[-1,1]`
   - `env_action` 是否是 Robotwin qpos/gripper
   - gripper 维度是否在 `[0,1]`
3. 如果要 BC，先把 Robotwin planner demo 转成 `env_action` 或直接转成 `model_action`，不要把绝对 qpos 伪装成 normalized `action`。
4. 再实现 QGF-IQL worker，把算法从“能跑 SAC 的 flow policy”升级成“严格 QGF”。
