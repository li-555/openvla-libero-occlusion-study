# Learned v2.1：遮挡下的抓取准备度与动作修正（独立增量）

**重要**：本版修复了夹爪开度、训练/执行候选动作一致性、低分有界回退。**先读 `CHANGES_v2_1_中文.md`**；此处下文为算法原始设计说明。

## 状态与边界

**已完成**：训练器、模型推理与动作干预模块、同状态动作探针协议、最小运行器注入器、CPU 单元测试。

**尚未完成**：港大服务器与实际 LIBERO 的探针适配器（`collect_probes.py` 指定接口，依赖已验证的 A3 恢复/分支代码）、真实抓取探针数据收集、真实模型训练、GPU 闭环验证。**没有证明 Learned 成功，不能保证成功。**

原始 Episode 5 遮挡方案不修改；旧 Clean / Occluded / Hold / Learned / Last-seen 结果保留。之前 `GraspRecovery_ActionPatch` 的 Clean 末端坐标方案**不要安装**。这次不使用它。

## 研究问题

输入只能是 **PRE-ACTION 公共观测**：可见阶段的目标像素估计、遮挡持续时间（相对上次可见时刻）、EEF 当前观测、EEF 速度、夹爪开合、本体历史动作和 OpenVLA 运动提议。严格不读仿真对象坐标、接触标注或 Clean 成功末端坐标。

执行闭爪前，小模型评分 8 种候选：

1. `close_now`：执行 OpenVLA 当前 XYZ + 立刻闭爪；
2. `move_then_close`：先按 OpenVLA 当前 XYZ 移动、保持张爪，下步闭爪；
3. 六个 `move_then_close`：在 XYZ 各轴加上正/负的小幅修正，再闭爪。

训练标签来自**同一个仿真快照分别真正执行**上述候选后的稳定抓持／提起结果。1个快照得到8个动作结果，不能把 Clean 的成功位置当标签，也不能把 `Attached` 当前状态标签假冒为“执行该动作后的抓取成功”标签。

随机森林集成（3个 ExtraTrees 模型）学习 `P(stable grasp after candidate | public pre-action state,candidate)` 的**未校准预测评分**。根据预测差值决定是否：允许闭爪、先小幅移动并暂缓闭爪、或者弃权。未经过验证时，分数**不能当作真实校准成功概率**。

**不依赖固定的 Step 36、Step 40、Episode 5 目标位置或成功轨迹。** 只在 `occluded` 且 OpenVLA 提议闭爪后激活；如果进行过修正，下一帧只执行与训练一致的预定闭爪动作，不再评分；之后不再干预本轮抓取。视觉锚点缺失或超过30步、模型不确定、低分暂缓闭爪最多2步；缺失锚点直接弃权，避免无依据的主动控制。`-1` 张爪、`+1` 闭爪必须在当前执行器再次核实。

## 文件

- `features.py`：严格的在线/训练输入白名单，仿真真值无法混入特征。
- `collect_probes.py`：真实动作探针驱动接口；依赖用户服务器现有验证过的 `paired_decision_eval` 快照恢复适配器，**不是已能直接运行的 LIBERO GPU 收集器**。
- `train.py`：只从 `development` 实际执行的 JSONL 探针训练；按 `initial_state_id` 分组留出验证状态；独立 `confirmation` 禁止训练。
- `controller.py`：轻量 CPU 推理，给出真实 7D 动作覆盖，不只是绘图。
- `install_runner.py`：只向 `run_gpu.py` 增加少量读取与动作覆盖钩子；默认 dry-run、`--apply` 自动备份、版本不兼容立即停止。适配两种本地快照：未装前一版 demo 补丁与已装 demo 补丁。
- `tests/`：CPU 测试（合成数据/模拟适配器仅检查代码逻辑，不代表真实机器人成功）。

## 使用流程

### 0. 复制覆盖层后运行兼容检查（不修改代码）

从项目根目录运行：

```bash
python -m experiments.grasp_v2.install_runner --runner experiments/occlusion_eval/run_gpu.py
python -m pytest -q experiments/grasp_v2/tests
```

若 dry-run 提示不兼容，**停止**；只提交需要的最小补丁，不整段替换运行器。

### 1. 收集真实 `grasp_probe_outcome_v1` 标签

服务器使用此前已经验证的 A3 快照保存/恢复流程，给 `collect_checkpoint(adapter, ...)` 提供适配器：

```python
from experiments.grasp_v2.collect_probes import collect_checkpoint,append_jsonl
# adapter 是服务器现有的真实 LIBERO 快照恢复器；须实现以下 6 个方法：
# capture_verified, restore_verified, public_preaction,
# openvla_action, step, stable_grasp_after_lift
rows = collect_checkpoint(adapter, initial_state_id=initial_state_id,
    episode_id=episode_id, seed=seed, split='development')
append_jsonl(rows, 'experiments/grasp_v2/data/probes_dev_state001.jsonl')
```

使用多个**不同 initial_state_id**、碗位置、抓取阶段、遮挡起始时刻；至少包含“抓稳／抓空”两类真实动作结果。至少5个初始状态才能运行训练器，实际可靠性需要更多独立状态。相同初始状态内8个动作探针必须从**完全相同、验证恢复成功**的快照出发，避免把不同状态差异当成动作影响。标签在执行候选和抓持提升测试后才读取，且绝不能输入在线策略。

`examples/probe_schema.json` **仅为格式示例，不是真实数据，禁止用于训练或论文**。

将真实开发标签合并：

```bash
cat experiments/grasp_v2/data/probes_dev_*.jsonl > experiments/grasp_v2/data/all_dev.jsonl
python -m experiments.grasp_v2.train \
  --probes experiments/grasp_v2/data/all_dev.jsonl \
  --output experiments/grasp_v2/models/model_v2
```

`training_report.json` 只报告开发验证 AUC/Brier，不是闭环成功率。没有真实标签不得训练、不得启动 Learned v2 比较。

### 2. 安装最小运行器钩子（仅准备好真实模型后）

```bash
python -m experiments.grasp_v2.install_runner --runner experiments/occlusion_eval/run_gpu.py --apply
```

自动备份 `run_gpu.py.pre_grasp_v2.bak`；安装器拒绝重复安装或覆写既有备份。保持 OpenVLA 原推理、图像预处理、评测成功判定、原始遮挡计划及原分支均不变。

### 3. 对冻结 Episode 5 做一次 DEVELOPMENT 闭环（非最终效果）

这里 `FROZEN_EP5_PLAN.json` 必须替换成**之前验证的原文件路径**，不新建遮挡框：

```bash
python experiments/occlusion_eval/run_gpu.py \
 --checkpoint <OPENVLA_CHECKPOINT> \
 --suite libero_spatial --task-ids 0 --episode-ids 5 --methods integrated \
 --pose-source rgb_plane --target-object akita_black_bowl_1 \
 --task-specs experiments/real_collection_runs/dev10/closed_loop_dev15/task_specs.json \
 --start-step 36 --end-step 51 --max-steps 220 --warmup 10 --seed 7 \
 --max-reserved-gib 15 --save-every 1 \
 --rgb-roi 0.08,0.62,0.32,0.86 \
 --local-mask-plan FROZEN_EP5_PLAN.json \
 --ai1-config experiments/ai1_protocol/examples/ai1_config.development.json \
 --unified-estimator memory --unified-intervention none \
 --grasp-v2-model experiments/grasp_v2/models/model_v2/candidate_model.joblib \
 --output-root experiments/grasp_v2/results_ep5_development --fail-fast
```

记录 `grasp_v2_event` 的 `prob_close_now`、`prob_best_move`、`chosen_action_xyz`、`modified` 和原因。若没有真实干预，不能声称算法使抓取成功。**不要重跑 Clean、Occluded、旧 Learned；不修改冻结遮挡。**

### 4. 科研验收

先用开发状态检查动作变化，随后冻结模型与参数，在全新初始状态、不同遮挡起止时机上对比：Occluded、固定等待、Last-seen / 简单规则、Learned v2。主指标是完整任务成功率；辅指标是稳定抓持成功率、EEF—碗口对准误差、提前闭爪比例、干预次数、弃权率、动作开销。采用同状态配对与独立状态置信区间，禁止只挑成功轨迹。

## 理论来源（借鉴方法，不复制第三方代码）

- FoundationPose (CVPR 2024): https://github.com/NVlabs/FoundationPose — 可见阶段的6D物体位姿/跟踪；本包目前仅复用既有像素记忆，**没有集成 FoundationPose**。
- AnyGrasp: https://github.com/graspnet/anygrasp_sdk — 候选抓取生成和评估；本包只有7个受限移动候选，**没有调用 AnyGrasp**。
- Diffusion Policy: https://diffusion-policy.cs.columbia.edu/ — 反馈式、滚动重规划思想；本包不训练扩散模型。
- robomimic: https://github.com/ARISE-Initiative/robomimic — 可重复操纵学习、分组对照实验思想；本包不依赖其训练栈。

## 当前限制

1. `memory` 依赖的现有暗色像素检测**不能证明目标身份**，目标检测误差（目前开发报告约13.72 px）可能严重影响动作选择。可替换成经过核验的目标跟踪器或 FoundationPose，但只能读取遮挡前可观测数据。
2. 现在还没有 GPU 实际探针数据，不能推断闭环成功率。
3. ExtraTrees 输出未经概率校准；需要较多执行标签，优先保证多个初始状态与遮挡时机。
4. 这个样本效率优先的轻量实现**不是已经验证的论文贡献或最优算法**；若结果不好，需要根据真实泛化评估决定是否增加物体中心协方差、运动学模型或在线位姿调整，而不是修改成功判定。
