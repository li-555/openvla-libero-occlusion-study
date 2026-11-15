# Learned v2.1：针对代码审查的最小修复

本包用于替换旧版 `experiments/grasp_v2/` 目录，**不修改或复制任何 Clean 成功抓取轨迹**，不修改 Episode 5 遮挡，不修改 OpenVLA 权重。

## 修复的三处问题

1. **夹爪开度校准**：新增 `aperture.py`；训练探针采集器要求 `adapter.gripper_joint_qpos()`，以当前已验证的 `qpos[0]-qpos[1]` 生成 `gripper_aperture_m`。运行器注入同一函数，不再使用 `np.mean`。缺少两指关节观测就报错，不猜。
2. **训练/部署候选动作一致**：统一成 `one_step_move_then_forced_next_close_v1`。`close_now` = 当前 OpenVLA XYZ + 立即闭合；`move_then_close` = 当前候选 XYZ + 张爪一步，然后**仅下一步**使用新的 OpenVLA XYZ/旋转动作，同时强制闭爪，之后立即恢复完全原始 OpenVLA 控制。不会每步重评分、不会无限尝试。这个显式的下一步强制闭爪是该候选动作本身的定义，不是额外的抓取后干预。
3. **低分回退**：具备可靠锚点时，若“现在闭爪”分数低、又没有高分位置修正候选，则最多暂缓闭爪 **2个控制周期**（默认）。这部分属于规则回退，须单独统计，不能算作学习打分的收益。过期/缺失锚点仍弃权并保留原指令。所有干预**只针对首次抓取尝试**；之后的张爪、提起、放置一律不碰。

## 代码协议变更（必须注意）

- **旧版模型文件不兼容**：模型 schema 升为 `grasp_v2_candidate_success_v2`，必须先有真实探针数据、重新训练；旧权重不会被错误加载。
- 真实探针每行新增 `candidate_protocol` 字段。训练器拒绝不带该字段的旧记录。这是为了不混合语义不同的标签。
- 真实 LIBERO 探针适配器**仍未完成**；收集器现在额外要求 `gripper_joint_qpos()` 从前动作公共观测获得两指关节位置。`public_preaction` 的其它字段保持原定义。
- `max_fallback_wait_steps=2` 可在控制器实例化时配置为0–10；默认仅为有界安全回退，不依赖固定遮挡起始步。

## Copilot 最小任务

先用本包替换 `experiments/grasp_v2/`（不改其他 AI 分支）。如原目录有未提交改动先备份。**不要安装旧版 XYZ Demo-Guided 补丁。**

项目根目录运行：

```bash
python -m pytest -q experiments/grasp_v2/tests
python -m experiments.grasp_v2.install_runner --runner experiments/occlusion_eval/run_gpu.py
```

仅有以上两条成功不等于可以跑 Learned；**仍然必须连接已有快照恢复接口、采集真实抓取成功/失败探针，训练得到新 schema 模型**。旧版如果已经向公共运行器写入 V1 钩子，先检查再手工最小升级，不能重复安装或覆盖别人工作。新模型和真实 LIBERO 验证之前，停止，不制造成功结论。

## 已经实际验证（本地、CPU）

- 15项测试通过：夹爪开度、训练数据字段校验、双步动作协议、后续张爪不被修改、低分有界回退、真实探针模拟适配、训练器与安装器 AST 检查等。
- 对当前提供的冻结 `run_gpu.py` 执行 dry-run，只读匹配 5处插入点，**未修改**该运行器。
- GPU / 真实 LIBERO 抓取：**没有运行**；不能声称成功或实现论文改进。

## 可发表性所需的下一步证据

从真实多个初始状态、多个遮挡时刻采集同状态候选动作探针；按初始状态划分开发/确认。部署用同样动作协议评测。除 Occluded/Fixed Delay，还需要同样位置候选/相同动作预算的规则基线，否则不能把动作修正本身带来的收益归功于 Learned 排序。
