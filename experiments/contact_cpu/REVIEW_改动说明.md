# Work 原版审查与 CPU 修订说明（v1.1）

## 已确认

- 原版 19 项合成测试通过；修订后 **22 项测试通过**（其中绘图测试仅在 Matplotlib 已安装时执行，否则自动跳过）。本地执行不等于港大远程测试通过。
- 已审查的 `pose()` / `inv()` / `Rigid Physics` 使用正确形式的 SE(3) 刚体关系：`T_world_object(t) = T_world_eef(t) @ inv(T_world_eef(anchor)) @ T_world_object(anchor)`。前提是两个位姿属于**同一世界坐标系**且四元数顺序正确；当前未访问真实 NPZ 和采集端代码，**尚未确认**这些前提。图表坐标单位也须先核实才能标为米。
- 遮挡前可见的物体真值仅作离线理想初始化；遮挡后 `ContactGate.predict()` 不接收物体状态。独立 `score()` 使用真值评价，与预测模块分开。

## 本次实际改动

1. **遮挡期 Memory 的失效边界**：如果遮挡前已知夹爪处于张开状态，遮挡后变为闭合或未知，不再一直断言静态 Memory 正确，而是锁定 `uncertain`（保持两种假设）。
2. **本体观测断步**：Memory / Physics 的确定状态遇到机器人遥测缺失，一律降为 `uncertain`，不能继续假设未观测期间未发生状态变化。
3. **参数校验**：`history` 和 `max_gap_steps` 必须是正整数，避免浮点值造成下游切片错误。
4. **增加 `plot_results.py`**：为各遮挡窗口生成误差折线图；对 M2 未判定帧保留空缺，不当作零误差。
5. **增强 `inspect_trace.py`**：输出原始 gripper 采样和碗-EEF 参考点距离以辅助排查，但**不把它们当成接触标签**。
6. **扩充 3 项合成测试**：增加抓取状态变化、观测断步、无头绘图的回归测试。
7. **补充 README**：清楚区分离线接触门控与尚未实现的 Decision-Aware/OpenVLA 策略补偿。

## 尚未完成、不能宣称已完成

- **没有港大真实 NPZ**：未测出真实 Contact-Aware M2 的确定率或位置误差，也未核实物体/末端四元数的来源及顺序。
- **没有夹爪开闭标定**：`config.example.json` 的 `grip_closed` / `grip_open` 保留 `null`。默认不会自动激活正向 Physics 门控；必须用独立的可见标定数据，而不是测试真值调节。
- **没有完整图像估计、实际 mask 和 VLA policy 接口**：`GPU_PLAN_中文.md` 是后续规范，不是已经部署的 GPU 实验。先用可观测视觉前缀替换 oracle 前缀，再进行真实闭环评估。
- **没有研究创新和成功率保证**：现有方法是启发式接触门控，不能仅凭单条79步演示推断多任务表现或新的论文贡献。

## 最省事的后续步骤

1. 在 HKU VS Code Remote-SSH 中将 `contact_cpu` 文件夹复制到 `~/openvla_eval/experiments/contact_cpu`。**不要覆盖**之前的 runner 或日志。
2. 让 Copilot 先静态核实本地 `run_libero_physics.py` 对四元数、EEF 参考点与机器人 gripper 的记录逻辑；只在学校允许的轻量 CPU 环境运行测试。
3. 根据真实采集代码确认 quaternion order 后，执行 `inspect_trace.py`、离线 `evaluate.py`；没有夹爪标定时明确汇报 M2 未标定，不调测试数据。
4. 若通过离线验收，再按 `GPU_PLAN_中文.md` 单独实现图像遮挡、策略可见的补偿接口和小规模闭环试验。只有最后一步使用获准的 GPU 作业。
