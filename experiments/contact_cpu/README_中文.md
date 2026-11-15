# HKU OpenVLA / LIBERO：CPU 接触门控实验包

原始 Work 包通过 19 项合成测试；本修订版在当前可用 Python 环境中通过 **22 项测试**（含可选绘图测试）。没有连接 HKU，没有读取你远端的 NPZ，没有生成或声称任何真实轨迹测量结果。它是可执行的离线状态预测原型，不是已证实的新方法，也未实现视觉感知或 OpenVLA 策略补偿。

**现在的算法与误差评估不需要 GPU。** 之后运行冻结的 OpenVLA 闭环任务需要另行申请学校允许的 GPU 资源。不重训 OpenVLA；本包也不会导入它、启动仿真或申请 GPU。

## 文件

- `contact_gate.py`：M0、M1、M2，只有 NumPy 依赖。
- `evaluate.py`：读 NPZ、构造因果输入、独立评分、输出 CSV / JSON / 两个预测假设。
- `inspect_trace.py`：只读检查字段、step、夹爪原始数值及时间采样（不作抓取标签推断）。
- `plot_results.py`：可选生成每个遮挡窗口的三方法位置误差图（仅需 Matplotlib；CPU/无头环境可用）。
- `REVIEW_改动说明.md`：本次代码审查与修改、已知限制。
- `config.example.json`：未验证的工程初值；夹爪标定为空，不自动宣称可靠抓取。
- `tests/test_contact.py`：22 项测试，含遮挡真值篡改、夹爪状态变化、机器人观测断步与可选绘图。
- `REMOTE_CODEX_PROMPT.md`：可直接交给 VS Code Remote-SSH 中 Codex 的任务说明。
- `GPU_PLAN_中文.md`：视觉 mask、真实策略输入补偿和正式闭环实验的后续规范。

## 1. 算法

统一采用世界坐标下的齐次变换 `T_world_from_local`，物体与 eef 必须在同一坐标系；位置按米、角度按弧度。四元数分别显式选择 `xyzw` 或 `wxyz`，不猜测。记最后可见步为 a。

- **M0**：`T_mem(t) = T_obj(a)`。
- **M1**：`T_phy(t) = T_ee(t) @ inv(T_ee(a)) @ T_obj(a)`。包含平移和旋转，不能简化成位置增量。
- **M2**：在遮挡前最近 K 个连续可见步上计算相对位姿 `inv(T_ee) @ T_obj` 的最大两两平移/转角变化、eef 与物体各自的运动范围、二者距离，以及标定后的夹爪状态。

门控规则：

1. eef 有足够运动、物体也有足够运动、相对位姿稳定、距离足够近、历史夹爪状态均符合闭合：选择 Physics。
2. eef 有足够运动，而物体保持静止且相对位姿不稳定：选择 Memory。
3. 其他情况：`uncertain`，保留 Memory 和 Physics，`selected=None`。

这不是接触概率模型。运动激励条件用于避免把“夹爪与碗一起静止”误判为抓牢；物体中心与 eef 的距离只是近邻证据，不是真正接触几何。共同运动也可能来自外力，不能据此证明刚性抓取。

遮挡后 `predict()` 只接受 `Robot(step, eef, gripper)`，没有物体字段。可靠 Physics 状态若失去闭合夹爪证据，退为不确定；**Memory 状态在遮挡期发生从已标定的张开到闭合/未知的变化也退为不确定**，不能假定物体一直未被抓取。任何确定状态遇到本体观测断步也退为不确定；遮挡期间不重新升级到可靠状态。**张开夹爪不表示物体回到了最后可见位置**，因此不自动切回 Memory。新一轮可见观测应建立新的 gate。

若滑落没有改变可测本体信息，抓牢和滑落两个世界可能产生完全一样的输入：本算法无法识别这种接触改变。Memory 与 Physics 都可能错；双假设不是包含真值的保证，也不是连续概率分布或完整可达集。

## 2. 在 Remote-SSH 部署

运行环境要求 Python 3.10 或更新版本；与 OpenVLA 环境不兼容时使用单独的 CPU 环境。

将整个 `contact_cpu` 文件夹上传到远端 `~/openvla_eval/experiments/contact_cpu`，可用 VS Code Remote-SSH 的资源管理器拖入。不要仅上传 ZIP 后让服务器猜路径。保留已有实验和日志。

在学校允许的小型 CPU 执行环境中运行；若登录网关不允许这些任务，使用学校规定的 CPU 作业/交互环境。本说明不假定 SSH 已连接，也不臆造 HKU 的队列名、分区、账号或调度命令。

```bash
cd ~/openvla_eval/experiments/contact_cpu
# 优先使用已有、可 import numpy 的 Python 环境。
python -c 'import sys, numpy; print(sys.version); print(numpy.__version__)'
python -m unittest discover -s tests -v
python inspect_trace.py ~/openvla_eval/experiments/logs/EVAL-libero_spatial-openvla-2026_11_06-00_52_46--physics_trace_01_ep0_states.npz
```

如果没有 NumPy，在允许联网安装的 CPU 环境创建独立环境，避免改变 OpenVLA 依赖：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 必须先核实的数据约定

让远端 Codex 只读检查原始轨迹采集代码，确认：

1. `bowl_pos/eef_pos` 坐标系、长度单位、eef 参考点；一帧数据是动作前还是动作后，所有字段必须对齐。
2. `bowl_quat` 与 `eef_quat` 各自来源和四元数顺序。LIBERO 的 object observation 与直接读 MuJoCo body quaternion 可能不同，不能按字段名判断。
3. 两个 `gripper_qpos` 的符号、单位、开闭端点，以及是否有夹持物体时的合适范围。

默认标量开度是 `dot([1,-1], gripper_qpos)`，只是可配置形式。若实际关节同号、或需要另一映射，修改权重。用独立、可见的标定序列测量 fully closed/open 对应标量，填入 `grip_closed`、`grip_open`；顺序可递增或递减。不能只看测试轨迹最大最小值就认定抓取成功，不能用测试期隐藏接触标签调阈值。

```bash
cp config.example.json config.local.json
# 用编辑器填入已核实的标定端点。所有门控阈值仅在独立验证集上确定。
```

端点仍为 null 时也可运行接口检查与 M0/M1；M2 的正向抓取选择被禁用，会保留不确定性。这不是算法失败结果或已完成标定。

### 执行真实离线评估

以下命令**仅适用于已确认两个四元数均为 xyzw** 的情况；否则分别替换。窗口使用 NPZ 的 **step 标签**，不是数组下标。

```bash
python evaluate.py \
  ~/openvla_eval/experiments/logs/EVAL-libero_spatial-openvla-2026_11_06-00_52_46--physics_trace_01_ep0_states.npz \
  --windows 1:14,15:29,30:44,45:59,49:63,60:74 \
  --object-quat-order xyzw --eef-quat-order xyzw \
  --config config.local.json --out results/cpu_trace01_v1
```

`15:29` 表示遮挡 15–29（含端点），最后可见 anchor 是 14，历史最多为 10–14。每个窗口是**独立的反事实遮挡重放**，不是先前窗口遮挡状态的延续；算法不跨窗口读取隐藏物体。

**不能直接重现原 0–14 窗口**：它需要 step=-1 的可见 anchor。现有数据从 0 开始时，用 step=0 初始化、评估 1–14；这与先前“0–14 共15步”的口径不同。要精确比较原表，先查清原脚本的 anchor、索引、误差定义、坐标系与单位。不要为了接近给定数值改阈值或索引。

你提供的历史结果（memory / physics：0–14 为 0/.02522；15–29 为 0/.09002；30–44 为 0/.08132；45–59 为 .02448/.01463；49–63 为 .05225/.01422；60–74 为 .06183/.00314）仅作待核对的原始记录，**不是本包实测结果**。

输出目录必须是新目录，防止覆盖：

- `predictions.jsonl`：逐帧 M0/M1 的 4×4 位姿、M2 状态、原因、确定时的选择，没有 ground truth。
- `frames.csv`：位置误差（欧氏范数，米）、旋转测地误差（弧度）、状态、两个假设间距。
- `summary.json`：配置、输入文件 SHA256、窗口/anchor、门控特征、逐窗口与逐轨迹指标。
- `plots/*.png`：如另行运行可选绘图脚本，会生成每个窗口的误差曲线（缺失的 M2 点表示不确定，**不算成零误差**）。

画图（可选；需要 Matplotlib，**不需要 GPU**）：

```bash
python plot_results.py results/cpu_trace01_v1
```

M2 的条件误差须与 `resolved_fraction` 一起报告，并对比同一批确定帧上的 M0/M1。`hypothesis_best_pos_m_ORACLE_BOUND` 是事后用真值选的理想下界，不能当成 M2 结果。不要把不确定帧算零误差。当前不提供任意全帧“融合坐标”来掩盖不确定性。

### 本阶段不能声称的结果

此时 M2 是**接触证据门控的状态预测器**，尚没有决策风险/回报定义，也没有进入 OpenVLA 的动作路径，**不能称为已验证的 Decision-Aware 算法或已经提升操作成功率**。正式的决策贡献需要后续冻结策略的实际输入补偿与配对任务结果，见 `GPU_PLAN_中文.md`。

## 3. 多轨迹 CPU 实验

该 79 步轨迹只能做开发诊断，不能验证泛化或新颖性。另采独立 episode、初始状态、随机种子，覆盖成功抓取、夹空、移动后脱落、静止无激励、旋转、部分可见和噪声。

先按 episode ID 划分开发/标定、验证、最终测试，再产生遮挡窗口；同一 episode 的重叠窗口不能跨集合。同一实际 rollout 的不同拷贝也不是独立样本。可先安排 10 条开发标定、10 条验证、30 条独立测试 episode 作试点，之后根据方差与所需区间精度确定规模，这不是样本量充分的保证。

固定验证集配置后，对每条测试轨迹配对重放相同窗口、长度、可见历史和观测噪声，比较 M0/M1/M2。多个输入文件可以一次处理：

```bash
python evaluate.py /path/to/test_episode_01.npz /path/to/test_episode_02.npz \
  --windows 15:29,45:59,60:74 \
  --object-quat-order xyzw --eef-quat-order xyzw \
  --config config.local.json --out results/cpu_independent_test_v1
```

每个文件必须包含这些窗口的连续 step；长度不同则分别运行适用窗口，预先登记排除规则。本工具拒绝重复路径/完全相同文件，但不能证明不同文件来自独立 episode。

报告每 episode、每遮挡条件的位置/旋转误差、M2 确定比例、同帧条件误差、失败例。统计比较按相同条件配对，以 episode 为重采样单位给区间；不能把重叠窗口中的每帧当独立样本。汇总的 pooled 指标只用于诊断，不代表独立样本显著性。

CPU 门控逻辑可得到测试覆盖，但 **NPZ 没有接触标签**：不能凭此报告抓取分类准确率。可见前缀用记录的 bowl pose 是“理想可见感知”代理；这仍属于特权前缀诊断，正式部署必须换成实际视觉估计并计入其误差。此 CPU 包只检验状态推断，不检验策略成功率或说明补偿一定有效。

## 4. 无泄漏边界

预测模块不接收整个 env/obs/NPZ：仅接收可见 `Visible` 序列和后续 `Robot`。评分模块可在独立文件/进程中保存目标真值，但它的结果不能返回预测模块或在线策略。

部署时还要阻断 `object-state`、所有目标位置/姿态/速度字段、`obj_to_eef` 等派生特权量、隐藏接触、MuJoCo body/site 状态、目标分割真值，以及可间接编码它们的提示词/日志。只阻断 `bowl_pos` 一个字段不够。冻结模型不自动消除输入泄漏。

合成测试已检查：任意替换遮挡后的 bowl 位置与四元数，预测和 gate metadata 不变，但评分改变；这也演示了隐藏滑落不可识别性。后续视觉/策略接口需重复同类审计。

## 5. 下一阶段及来源

先阅读 `GPU_PLAN_中文.md`，在远端完成视觉和策略接口实现后，再申请学校规定的 GPU 配额。不要在 `<login-gateway>` 登录网关运行 OpenVLA 推理、仿真批量评估或训练。

已核对的上游接口仅用于设计后续方案；远端实际 checkout 需再次核对并记录 commit：

- [OpenVLA LIBERO runner](https://github.com/openvla/openvla/blob/main/experiments/robot/libero/run_libero_eval.py)：组装观测并调用 action 路径。
- [OpenVLA action utilities](https://github.com/openvla/openvla/blob/main/experiments/robot/openvla_utils.py)：图像/任务文本进入 processor 与 predict_action。
- [LIBERO observation implementation](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/envs/bddl_base_domain.py)：object observation 及四元数转换来源。采集脚本若直接读取模拟器可能采用不同顺序。
