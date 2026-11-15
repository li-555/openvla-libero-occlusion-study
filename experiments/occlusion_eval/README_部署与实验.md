# OpenVLA / LIBERO 视觉遮挡闭环评估 v1

> **状态**：本包的新增 CPU/模拟测试通过；**尚未连接 HKU GPU，也没有运行真实 OpenVLA-7B/MuJoCo 闭环**。本包是实验工具，不是已验证的性能改进或新颖算法成果。

## 目录与复用策略

本 ZIP 只新增 `experiments/occlusion_eval/`。它依赖港大服务器上**已经存在**的 `experiments/contact_cpu/`、`experiments/physics_occlusion/`、OpenVLA 的 `experiments/robot/` 和本地 LIBERO 环境。**不覆盖**原始 `run_libero_eval.py`、`libero_utils.py`、`openvla_utils.py`、检查点或日志。

- `run_gpu.py`：同一个冻结 OpenVLA 4-bit 权重，使用原始 `robot_utils.get_action()`；同一初始状态配对跑 clean / occluded / m0 / m1 / m2 / cue_control，写每步动作、图像哈希、成功标志、实际处理器输入图片、峰值显存。
- `core.py`：从可用 RGB 与机器人本体状态更新物体假设；包装已有 ContactGate；禁止遮挡后透传物体真值。
- `analyze.py`：按 episode 配对汇总成功率、Wilson 区间、配对 Bootstrap 差异；按 task 分层；有失败的运行不算作策略失败。
- `tests/`：纯 CPU 单元与假策略集成测试。
- `TASK_SPEC.example.json`：单任务配置示例，**需独立检查遮挡实际覆盖率和任务所需物体**。

**特别提醒**：目前 `physics_occlusion/pipeline.py` 的补偿是在预测位置画一个绿色小圆点，而不是逼真的黑碗重建！这个机制是否能让冻结 OpenVLA 获益未知，`cue_control` 是为排除简单视觉提示影响而设的对照。

## A. 上传/安装（无需 Copilot）

在本地下载本 ZIP。在 VS Code 远程 `openvla_eval` 项目根目录上传 ZIP 到 `~/openvla_eval/`，然后在远程终端运行：

```bash
cd ~/openvla_eval
python -m zipfile -e openvla_gpu_eval_v1.zip .
```

终端也可只读验证：

```bash
ls experiments/occlusion_eval
```

这样不会修改原来其他目录。

## B. CPU 测试（仅在学校许可的 CPU 环境）

项目根目录执行：

```bash
PYTHONPATH=. python -m unittest discover -s experiments/occlusion_eval/tests -v
```

这些测试用合成数据和假 policy，不启动实际 MuJoCo/GPU。

## C. GPU 连接与权限

**先连接学校分配给你的计算节点**：`<allocated-compute-node>` 是登录网关，不能直接在上面加载7B模型。仅因显存不超过16GB，不代表可以绕过 GPU Farm 的作业/分配规定。请按照 HKU 文档或已分配的会话进入计算节点；此包不猜测调度命令、节点 IP 或队列。

在实际获分配的节点上执行：

```bash
hostname
nvidia-smi
conda activate libero-eval
cd ~/openvla_eval
```

如果已经以 Remote-SSH 打开共享 home，编辑文件可在 gateway；运行推理必须从实际获分配的 compute 节点执行。运行器会拒绝主机名带 `gate`/`login`/`head` 的节点，也要求 CUDA 可用，但**不负责证明授权**。

## D. 最小 GPU Smoke Test（先跑 2 条）

先不运行物理方法，测试原始输入和固定遮挡是否能完成闭环，且验证真实 processor 收到的图像：

```bash
python experiments/occlusion_eval/run_gpu.py --pose-source oracle_visible --allow-oracle-upper-bound --task-ids 0 --episode-ids 0 --methods clean,occluded --start-step 35 --end-step 85 --mask 0.32,0.32,0.70,0.70 --max-steps 100 --save-every 20 --max-reserved-gib 15 --fail-fast
```

这只是 smoke test，截短的100步回合不能当完整成功率基准。注意 mask 范围是**示例参数，不保证覆盖黑碗**。运行后先查看结果的 `target_center_mask_coverage`（评估侧分离真值标签）、原始/遮挡/processor 图片，确认固定 ROI 与真实目标位置匹配；不能根据最终测试表现反向挑选遮挡。还应核对 `model.norm_stats`、中心裁剪、模型动作输出和执行动作、CUDA reserved 峰值。

如果你之前设置过 `VLA_OCCLUSION=center`，请在终端执行 `unset VLA_OCCLUSION`，避免两处重复遮挡。

## E. 六组方法试点（上界研究，不可宣称可部署）

满足上一步才执行（XML 夹爪数值是**参考限位，不是独立接触标定**）：

```bash
python experiments/occlusion_eval/run_gpu.py --pose-source oracle_visible --allow-oracle-upper-bound --gate-config experiments/contact_cpu/v2_gripper_audit_20261106/config.xml_reference.json --gate-status xml_reference --task-ids 0 --episode-ids 0,1 --methods clean,occluded,m0,m1,m2,cue_control --start-step 35 --end-step 85 --max-steps 220 --save-every 20 --max-reserved-gib 15 --fail-fast
```

**注意**：这里使用**遮挡前的模拟器真实碗位置**作为理想可见起点，只用于物理机制的上界与接口开发，不能称为仅依赖 RGB 的机器人策略，也不能作为主要论文结果。若碗在真实遮挡中失去抓取，gate 未必能察觉（没有接触传感器）。

这些实验采用冻结模型、单份4-bit权重，没有训练操作；每条任务回合独立 reset 到相同的初始状态，同一 `env` 按方法顺序复用。请用多个种子或打乱执行顺序排除隐藏顺序效应；检查 `env.check_success` 在已安装 LIBERO 版本的含义。

## F. 仅使用可见 RGB 的初步验证（不是完整 6DoF）

`--pose-source rgb_plane` 使用预先选定的**可见帧 RGB 固定 ROI 中暗像素质心**和**独立标定的物体中心高度**，反投影到固定平面。它不调用大型视觉模型；对光照、黑色背景、夹爪遮挡和多个黑碗可能严重失败；物体旋转估计为未知（固定单位旋转），不能当作 FoundationPose 6D。只用来验证 RGB-only 观测能否走通。

使用前独立确定在 `get_libero_image(...,224)` 图像坐标系下的可见黑碗 ROI 与碗中心高度，例如：

```bash
python experiments/occlusion_eval/run_gpu.py --pose-source rgb_plane --rgb-roi X0,Y0,X1,Y1 --object-center-height YOUR_CALIBRATED_METER_VALUE --gate-config experiments/contact_cpu/v2_gripper_audit_20261106/config.xml_reference.json --task-ids 0 --episode-ids 0 --methods occluded,m0,m1,m2 --max-steps 110 --fail-fast
```

把占位符替换为独立确定的参数；**不要直接复制包含占位符的命令**。在做机器人成功率比较前，必须检查检测是否实际锁定目标、投影是否正确、成功/失败检测比例，并冻结参数。更强的真实视觉结果需要用实际 RGB 姿态估计器（如独立运行的 FoundationPose/SAM pipeline）替换此简化代理，并记录它的误差；该接口尚未实现。

## G. 多任务需要逐任务规格

不能把 `akita_black_bowl_1` 用到其他 LIBERO 任务。可复制 `TASK_SPEC.example.json` 并为每个任务 ID 设置真实目标字段、遮挡区域、时段以及 RGB 参数，然后传：

```bash
--task-ids 0,1 --task-specs experiments/occlusion_eval/task_spec.local.json
```

每个 task 应独立在开发/验证集确定 ROI、窗口、颜色与标定；正式测试锁定配置。未经核实的任务目标名或 ROI 不得用于正式实验。

## H. 输出与论文标准

每次运行创建唯一的 `experiments/occlusion_eval/results/eval_UTC/`，包括：

- `manifest.json`：版本、设置、上界/可见 RGB 属性及所有门控配置。
- `task*_ep*_{method}.json`：每步真实动作、输入哈希、遮挡覆盖评估标签、门控选择、推理时长等。
- `episodes.json`、`analysis.json`、`summary.csv`：按**初始状态/episode**配对；多任务结果按任务分层；无独立回合绝不报独立样本显著性。
- `images/`：raw、occluded、action_input（到原始 OpenVLA 的 image）与 processor 最终 PIL 图像。
- `*_ERROR.json`：环境兼容或 OOM 错误单列，不将代码异常算成算法失败。

正式可争取 PhD 核心研究项目的证据至少应包含：独立任务、多个初始状态、独立接触标定、真实视觉位姿、同风格无信息提示对照、预注册固定遮挡、正反例分析、配对成功率/置信区间、代码/版本复现。真正的算法创新不能仅靠 M0/M1/M2 重新命名；决策风险门控应在可靠闭环基线建立并进行文献新颖性核验后开发。

## I. 服务器兼容性待核实（交给 Copilot 的最小范围）

仅要求 Copilot 修复：1) 已安装 LIBERO `OffScreenRenderEnv` 的 `env.env.sim` 或 sim 访问方式；2) `robosuite.utils.camera_utils` 与180°图像旋转后的投影是否匹配；3) `get_model / get_action / normalize_gripper_action` 在远端 checkout 的真实签名和动作方向；4) 验证初始状态重置、成功标签；5) 某个 GPU 节点的实际分配方式。不要让 Copilot 重写核心算法或虚构成功率。
