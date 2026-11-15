# 80–87 步抓取后 M1/M2 探索实验

**目的**：一次检验 occluded / m0 / m1_2d / m2_2d 四种模式是否真实闭环运行，并观察遮挡期间的像素预测、门控、夹爪和动作。不是正式发表级的遮挡因果证据。

在港大项目根目录解压 ZIP，使文件位于 `experiments/occlusion_eval/postgrasp_dev/`。先在 `<login-gateway>` 申请 `gpu-interactive`，进入分配节点后 `conda activate libero-eval`，再执行 `bash experiments/occlusion_eval/postgrasp_dev/run_postgrasp_exploratory.sh`。

## 预先固定的配置
- Task 0 / Episode 0，Seed 7（runner 默认）。
- 第80–87步固定灰块；ROI x=42:71, y=119:141（224x224原图），面积约1.27%。灰块可能遮住部分手指，不能称为合格目标独占遮挡。
- `anchor_step=35`：复用先前经过同初始状态匹配的无遮挡第35帧的 SHA256 校验。它验证相同前缀，但不是第79帧的无损哈希。第79步是实际算法使用的最后可见估计。
- M1/M2可见RGB搜索ROI沿用原来 (0.16518,0.52679,0.31696,0.66071)，第75–79帧粗检有运动一致性，仍可能包括夹爪暗色像素。
- M2 临时夹爪校准 `grip_closed=0.007`, `grip_open=0.080`，根据这条 Clean 开发轨迹的夹爪关节范围选取，**不是独立校准**；仅供开发确认门控能否切换。
- M0的 `object-center-height=0.8984` 来自既有模拟器开发轨迹，**不属于独立标定**；本轮不能作为无真值性能证据。
- 110步上限是开发诊断：覆盖完整遮挡和恢复，但任务未成功时不能称为最终失败。若需终局成功率，之后用新预注册状态跑220步。

## 输出要看什么
结果目录下的 `task0_ep0_{method}.json`：`steps`, `termination`, `masked_frames`, `compensated_frames`, `gate_counts`, 遮挡期 `frames[*].projected_center_px`、`gate`, `gate_reason` 和 `action_executed`。`eval_only/*state_trace.jsonl` 只用于后验计算预测误差，绝不能传入控制器。

如果第35步 anchor SHA 不匹配，runner 会拒绝继续；不要绕过校验，而应查明是初始化变化还是参考轨迹差异。若 M2 完全 uncertain，请保留日志，不要自动改阈值。
