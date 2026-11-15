# 现有数据离线审查（不加载 OpenVLA / 不运行 GPU）

## 核实过的范围
- 新版实验目录：`<source-repository>/experiments/unified_eval/outputs/gpu_smoke_20261108T174731Z`
- 旧版诊断目录：`<source-repository>/experiments/occlusion_eval/results/diagnostic_v2_20261108T104823Z_2635356/eval_20261108T105011159334Z`
- 找到现有 MP4：10 个。索引：`index.html`
- 新旧 Clean 比较：`100` 个公共步骤。
- Clean 第100步终止与旧版第102步成功 **不是可比的退化结论**。

## 规则无效的定位
- `rule_steps.csv` 报告所需步骤的当前日志值；未记录的锚点、规则内部状态、弃权理由一律标明 `NOT_RECORDED`。
- 旧轨迹 EVAL_ONLY 物体运动仅用于离线诊断，严禁输送给在线估计器。
- 不能根据无效输出武断判断是接触阈值、目标检测或渲染故障。

## 视觉提示离线检查
- 当前结果：`RENDERER_ONLY_PASS`。
- 如生成 `memory_proxy_cue.png`，这只检查候选位置的绘图，不证明完整 Memory 在线重放或真实 processor 已接收。
- CPU processor 对照状态：`PASS_DIFFERENT_PROCESSOR_PIXEL_VALUES`。
- 若要证明实际 Memory 历史和 GPU 策略真正使用提示，需要保存完整公共历史、实际 processor 张量，并执行专门的真实 GPU 验证。

## 后续 GPU 实验许可边界
- 禁止自动补跑或更改规则阈值，GPU 预算需经批准。
- Rule 的规则弃权可标为 `PASS_EXECUTION + ABSTAINED + NOT_EXERCISED`，不必伪造提示。
- 先完成离线证据核验，再单独跑 Memory Visual / Fixed Hold。
