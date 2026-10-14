# PIP-constraint A-v1 适配判定记录（判死，2026-09-29 复认）

- 方法：Proactive Infeasibility Prevention（PIP / PIP-D，NeurIPS 2024，
  Bi et al.，arXiv:2410.21066），POMO / AM / GFACS 上的约束感知掩码框架。
- 本地资产：`CC_Compare/PIP-constraint/`（README + LICENSE；完整代码未 clone）。

## 判死依据（沿用 §5.1 与工作包决定，复认维持）

1. **任务形态不匹配**：PIP 官方演示任务为单车辆 TSPTW / 带复杂约束单路径构造
   （POMO/AM 单旅行商路线 + 拉格朗日/预防性不可行掩码）；A-v1 是**多车**（15 车、
   容量 50）pickup-to-depot + 揭示即接受/拒绝 + C0 预算，PIP 掩码的目标结构
   （单路线可行性）不覆盖 A-v1 的决策空间。
2. **缺容量/多车机制**：其约束掩码针对 TSPTW 类单路径；A-v1 的容量、多车、承诺
   约束由外层 certify 处理，PIP 的核心机制（学掩码防不可行）无对应可保留机制。
3. **无权重/无对应训练档**：本地无 PIP 训练产物；即使重训，训练任务 = 重新定义
   PIP 的约束集，等价于发明新方法而非"保留原机制适配"。

## 结论

- **判死维持**：不进入 A-v1 对比批次（含 P1–P4）；论文主表不列，如需引用按
  "约束感知掩码框架（任务形态不匹配，未适配）"文献锚点处理。
- 复认时间：2026-09-29（工作包全方法复核后维持原判定）。

## 复核更正（2026-10-13）

以 [`a1_accept_v1/RESULTS.md`](a1_accept_v1/RESULTS.md) 为准（含全部命令与原始数字）：

1. **「本地资产 = README + LICENSE，完整代码未 clone」不准确**：`POMO+PIP/`、`AM+PIP/`、
   `GFACS+PIP/`、`baselines/`、`data/` 均完整存在（含 `utils.load_model`、`nets/`、
   `models/SINGLEModel.py`、`problems/tsptw/`）。
2. **「本地无 PIP 训练产物」不准确**：48 个 `.pt` 中 **43 个是合法 PyTorch 归档**，
   `POMO_star_PIP-D` 的 TSPTW50/100、TSPDL100 权重 `torch.load` 实测成功；与官方
   `SINGLEModel` 逐形状对账 86/86 吻合（4 个差异键为未开启的 PIP-D 辅助双头参数）。
   损坏的 3 个是 `AM+PIP/pretrained/tsptw100_{hard,easy}/AM_star_PIP[-D]/epoch-99.pt`
   （非合法 zip，本地/服务器 sha256 相同），另有 2 个 0 字节 `args.json`。
3. **判死依据（①任务形态、②缺容量/多车）复认成立**，且给出字段级证据：
   `POMO+PIP/envs/{TSPTWEnv,TSPDLEnv}.py` 中 `num_vehicles/vehicle_id` 与 `capacity`
   出现次数**均为 0**；env 目录只有 TSPTW/TSPDL；`AM+PIP/problems/` 只有 `tsptw`。
4. **结论**：仍不进 A-v1 主表，但理由更正为「**单路线任务形态不匹配（资产可用）**」。
   用户提出的「单车单路线机制参考」在技术上可行（权重可用），但会把 PIP 掩码限定在
   单条 TW 路线上、多车容量维度无作用对象，本轮**不实施**，记录为可按需另立项。
