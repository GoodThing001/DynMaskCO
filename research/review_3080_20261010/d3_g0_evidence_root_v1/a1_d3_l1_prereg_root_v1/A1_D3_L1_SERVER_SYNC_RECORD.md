# D3-L1 服务器同步记录（5090 机器）

- 目的：把 D3-L1 的工具与证据同步到 5090 那台服务器，使服务器工作副本与已提交状态一致，
  并修掉上一轮 `Q_identity=false` 的**装配缺陷**（身份文件当时不在服务器上）
- 结果：**19/19 文件上传成功**；服务器身份文件 sha256 `f44e3456…5172c` 与本地**逐字节一致**
- 未改动：历史合同、profile、冻结源码、归档报告；**未**重跑计时资格运行（授权仅一次）

## 1. 服务器事实（本次实测）

| 项 | 值 |
|---|---|
| git 仓库 | **否**——`/home/hzeng/project/MASKCO-Main` 是纯文件副本，`git fetch/merge` 报 "not a git repository" ⇒ 同步只能走 sftp |
| GPU 0 | NVIDIA GeForce RTX 5090，**0% 利用率**，673 MiB / 32,607 MiB |
| GPU 1 | NVIDIA GeForce RTX 5090，**0% 利用率**，21 MiB / 32,607 MiB |
| CPU / 负载（计时运行时） | 96 CPU，loadavg 0.19 / 0.09 / 0.08 |

## 2. 同步内容（`a1_d3_l1_prereg_root_v1/`）

- `l1_gates/`：`A1_D3_L1_CONTRACT_PROTOCOL_IDENTITY.json`（**关键**，修复身份门）、
  `A1_D3_L1_CAUSAL_GATE.json`、`A1_D3_L1_COLLECTOR_QUALIFICATION.json`、
  `A1_D3_L1_RESUME_PARITY_ATTRIBUTION.json`、`A1_D3_L1_SERVER_TIMING_QUALIFICATION.json`
- 文档：合同协议冻结、合同来源溯源、服务器计时资格、轨迹等价性归因、采集器资格更正、
  L1 七要素预登记、状态重采集预登记
- 工具（`scripts/tests/`）：合同协议冻结/来源审计、采集器资格、因果门、F 评分接口、
  动作一致性审计、导入证明
- **未同步**（有意）：早期轮次的冻结树副本与大批 raw 日志——它们不参与服务器执行，
  留在仓库由 Git 承载

## 3. 只读身份复核（服务器实算，**不含 rollout**，故不构成第二次计时运行）

| 文件 | 服务器 sha256（前 16） | 冻结值 |
|---|---|---|
| `coldchain_contract.py` | `0fb9e6e8fcff11c5` | 一致 |
| `coldchain_state.py` | `a2ee98dcb7290f52` | 一致 |
| `strict_online_env.py` | `fd17fe77f8375463` | 一致 |
| `recourse_snapshot.py` | `0af3535acaf6a849` | 一致 |
| `d3_l1_state_collector.py` | `849667e4d572e825` | 一致 |
| `dcc_50_r1_edod05_cal_teacher.npz` | `c35fa6834b800792` | 一致 |
| `objective_profile.json` | `63e56212174c9ce3` | 一致 |
| 身份文件自身 | `f44e3456db5fac67`（= 本地） | 一致 |

⇒ 下一次服务器运行将读到冻结身份文件并得到 **`Q_identity = true`**（上一轮的 false 纯属
我方上传失败，已在 `A1_D3_L1_SERVER_TIMING_QUALIFICATION.md` §3 保留原文并更正）。

## 4. GPU 可用性与使用口径

两张 5090 空闲。但须注意：
- **L1 的 F-free 采集/计时链路不使用加速器**（纯 NumPy 基线 + 快照），GPU 对本阶段无增益；
- GPU 真正有用的是**后续 F 侧**（MaskCO 评分接口/机制诊断的模型前向）。届时须遵守既有环境口径：
  `jax-cuda12-plugin` 与 `nvidia-*-cu12`、`flax==0.10.4`（0.10.7 会破坏 `SwiGLU`）、
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`，并注意共享机器上的编译/显存竞争；
  对小型 MLP 运行时 `--xla_gpu_autotune_level=0` 需谨慎（会触发 contracting dimension 报错）。
- 本轮**未**启动任何 GPU 任务；`F_SCORING_INTERFACE` 的验证此前是在**本机 CPU** 上完成的。

## 5. 边界

服务器同步**不构成**任何新资格：仍需用户签署**正式状态采集预登记**（`dev_check × K=4`、≤64 槽位）
之后才能采集；F−D-Geo 比较仍需单独授权。
`states_collected_for_L1 = 0`；`algorithmic_gain_claim: none`；D3 G0 未通过、D8 阻断、E2 缺失、
`stable_online_gain=False`、I3 正式负结果与旧 D2 `not_qualified` 全部不变。
