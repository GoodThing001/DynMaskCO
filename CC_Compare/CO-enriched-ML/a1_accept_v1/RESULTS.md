# CO-enriched-ML A-v1 适配结果（2026-10-13）

> 全部数字来自服务器 1003 实测（`cc_compare` env；F1 正式批次同时在跑，本方法只用
> CPU、未占用 GPU、未改动任何计算链文件）。原始日志：`/tmp/pchgs_*.log`、`/tmp/coenrich_export.log`。

## 0. 结论摘要

| 问题 | 结论 | 证据 |
|---|---|---|
| 官方 FeatureComputer / 学习策略 / PC-HGS 入口与依赖 | **全部核实可用** | 见 §1、§2 |
| PC-HGS（C++20）能否在 1003 编译 | **能**（无需替代件） | §1 |
| 官方权重能否真实加载 + 推理 | **能**（NumPy 复刻 max\|Δ\|=4.8e-7） | §2 |
| A-v1 桥接是否可跑 | **能**（烟测全绿，含 PC-HGS 模式） | §4 |
| 是否按 A-v1 目标重训 | **已重训**（预声明预算 3000 步/4.4 s；val acc 0.26→0.84） | §3 |
| 是否进入 40 天正式臂 | **可进入，尚未接线**（等 F1 结束再加注册表一行） | `README.md` §5 |

---

## 1. PC-HGS 编译（**成功**，先前评估的「需 cmake-20 编译，可能不可行」已解决）

环境：`cmake 3.31.6`、`g++ (Ubuntu 15.2.0)`（C++20）、96 核、网络可达 github/codeload。

```
CFG_EXIT=0
[100%] Linking CXX executable PCHGS
[100%] Built target PCHGS
BUILD_EXIT=0
-rwxrwxr-x 529960  .../a1_accept_v1/build_pchgs/PCHGS
```

过程中遇到的三个真实障碍与处理（**均不修改第三方原文件**）：

1. **CPM.cmake 自举下载失败**：`pchgs/cmake/CPM.cmake` 用 `file(DOWNLOAD)` 拉
   `CPM_0.35.0.cmake`，静默失败后报 `Unknown CMake command "CPMAddPackage"`。
   → 用 curl 预置 `$CPM_SOURCE_CACHE/cpm/CPM_0.35.0.cmake`，configure 时传
   `-DCPM_SOURCE_CACHE=/home/hzeng/pchgs_deps`。
2. **三个依赖（fmt 8.1.1 / nlohmann_json v3.9.1 / dynamic_bitset v1.2.1）git clone 超时**：
   → 改用 `codeload.github.com` tarball 预取 + `-DFETCHCONTENT_SOURCE_DIR_<NAME>` 指向本地。
3. **`--target all` 失败**：`nlohmann_json` 自带 doctest 测试在 glibc≥2.34 上编译失败
   （`static char altStackMem[4*SIGSTKSZ]`，`SIGSTKSZ` 已非常量表达式）。这与 PC-HGS 无关。
   → `cmake --build <dir> --target PCHGS`（只建目标）。

**结论：PC-HGS 直接用原二进制作 A-v1 的 CO 层，不需要 OR-Tools/PyVRP 替代件。**

## 2. 官方权重加载 + 推理复刻对账

官方资产是 Keras 2.11 写出的 TF SavedModel。**Keras 2.15 无法解析其 `keras_metadata.pb`**
（`Cannot parse keras metadata`），且 PyPI 无 cp311 的 tensorflow 2.11 wheel → `keras.models.load_model`
走不通。处理：在**独立 venv**（`/home/hzeng/venvs/coenrich`，`tensorflow-cpu==2.15.1`，
不触碰正在跑 F1 的 `cc_compare`）用 `tf.saved_model.load` + `serving_default` 签名加载。

```
加载路径: saved_model.load   serving_default 输入参数名 = input_1
layers: dense(20,10,relu) dense_1(10,10,relu) dense_2(10,10,relu) dense_3(10,10,relu) dense_4(10,1,linear,no-bias)
kernel 形状序列: [(20,10),(10,10),(10,10),(10,10),(10,1)]
```

- 架构与官方 `training/prediction.py::NeuralNetwork.create_model` 一致；
- 输入维 20 = featureset-dynamicstatic 的 11 dynamic + 9 static → **反向验证了本适配的
  A-v1 特征管线产出的列数与官方模型期望一致**；
- NumPy 复刻前向 vs 官方 TF 前向（257 个固定随机样本）：**max|Δ| = 4.768e-07**（float32 舍入量级）。

## 3. 按 A-v1 目标重训

**预声明预算**（写死在 `train_coenrich_a1.py::BUDGET`）：同官方架构 `[20,10,10,10,10,1]`、
Adam(lr=1e-3)、batch 256、≤3000 步、CPU；实测 **4.4 s**（远低于「≤1–2 小时」上限）。

**数据**（`gen_coenrich_train_data.py`，与外部臂同口径，**训练天 seed 20260920**，
≠ 开发集 20260926 ≠ 保留测试 20260927/28）：
- 8 天 × 200 订单；合同 `make_c0_contract_v2()`；B=710.6548 / cooling_share=3.7209
  （取自既有 `results/a1_step2_gate_c1_20260926/gate.json`，只读）；
- 特征 = **部署同源**：未绑定调用冻结的 `OrderingAcceptReplanner._build_view` →
  `build_subproblem` → `compute_features`（零重复实现，杜绝训练/部署特征不一致）；
- 标签 = 参考臂（`UncondDynamicReplanner`，强非学习 A-v1 臂）在该事件是否接受该订单
  （`--label-scope new`：只标注本次新揭示订单，避免「已接受订单必然为 1」的平凡标签）；
- 产出 **1567 样本 × 20 特征，正例率 0.4435**（第一次尝试用 `--label-scope pool` 得
  10894 样本 / 正例率 0.92 = 标签被「已接受」主导，已弃用并改口径）。

**损失曲线**（train 天 0–5 = 1167，val 天 6–7 = 400）：

| step | 1 | 500 | 1000 | 2000 | 3000 |
|---|---|---|---|---|---|
| train MSE | 0.3836 | 0.1324 | 0.1277 | 0.1063 | 0.0984 |
| val MSE | 0.3334 | 0.1251 | 0.1250 | 0.1214 | **0.1172** |
| val acc | 0.2625 | 0.8425 | 0.8525 | 0.8675 | **0.8375** |

- 多数类基线 = 0.5565 → 重训模型 **0.8375**，确有信号；
- **真的改变了决策**：与官方权重在同一 val 集上分数相关 **r = −0.2205**（基本无关），
  官方权重 val acc = **0.4375**（低于多数类基线），重训 = 0.8375。

**必须标注的口径替换**：原文训练损失是**结构化学习**（profit 加噪 → 反复调 PC-HGS →
用扰动后派车集合的目标值差做次梯度）；本适配退化为**对参考臂接单标签的 MSE 回归**。
因此「重训」证明的是「ML-CO 的预测层能被小预算重新拟合到 A-v1 的特征/目标空间并改变
决策」，**不是**复现原文的 SL 训练闭环。另外标签是**参考臂行为克隆**，不是原文的
anticipative-lower-bound 最优派车集合（A-v1 无该预解资产）。

## 4. 桥接烟测（`smoke_coenrich.py`，两次全绿）

**官方权重模式**（`--artifacts artifacts/`）与 **A-v1 重训权重模式**（`--artifacts artifacts_a1/`）
各自跑过；下表为 A-v1 重训权重 + 真实 PC-HGS 的结果：

| 检查 | 结果 |
|---|---|
| 权重加载 + 层结构 | 5 层 `[20,10,10,10,10,1]` |
| 空池 / 单客户池 | `()` / `(101,)` |
| `order` 覆盖全池且无重复 | n=60 uniq=60 ✔ |
| 确定性（同 sp 两次） | 一致 ✔ |
| **偏好确实改变排序** | 与 EDD 不同位置 **58/60**，Spearman 0.419（官方权重模式 59/60、0.098） |
| **未来信息不变性** | 只扰动不可见节点字段 → `canonical_hash` 与排序都不变（hash_eq=True）✔ |
| 纯 NumPy 决策耗时 | **0.020–0.026 s**（pool 5/25/60） |
| PC-HGS 模式耗时 | **≈2.03–2.12 s/次调用**（`-t 2`），`n_pchgs_calls=2`、`n_pchgs_error=0` |
| PC-HGS 模式排序仍为合法排列 | ✔ |

**缓存语义**：provider 按可见子问题内容（池/坐标/需求/TW/服务时间/距离矩阵/容量/视界）做
SHA256 缓存；同一决策内 adapter 对 15 辆车各调一次 `order(sp)`，池相同 → **1 次 PC-HGS**
（不改变任何决策，只省重复计算），与 MVMoE 的跨车缓存先例一致。

## 5. 已知局限

1. **未跑 40 天正式臂**：F1 正式批次（`results/a1_ext_formal2/`）运行期间禁止改
   `ordering_providers.py` 等计算链文件，故只交付「独立文件 + 烟测」；接线命令见 `README.md` §5。
2. **PC-HGS 计时**：2.1 s/决策 × ~200 决策/天 × 40 天，在 6 workers 下约 45–50 min/臂；
   纯排序模式（不传 PC-HGS）≈0.02 s/决策。两模式的**差异本身就是「原 CO 层贡献多少」的
   消融**，建议正式臂两模式都跑并如实报告。
3. **特征口径**：`static` 分位数特征改为可见池内计算（因果协议要求）；时间单位由秒改小时。
   这两项使官方权重分布与训练分布不匹配 → 官方权重零样本迁移的 val acc 仅 0.4375，属预期。
4. 重训标签为参考臂行为克隆（见 §3），不是原文的最优派车目标。
5. `--provider-device` 对本 provider 无意义（纯 NumPy），传入被忽略。
