# CO-enriched-ML → A-v1 适配（`a1_accept_v1/`）

**方法**：Combinatorial-Optimization-enriched Machine Learning（ML-CO），Baty et al.,
*Transportation Science* 2024，arXiv:2304.00789；第三方仓库 `CC_Compare/CO-enriched-ML/`
（EURO Meets NeurIPS 2022 动态 VRPTW 挑战协议）。**本目录不修改该仓库任何原文件**。

## 1. 原方法机制（原文三步闭环）

```
epoch_instance ──► FeatureComputer（节点特征 20 维 = 11 dynamic + 9 static）
              ──► 学习策略 NeuralNetwork（20→10→10→10→10→1）预测「每个请求的 profit」
              ──► PC-HGS（profit-collecting HGS，C++20）在给定 profit 下决定
                  「本 epoch 派哪些请求 + 怎么路由」
              ──► 下一 epoch（未派请求顺延）
```
训练信号 = **辅助下界解**（`experiments/*/best-sol.json`，全天已知的预解）在该 epoch 的
派车集合 `n_i`；损失 = 结构化学习（对 profit 加噪 → 反复调 PC-HGS → 用扰动后派车集合的
目标值差做次梯度）。

## 2. A-v1 桥接（本目录实现）

A-v1 的决策接口是「每次揭示 → 接/拒 + 计划写回」，项目侧统一 provider 约定为
`provider.order(sp) -> tuple[订单序号...]`（`scripts/evaluation/ordering_providers.py` +
`OrderingAcceptReplanner`，本目录不改这两个文件）。

| 原文 | A-v1 桥接 | 位置 |
|---|---|---|
| 一个 epoch | 一次揭示事件 | `coenrich_provider.order()` |
| `epoch_instance` 请求集合 | `SubProblem` 的可见池（自包含、无未来信息） | `coenrich_features.build_mlco_inputs` |
| 学习策略预测 profit | 同一 MLP 架构，NumPy 前向（部署零 TF 依赖） | `coenrich_model.NumpyMLP` |
| PC-HGS 决定派车 + 路由 | 同二进制作**优先级生成器**：`order = PC-HGS 服务集(路线序) ++ 其余(profit 降序)` | `coenrich_provider._pchgs_order` |
| 波次派车 / 顺延 | 逐揭示排序 → 接/拒由共享 `certify_plan` 决定 | `OrderingAcceptReplanner`（项目侧，只读） |
| 多趟回仓静态 VRPTW | 单车 pickup-to-depot + 在途接单；多车分配/容量/TW/C0 预算全交共享层 | 同上 |

**保留的原方法机制**：特征工程逐行调用官方 `features/FeatureComputer.py`（不重写公式）；
官方 MLP 架构；官方 PC-HGS 二进制与官方 JSON 接口；官方 `evaluation/tools.py` 的
payload 打包函数。

**替代件（必须标注）**：
1. **静态实例口径**：原文 `static` 分位数特征在全天实例（含未揭示订单）上计算；A-v1 是
   严格因果协议，本实现把「静态实例」限定为 **depot + 当前可见池**（同一公式、可见口径）。
2. **时间单位**：官方 `EPOCH_DURATION=3600s`、`MARGIN_DISPATCH=3600s`；A-v1 单位为小时
   → 构造后覆盖为 `1.0 / 0.0`，`end_epoch` = depot 时间窗终点。
3. `planning_starttime` 用 `sp.ready_time` 代理；`must_dispatch` 全 0（A-v1 无此概念）。
4. **PC-HGS 的作用被收窄**：只用于生成可见池优先级（哪些值得派 + 访问顺序偏好）；
   原文的多趟回仓路由/车队语义由项目共享协调器 + `certify_plan` 取代。
5. **重训目标**：原文结构化学习损失不可复用 → 退化为「参考臂接单标签」的 MSE 回归
   （见 `RESULTS.md` §3，明确标注为替代目标）。

## 3. 文件

| 文件 | 作用 |
|---|---|
| `export_official_weights.py` | 独立 venv 加载官方 TF SavedModel → 导出 `npz` + 层结构 + 官方前向对账样本（**只读官方资产**） |
| `coenrich_features.py` | A-v1 `SubProblem` → 官方 FeatureComputer 期望的输入（可见池静态实例） |
| `coenrich_model.py` | 官方 MLP 的 NumPy 复刻（部署侧零 TF 依赖） |
| `coenrich_provider.py` | `COEnrichedMLProvider.order(sp)`：特征 → 预测 profit → （可选）PC-HGS → 池排序 |
| `gen_coenrich_train_data.py` | 训练天（seed 20260920）生成可见池特征 + A-v1 接单标签 |
| `train_coenrich_a1.py` | 按 A-v1 目标重训（同官方架构，纯 NumPy Adam，预声明预算） |
| `smoke_coenrich.py` | 烟测：权重复刻对账 / 覆盖性 / 确定性 / 偏好改变 / 未来信息不变性 / 计时 / PC-HGS |
| `artifacts/` | 官方权重导出物（`official_weights.npz` 等） |
| `artifacts_a1/` | A-v1 重训权重（`a1_weights.npz`） |
| `data/train_pool.npz` | 重训数据（1567 样本 × 20 特征，seed 20260920 训练天） |
| `build_pchgs/` | PC-HGS 的**源外**构建目录（不污染 `pchgs/` 原文件） |

## 4. 复现命令（服务器，`cc_compare` env）

```bash
cd /home/hzeng/project/MASKCO-Main
COE=CC_Compare/CO-enriched-ML/a1_accept_v1

# (a) 官方权重 → npz（独立 venv /home/hzeng/venvs/coenrich，tensorflow-cpu==2.15.1）
/home/hzeng/venvs/coenrich/bin/python $COE/export_official_weights.py \
  --model CC_Compare/CO-enriched-ML/training/models/NeuralNetwork_samples-50_instances-15_runtime-3600_featureset-dynamicstatic_iteration-29 \
  --out $COE/artifacts

# (b) 编译 PC-HGS（源外；依赖已缓存在 /home/hzeng/pchgs_deps）
cmake -S CC_Compare/CO-enriched-ML/pchgs -B $COE/build_pchgs -DCMAKE_BUILD_TYPE=RELEASE \
  -DCPM_SOURCE_CACHE=/home/hzeng/pchgs_deps \
  -DFETCHCONTENT_SOURCE_DIR_FMT=/home/hzeng/pchgs_deps/fmt \
  -DFETCHCONTENT_SOURCE_DIR_NLOHMANN_JSON=/home/hzeng/pchgs_deps/nlohmann_json \
  -DFETCHCONTENT_SOURCE_DIR_DYNAMIC_BITSET=/home/hzeng/pchgs_deps/dynamic_bitset
cmake --build $COE/build_pchgs --target PCHGS -j 10     # 只建 PCHGS 目标，见 RESULTS §1

# (c) 烟测（官方权重 / 重训权重；可选 PC-HGS）
/home/hzeng/envs/cc_compare/bin/python $COE/smoke_coenrich.py \
  --artifacts $COE/artifacts_a1 --pchgs-exe $PWD/$COE/build_pchgs/PCHGS --pchgs-seconds 2

# (d) 重训（先造数据，再训练）
cd C-VRP_Cold-chainVehicleRoutingProblem
/home/hzeng/envs/cc_compare/bin/python ../$COE/gen_coenrich_train_data.py \
  --days 8 --label-scope new --out ../$COE/data/train_pool.npz
/home/hzeng/envs/cc_compare/bin/python ../$COE/train_coenrich_a1.py \
  --data ../$COE/data/train_pool.npz --out ../$COE/artifacts_a1
```

## 5. 进入 40 天正式臂的状态

**尚未接线**（用户约束：F1 正式批次 `results/a1_ext_formal2/` 跑完前不得修改
`ordering_providers.py` 等计算链文件）。接线只需在 `ordering_providers._REGISTRY` 加一行：

```python
'coenrich': ('CC_Compare/CO-enriched-ML/a1_accept_v1',
             'coenrich_provider', 'COEnrichedMLProvider'),
```

随后用统一入口跑两臂（与其它方法完全同口径）：

```bash
cd C-VRP_Cold-chainVehicleRoutingProblem
PY=/home/hzeng/envs/cc_compare/bin/python
$PY scripts/evaluation/run_a1_external_accept.py --solver ordering --provider coenrich \
   --provider-ckpt ../CC_Compare/CO-enriched-ML/a1_accept_v1/artifacts_a1 \
   --provider-device cpu --baseline-from results/a1_c1fresh_reveal_40/gate.json \
   --dev-instances 40 --workers 6 --penalty "p_c=0" --out results/a1_ext_extra/coenrich_0_f
# 主档同法，--penalty "p_c=(5,10,15)"
```

⚠️ **计时预算**：PC-HGS 单次调用 `-t 2` ≈ 2.1 s；provider 按可见子问题内容缓存
（同决策内 15 辆车 → 1 次 PC-HGS），因此单决策 ≈ 2.1 s + 共享认证 ≈ 10 s 预算内，
但 40 天 × ~200 决策/天 的墙钟显著高于纯 NumPy 模式（0.02 s/决策）。若要低风险起跑，
可先用 `--no-pchgs`（不传 `--pchgs-exe` / 不设 `COENRICH_PCHGS_EXE`）= 纯学习排序模式。
