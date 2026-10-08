# LIH（Learn-to-Improve-Heuristics / L2I）A-v1 对比臂结果（诊断批次）

> **⚠️ 诊断批次声明（2026-09-30 主线协议更新）**：本批四条臂运行于**时限修复前**的
> OrderingAcceptReplanner（旧语义：越限决策仍先提交接单、再计超时），且 pool 含
> 在途 `committed_next` 客户（主线已修复为 pool 减在途）。若 40 天中出现
> `duplicate_service` 硬失败（POMO 诊断臂 40/40 天复现），即该机制的实锤证据；
> 本批数字**仅诊断，不构成正式裁决**。正式重跑由主线在修复后统一执行，本批产物保留。

## 1. 方法与档位

| 项 | 值 |
|---|---|
| 方法 | Learn-to-Improve-Heuristics（L2I，TNNLS 2022）CVRP50：AM 编码器 + NeuRewriter 成对局部算子改进步 |
| provider | `CC_Compare/Learn-Improvement-Heuristics/a1_accept_v1/lih_provider.py`（类 `LIHProvider`） |
| checkpoint | `CVRP/CVRP50/outputs/cvrp_50/run/epoch-199.pt`（官方 CVRP50；args: embedding/hidden 128、3 层、batch-norm、steps=100） |
| 档位 | 官方 test.py 硬编码 1000 改进步，CPU 实测 26s(n=50)/53s(n=90) 超预算 → **官方参数空间最小档 steps=100**（options.py 默认值 = epoch-199.pt 训练配置 args.json 的 steps 值） |
| 设备 | CPU（`--provider-device cpu`，禁止 GPU） |
| 输入规范化 | 官方 VRPDataset 口径：坐标 [0,1]、demand/capacity；100-token 循环序列（≤50 客户 token + depot token；seq_tensor2 硬编码 token>50 为 depot） |
| 初始解桥接 | 见 §5.1（官方初始解在其自身数据上容量不可行 → 贪心装箱可行循环序列） |
| 规模桥接 | 官方模型硬编码 50 客户槽 → pool > 50 按 50 分块、逐块 LIH 后拼接（模型规模上限的必要桥接，非超参数） |

## 2. 烟测（服务器，CPU）

mock sp 用**非连续真实风格 id**（101..299），断言 `sorted(order) == sorted(pool_customer_ids)`：

| pool | 求解耗时 (steps=100) | 覆盖 |
|---|---|---|
| 5 | 3.06s | ✓ |
| 25 | 2.78s | ✓ |
| 60（分块 50+10） | 5.28s | ✓ |

## 3. n=90 无竞争最大池探针（2026-09-30，服务器 load ≈200/96 共享窗口）

- 3 次单决策 `provider.order`（n=90，分块 50+40，steps=100）中位数 **5.52s**
  （4.80–5.74s）→ ≤7s 探针线，**正式档 = steps=100 不动**。
- 参考：steps=1000（官方 test.py 硬编码）n=90 中位数 52.9s、n=50 单解 26.0s —— 官方档在本任务 CPU 预算外的实测依据。

## 4. 结果（seed 20260926 开发集 40 天；`results/a1_ext_extra/lih_{0,51015}/gate.json`）

### p_c=0

| 指标 | 值 |
|---|---|
| utility_all_days mean [ci_lo, ci_hi] | **741.25** [599.81, 897.20]（n=40）⚠️ 高负载污染（见下） |
| n_days / adjudicable | 40 / True |
| hard_feasible_rate / failures | 1.0 / `{}`（无 duplicate_service） |
| mean_timeouts | 107.8/天（总计 4312 次；max_decide_s=261s——该臂在服务器负载峰值窗口运行，10s 预算大量超限） |
| reject_reasons (solver_meta) | `{'timeout': 4312, 'deferred': 5, 'certify_budget': 1865}`（n_solves=7761, n_fail=0, mean_solve 19.40s） |
| ordering_accept − cond_hist | **−743.28** [−889.57, −581.97]（n_paired=40, adjudicable=True） |
| ordering_accept − uncond_hist | −703.47 [−835.48, −557.72] |
| ordering_accept − explicit_feat | −706.26 [−846.04, −554.23] |

### p_c=(5,10,15)

| 指标 | 值 |
|---|---|
| utility_all_days mean [ci_lo, ci_hi] | **−71.03** [−202.36, 56.76]（n=40） |
| n_days / adjudicable | 40 / True |
| hard_feasible_rate / failures | 1.0 / `{}` |
| mean_timeouts | 6.73/天（总计 269 次；该臂在较低负载窗口运行） |
| reject_reasons (solver_meta) | `{'timeout': 269, 'certify_budget': 4849, 'deferred': 18}`（n_solves=7761, n_fail=0, mean_solve 3.23s） |
| ordering_accept − cond_hist | **−379.60** [−428.55, −331.83] |
| ordering_accept − uncond_hist | −308.11 [−389.32, −221.74] |
| ordering_accept − explicit_feat | −315.84 [−392.86, −236.92] |

（配对块 `formal_adjudication=False`、`identity_verified=False`（Q-04 门控：诊断批次
驱动源码与步骤 2 基线身份不匹配，仅诊断口径）。两臂为同数据独立运行；p_c=0 臂在
服务器负载峰值（load≈190/96，与 xrf/a1c1r/a1s3r2 及兄弟方法臂共享 CPU）窗口完成，
其 mean_solve 19.4s、4312 次超时即该窗口的高负载污染证据；p_c=(5,10,15) 臂在较低
负载窗口完成（mean_solve 3.2s）。正式 `lih_*_f` 腿由主线在时限修复后运行，正式
数字以主线同步为准。）

## 5. 适配日志（错误与修复）

1. **官方初始解容量不可行（实锤发现）**：官方 `test_function.validate()` 把实例行
   按需求排序后令初始 `rec=[1..100]`（全体客户一段 + depot 段）——在其自身
   CVRP50 数据上（50 客户总需求 ≈ 6.25×容量）段和 >1 → `seq_tensor2` 的
   che_mask **全 False** → `MultiHeadAttention_to_attn` softmax 输入全 -inf →
   NaN → multinomial 崩溃（官方代码在其自身数据上也会崩溃；旧 torch 的 uint8
   掩码语义掩盖了这一点）。**桥接**：provider 构造容量可行的初始循环序列
   （按 chunk 序贪心装箱，段和 ≤1，段满插 depot token 51..100，尾部补 depot +
   n<50 时的零需求假客户 token 凑满 100 位）——NeuRewriter 改进步与官方完全一致。
2. **上游 CUDA 时代代码**：`problem_vrp.get_costs`/`seq_tensor2`/`emdedding` 多处
   硬编码 `.cuda()` → 进程内 monkey-patch `torch.Tensor.cuda` 为恒等（本臂纯 CPU，
   不改上游文件）。
3. **torch 2.x 不兼容**：`Tensor.byte` 已移除（`1 - mask.byte()` 掩码索引退化为
   整数索引）→ provider 进程内补 `Tensor.byte`（返回 bool）并替换
   `MultiHeadAttention_to_attn.forward` 为 bool 掩码等价实现；采样分支
   `att.squeeze()` 在 batch=1、n_heads=1 时挤掉 batch 维崩溃（上游
   eval_batch_size=128 才成立）→ 替换 `GraphAttentionEncoder.forward` 为
   batch=1 安全等价版（显式 head-0 分布采样）。以上均为进程内桥接，上游文件不改。
4. **性能**：逐 token 的 `emdedding` Python 循环（100 次 nonzero）+ 每步重建
   `position_encoding_init` 表 → 向量化 emdedding（数值逐元素一致）+ 位置编码缓存
   → 100 步由 20-40s 降至 3-5s。
5. **烟测 mock 升级**（MVMoE 教训）：非连续真实风格 id + 真实 id 集合覆盖断言；
   provider 输出 `chunk[t-1]` 真实 id，无 index/id 混淆。
6. **共享机负载污染（诊断批次实录）**：p_c=0 臂在服务器负载峰值窗口运行，
   `torch.set_num_threads(8)` 的 OpenMP 空转 + 96 核机器 load≈190 的超订使
   每决策墙钟放大数倍（max_decide_s=261s、4312 次超时、mean_solve 19.4s）。
   同期 1 天诊断（workers=1，3000 次 provider 调用 / 2800 次 memoize 命中、
   200 决策 200 求解）实测每决策 ≈2.9s（p_c=(5,10,15) 臂较低负载窗口
   mean_solve 3.2s 与之一致）。正式批次定档时可把 provider 线程数降为 2
   （运行期参数，不改算法）以削减空转。

## 6. 已知局限（诚实口径）

- **CVRP-only 零样本跨任务**：原生无 TW/冷链/多温区/在途锚点；本臂按 CVRP 求解
  排序、忽略 TW，真实 TW/C0 可行性由下游冻结 dcc_rh_v4 协调器 + certify_plan
  硬认证兜底。
- steps=100 为官方参数空间最小档（官方 test.py 硬编码 1000 档在本任务 CPU 预算外，
  实测 26s/53s）；初始解为容量可行贪心装箱桥接（官方初始解在其自身数据上不可行，
  §5.1）；pool > 50 按 50 分块（官方模型 50 客户槽硬编码）。
- 官方采样档（test=False）随机性：NeuRewriter 每步 multinomial 采样，同决策内
  15 辆车共享同一 pool memoize 保证一致性；不同决策间随机。
- 诊断批次时限与 pool 语义见文件头声明。
