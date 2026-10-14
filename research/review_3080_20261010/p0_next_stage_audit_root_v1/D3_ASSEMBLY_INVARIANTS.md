# D3 装配层不变量（A1–A4）与两项新风险

**日期**：2026-10-14 · **基准**：`95c11d7`（本轮起点）· **性质**：装配层代码 + 合成反例 + 独立回归测试
**授权**：用户 2026-10-14 裁定（优先第 1 项：D3 装配层不变量）
**边界**：不运行真实 F−D 比较、不执行 32 状态比较、不训练、不采集、不触碰最终测试种子、
不修改既有冻结树或历史裁决。

---

## 0. 结论

用户从源码发现的两项新风险**全部成立**，测试过程中又暴露**第三项**；三项均已修复并有反例。

| 套件 | 修复前 | 修复后 |
|---|---|---|
| `scripts/tests/test_d3_assembly_invariants.py`（A1–A4，本轮新增） | **5 通过 / 17 失败** | **23 / 0** |
| `scripts/tests/test_d3_adapter_risks.py`（R1–R4） | 10 / 0 | **10 / 0** |
| `scripts/tests/test_d3_geo_adapter.py`（T-G1..T-G9 + X1/X2） | 33 / 0 | **36 / 0** |

原件：`raw/A_invariants_PREFIX.txt`（修复前反例）、`raw/A_invariants_POSTFIX.txt`、`raw/T_G_POSTFIX.txt`。

## 1. 新风险 N1：采样器默认容差仍大于零（成立）

`stepwise_sample_d3()` 的 `eps_rel/eps_abs` 默认仍是 `1e-9/1e-12`，而 `select_geo()` 已改为 0
⇒ **装配层若不显式传零，"主对照 eps=0" 不会自动兑现**。
**修复**：两个函数的默认值统一为 `EPS_D_PRIMARY_REL/ABS = 0.0`；
原暂定值保留为 `EPS_D_*_PROVISIONAL`，仅供**非主对照**的稳健性变体显式启用。
反例断言：`inspect.signature` 上两者的默认值都必须为 `0.0`。

## 2. 新风险 N2：`eps=0` **也不**足以保证 F 与 D-Geo 平局一致（成立）

`select_geo` 原本用**规范动作键最小**做平局，而 F 的历史规则是
**`legal` 顺序下的首个最大值**。当实际 `legal` 顺序与规范键序**不一致**且出现精确同分时，
**即使 `eps=0` 两者仍可选不同动作**。原 P0 报告"规范顺序下等价"的结论**有前提**，
不能升级为实际装配路径的无条件等价——用户判断正确。

### 2.1 设计取舍（按用户裁定实现）

> **唯一主对照保留 F 的历史 `legal` 顺序与首个最大值规则。**
> D-Geo 在 `eps=0` 时**也按同一 `legal` 顺序选第一个最小距离动作**；
> 规范动作键**仍然记录**用于审计与跨进程核查，但**不得**擅自取代 F 的历史平局规则。

**实现**：新增 `TIE_BREAK_LEGAL_ORDER`（主判）/ `TIE_BREAK_CANONICAL`（审计与变体）。
`single_factor_with_F` 仅在 `eps == 0 且 tie_break == legal_order_first` 时为 `True`。
把两臂都改成"先按规范键重排序"虽然形式上对称，却可能改变 F 原本的动作选择、
从而引入**新的策略版本**——本轮明确不这么做。

### 2.2 构造性反例

`legal` 顺序刻意与规范键序**相反**（下标 0 = customer 5，其键更大）且距离精确同分：

* F：`argmax` 取**首个最大** ⇒ 下标 0 ⇒ **customer 5**
* 旧 D-Geo（规范键）⇒ 下标 1 ⇒ **customer 4** ⇒ **动作不同**（反例成立）
* 新 D-Geo（legal 顺序 + `eps=0`）⇒ 下标 0 ⇒ **customer 5** ⇒ 与 F 一致

## 3. 新风险 N3：`SELECTOR_GEO` 在 `deterministic=False` 时**静默退化**（测试中发现）

该组合原本落入历史 softmax 采样分支 ⇒ **请求的 geo 规则被忽略**；
且当评分向量长度与 `legal` 不符时在 `rng.choice` 内**崩溃**（`ValueError: a and p must have same size`），
而不是 fail-closed。
**修复**：`selector == SELECTOR_GEO and not deterministic` **直接抛错**（拒绝静默退化）。
反例断言该组合必须抛 `ValueError`。

## 4. A1–A4 验收

| 类别 | 覆盖 |
|---|---|
| **A1 主对照参数传递** | 两个函数默认 `eps=0`；主判 tie_break = legal-order-first；正容差仍存在但仅供变体；安装事务 API 存在；**GEO 非确定性被拒** |
| **A2 F 历史选择等价** | 逆序规范键 + 精确同分 ⇒ 新装配层与 F 选同一动作；规范键仍被记录；`single_factor_with_F=True`；**端到端**与原始 `stepwise_sample` 逐步一致（投影到历史字段，且唯一增量为已固定的 `incremental_distance`） |
| **A3 安装事务安全** | `no_legal` / `nan` / `timeout` / 注入异常 四类 ⇒ **安装次数 0**；异常**不外泄**；成功后**恰好安装 1 次**；失败时原计划不变 |
| **A4 记录与计时** | 每步落盘 `legal_order` / `canonical_keys` / `distance_ties` / `score_exact_ties` / `chosen`；六项工作量与墙钟分别报告；**认证之后越过上限 ⇒ 判 `timeout`，不得计为正常完成** |

### 4.1 安全改进声明（**不是**历史行为等同）

用户指出："F 的选择规则未变"与"F 的失败后安装行为未变"是**两个不同命题**。本轮如实区分：

* **F 的选择规则未变** —— **严格保持**，由 A2 与 X2 断言（F 仍是首个最大值 `argmax`，
  1e-13 间距**不**视为平局）。
* **F 的失败后安装行为未变** —— **不主张**。新增的 `InstallLedger` / `run_repair_transaction`
  给装配层加了 **fail-closed 保护**（仅 `failure_reason is None` 才允许安装，
  异常转为 `exception:<Type>` 且安装次数为 0）。这是**安全性改进**，
  **不是**历史行为完全相同。

### 4.2 认证后超时（用户指出）

`stepwise_sample_d3` 原本只在每轮**开始**检查墙上限；认证成功但已越过上限时会被当作**正常完成**。
**修复**：认证通过后**再检查一次**，命中则 `diag['timed_out']=True` +
`diag['timed_out_after_certification']=True` 并回退原计划。A4 用 `certify_fn` 内 `sleep(0.02)`
配 `wall_clock_limit_s=0.005` 构造反例。

## 5. 测试命令

```bash
python C-VRP_Cold-chainVehicleRoutingProblem/scripts/tests/test_d3_assembly_invariants.py
python C-VRP_Cold-chainVehicleRoutingProblem/scripts/tests/test_d3_adapter_risks.py
python C-VRP_Cold-chainVehicleRoutingProblem/scripts/tests/test_d3_geo_adapter.py
```

解释器：`D:\AA_Py\A_Anaconda\python.exe`（jax 0.5.0 / numpy 1.26.4；`mpre_policy` 需要 jax）。

## 6. 失败用例处理声明

本轮所有失败都用**修代码**解决，**没有**通过放宽容差、重排 F 历史候选或删减失败用例来变绿。
两处测试期望的修改都是**契约变更的如实记录**并已留痕：
T-G5 的两条断言原按"规范键为主判"编写，现按新契约拆成**主判（legal 顺序，顺序相关是设计意图）**
与**规范键变体（顺序无关）**两条分支分别断言；A2 端到端按 X1 的同一处理投影掉已固定的
`incremental_distance` 增量并断言该增量恰好只有一个字段。

## 7. 停止条件与不主张

装配不变量全绿 ⇒ **本轮到此停止**。
**测试通过不意味着 D3-Science 获得放行资格**：E2 训练资格、`P_after_install` 前瞻采集资格、
32 状态可恢复性、完整 8+2 秒发布门**均未满足**。
不主张 F 与 D-Geo 有任何效果差异；不把"工程实现完成"升级为"主模型有效性获得证明"；
**MaskCO 稳定核心在线增益仍未建立（`False`）**。
