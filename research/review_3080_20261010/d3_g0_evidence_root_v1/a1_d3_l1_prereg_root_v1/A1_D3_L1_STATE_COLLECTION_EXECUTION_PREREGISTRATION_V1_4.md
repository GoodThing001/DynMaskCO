# A1_D3_L1_..._V1.4：P0-B 方案 B 通过（距离合同逐位验证 + 扰动修正 + 投影假通过修复）

- 追加记录，**V1/V1.1/V1.2/V1.3 与既有产物全部不改写**；本文件包含对 V1.3 结论的**更正**
- 证据：`scripts/simulation/d3_l1_causal_perturbation.py`（重写）、
  `l1_gates/A1_D3_L1_P0B_CAUSAL_PERTURBATION.{json,_stdout.txt,_stderr.txt}`
- 本轮结果：**5 行 / 5 PASS / 0 FAIL**；`DIST_MAT_GENERATION_CONTRACT = VERIFIED`

## 1. 更正我在 V1.3 的判断（你的纠正是对的）

V1.3 说"`dist_mat` 不等于 coords 的欧氏距离 ⇒ 可能是外部路网矩阵"。**该结论错误**：
我用 **float64 + `np.linalg.norm`** 计算，而生成器用 **float32** 运算顺序——两者不是同一计算路径，
据此**不能**推断数据合同。V1.3 产物原样保留，本文件为更正。

按你的要求改用生成器原序复算：

```python
coords32 = np.asarray(dataset["coords"][day], dtype=np.float32)
diff = coords32[:, None, :] - coords32[None, :, :]
dist_rebuilt = np.sqrt((diff ** 2).sum(axis=-1))
```

**逐日 `np.array_equal` 结果：16/16 天逐位相等，0 差异，dtype 与 shape 均一致** →
`DIST_MAT_GENERATION_CONTRACT = VERIFIED`（**未放宽容差**；若有差异会输出首个差异位置/数值/位模式并保持 `NOT_VERIFIABLE`）。

**生成器身份**：`M0_TEACHER_MANIFEST.json` 的 `generator_sha256` = `43cf781bec078dd4…` = 实际
`generate_coldchain_data.py` 的 sha256；数据集 sha = 已登记的 dev_check `66573ddc…39c89`（B0 PASS）。
（数据内容仍以**服务器实算**为准；GitHub 侧不含 NPZ 原件。）

## 2. 修好的实现错误（你指出的第二点）

旧代码在扰动通过后把**由原始坐标算出的 `eu`** 写回扰动后的 `dist_mat`。现在：**用扰动后的 `coords`
按同一 float32 顺序重算**，实测：
`rebuilt_from_perturbed_coords = true`、`entries_changed = 1848`、
**可见×可见项逐位不变**（`visible_visible_unchanged = true`）、dtype 保留。
`energy_mat` **未改动**并显式记录"其合同需先声明才可重建"（避免相互矛盾的派生输入）。
历史 teacher 标签未进入决策投影。

## 3. 修好的投影假通过风险（你指出的第三点）

1. **`plan()` 原地修改传入车辆、无返回值** → 现改为 `deepcopy` 车辆、调用 `plan()`、**忽略返回值**、
   从**克隆对象**读取 `mutable_suffix`；两侧各 `replanner_calls = 1`，无异常；
2. **`complete` 不再把"存在 error 键"当作完成**：要求三项输出（可见输入 / 合法候选 / planned suffix）
   **真实产生且无 error 键**，否则 `NOT_VERIFIABLE`；
3. **删除 B5 的无条件 `True`**，改为真实逐项比较；
4. **导入边界已解决**：投影使用的 `decision_pool_from_vehicles` 来自 `dynmaskco_cc_context`
   （sha `4eebb2a51a024a9e…`）→ 已**显式登记为 F-free 执行链依赖**（不再"一边排除一边导入"），
   且进程内断言 `process_stayed_f_free = true`。

## 4. 本批实测结论（F-free 边界，范围严格限定）

原始与扰动两侧独立重建的**可见输入 SHA、合法候选 SHA、planned suffix SHA 全部相等**
（`complete = true`）⇒ 在**本预声明扰动族**（24 个不可见客户的 coords/demands/tw/service_time，
distance 由扰动坐标按 float32 重算）下，**该无模型决策信息边界不变**。
**F 侧边界（按你的要求）**：这只证明 F-free 边界；真正的 F 特征提取器仍须在
**F−D-Geo 机制诊断资格门**内单独核验，**不得**据此声称"模型特征因果安全已证明"。

## 5. 状态与下一步

```
P0_A = CLOSED_ON_COUNTEREXAMPLE_GATE
P0_B = PASSED_ON_PLAN_B_GATE            # 5/5；距离合同逐位验证 + 扰动修正 + 投影修复
P0_C = PENDING                          # 正式 worker/supervisor + 真实入口闭包
FORMAL_COLLECTION_AUTHORIZATION = NOT_AUTHORIZED
```
方案 A / C **未实施**（按你的裁定，B 未失败前不提）；120/60/3600 三硬截止、原子快照发布、manifest、
失败不可重采、64 槽位分母、合同与数据身份**全部不变**；未重跑服务器计时样本。
下一步（需你授权）：进入 **P0-C** —— 实现正式 worker 与独立 supervisor，并以**真实入口**重建源码闭包
（含本轮新增的 `dynmaskco_cc_context` 依赖与新的因果投影/比较器模块 SHA）。

## 6. Claim limits

未采集状态、未调用 F/D-Geo、未改历史合同/profile/冻结源码/归档报告。
`states_collected_for_L1 = 0`；`algorithmic_gain_claim: none`；D3 G0 未通过、D8 阻断、E2 缺失、
`stable_online_gain=False`、I3 正式负结果与旧 D2 `not_qualified` 全部不变。
