# MaskCO 直接目标训练 — 搜索增强验证与收束

日期：2026-09-22。性质：结果冻结与退出判定，不是新实验计划。

## 结论（一句话）

在干净 checkpoint（`f5acf83f…`）上，**逐层 SGBS 式搜索能把 learned 策略拉到 ≈ regret-2 的水平，但仍被「距离启发式 + 相同搜索」显著支配**；learned 提议不构成部署增量，**收束本 checkpoint 的搜索增强路线**（不加 beam、不加 EAS、不追训）。

> 这不是「MaskCO 不可学习」：干净训练确实产生了「优于 M-pre、缩小 ~61–66% 与 R 差距」的信号；但该信号在公平搜索下未能超过廉价距离启发式，因此方法创新的性能优势尚未成立。

## 两处命名修正（执行端审计）

1. **首动作探针 ≠ SGBS**。第一版 `_sgbs` 只做「greedy + 首动作 top-2 + 贪心补全」，未扩展第二层动作；第二层排名次高动作可能产生更低 `J_vis`，第一版不会尝试。因此早期「搜索已排除」结论过强，已撤回。
2. **计时口径**：wallclock 受 JIT 编译（unpadded 每遇新 shape 重编译）与执行顺序污染，429s/101s 不能当稳态延迟比。

## 最终结果（逐层 beam：greedy + 2 首动作 + 4 末步 = 7 次/状态，去重 ≈4 唯一）

接受后 J（实例级均值，n=32×2，mask=2）：

| 方法 | TRAIN | CAL |
|---|---|---|
| **距离提议 + SGBS** | **1.0617** | **1.1447** |
| R（regret-2 单次） | 1.0697 | 1.1515 |
| M-trained + SGBS | 1.0789 | 1.1528 |
| M-trained greedy+7采样 | 1.0779 | 1.1528 |
| M-trained greedy | 1.0937 | 1.1663 |
| M-pre + SGBS | 1.0890 | 1.1629 |

配对 CI（实例聚类 bootstrap）：

| 对比 | TRAIN | CAL |
|---|---|---|
| 距离SGBS − R | −0.008 [−0.014, −0.003] | −0.007 [−0.016, −0.002] |
| M-trained SGBS − 距离SGBS | +0.017 [0.010, 0.024] | +0.008 [0.003, 0.014] |
| M-trained SGBS − R | +0.009 [0.000, 0.017] | +0.001 [−0.010, +0.010] |
| M-trained SGBS − greedy | −0.015 [−0.023, −0.007] | −0.014 [−0.021, −0.007] |

补全统计（32 状态合计，三种 SGBS 方法）：`n_total≈224`（=7/状态），`n_unique≈119–126`，`n_fail=0`。

## 判定（对齐事先固定的退出标准）

命中 **「learned 搜索仍输距离搜索」**：

- 距离提议 + 相同搜索在两分片都显著优于 learned 提议（+0.017/+0.008）和 R（−0.008/−0.007）；
- learned 提议只在 greedy 基础上被 beam 改善（−0.015/−0.014），但这改善是「搜索」贡献——换成距离启发式后更好；
- M-trained SGBS ≈ 8 次盲采样（跨零），beam 不比采样更聪明。

→ **收束当前 checkpoint 的搜索增强路线。** 不加 beam、不加 EAS、不追训。

## 实现验收（已通过）

- **搜索验收**：两步合成例中，最优完整方案必须用第二层高分动作（greedy=10.0，最优=5.0）；`sgbs_search` 正确找到 5.0，n_total=7。
- **缓存验收**：`mpre_score_fn_cached`/`mtrained_score_fn_cached` 与未缓存 bit-identical（合成 64 例 worst |Δ|=0；真实 4 状态 worst |Δ|=0，完整计划 hash 一致），且同一 H0 下改变动态特征后分数变化（|Δ|>0），确认 C/decoder 重算。

## 产物身份

- 干净 checkpoint：`results/m0_scale/mpre_reinforce_s42_clean/model.ckpt`，SHA256 `f5acf83fa2519020aeea274df5ecf740a6ad809b215d2d270209b7901903a7f5`
- 训练数据：`data/m0_scale/dcc_50_r1_edod05_train_teacher.npz`，SHA256 `c70258c4fde86e5ebfb08c17e4f8ccacac5f1b18319dc180eaa048e8ae771d01`
- 逐层 SGBS 结果：`results/m0_scale/sgbs_fixed_train_v2/`、`sgbs_fixed_cal_v2/`（`summary.json` + `per_state.json`）
- 首动作探针（已废弃参照）：`results/m0_scale/sgbs_fixed_train/`、`sgbs_fixed_cal/`
- 采样诊断：`results/m0_scale/fsq_clean_{train,cal}/sampling_diag.json`
- 缓存/搜索验收：`results/m0_scale/_cache_real_verify.log`；`verify_cache_equiv.py` / `verify_cache_real.py` / `verify_sgbs_search.py`

## 复现命令

```bash
# 逐层 SGBS 固定状态对照（TRAIN / CAL）
python scripts/evaluation/run_sgbs_fixed_state.py \
    --data data/m0_scale/dcc_50_r1_edod05_train_teacher.npz \
    --cvrp-ckpt ../MASKCO_code/ckpts/cvrp100.ckpt \
    --model-ckpt results/m0_scale/mpre_reinforce_s42_clean/model.ckpt \
    --objective-profile results/o0cc/scale_v2/objective_profile.json \
    --split train --out results/m0_scale/sgbs_fixed_train_v2
# CAL：--data dcc_50_r1_edod05_cal_teacher.npz --split cal --out sgbs_fixed_cal_v2
```

## 下一步（供后续，不本轮执行）

搜索增强被距离启发式支配后，**下一项是重新定义学习模块相对 R/距离启发式的职责**（候选方向：车辆—部分路线条件化）。但需先写清：它比现有 δ 多解决什么可观察问题、如何同训练预算比较、如何战胜距离搜索。搜索失败本身不自动支持加辅助 CE 或换注意力层。
