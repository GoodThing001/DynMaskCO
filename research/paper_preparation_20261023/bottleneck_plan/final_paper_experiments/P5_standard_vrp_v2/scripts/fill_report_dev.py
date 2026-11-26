# -*- coding: utf-8 -*-
"""把开发面板与固定候选诊断结果写入 FINAL_REPORT_V2.md（替换占位符）。"""
import json
import statistics as st
from pathlib import Path

V2 = Path(__file__).resolve().parent.parent
rep = V2 / "FINAL_REPORT_V2.md"
text = rep.read_text(encoding="utf-8")

dev = json.loads((V2 / "development_results/dev01_final/analysis_dev.json").read_text(encoding="utf-8"))
diag = json.loads((V2 / "tables/analysis_diag.json").read_text(encoding="utf-8"))
cells = dev["cell_table"]
agg = dev["per_version"]
pairs = dev["pairs"]
mech = dev["mechanisms"]

lines = ["### 7.1 五版本聚合（240 端点，零失败、零复算不一致）", "",
         "| 版本 | 端点 | 平均距离 | 中位距离 | 相对初解改善% | 平均车辆 | 平均 epochs | 平均在线 s |",
         "|---|---|---|---|---|---|---|---|"]
order = ["STD_OLD", "STD_BUGFIX", "STD_FIX", "STD_NORM", "STD_DIST"]
for v in order:
    a = agg[v]
    lines.append("| %s | %d | %.1f | %.1f | %+.2f | %.2f | %.1f | %.2f |" % (
        v, a["endpoints"], a["mean_distance"], a["median_distance"],
        a["mean_gain_vs_initial_pct"], a["mean_vehicles"], a["mean_epochs"], a["mean_online_s"]))
lines += ["", "### 7.2 配对比较（后者相对前者的距离变化%，负值=后者更优）", "",
          "| 对比 | n | 平均Δ% | 中位Δ% | 后者更优 |", "|---|---|---|---|---|"]
for k, p in pairs.items():
    lines.append("| %s | %d | %+.3f | %+.3f | %d |" % (
        k, p["n"], p["mean_delta_pct"], p["median_delta_pct"], p["b_better"]))
lines += ["", "### 7.3 机制计数（校准与双模式确实被使用）", "",
          "| 版本 | 试选选项采用 [距离, 原始热图, 校准热图] | 掩码模式 [随机, 相关] | regret epochs |",
          "|---|---|---|---|"]
for v in order:
    m = mech[v]
    rc = m["ranking_counts"]
    mm = m["mask_mode_counts"]
    lines.append("| %s | [%s, %s, %s] | [%s, %s] | %d |" % (
        v, rc.get("0", 0), rc.get("1", 0), rc.get("2", 0),
        mm.get("0", 0), mm.get("1", 0), m["regret_epochs"]))

lines += ["", "### 7.4 按目标拆分（均值，CVRP 距离/车辆、VRPTW 距离/车辆）", "",
          "| 版本 | CVRP n=24 平均距离 | CVRP 平均 K | VRPTW n=24 平均距离 | VRPTW 平均 K |",
          "|---|---|---|---|---|"]
for v in order:
    cv = [c["versions"][v] for c in cells if c["objective"] == "cvrp" and v in c["versions"]]
    vt = [c["versions"][v] for c in cells if c["objective"] == "vrptw" and v in c["versions"]]
    lines.append("| %s | %.1f | %.2f | %.1f | %.2f |" % (
        v, st.mean([x["distance"] for x in cv]), st.mean([x["vehicles"] for x in cv]),
        st.mean([x["distance"] for x in vt]), st.mean([x["vehicles"] for x in vt])))

lines += ["", "### 7.5 固定候选数量诊断（48 单元 = 24 实例 × 2 种子）", "",
          "同一初解、同一 40 客掩码、同一候选集（全部可行插位 + 新开单车）、同一插入顺序，",
          "仅改变排序规则，统计每个客户的插位代价后悔值（所选 − 最优）与最优插位召回率。", "",
          "| 排序 | 平均后悔值 | 中位后悔值 | recall@1 | recall@3 | 平均最终距离 | 相对距离排序 |",
          "|---|---|---|---|---|---|---|"]
pv = diag["per_variant"]
pr = diag["pairs"]
label = dict(distance="距离（参照）", heat_raw="原始热图", heat_calibrated="校准热图",
             heat_norm="归一化热图")
for k in ("distance", "heat_raw", "heat_calibrated", "heat_norm"):
    d = pv[k]
    rel = "—" if k == "distance" else "%+.2f%%（%d 优 / %d 劣）" % (
        pr[k + "_vs_distance"]["mean_delta_pct"], pr[k + "_vs_distance"]["better"],
        pr[k + "_vs_distance"]["worse"])
    lines.append("| %s | %.2f | %.2f | %.3f | %.3f | %.1f | %s |" % (
        label[k], d["mean_regret"], d["median_regret"], d["mean_recall1"],
        d["mean_recall3"], d["mean_final"], rel))
lines += ["",
          "> **口径说明**：单步 oracle 定义为\"插入代价最小\"，因此距离排序在单步上按构造最优",
          "> （recall@1 = 1、后悔值 = 0）。该诊断度量的是**排序偏离立即代价最优的程度**，",
          "> 不是神经排序的潜在长期收益；后者只能由 60 秒面板（§7.1–7.2）回答。",
          "",
          "**读法**：校准把神经排序的立即代价惩罚从 +4.80% 降到 +2.93%（中位 +3.67% → +1.96%），",
          "方向与锁版机制（距离条件残差校准）一致，但仍未超过纯距离排序；",
          "面板层面 STD_FIX 相对 STD_DIST 为 +0.74%（48 单元，24 优于 DIST），",
          "两者方向一致、量级不同（面板有搜索与跨车 Split 补偿）。"]

new_dev = "\n".join(lines)
if "<!-- DEV_RESULTS -->" in text:
    text = text.replace("<!-- DEV_RESULTS -->", new_dev)
else:
    raise SystemExit("placeholder missing")
rep.write_text(text, encoding="utf-8")
print("dev + diag sections written, %d chars" % len(new_dev))
