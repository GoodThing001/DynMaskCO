# -*- coding: utf-8 -*-
"""把正式面板分析结果写入 FINAL_REPORT_V2.md（替换 <!-- FORMAL_RESULTS --> 与审计占位）。"""
import json
from pathlib import Path

V2 = Path(__file__).resolve().parent.parent
rep = V2 / "FINAL_REPORT_V2.md"
a = json.loads((V2 / "tables/analysis_formal.json").read_text(encoding="utf-8"))
ext = json.loads((V2 / "tables/analysis_external.json").read_text(encoding="utf-8"))
INTERNAL = ["ORDER_SPLIT", "ORDER_SPLIT_D", "MASK", "SLOT", "DIST", "DIST_SLOT"]
lines = []

rep_txt = a.get("report") or {}
lines += ["### 8.1 覆盖与完整性", "",
          "- 端点数：%s（CVRP %s + VRPTW %s + STD_NORM %s）" % (
              rep_txt.get("endpoints"), len(a.get("v2_1") or []) and "见 V2-1" or "0",
              "见 V2-4", "见 V2-6"),
          "- 独立复算：全部端点由 `scripts/p5v2_audit.py` 重算（覆盖性/容量/时窗/车辆数/距离），",
          "  差异清单见 `formal_results/*/audit.json`。", ""]

v22 = a.get("v2_2") or []
if v22:
    lines += ["### 8.2 V2-2 CVRP 规模段（平均 Gap 到官方 BKS，逐例子差后等权）", "",
              "| 规模段 | 例数 | " + " | ".join(INTERNAL) + " |",
              "|---" * (len(INTERNAL) + 2) + "|"]
    for r in v22:
        vals = " | ".join(("%.3f" % r[m + "_mean_gap%"]) if m + "_mean_gap%" in r else "-"
                          for m in INTERNAL)
        lines.append("| %s | %s | %s |" % (r["bucket"], r["instances"], vals))
    lines += ["", "平均车辆数（同段）：" + "；".join(
        "%s %s" % (m, ", ".join("%s=%.2f" % (r["bucket"], r.get(m + "_mean_K", 0)) for r in v22))
        for m in INTERNAL), ""]

v27 = a.get("v2_7") or []
if v27:
    lines += ["### 8.3 V2-7 神经引导净贡献（配对，负值=神经更优）", "",
              "| 目标 | 神经版 | 距离版 | n | 神经更优 | 距离更优 | 平均Δ | 中位Δ | 符号检验 p |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in v27:
        st_ = r.get("sign_test") or {}
        lines.append("| %s | %s | %s | %d | %d | %d | %+.3f | %+.3f | %s |" % (
            r["objective"], r["neural"], r["distance_only"], r["n"], r["neural_better"],
            r["distance_better"], r["mean_delta"], r["median_delta"], st_.get("p")))
    lines.append("")

v28 = a.get("v2_8") or []
if v28:
    lines += ["### 8.4 V2-8 机制消融（开启 − 关闭）", "",
              "| 目标 | 消融 | n | 开启更优 | 关闭更优 | 平均Δ | 中位Δ | 符号检验 p |",
              "|---|---|---|---|---|---|---|---|"]
    for r in v28:
        st_ = r.get("sign_test") or {}
        lines.append("| %s | %s | %d | %d | %d | %+.3f | %+.3f | %s |" % (
            r["objective"], r["ablation"], r["n"], r["on_better"], r["off_better"],
            r["mean_delta"], r["median_delta"], st_.get("p")))
    lines.append("")

v26 = a.get("v2_6") or []
if v26:
    lines += ["### 8.5 V2-6 STD_NORM（编码对齐）相对 ORDER_SPLIT", "",
              "| 目标 | n | NORM 更优 | ORDER_SPLIT 更优 | 平均Δ% / 平均ΔK | 符号检验 p |",
              "|---|---|---|---|---|---|"]
    for r in v26:
        st_ = r.get("sign_test") or {}
        delta = ("%.3f%%" % r["mean_delta_pct"]) if "mean_delta_pct" in r else \
            ("ΔK=%+.3f" % r.get("mean_Kdiff", 0))
        lines.append("| %s | %d | %d | %d | %s | %s |" % (
            r["objective"], r["n"], r["norm_better"], r["split_better"], delta, st_.get("p")))
    lines.append("")

v25 = a.get("v2_5") or []
if v25:
    lines += ["### 8.6 V2-5 VRPTW 类别聚合（平均车辆数与 BKS 车对比）", "",
              "| 类别 | 例数 | BKS K | " + " | ".join("%s K" % m for m in INTERNAL) + " |",
              "|---" * (len(INTERNAL) + 3) + "|"]
    for r in v25:
        vals = " | ".join(("%.2f" % r[m + "_mean_K"]) if m + "_mean_K" in r else "-"
                          for m in INTERNAL)
        lines.append("| %s | %s | %.2f | %s |" % (r["category"], r["instances"],
                                                 r.get(INTERNAL[0] + "_bks_K", 0), vals))
    lines += ["", "K 胜负（6 组合汇总，相对 BKS 路线 K）：" + "；".join(
        "%s 优%d/平%d/劣%d" % (r["category"],
                              sum(r.get(m + "_K_win", 0) for m in INTERNAL),
                              sum(r.get(m + "_K_tie", 0) for m in INTERNAL),
                              sum(r.get(m + "_K_loss", 0) for m in INTERNAL)) for r in v25), ""]

v23 = a.get("v2_3") or []
if v23:
    lines += ["### 8.7 V2-3 CVRP 组合两两配对（前 10 行）", "",
              "| 对比 | n | 平均Δ% | 中位Δ% | 后者更优 | p |", "|---|---|---|---|---|---|"]
    for r in v23[:10]:
        st_ = r.get("sign_test") or {}
        lines.append("| %s → %s | %d | %+.3f | %+.3f | %d | %s |" % (
            r["a"], r["b"], r["n"], r["mean_delta_pct"], r["median_delta_pct"],
            r["b_better"], st_.get("p")))
    lines.append("")

v29 = a.get("v2_9") or []
if v29:
    lines += ["### 8.8 V2-9 计时与预测器审计", "",
              "| 目标 | 方法 | 端点 | 平均在线 s | 最大在线 s | 平均 epochs | 前向尝试 | 前向成功 | 回退 |",
              "|---|---|---|---|---|---|---|---|---|"]
    for r in v29:
        lines.append("| %s | %s | %d | %.2f | %.2f | %.1f | %d | %d | %d |" % (
            r["objective"], r["method"], r["endpoints"], r["mean_online_s"],
            r["max_online_s"], r["mean_epochs"], r["forward_attempts"],
            r["forward_success"], r["fallbacks"]))
    lines.append("")

s = ext["summary"]
lines += ["### 8.9 外部基线（PyVRP VRPTW，168 端点，单列不并入内部排名）", "",
          "- 全部 AUDITED、双精度可行；平均在线 %.2f s、最大 %.2f s；" % (
              s["mean_online_s"], s["max_online_s"]),
          "- 相对 BKS：K 更优 %d / 相同 %d / 更劣 %d；同 K 子集平均距离差 %+.2f。" % (
              s["K_better"], s["K_tie"], s["K_worse"], s["mean_D_delta_sameK"]), ""]

new = "\n".join(lines)
text = rep.read_text(encoding="utf-8")
if "<!-- FORMAL_RESULTS -->" in text:
    text = text.replace("<!-- FORMAL_RESULTS -->", new)
else:
    raise SystemExit("formal placeholder missing")
rep.write_text(text, encoding="utf-8")
print("formal section written, %d chars" % len(new))
