# -*- coding: utf-8 -*-
"""表7（P3 双因素消融）与表8（P2 八城十种子）MD/CSV，以及导师版 Excel 汇总。"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FP = HERE.parent
TAB = FP / "tables"
P2 = FP / "P2_retail_analysis"
P3 = FP / "P3_factorial_ablation"
TAB.mkdir(parents=True, exist_ok=True)


def table7():
    rows = list(csv.DictReader((P3 / "P3_双因素消融汇总表.csv").open(encoding="utf-8-sig")))
    eq = list(csv.DictReader((P3 / "P3_三来源等权汇总.csv").open(encoding="utf-8-sig")))
    out = []
    for r in rows:
        out.append(dict(source=r["source"], quantity=r["quantity"], mean=float(r["mean"]),
                        ci95_lo=float(r["ci95_lo"]), ci95_hi=float(r["ci95_hi"]),
                        wins=int(r["wins"]), ties=int(r["ties"]), losses=int(r["losses"]),
                        units_pos=int(r["units_pos"]), units_total=int(r["units_total"]),
                        by_seed=r["by_seed"]))
    with (TAB / "表7_双因素消融.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    md = ["# 表7 学习引导与经济分车双因素消融", "",
          "A=HYBRID−HYBRID_D；B=ORDER_SPLIT−ORDER_SPLIT_D；C=ORDER_SPLIT−HYBRID；D=ORDER_SPLIT_D−HYBRID_D；"
          "交互=B−A。95% 区间为几何整簇重采样 20,000 次（种子 20261127）。", "",
          "| 来源 | 量 | ΔU | 95% 区间 | 胜/平/负几何 | 改善单位 |",
          "|---|---|---:|---|---|---:|"]
    for r in out:
        md.append(f"| {r['source']} | {r['quantity']} | {r['mean']:+,.4f} | "
                  f"[{r['ci95_lo']:+,.4f}, {r['ci95_hi']:+,.4f}] | "
                  f"{r['wins']}/{r['ties']}/{r['losses']} | {r['units_pos']}/{r['units_total']} |")
    md += ["", "| 量 | 三来源等权 ΔU |", "|---|---:|"]
    for r in eq:
        md.append(f"| {r['quantity']} | {float(r['equal_source_mean']):+,.4f} |")
    (TAB / "表7_双因素消融.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("表7 written")


def table8():
    rows = list(csv.DictReader((P2 / "P2_八城十种子逐端点结果.csv").open(encoding="utf-8-sig")))
    dec = list(csv.DictReader((P2 / "P2_逐城市经济收益分解.csv").open(encoding="utf-8-sig")))
    import numpy as np
    city = {}
    for r in rows:
        city.setdefault(r["city"], []).append(r)
    dec_agg = {}
    for r in dec:
        if r["comparator"] == "ALNS":
            dec_agg.setdefault(r["city"], []).append(r)
    out = []
    for c in sorted(city):
        sub = city[c]
        for m in ("ORDER_SPLIT", "ORDER_SPLIT_D", "ALNS"):
            ms = [r for r in sub if r["method"] == m]
            out.append(dict(city=c, method=m, n_seeds=len(ms),
                            mean_U=round(float(np.mean([float(r["U"]) for r in ms])), 4),
                            min_U=round(float(np.min([float(r["U"]) for r in ms])), 4),
                            max_U=round(float(np.max([float(r["U"]) for r in ms])), 4),
                            sd_U=round(float(np.std([float(r["U"]) for r in ms], ddof=1)), 4),
                            mean_vehicles=round(float(np.mean([float(r["vehicles"]) for r in ms])), 2),
                            mean_distance_km=round(float(np.mean([float(r["distance_km"]) for r in ms])), 2),
                            mean_quality_loss=round(float(np.mean([float(r["quality_loss_U"]) for r in ms])), 2),
                            mean_refrigeration=round(float(np.mean([float(r["refrigeration_U"]) for r in ms])), 2)))
    with (TAB / "表8_八城十种子.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    md = ["# 表8 八城零售门店十种子实验结果（RWVRP八城零售门店冷链投影数据集）", "",
          "城市内十种子（65001–65010）均值；完整方法对 ALNS 的经济分解见 P2_八城稳定性分析.md。", "",
          "| 城市 | 方法 | 种子数 | 平均 U | 最小 U | 最大 U | 标准差 | 车辆 | 距离 km | 品质损失 | 制冷 |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in out:
        md.append(f"| {r['city']} | {r['method']} | {r['n_seeds']} | {r['mean_U']:,.2f} | "
                  f"{r['min_U']:,.2f} | {r['max_U']:,.2f} | {r['sd_U']:,.2f} | {r['mean_vehicles']} | "
                  f"{r['mean_distance_km']:,.2f} | {r['mean_quality_loss']:,.2f} | {r['mean_refrigeration']:,.2f} |")
    (TAB / "表8_八城十种子.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("表8 written")


def excel():
    from openpyxl import Workbook
    wb = Workbook()
    default = wb.active
    default.title = "说明"
    default["A1"] = "MASKCO 静态冷链多车配送优化——论文实验汇总（表1–表10）"
    default["A2"] = "生成时间：2026-11-26；数据来源：统一16法 2,976 终点 + P1 372 + P2 240 + P3 156；全部经独立复算。"
    default["A3"] = "各工作表与表1–表10 对应；Markdown/LaTeX 版本见 tables/ 目录。"
    sources = {
        "表1_数据集统计": TAB / "表1_数据集统计.csv",
        "表2_方法发表信息": TAB / "表2_对比方法及发表信息.csv",
        "表3_四数据集18法": TAB / "表3_四数据集统一18法比较表.csv",
        "表4_既有三来源复测": TAB / "表4_既有三来源复测表.md",
        "表5_三前瞻来源": TAB / "表5_三前瞻来源表.md",
        "表6b_对原版增量": TAB / "表6b_相对MASK与SLOT配对增量表.csv",
        "表7_双因素消融": TAB / "表7_双因素消融.csv",
        "表8_八城十种子": TAB / "表8_八城十种子.csv",
        "表9_预算配对收益": TAB / "表9_不同计算预算下的配对收益.csv",
        "表10_经济分解": TAB / "表10_经济分解与损失降低率.md",
    }
    for name, path in sources.items():
        if not path.is_file():
            continue
        if path.suffix == ".csv":
            ws = wb.create_sheet(name)
            with path.open(encoding="utf-8-sig", newline="") as f:
                for row in csv.reader(f):
                    ws.append(row)
        else:
            ws = wb.create_sheet(name)
            for line in path.read_text(encoding="utf-8").splitlines():
                ws.append([line])
    wb.save(TAB / "论文实验汇总_导师版.xlsx")
    print("excel written")


if __name__ == "__main__":
    table7()
    table8()
    excel()
