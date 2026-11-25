"""Describe rank coverage across geometries and the reduction in controllable economic losses."""
import csv
from collections import defaultdict
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
sys.path.insert(0, str(STUDY))
from unified_report import EXTERNAL, OUR, csv_rows, ranks, read, write
from statistics import fmean


def geometric_profiles(rows):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        loss = sum(float(row[k]) for k in ("quality_loss_U", "distance_km", "refrigeration_U"))
        grouped[row["geometry"]][row["method"]].append((float(row["U"]), loss))
    profiles = []
    for geometry, methods in sorted(grouped.items()):
        profiles.append((geometry, {m: fmean(u for u, loss in pairs) for m, pairs in methods.items()},
                         {m: fmean(loss for u, loss in pairs) for m, pairs in methods.items()}))
    return profiles


def main():
    target = STUDY / "results/prospective01"
    summary = read(target / "analysis.json")
    meta = read(HERE / "metadata.json")
    with (target / "endpoints.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1152 or read(target / "independent_audit.json")["status"] != "PASS":
        raise ValueError("Incomplete independent prospective audit")
    panels = {k: set(v) for k, v in {**meta["panels"], **meta["additional_panels"]}.items()}
    coverage = {}; details = []
    if set(panels) != set(summary["panels"]):
        raise ValueError("Panel membership differs from the complete comparison")
    for name, registered in panels.items():
        strongest = summary["panels"][name]["strongest_external"]
        profiles = geometric_profiles(row for row in rows if row["case"] in registered)
        net_gains = []; base_losses = []; scores = []; all_scores = []
        for geometry, net, loss in profiles:
            rank = ranks({m: net[m] for m in [OUR, *EXTERNAL]})[OUR]
            rank_all = ranks(net)[OUR]
            gain = net[OUR] - net[strongest]
            if abs((loss[strongest] - loss[OUR]) - gain) > 1e-6:
                raise ValueError(f"Economic loss does not reconcile for {name} / {geometry}")
            net_gains.append(gain); base_losses.append(loss[strongest]); scores.append(rank); all_scores.append(rank_all)
            labels = sorted(c for c in registered if meta["cases"][c]["geometry"] == geometry)
            details.append(dict(panel=name, cases="|".join(labels), geometry=geometry, rank_13=rank, rank_16=rank_all,
                                reference=strongest, candidate_minus_reference_U=gain,
                                controllable_loss_reduction_percent=100*gain/loss[strongest]))
        result = dict(geometries=len(scores), top3_13=sum(x<=3 for x in scores), top3_16=sum(x<=3 for x in all_scores),
                      first_13=sum(x==1 for x in scores), strongest_external=strongest,
                      reference_controllable_loss_U=fmean(base_losses),
                      mean_delta_U=fmean(net_gains), controllable_loss_reduction_percent=100*fmean(net_gains)/fmean(base_losses))
        if abs(result["mean_delta_U"] - summary["panels"][name]["contrasts"][strongest]["mean"]) > 1e-6:
            raise ValueError(f"Geometry weighting differs for {name}")
        coverage[name] = result
    csv_rows(target / "geometry_ranks.csv", details)
    write(target / "rank_coverage.json", coverage)
    lines = ["# 新面板上各几何的前三覆盖率与经济收益", "",
             "每个几何先平均三个种子。GH 中客户坐标相同的两个实例先在同一几何内平均。外部名次统计包含锁定 MASKCO 和十二个外部方法，全体名次另含三个内部对照，并列取并列组中最末名次。", "",
             "| 来源 | 评价组数 | 外部比较中第一 | 外部比较中前三 | 全部十六法中前三 | 来源内最强外部方法 | 品质、里程及制冷损失降幅 |",
             "|---|---:|---:|---:|---:|---|---:|"]
    for name, row in coverage.items():
        lines.append(f"| {name} | {row['geometries']} | {row['first_13']} | {row['top3_13']} | "
                     f"{row['top3_16']} | {row['strongest_external']} | {row['controllable_loss_reduction_percent']:+.2f}% |")
    lines += ["", "损失降幅将 MASKCO 的几何等权净收益差除以该来源最强外部方法的几何等权总损失。总损失包括品质货损、里程和制冷费用。", "",
              "## 样本范围", "",
              "比利时和荷兰各含同一运营商的周一至周五五个配送日。订单日可能共享运营模式。按订单日计算的区间是对这些日期的描述性区间，跨运营商推广需要独立样本。XTW 使用八个低重叠客户几何。GH 的六个第二客户块来自五个已使用过的母几何，仅作跨客户块压力测试。",
              "", "[逐几何排名及收益](geometry_ranks.csv) · [机器可读覆盖率](rank_coverage.json)"]
    (target / "DISTRIBUTION.md").write_text("\n".join(lines) + "\n", encoding="utf8")
    results = target / "RESULTS.md"
    report = results.read_text(encoding="utf8")
    if "DISTRIBUTION.md" not in report:
        with results.open("a",encoding="utf8") as out:
            out.write("\nVRPPDL 含同一运营商在两国的十个订单日，其重采样区间为描述性区间。GH 客户块用于已知母几何压力测试。参见[逐几何排名、费用降幅及样本范围](DISTRIBUTION.md)。\n")
    print(coverage)


if __name__ == "__main__":
    main()
