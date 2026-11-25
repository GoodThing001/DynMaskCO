"""Count case identities and completed method runs across the external panels."""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXT = ROOT / "maskco_perishable_external_20261119"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def summarize():
    cross = ROOT / "maskco_cross_dataset_20261119"
    heat = ROOT / "maskco_heatmap_calibration_20261119"
    recent = ROOT / "recent5_20261120"
    sync = ROOT / "supplement_sync_20261120"
    sets = [
        ("crossmain01", cross / "matrix.json", cross / "analysis.json", "complete", 7),
        ("confirm01", heat / "confirm01.json", heat / "results/confirm01/analysis.json", "complete", 7),
        ("replicate01", EXT / "replicate01.json", EXT / "results/replicate01/analysis.json", "complete", 7),
        ("reserve02", EXT / "reserved_validation/reserve02.json", EXT / "reserved_validation/results/reserve02/analysis.json", "complete", 7),
        ("xtwsupp01", EXT / "xtw_low_overlap/xtwsupp01.json", EXT / "xtw_low_overlap/results/xtwsupp01/status.json", "complete", 7),
        ("recent5/benchmark01", recent / "matrix.json", recent / "results/benchmark01/recent_summary.json", "endpoints", 5),
        ("sync01", sync / "phase.json", sync / "results/sync01/independent_audit.json", "endpoints", 2),
        ("benchmark01", HERE / "matrix.json", HERE / "results/benchmark01/independent_audit.json", "endpoints", 12),
        ("prospective01", HERE / "new_panel/matrix.json", HERE / "results/prospective01/independent_audit.json", "endpoints", 12),
        ("trajectory01", HERE / "trajectory_matrix.json", HERE / "results/trajectory01/analysis.json", "endpoints", 2),
    ]
    retail_endpoint_file=HERE/"results/retail01/independent_audit.json"
    if retail_endpoint_file.exists():
        sets.append(("retail01", HERE/"retail_source/matrix.json", retail_endpoint_file, "endpoints", 12))
    origin_status=HERE/"results/origin01/status.json"
    if origin_status.exists():
        sets.append(("origin01 原版消融", HERE/"origin_matrix.json", origin_status, "cells", 0))
    cases_seen = set()
    evidence = []
    rows = []
    for label, matrix_path, result_path, count_field, external in sets:
        if not result_path.exists():
            raise FileNotFoundError(result_path)
        blocks = read(matrix_path)["blocks"]
        cases = {b["case"] for b in blocks}
        endpoints = read(result_path)[count_field]
        expected_arms = len(blocks[0].get("methods", blocks[0].get("arms", [])))
        if label == "xtwsupp01":
            assert endpoints == len(blocks)
            endpoints *= expected_arms
        assert endpoints == len(blocks) * expected_arms, (label, endpoints)
        new = cases - cases_seen
        rows.append(f"| {label} | {len(cases)} | {len(new)} | {external} | {expected_arms} | {endpoints:,} |")
        evidence += [(label, c, c in new) for c in sorted(cases)]
        cases_seen.update(cases)
    with (HERE/"cases_by_campaign.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(("campaign", "case", "first_in_reviewed_series"))
        w.writerows(evidence)
    lines = ["# 实例和外部对比方法的覆盖", "",
             f"下列完整的跨来源对照及重复批次合计覆盖 {len(cases_seen)} 个不同案例标识。早期阶段包含不同方法版本。每个标识对应一份输入，同母几何的不同时间窗、订单和客户块另外分组。", "",
             "| 批次 | 案例数 | 首次出现在这组批次的案例 | 同批外部法 | 同批总方法 | 求解终点 |",
             "|---|---:|---:|---:|---:|---:|", *rows, "",
             "锁版 ORDER_SPLIT 的十二法统一比较覆盖已测输入 30 例以及 10 个新配送日、8 个低重叠几何和 6 个同母几何压力实例。"
             +(f"另在独立零售来源比较 {read(retail_endpoint_file)['cases']} 个百客案例。" if retail_endpoint_file.exists() else "")
             +"十种子与四法复核使用其中既有案例。三个开发例的两种子首次请求对照另外保存在[部署复核](../maskco_followup_dev_20261120/startup/RESULTS.md)。",
             "", "## 尚未评分的实例", "",
             "原 FLEET_PRICE 的 32 个验证例因开发门槛未通过而保持封存。公开荷比货运包有 48 个大订单文件，其中 10 个属于上述封存名单，本轮另取 10 个。其余 28 个未纳入这两份名单，来源相同，具体独立性需逐例核查。",
             "", "GH 公开库有 300 个时间窗实例，已登记与封存的 60 个变体覆盖全部 20 个母客户集合。按既定去重条件，剩余变体没有独立新母几何。XTW 从 83 个已登记或封存原件出发，新增 8 个与旧面板及彼此重合客户坐标均不超过 10 个的百客几何。", "",
             "[逐案例批次表](cases_by_campaign.csv) · [新增实例与重叠审查](new_panel/准入.md) · [原 32 例登记](../maskco_perishable_external_20261119/下一次验证样本.md)"]
    (HERE/"实例与方法覆盖.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__":
    summarize()
