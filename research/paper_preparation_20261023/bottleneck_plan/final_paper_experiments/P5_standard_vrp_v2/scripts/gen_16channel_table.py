# -*- coding: utf-8 -*-
"""Phase 4：16 通道训练语义核查与输入分布统计（生成 neural_diagnostics/16channel_table.md）。

数据来源：
  * 冷链锁版：冻结项目的 partial_batch 源码（通道定义）+ reserved_validation 输入（分布代理）
  * 标准任务：CVRPLIB X（CVRP）、Solomon（VRPTW）
"""
import json
import statistics
from pathlib import Path

ROOT = Path(r"D:\PyCharm_\MASKCO-Main")
RES = ROOT / "research/paper_preparation_20261023/bottleneck_plan/final_paper_experiments"
V1DATA = RES / "P5_standard_vrp/data"
V2 = RES / "P5_standard_vrp_v2"
COLD_INPUTS = ROOT / "research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/reserved_validation/inputs"


def cold_stats():
    xs, ys, ds, ls, rs, ss = [], [], [], [], [], []
    for f in sorted(COLD_INPUTS.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        nodes = d["nodes"]
        x0, y0 = nodes[0][1], nodes[0][2]
        for row in nodes[1:]:
            xs.append(abs(row[1] - x0) / 1440.0)
            ys.append(abs(row[2] - y0) / 1440.0)
            ds.append(row[3] / 20000.0)
            ls.append(row[4] / 1440.0)
            rs.append(row[5] / 1440.0)
            ss.append(row[6] / 1440.0)
    return xs, ys, ds, ls, rs, ss


def parse_vrp_stats(path):
    text = path.read_text(encoding="utf-8")
    import re
    coords = {}
    demands = []
    section = None
    for raw in text.splitlines():
        line = raw.strip()
        if line.endswith("_SECTION"):
            section = line
            continue
        if section == "NODE_COORD_SECTION":
            p = line.split()
            if len(p) == 3:
                coords[int(p[0])] = (float(p[1]), float(p[2]))
        elif section == "DEMAND_SECTION":
            p = line.split()
            if len(p) == 2:
                demands.append(float(p[1]))
    cap = float(re.search(r"CAPACITY\s*:\s*(\d+)", text, re.I).group(1))
    ids = sorted(coords)
    depot = ids[0]
    x0, y0 = coords[depot]
    xs = [abs(coords[i][0] - x0) for i in ids[1:]]
    ys = [abs(coords[i][1] - y0) for i in ids[1:]]
    span = max(max(xs), max(ys))
    return dict(name=path.stem, cap=cap, span=span,
                coord_med=statistics.median(xs + ys) / 1440.0,
                demand_ratio=statistics.median([d for d in demands if d > 0]) / cap,
                demand_20000=statistics.median([d for d in demands if d > 0]) / 20000.0)


def solomon_stats(path):
    text = path.read_text(encoding="utf-8")
    lines = [l for l in text.splitlines() if l.strip()]
    cap = int(lines[3].split()[1])
    coords = {}
    demands = []
    windows = []
    for raw in text.splitlines():
        p = raw.split()
        if len(p) < 7:
            continue
        try:
            i = int(p[0])
        except ValueError:
            continue
        if i > 1000:
            continue
        coords[i] = (float(p[1]), float(p[2]))
        demands.append(float(p[3]))
        windows.append((float(p[4]), float(p[5])))
    x0, y0 = coords[0]
    xs = [abs(coords[i][0] - x0) for i in coords if i]
    ys = [abs(coords[i][1] - y0) for i in coords if i]
    return dict(name=path.stem, cap=cap, span=max(max(xs), max(ys)),
                coord_med=statistics.median(xs + ys) / 1440.0,
                demand_ratio=statistics.median(demands) / cap,
                demand_20000=statistics.median(demands) / 20000.0,
                win_r_med=statistics.median([r for _, r in windows]) / 1440.0)


def main():
    xs, ys, ds, ls, rs, ss = cold_stats()
    def st(v):
        return "%.4f / %.4f / %.4f" % (min(v), statistics.median(v), max(v))
    x_files = sorted((V1DATA / "cvrplib/instances").glob("X-n*.vrp"),
                     key=lambda p: int(p.stem.split("-n")[1].split("-")[0]))
    sol = sorted((V1DATA / "solomon/instances").glob("*.vrp"))
    x_rows = [parse_vrp_stats(f) for f in x_files[:1] + x_files[-1:]]
    s_rows = [solomon_stats(sol[0]), solomon_stats(sol[18])]  # C101, R101
    lines = ["# 16 通道训练语义对照表（Phase 4 核查）", "",
             "> 权威来源：冻结 `profit_core.py::partial_batch`（feature_schema=v0-16channels-partial-only）。",
             "> 分布代理：冷链 reserved_validation 输入（%d 件）；标准：CVRPLIB X、Solomon。" % len(list(COLD_INPUTS.glob('*.json'))),
             "",
             "| 通道 | 训练期语义（冻结源码） | 冷链取值范围（min/med/max） | P5-V1 实现 | P5-V2 STD_FIX | P5-V2 STD_NORM |",
             "|---|---|---|---|---|---|"]
    table = [
        ("0–1", "(x−x0)/1440, (y−y0)/1440（km）", st(xs + ys), "同冻结", "同冻结", "按跨度归一化（见 §2）"),
        ("2", "需求(kg)/20000（=需求/容量，冷链容量 20000）", st(ds), "需求/20000", "需求/20000（登记原编码）", "**需求/容量**"),
        ("3–5", "l/1440, r/1440, service/1440（分钟）", "l:" + st(ls) + " r:" + st(rs) + " s:" + st(ss), "CVRP 中性(0,1,0)；VRPTW 原始值/1440", "同冻结", "同冻结"),
        ("6", "rate×1440 = 0.266112（冻结常数）", "0.266112", "同冻结", "同冻结", "同冻结"),
        ("7", "最低品质 0.8（冻结常数）", "0.8", "同冻结", "同冻结", "同冻结"),
        ("8", "log1p(price×20000/1440) = 3.1842077（冻结常数）", "3.1842077", "同冻结", "同冻结", "同冻结"),
        ("9", "cold = 0.0972（冻结常数）", "0.0972", "同冻结", "同冻结", "同冻结"),
        ("10", "**float(i==0 or i not in removed)**（仓库恒 1）", "0/1", "仓库 0、可见 1（偏差）", "冻结语义（仓库=1）", "冻结语义"),
        ("11–15", "maskco_base 变体：**全零**", "0", "动态值+ch12=1（偏差）", "**全零**（冻结语义）", "全零"),
        ("timestep", "adjacency[1:].sum()/(n×(1 or 2))", "随 n", "/200 固定（偏差）", "/(n×2)", "/(n×2)"),
    ]
    for row in table:
        lines.append("| " + " | ".join(row) + " |")
    lines += ["", "## 标准任务分布（实测）", "",
              "| 实例 | 容量 | 空间跨度 | 坐标中位/1440 | 需求中位/容量 | 需求中位/20000 | 备注 |",
              "|---|---:|---:|---:|---:|---:|---|"]
    for r in x_rows:
        lines.append("| %s | %d | %.0f | %.4f | %.4f | %.6f | CVRP |" % (r["name"], r["cap"], r["span"], r["coord_med"], r["demand_ratio"], r["demand_20000"]))
    for r in s_rows:
        lines.append("| %s | %d | %.0f | %.4f | %.4f | %.6f | VRPTW r_med/1440=%.3f |"
                     % (r["name"], r["cap"], r["span"], r["coord_med"], r["demand_ratio"], r["demand_20000"], r["win_r_med"]))
    lines += ["", "## §2 STD_NORM 编码规则（本阶段登记，先于正式实验锁定）", "",
              "- ch2：`需求/容量`（与训练语义一致：训练容量 20000 kg 即需求/20000 的比例形式）；",
              "- ch0–1：`(x−x0)/coord_scale`，`coord_scale = 实例坐标通道中位(原始单位)/0.0475`（0.0475 = 训练中位）；",
              "  即把实例坐标通道的分布中位对齐训练中位（X 集 → 约 5600；Solomon → 约 316；冷链实例保持 1440 不变）；",
              "  （同时消除 100 单位方格实例坐标通道整体缩小 10 倍的偏差）；",
              "- ch3–5、6–9、10、11–15、timestep：与 STD_FIX 相同（冻结语义）。",
              "- 该规则在开发数据上验证前锁定，不在正式测试面板上调整。"]
    out = V2 / "neural_diagnostics/16channel_table.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:12]))
    print("saved", out)


if __name__ == "__main__":
    main()
