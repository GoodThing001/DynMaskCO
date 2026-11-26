# -*- coding: utf-8 -*-
"""Phase 5：24 个开发实例生成（CVRP 12 + VRPTW 12，种子 71001/71002）。

生成规则先于比较实验登记（本脚本即登记物）：
CVRP（12 例）：
  * 坐标 U[0,1000]²；需求 U{1..2000} 整数；车队目标 k = max(2, round(n/4))；
    容量 = ceil(总需求 / k)（保证 k 车可行且贴近标准 X 的紧度）；
  * 规模：100 客户 4 例、300 客户 4 例、800 客户 4 例（seed 71001 两例、71002 两例/规模）。
VRPTW（12 例）：
  * 100 客户；布局 3 类各 4 例：聚簇（3–5 簇）、均匀、混合（聚簇+均匀各半）；
  * 容量 200、需求 U{1..10}、服务 10；
  * **可行性构造**：先按角度扫描分成 k=10 条路线（容量约束），组内最近邻排序，
    以距离=时间（速度 1）模拟，客户窗 [到达−w/2, 到达+w/2]（紧凑 w=20、宽松 w=60 各半），
    仓库窗 [0, 最长路线结束+100]；构造解必须经 route_schedule 复核可行，否则加大 w 重试。
输出：development/instances/*.vrp（CVRPLIB 格式）+ development/manifest.json（含构造解与 SHA256）。
"""
import hashlib
import json
import math
import random
from pathlib import Path

V2 = Path(r"D:\PyCharm_\MASKCO-Main\research\paper_preparation_20261023\bottleneck_plan\final_paper_experiments\P5_standard_vrp_v2")
OUT = V2 / "development/instances"
MANIFEST = V2 / "development/manifest.json"
SEEDS = (71001, 71002)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write_cvrp(name, coords, demands, capacity):
    lines = ["NAME : " + name, "COMMENT : P5-V2 development instance (registered generator)",
             "TYPE : CVRP", "DIMENSION : %d" % len(coords), "EDGE_WEIGHT_TYPE : EUC_2D",
             "CAPACITY : %d" % capacity, "NODE_COORD_SECTION"]
    for i, (x, y) in enumerate(coords, 1):
        lines.append("%d %.4f %.4f" % (i, x, y))
    lines.append("DEMAND_SECTION")
    for i, d in enumerate(demands, 1):
        lines.append("%d %d" % (i, d))
    lines.append("DEPOT_SECTION")
    lines.append("1")
    lines.append("-1")
    lines.append("EOF")
    return "\n".join(lines) + "\n"


def write_solomon(name, coords, demands, windows, service, capacity, k):
    lines = [name, "", "VEHICLE", "NUMBER CAPACITY", "%d %d" % (k, capacity), "",
             "CUSTOMER",
             "CUST NO.  XCOORD.   YCOORD.    DEMAND   READY TIME  DUE DATE   SERVICE TIME"]
    for i in range(len(coords)):
        x, y = coords[i]
        lines.append("%4d %8.2f %8.2f %8d %10.2f %10.2f %10.2f"
                     % (i, x, y, demands[i], windows[i][0], windows[i][1], service[i]))
    return "\n".join(lines) + "\n"


def gen_cvrp(n, seed, tag):
    rng = random.Random(seed * 7 + tag)
    coords = [(rng.uniform(0, 1000), rng.uniform(0, 1000)) for _ in range(n + 1)]
    demands = [0] + [rng.randint(1, 2000) for _ in range(n)]
    k = max(2, round(n / 4))
    cap = int(math.ceil(sum(demands) / k))
    cap = max(cap, max(demands))
    return "XDEV-n%d-s%d-r%d" % (n, seed, tag % 2), coords, demands, cap



def schedule_feasible(xs, windows, service, routes):
    """校验给定路线在窗口下可行（构造解自检）。"""
    n = len(xs) - 1
    for r in routes:
        t = windows[0][0]
        prev = 0
        for i in r:
            d = math.hypot(xs[prev][0] - xs[i][0], xs[prev][1] - xs[i][1])
            t += d
            if t < windows[i][0]:
                t = windows[i][0]
            if t > windows[i][1] + 1e-9:
                return False
            t += service[i]
            prev = i
        d = math.hypot(xs[prev][0] - xs[0][0], xs[prev][1] - xs[0][1])
        t += d
        if t > windows[0][1] + 1e-9:
            return False
    return True


def _schedule_ok(coords, demands, windows, service, capacity):
    """构造解可行性检查（独立于标准评价器的最简实现）。"""
    n = len(coords) - 1
    routes = []
    order = sorted(range(1, n + 1), key=lambda i: math.atan2(coords[i][1] - coords[0][1],
                                                             coords[i][0] - coords[0][0]))
    k = 10
    group, load = [], 0.0
    groups = []
    for c in order:
        if group and load + demands[c] > capacity:
            groups.append(group)
            group, load = [], 0.0
        group.append(c)
        load += demands[c]
    if group:
        groups.append(group)
    for g in groups:
        r, cur = [], 0
        left = set(g)
        while left:
            nxt = min(left, key=lambda i: (coords[cur][0] - coords[i][0]) ** 2
                      + (coords[cur][1] - coords[i][1]) ** 2)
            r.append(nxt)
            left.remove(nxt)
            cur = nxt
        routes.append(r)
    # 时刻表
    t = windows[0][0]
    for r in routes:
        t = windows[0][0]
        prev = 0
        for i in r:
            d = math.hypot(coords[prev][0] - coords[i][0], coords[prev][1] - coords[i][1])
            t += d
            if t < windows[i][0]:
                t = windows[i][0]
            if t > windows[i][1] + 1e-9:
                return None
            t += service[i]
            prev = i
        d = math.hypot(coords[prev][0] - coords[0][0], coords[prev][1] - coords[0][1])
        t += d
        if t > windows[0][1] + 1e-9:
            return None
    return routes


def gen_vrptw(layout, seed, tag):
    rng = random.Random(seed * 7 + tag)
    n = 100
    centers = []
    if layout == "clustered":
        centers = [(rng.uniform(100, 400), rng.uniform(100, 400)) for _ in range(4)]
    elif layout == "mixed":
        centers = [(rng.uniform(100, 400), rng.uniform(100, 400)) for _ in range(2)]
    xs = [(rng.uniform(200, 800), rng.uniform(200, 800))]
    for i in range(n):
        if layout == "uniform":
            xs.append((rng.uniform(0, 1000), rng.uniform(0, 1000)))
        elif layout == "clustered":
            cx, cy = centers[rng.randrange(len(centers))]
            xs.append((min(1000, max(0, rng.gauss(cx, 40))), min(1000, max(0, rng.gauss(cy, 40)))))
        else:
            if i % 2 == 0:
                cx, cy = centers[rng.randrange(len(centers))]
                xs.append((min(1000, max(0, rng.gauss(cx, 40))), min(1000, max(0, rng.gauss(cy, 40)))))
            else:
                xs.append((rng.uniform(0, 1000), rng.uniform(0, 1000)))
    demands = [0] + [rng.randint(1, 10) for _ in range(n)]
    service = [0] + [10.0] * n
    capacity = int(math.ceil(550 / 10))
    width = 20.0 if (tag % 2 == 0) else 60.0
    for attempt in range(6):
        w = width + 20 * attempt
        # 预估到达时间：用构造路由的扫描分组（与 _schedule_ok 相同）反推
        tmp_windows = [(0.0, 1e9)] * (n + 1)
        base = _schedule_ok(xs, demands, tmp_windows, service, capacity)
        if base is None:
            tmp_windows = [(0.0, 1e9)] * (n + 1)
            continue
        # 用构造路由的到达时间生成窗口
        windows = [(0.0, 1e9)] * (n + 1)
        longest = 0.0
        for r in base:
            t = 0.0
            prev = 0
            for i in r:
                d = math.hypot(xs[prev][0] - xs[i][0], xs[prev][1] - xs[i][1])
                t += d
                a = t
                windows[i] = (max(0.0, a - w / 2), a + w / 2)
                t += service[i]
                prev = i
            d = math.hypot(xs[prev][0] - xs[0][0], xs[prev][1] - xs[0][1])
            t += d
            longest = max(longest, t)
        windows[0] = (0.0, longest + 100.0)
        if schedule_feasible(xs, windows, service, base):
            return "RDEV-%s-s%d-r%d" % (layout[0].upper(), seed, tag % 2), xs, demands, windows, service, capacity, len(base), base
    return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = dict(generator="P5-V2 development v1", seeds=list(SEEDS), instances=[])
    specs = []
    for n in (100, 300, 800):
        for j, seed in enumerate(SEEDS):
            for m in range(2):
                specs.append(("cvrp", n, seed, m))
    for layout in ("clustered", "uniform", "mixed"):
        for j, seed in enumerate(SEEDS):
            for m in range(2):
                specs.append(("vrptw", layout, seed, m))
    for kind, a, seed, idx in specs:
        if kind == "cvrp":
            name, coords, demands, cap = gen_cvrp(a, seed, idx)
            p = OUT / (name + ".vrp")
            p.write_text(write_cvrp(name, coords, demands, cap), encoding="utf-8")
            manifest["instances"].append(dict(kind="cvrp", name=name, n=a, seed=seed,
                                              capacity=cap, file=p.name, sha256=sha(p)))
            print("CVRP", name, "cap", cap, "file", p.name)
        else:
            res = gen_vrptw(a, seed, idx)
            if res is None:
                print("VRPTW FAILED", a, seed)
                continue
            name, xs, demands, windows, service, cap, k, base = res
            p = OUT / (name + ".vrp")
            p.write_text(write_solomon(name, xs, demands, windows, service, cap, k), encoding="utf-8")
            manifest["instances"].append(dict(kind="vrptw", name=name, layout=a, seed=seed,
                                              capacity=cap, k=k, file=p.name, sha256=sha(p),
                                              constructed_routes=base))
            print("VRPTW", name, "k", k, "instances", len(base), "file", p.name)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print("instances:", len(manifest["instances"]), "manifest:", MANIFEST)


if __name__ == "__main__":
    main()

