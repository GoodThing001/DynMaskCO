# -*- coding: utf-8 -*-
"""P5-V2 独立复算审计：不导入 standard_vrp 任何模块，自带实例解析与评价。

用法：python p5v2_audit.py <phase.json> <out_dir> [audit.json]

逐端点独立重算：客户覆盖、容量、时窗（含服务时间与仓库回程）、车辆数、
距离目标；与端点记录值比对，输出差异清单。
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

TOL = 1e-6


# ------------------------------------------------------------------ 独立解析
def _blocks(text):
    out, cur = {}, None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if re.fullmatch(r"[A-Z_]+_SECTION", line) or line == "EOF":
            cur = line
            out.setdefault(cur, [])
            continue
        if cur:
            out[cur].append(line)
    return out


def parse_vrp(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    dim = int(re.search(r"DIMENSION\s*:\s*(\d+)", text, re.I).group(1))
    cap = float(re.search(r"CAPACITY\s*:\s*(\d+(?:\.\d+)?)", text, re.I).group(1))
    sec = _blocks(text)
    coords = {}
    for line in sec.get("NODE_COORD_SECTION", []):
        p = line.split()
        if len(p) >= 3:
            coords[int(p[0])] = (float(p[1]), float(p[2]))
    dem = {}
    for line in sec.get("DEMAND_SECTION", []):
        p = line.split()
        if len(p) >= 2:
            dem[int(p[0])] = float(p[1])
    depots = [int(x) for x in " ".join(sec.get("DEPOT_SECTION", [])).split()
              if re.fullmatch(r"-?\d+", x)]
    depot = depots[0] if depots else 1
    order = [depot] + [i for i in sorted(coords) if i != depot]
    if len(order) != dim:
        raise ValueError("dimension mismatch: %s" % path)
    # 内部编号：0=仓库，1..n 客户
    return dict(kind="cvrp", name=Path(path).stem, n=dim - 1, capacity=cap,
                xy={k: coords[order[k]] for k in range(dim)},
                demand={k: dem.get(order[k], 0.0) for k in range(dim)},
                windows=None, service=None, depot_window=None)


_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def parse_solomon(path):
    lines = [l for l in Path(path).read_text(encoding="utf-8", errors="replace").splitlines()]
    cap = None
    rows = []
    pend = False
    for line in lines:
        s = line.strip()
        if not s:
            continue
        low = s.lower()
        if low.startswith("number") and "capacity" in low:
            pend = True          # 容量数值在下一行
            continue
        if pend and cap is None:
            nums = _NUM.findall(s)
            if len(nums) >= 2:
                cap = float(nums[-1])
                pend = False
                continue
        nums = _NUM.findall(s)
        if len(nums) == 7 and cap is not None:
            rows.append([float(x) for x in nums])
    if cap is None or len(rows) < 2:
        raise ValueError("solomon parse failed: %s" % path)
    # 第一行 = 仓库（id 0）
    xy = {i: (r[1], r[2]) for i, r in enumerate(rows)}
    return dict(kind="vrptw", name=Path(path).stem, n=len(rows) - 1, capacity=cap,
                xy=xy,
                demand={i: r[3] for i, r in enumerate(rows)},
                windows={i: (r[4], r[5]) for i, r in enumerate(rows)},
                service={i: r[6] for i, r in enumerate(rows)},
                depot_window=(rows[0][4], rows[0][5]), max_vehicles=25)


# ------------------------------------------------------------------ 独立评价
def dist(inst, a, b):
    """距离口径：CVRP 逐边整数舍入（EUC_2D，与官方 BKS Cost 一致）；VRPTW 双精度。"""
    xa, ya = inst["xy"][a]
    xb, yb = inst["xy"][b]
    d = math.hypot(xa - xb, ya - yb)
    if inst.get("kind") == "cvrp":
        return float(int(d + 0.5))
    return d


def audit_routes(inst, routes):
    """返回 (ok, 距离, 车辆数, 问题清单)。"""
    n = inst["n"]
    problems = []
    seen = []
    total = 0.0
    for ridx, route in enumerate(routes):
        if not route:
            problems.append("route %d empty" % ridx)
            continue
        load = 0.0
        prev = 0
        t = None
        if inst["kind"] == "vrptw":
            t = inst["depot_window"][0]
        seg = 0.0
        for c in route:
            if not (1 <= c <= n):
                problems.append("route %d bad customer %r" % (ridx, c))
                continue
            seen.append(c)
            seg += dist(inst, prev, c)
            load += inst["demand"][c]
            if inst["kind"] == "vrptw":
                arrive = t + dist(inst, prev, c)
                lo, hi = inst["windows"][c]
                start = max(arrive, lo)
                if start > hi + 1e-9:
                    problems.append("route %d cust %d window violated (%.4f>%.4f)"
                                    % (ridx, c, start, hi))
                t = start + inst["service"][c]
            prev = c
        seg += dist(inst, prev, 0)
        total += seg
        if load > inst["capacity"] + 1e-9:
            problems.append("route %d capacity %.4f>%.4f" % (ridx, load, inst["capacity"]))
        if inst["kind"] == "vrptw":
            back = t + dist(inst, prev, 0)
            if back > inst["depot_window"][1] + 1e-6:
                problems.append("route %d depot return %.4f>%.4f"
                                % (ridx, back, inst["depot_window"][1]))
    if sorted(seen) != list(range(1, n + 1)):
        miss = sorted(set(range(1, n + 1)) - set(seen))
        dup = sorted({c for c in seen if seen.count(c) > 1})
        problems.append("coverage missing=%s duplicated=%s" % (miss[:8], dup[:8]))
    k = len([r for r in routes if r])
    if inst["kind"] == "vrptw" and inst.get("max_vehicles") and k > inst["max_vehicles"]:
        problems.append("fleet %d > %d" % (k, inst["max_vehicles"]))
    return (not problems), total, k, problems


def close(a, b):
    return abs(a - b) <= max(TOL, TOL * max(abs(a), abs(b)))


# ------------------------------------------------------------------ 主流程
def main(phase_path, out_dir, dest=None):
    phase = json.loads(Path(phase_path).read_text(encoding="utf-8"))
    blocks = phase["blocks"]
    out = Path(out_dir)
    rows = []
    for name in sorted(out.rglob("*.json")):
        if any(k in name.name for k in ("phase", "status", "audit")):
            continue
        rows.append(json.loads(name.read_text(encoding="utf-8")))
    cache = {}
    results = []
    fails = 0
    for row in rows:
        ver = row.get("version") or row.get("method")
        row["_ver"] = ver
        idx = None
        for i, b in enumerate(blocks):
            if (b["case"] == row["case"] and b["seed"] == row["seed"]
                    and (ver in b["methods"] or not b.get("methods"))):
                idx = i
                break
        if idx is None:
            results.append(dict(case=row["case"], version=ver,
                                blocked="no matching block"))
            fails += 1
            continue
        path = blocks[idx].get("path")
        local = Path(__file__).resolve().parent.parent
        if not path or not Path(str(path)).exists():
            # 远端路径不可达：按 case 名在本地数据/开发实例中解析
            cands = [local / "development/instances" / (row["case"] + ".vrp"),
                     local / "data/cvrplib/instances" / (row["case"] + ".vrp"),
                     local / "data/solomon/instances" / (row["case"] + ".vrp")]
            path = None
            for cand in cands:
                if cand.exists():
                    path = str(cand)
                    break
            if path is None:
                raise FileNotFoundError("instance not found for " + str(row["case"]))
        if path not in cache:
            cache[path] = (parse_vrp(path) if blocks[idx]["objective"] == "cvrp"
                           else parse_solomon(path))
        inst = cache[path]
        if row.get("status") not in ("AUDITED", "TRUNCATED_OK"):
            results.append(dict(case=row["case"], seed=row["seed"], version=row["version"],
                                status=row["status"], feasible=False,
                                problems=[row.get("error", "non-audited endpoint")],
                                pass_=True, pass_note="登记的非审计端点（不参与可行性复算）"))
            continue
        ok, val, k, problems = audit_routes(inst, row["routes"])
        ok0, val0, k0, problems0 = audit_routes(inst, row["initial_routes"])
        rec = dict(case=row["case"], seed=row["seed"], version=ver,
                   kind=inst["kind"], n=inst["n"], status=row["status"],
                   recorded_distance=row["distance"], audited_distance=val,
                   recorded_vehicles=row["vehicles"], audited_vehicles=k,
                   online_s=row["online_s"],
                   distance_match=close(row["distance"], val),
                   vehicles_match=int(row["vehicles"]) == int(k),
                   initial_ok=ok0, initial_distance_audit=val0,
                   initial_distance_match=close(row["initial_U"]["distance"], val0),
                   initial_vehicles_match=int(row["initial_U"]["vehicles"]) == int(k0),
                   feasible=ok, problems=problems,
                   initial_problems=problems0[:6])
        bad = (not ok) or (not rec["distance_match"]) or (not rec["vehicles_match"])
        bad = bad or (not ok0) or (not rec["initial_distance_match"]) \
            or (not rec["initial_vehicles_match"]) or row["online_s"] > 60.0
        rec["pass"] = not bad
        rec["pass_"] = not bad
        if bad:
            fails += 1
        results.append(rec)
    summary = dict(phase=phase.get("name"), endpoints=len(results), fails=fails,
                   versions=sorted({r.get("version") for r in results if r.get("version")}),
                   cases=len({r.get("case") for r in results}),
                   cells=sorted({(r.get("case"), r.get("seed")) for r in results}))
    summary["cells"] = len(summary["cells"])
    payload = dict(summary=summary, results=results)
    if dest is None:
        dest = str(out.parent / "audit_v2.json")
    Path(str(dest)).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                               encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    for r in results:
        if r.get("pass") is False:
            print("FAIL", r.get("case"), r.get("seed"), r.get("version"),
                  (r.get("problems") or [])[:3], r.get("initial_problems")[:2])
    return fails


if __name__ == "__main__":
    sys.exit(1 if main(sys.argv[1], sys.argv[2],
                       sys.argv[3] if len(sys.argv) > 3 else None) else 0)
