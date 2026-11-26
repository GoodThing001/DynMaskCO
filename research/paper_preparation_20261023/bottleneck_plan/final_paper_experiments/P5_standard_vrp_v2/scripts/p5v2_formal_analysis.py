# -*- coding: utf-8 -*-
"""P5-V2 正式面板分析：生成 V2-1..V2-10 表（MD/CSV/JSON）。

用法：python p5v2_formal_analysis.py <cvrp_out_dir> <vrptw_out_dir> <norm_out_dir> [dest]

只读端点 JSON（p5v2_worker 产物）。所有统计量写入 tables/ 供报告与图引用。
"""
from __future__ import annotations

import csv
import json
import math
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
INTERNAL = ["ORDER_SPLIT", "ORDER_SPLIT_D", "MASK", "SLOT", "DIST", "DIST_SLOT"]
PAIRS_NEURAL = [("ORDER_SPLIT", "ORDER_SPLIT_D"), ("MASK", "DIST"), ("SLOT", "DIST_SLOT")]


def load(dirs):
    rows = []
    for d in dirs:
        p = Path(d)
        if not p.exists():
            continue
        for f in sorted(p.rglob("*.json")):
            if any(k in f.name for k in ("phase", "status", "audit")):
                continue
            row = json.loads(f.read_text(encoding="utf-8"))
            if "objective" not in row:
                continue
            rows.append(row)
    return rows


def sign_test(wins, losses):
    """双侧符号检验（正态近似，无 scipy 依赖）。"""
    n = wins + losses
    if n == 0:
        return None
    z = (wins - n / 2) / math.sqrt(n / 4)
    # 双侧 p（正态近似）
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return dict(wins=wins, losses=losses, z=round(z, 3), p=round(p, 6))


def bks_tables():
    import re as _re
    cvrp = json.loads((V2 / "data/bks_derived.json").read_text(encoding="utf-8"))["cvrp"]
    vrptw = json.loads((V2 / "data/solomon_bks_derived.json").read_text(encoding="utf-8"))["rows"]
    cv = {}
    for r in cvrp:
        name = r.get("name") or r["instance"]
        m = _re.search(r"-n(\d+)-k(\d+)", name)
        n = int(m.group(1)) - 1 if m else None     # 实例维度 - 1 个仓库
        cv[name] = dict(bks=r.get("rounded") or r.get("official_cost"),
                        k_sol=r.get("k_sol") or r.get("k"),
                        n=n)
    vt = {r.get("name") or r["instance"]: r for r in vrptw}
    return cv, vt


def group(rows, objective):
    out = {}
    for r in rows:
        if r["objective"] != objective:
            continue
        out.setdefault((r["case"], r["seed"]), {})[r["version"]] = r
    return out


def mean_of(d, keys):
    vals = [d[k]["distance"] for k in keys if k in d]
    return st.mean(vals) if vals else None


def main(cvrp_dir, vrptw_dir, norm_dir=None, dest=None):
    dest = Path(dest) if dest else V2 / "tables"
    dest.mkdir(parents=True, exist_ok=True)
    bks_c, bks_v = bks_tables()
    rows = load([cvrp_dir, vrptw_dir])
    norm_rows = load([norm_dir]) if norm_dir and Path(norm_dir).exists() else []
    cv = group(rows, "cvrp")
    vt = group(rows, "vrptw")
    nm = {}
    for r in norm_rows:
        nm.setdefault((r["case"], r["seed"]), {})[r["version"]] = r

    report = dict(endpoints=len(rows) + len(norm_rows), cells=len(cv) + len(vt) + len(nm))

    # ---------------- V2-1 CVRP 逐实例 ----------------
    v21 = []
    for (case, seed), d in sorted(cv.items()):
        b = bks_c.get(case, {})
        ref = b.get("bks") or b.get("rounded")
        entry = dict(instance=case, seed=seed, bks=ref, bks_k=b.get("k_sol"), n=b.get("n"))
        for m in INTERNAL:
            if m in d:
                entry[m] = d[m]["distance"]
                entry[m + "_K"] = d[m]["vehicles"]
                entry[m + "_gap%"] = (100.0 * (d[m]["distance"] - ref) / ref
                                      if ref else None)
        v21.append(entry)
    write_csv(dest / "V2-1_cvrp_per_instance.csv", v21)
    report["v2_1"] = dict(rows=len(v21))

    # ---------------- V2-2 CVRP 规模段聚合 ----------------
    def bucket(n):
        if n is None:
            return "unknown"
        if n < 200:
            return "100-199"
        if n < 500:
            return "200-499"
        return "500-1000"

    v22 = []
    for name in ("100-199", "200-499", "500-1000"):
        sub = [e for e in v21 if bucket(e["n"]) == name]
        if not sub:
            continue
        rec = dict(bucket=name, instances=len({e["instance"] for e in sub}), endpoints=len(sub))
        for m in INTERNAL:
            gaps = [e[m + "_gap%"] for e in sub if e.get(m + "_gap%") is not None]
            ks = [e[m + "_K"] for e in sub if e.get(m + "_K") is not None]
            if gaps:
                rec[m + "_mean_gap%"] = round(st.mean(gaps), 4)
                rec[m + "_median_gap%"] = round(st.median(gaps), 4)
                rec[m + "_mean_K"] = round(st.mean(ks), 3)
                rec[m + "_bksK_mismatch"] = sum(1 for e in sub
                                                if e.get(m + "_K") != e.get("bks_k"))
        v22.append(rec)
    write_csv(dest / "V2-2_cvrp_by_size.csv", v22)

    # ---------------- V2-3 CVRP 方法两两配对 ----------------
    v23 = []
    per_cell = {(e["instance"], e["seed"]): e for e in v21}
    for i, a in enumerate(INTERNAL):
        for b in INTERNAL[i + 1:]:
            diffs, wins, losses = [], 0, 0
            for e in per_cell.values():
                if e.get(a) is None or e.get(b) is None:
                    continue
                diffs.append(100.0 * (e[b] - e[a]) / e[a])
                if e[b] < e[a]:
                    wins += 1
                elif e[b] > e[a]:
                    losses += 1
            if diffs:
                v23.append(dict(a=a, b=b, n=len(diffs), mean_delta_pct=round(st.mean(diffs), 4),
                                median_delta_pct=round(st.median(diffs), 4),
                                b_better=wins, a_better=losses,
                                sign_test=sign_test(wins, losses)))
    write_csv(dest / "V2-3_cvrp_pairs.csv", v23)

    # ---------------- V2-4 VRPTW 逐实例 ----------------
    v24 = []
    for (case, seed), d in sorted(vt.items()):
        b = bks_v.get(case, {})
        e = dict(instance=case, seed=seed, bks_K=b.get("k_bks"),
                 bks_D=b.get("distance_double"), ref_feasible_double=b.get("feasible_double"))
        for m in INTERNAL:
            if m in d:
                e[m + "_K"] = d[m]["vehicles"]
                e[m + "_D"] = d[m]["distance"]
                e[m + "_Kdiff"] = d[m]["vehicles"] - (b.get("k_bks") or 0)
                e[m + "_Ddelta"] = (d[m]["distance"] - b["distance_double"]
                                    if b.get("distance_double") else None)
        v24.append(e)
    write_csv(dest / "V2-4_vrptw_per_instance.csv", v24)

    # ---------------- V2-5 VRPTW 类别聚合 ----------------
    def cat(name):
        return name[0] + (name[1] if name[0] == "R" and name[1] in "12" and name.startswith("R")
                          else "")
    v25 = []
    for c in sorted({cat(n) for n in {e["instance"] for e in v24}}):
        sub = [e for e in v24 if cat(e["instance"]) == c]
        if not sub:
            continue
        rec = dict(category=c, instances=len({e["instance"] for e in sub}), endpoints=len(sub))
        for m in INTERNAL:
            ks = [e[m + "_K"] for e in sub if e.get(m + "_K") is not None]
            ds = [e[m + "_D"] for e in sub if e.get(m + "_D") is not None]
            kd = [e[m + "_Kdiff"] for e in sub if e.get(m + "_Kdiff") is not None]
            dd = [e[m + "_Ddelta"] for e in sub if e.get(m + "_Ddelta") is not None]
            if ks:
                rec[m + "_mean_K"] = round(st.mean(ks), 3)
                rec[m + "_mean_Kdiff"] = round(st.mean(kd), 3)
                rec[m + "_K_win"] = sum(1 for x in kd if x < 0)
                rec[m + "_K_tie"] = sum(1 for x in kd if x == 0)
                rec[m + "_K_loss"] = sum(1 for x in kd if x > 0)
                rec[m + "_mean_D"] = round(st.mean(ds), 3)
                rec[m + "_mean_Ddelta_sameK"] = round(
                    st.mean([x for x, e2 in zip(dd, sub)
                             if e2.get(m + "_Kdiff") == 0] or [0.0]), 3)
                rec[m + "_bks_K"] = round(st.mean([e["bks_K"] for e in sub
                                                  if e.get("bks_K")]), 3)
        v25.append(rec)
    write_csv(dest / "V2-5_vrptw_by_category.csv", v25)

    # ---------------- V2-6 STD_NORM 效果（配对） ----------------
    v26 = []
    for obj, base in (("cvrp", cv), ("vrptw", vt)):
        diffs, wins, losses, kdiffs = [], 0, 0, []
        for key, d in nm.items():
            if key not in base:
                continue
            n = d.get("STD_NORM")
            ref = base[key].get("ORDER_SPLIT")
            if not n or not ref:
                continue
            if obj == "cvrp":
                diffs.append(100.0 * (n["distance"] - ref["distance"]) / ref["distance"])
                if n["distance"] < ref["distance"]:
                    wins += 1
                elif n["distance"] > ref["distance"]:
                    losses += 1
            else:
                kdiffs.append(n["vehicles"] - ref["vehicles"])
                if (n["vehicles"], n["distance"]) < (ref["vehicles"], ref["distance"]):
                    wins += 1
                elif (n["vehicles"], n["distance"]) > (ref["vehicles"], ref["distance"]):
                    losses += 1
        rec = dict(objective=obj, n=wins + losses, norm_better=wins, split_better=losses,
                   sign_test=sign_test(wins, losses))
        if obj == "cvrp" and diffs:
            rec.update(mean_delta_pct=round(st.mean(diffs), 4),
                       median_delta_pct=round(st.median(diffs), 4))
        if obj == "vrptw" and kdiffs:
            rec.update(mean_Kdiff=round(st.mean(kdiffs), 3),
                       K_better=sum(1 for x in kdiffs if x < 0),
                       K_tie=sum(1 for x in kdiffs if x == 0),
                       K_worse=sum(1 for x in kdiffs if x > 0))
        v26.append(rec)
    write_csv(dest / "V2-6_norm_effect.csv", v26)

    # ---------------- V2-7 神经引导净贡献 ----------------
    v27 = []
    for obj, base in (("cvrp", cv), ("vrptw", vt)):
        for neural, dist in PAIRS_NEURAL:
            diffs, wins, losses = [], 0, 0
            for key, d in base.items():
                a, b = d.get(neural), d.get(dist)
                if not a or not b:
                    continue
                if obj == "cvrp":
                    diffs.append(100.0 * (a["distance"] - b["distance"]) / b["distance"])
                    if a["distance"] < b["distance"]:
                        wins += 1
                    elif a["distance"] > b["distance"]:
                        losses += 1
                else:
                    if a["vehicles"] != b["vehicles"]:
                        if a["vehicles"] < b["vehicles"]:
                            wins += 1
                        else:
                            losses += 1
                        diffs.append(float(a["vehicles"] - b["vehicles"]))
                    else:
                        diffs.append(100.0 * (a["distance"] - b["distance"]) / b["distance"])
                        if a["distance"] < b["distance"]:
                            wins += 1
                        elif a["distance"] > b["distance"]:
                            losses += 1
            if diffs:
                v27.append(dict(objective=obj, neural=neural, distance_only=dist, n=len(diffs),
                                neural_better=wins, distance_better=losses,
                                mean_delta=round(st.mean(diffs), 4),
                                median_delta=round(st.median(diffs), 4),
                                sign_test=sign_test(wins, losses)))
    write_csv(dest / "V2-7_neural_contribution.csv", v27)

    # ---------------- V2-8 机制消融 ----------------
    v28 = []
    ablations = [("split_on", "ORDER_SPLIT", "MASK"), ("split_on_dist", "ORDER_SPLIT_D", "DIST"),
                 ("slot_on", "SLOT", "MASK"), ("slot_on_dist", "DIST_SLOT", "DIST")]
    for obj, base in (("cvrp", cv), ("vrptw", vt)):
        for label, on, off in ablations:
            row = dict(objective=obj, ablation=label, on=on, off=off)
            diffs, wins, losses = [], 0, 0
            for key, d in base.items():
                a, b = d.get(on), d.get(off)
                if not a or not b:
                    continue
                if obj == "cvrp":
                    diffs.append(100.0 * (a["distance"] - b["distance"]) / b["distance"])
                    if a["distance"] < b["distance"]:
                        wins += 1
                    elif a["distance"] > b["distance"]:
                        losses += 1
                else:
                    kd = a["vehicles"] - b["vehicles"]
                    diffs.append(float(kd) if kd else 100.0 * (a["distance"] - b["distance"])
                                 / b["distance"])
                    if (a["vehicles"], a["distance"]) < (b["vehicles"], b["distance"]):
                        wins += 1
                    elif (a["vehicles"], a["distance"]) > (b["vehicles"], b["distance"]):
                        losses += 1
            if diffs:
                row.update(n=len(diffs), on_better=wins, off_better=losses,
                           mean_delta=round(st.mean(diffs), 4),
                           median_delta=round(st.median(diffs), 4),
                           sign_test=sign_test(wins, losses))
                v28.append(row)
    write_csv(dest / "V2-8_mechanism_ablation.csv", v28)

    # ---------------- V2-9 计时与预测器审计 ----------------
    v29 = []
    for obj, base in (("cvrp", cv), ("vrptw", vt)):
        for m in INTERNAL:
            rs = [d[m] for d in base.values() if m in d]
            if not rs:
                continue
            att = sum((r.get("stats") or {}).get("predictor", {}).get("attempts", 0) for r in rs)
            suc = sum((r.get("stats") or {}).get("predictor", {}).get("success", 0) for r in rs)
            fwd_s = sum((r.get("stats") or {}).get("predictor", {}).get("forward_s", 0) or 0
                        for r in rs)
            v29.append(dict(objective=obj, method=m, endpoints=len(rs),
                            mean_online_s=round(st.mean([r["online_s"] for r in rs]), 3),
                            max_online_s=round(max(r["online_s"] for r in rs), 3),
                            mean_epochs=round(st.mean([(r.get("stats") or {}).get("epochs") or 0
                                                       for r in rs]), 2),
                            mean_iterations=round(st.mean([(r.get("stats") or {}).get("iterations")
                                                           or 0 for r in rs]), 2),
                            forward_attempts=att, forward_success=suc,
                            forward_s=round(fwd_s, 2),
                            fallbacks=sum((r.get("stats") or {}).get("fallbacks") or 0 for r in rs)))
    write_csv(dest / "V2-9_timing_predictor.csv", v29)

    # ---------------- V2-10 历史快照（5/15/30/60 秒） ----------------
    v210 = []
    for obj, base in (("cvrp", cv), ("vrptw", vt)):
        for m in INTERNAL:
            rec = dict(objective=obj, method=m)
            for cp in ("5", "15", "30", "60"):
                vals = []
                for d in base.values():
                    hist = (d.get(m, {}).get("stats") or {}).get("history") or {}
                    if cp in hist:
                        vals.append(hist[cp]["distance"])
                if vals:
                    rec["mean_D_t%s" % cp] = round(st.mean(vals), 3)
                    rec["n_t%s" % cp] = len(vals)
            if len(rec) > 2:
                v210.append(rec)
    write_csv(dest / "V2-10_history.csv", v210)

    payload = dict(report=report, v2_2=v22, v2_3=v23, v2_5=v25, v2_6=v26, v2_7=v27,
                   v2_8=v28, v2_9=v29, v2_10=v210)
    (dest / "analysis_formal.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(endpoints=report["endpoints"], cells=report["cells"],
                          cvrp_cells=len(cv), vrptw_cells=len(vt), norm_cells=len(nm),
                          tables=len(list(dest.glob("V2-*.csv")))), ensure_ascii=False))
    return payload


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in keys})


if __name__ == "__main__":
    main(*(sys.argv[1:5]))
