# -*- coding: utf-8 -*-
"""P5 收据独立审计：本地标准评价器重算每条路线（1e-6 容差）与协议核验。

逐格检查：
  1. 身份：case/seed/method/objective 与 phase.json 块一致；
  2. 状态：AUDITED（搜索族与已认证构造式）/ TRUNCATED_OK（预登记 §6 的 8 个截断实例，
     双精度超窗 ≤0.11）/ INFEASIBLE（仅构造式方法可登记，如实计入）；
  3. 预算：搜索族 online_s ≤ 60；构造式 ≤ 1200（作者默认推理，原生耗时如实记录）；
  4. 重算：distance/vehicles 与本地评价器一致（CVRP 取整口径 / VRPTW 双精度，容差 1e-6）；
  5. 共同初解：搜索族同一 case 的 initial_routes 相同且 initial_U 与重算一致
     （构造式方法 initial=None，不参与共同初解核验）。
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
P5 = HERE.parent / "P5_standard_vrp"
sys.path.insert(0, str(P5))
from standard_vrp.core.standard_evaluator import (CVRPInstance, VRPTWInstance,
                                                  evaluate_cvrp, evaluate_vrptw)

TOL = 1e-6
CONSTRUCTORS = {"AttentionModel", "POMO", "DeepACO", "CaR", "MVMoE", "CaDA",
                "RouteFinder", "RRNCO"}
TRUNCATED_CASES = {"R102", "R105", "R107", "R108", "R112", "R211", "RC101", "RC105"}
TRUNCATION_TOL = 0.11


def load_inst(name, objective):
    if objective == "cvrp":
        for cand in (P5 / "data/cvrplib/instances" / (name + ".vrp"),
                     P5 / "cvrplib/instances" / (name + ".vrp")):
            if cand.exists():
                return CVRPInstance.parse_vrp(cand.read_text(encoding="utf-8"))
        raise FileNotFoundError(name)
    return VRPTWInstance.parse_solomon(P5 / "data/solomon/instances" / (name + ".vrp"))


def max_window_violation(inst, routes):
    worst = 0.0
    for r in routes:
        t = inst.depot_window[0]
        for idx, i in enumerate(r):
            t += inst.dist[0 if idx == 0 else r[idx - 1]][i]
            if t < inst.windows[i][0]:
                t = inst.windows[i][0]
            worst = max(worst, t - inst.windows[i][1])
            t += inst.service[i]
        t += inst.dist[r[-1]][0]
        worst = max(worst, t - inst.depot_window[1])
    return worst


def main(results_dir):
    base = Path(results_dir)
    phase = json.loads((base / "phase.json").read_text(encoding="utf-8"))
    out_dir = base / "out"
    if not out_dir.exists():
        out_dir = base / "endpoints"
    issues = []
    rows = []
    for f in sorted(out_dir.glob("*.json")):
        rows.append(json.loads(f.read_text(encoding="utf-8")))
    rows.sort(key=lambda r: (r["case"], r["seed"], r["method"]))
    plan = {(b["case"], b["seed"]): b for b in phase["blocks"]}
    registered_infeasible = 0
    truncated = 0
    extras = 0
    for r in rows:
        b = plan.get((r["case"], r["seed"]))
        cid = "%s/%s/%s" % (r["case"], r["seed"], r["method"])
        if b is None:
            extras += 1  # 多期合并目录：块不在当期 phase（信息性，不判失败）
        elif r["method"] not in b["methods"]:
            extras += 1
        if b is not None and r["objective"] != b["objective"]:
            issues.append("cell %s objective mismatch" % r["case"])
        cap = 1200 if r["method"] in CONSTRUCTORS else 60
        if not 0 <= r["online_s"] <= cap:
            issues.append("cell %s online_s=%.2f > cap %d" % (cid, r["online_s"], cap))
        inst = load_inst(r["case"], r["objective"])
        routes = [tuple(x) for x in r["routes"]]
        if r["objective"] == "cvrp":
            ok, d, loads, k = evaluate_cvrp(inst, routes, rounded=True)
            if not ok:
                if r["method"] in CONSTRUCTORS and r["status"] == "INFEASIBLE":
                    registered_infeasible += 1
                    continue  # 构造式不可行已登记
                issues.append("cell %s re-audit infeasible (status=%s)" % (cid, r["status"]))
                continue
            if r["status"] != "AUDITED":
                issues.append("cell %s status=%s but re-audit feasible" % (cid, r["status"]))
            if abs(d - r["distance"]) > TOL:
                issues.append("cell %s distance mismatch %.6f vs %.6f" % (cid, d, r["distance"]))
            if k != r["vehicles"]:
                issues.append("cell %s K mismatch %d vs %d" % (cid, k, r["vehicles"]))
            if r["initial_routes"] is None:
                if r["method"] not in CONSTRUCTORS:
                    issues.append("cell %s missing initial_routes" % cid)
            else:
                ok0, d0, _, _ = evaluate_cvrp(inst, [tuple(x) for x in r["initial_routes"]], rounded=True)
                if not ok0 or abs(d0 - r["initial_U"]) > TOL:
                    issues.append("cell %s initial_U mismatch" % cid)
        else:
            ok, k, d, w, loads = evaluate_vrptw(inst, routes)
            if not ok:
                if r["method"] in CONSTRUCTORS and r["status"] == "INFEASIBLE":
                    registered_infeasible += 1
                    continue
                if r["status"] == "TRUNCATED_OK":
                    viol = max_window_violation(inst, routes)
                    if r["case"] in TRUNCATED_CASES and viol <= TRUNCATION_TOL:
                        truncated += 1
                        if abs(d - r["distance"]) > TOL or k != r["vehicles"]:
                            issues.append("cell %s truncated row value mismatch" % cid)
                        continue
                issues.append("cell %s re-audit infeasible (status=%s)" % (cid, r["status"]))
                continue
            if r["status"] == "INFEASIBLE":
                issues.append("cell %s status=INFEASIBLE but re-audit feasible" % cid)
            elif r["status"] == "TRUNCATED_OK":
                issues.append("cell %s TRUNCATED_OK but re-audit feasible" % cid)
            if abs(d - r["distance"]) > TOL or k != r["vehicles"]:
                issues.append("cell %s vrptw distance/K mismatch" % cid)
            if r["initial_routes"] is None:
                if r["method"] not in CONSTRUCTORS:
                    issues.append("cell %s missing initial_routes" % cid)
            else:
                ok0, k0, d0, _, _ = evaluate_vrptw(inst, [tuple(x) for x in r["initial_routes"]])
                u0 = r["initial_U"]
                if not ok0 or k0 != u0["vehicles"] or abs(d0 - u0["distance"]) > TOL:
                    issues.append("cell %s initial_U mismatch" % cid)
    # 共同初解（仅搜索族）
    initials = {}
    for r in rows:
        if r["initial_routes"] is None:
            continue
        key = (r["case"], r["objective"])
        sig = tuple(tuple(x) for x in r["initial_routes"])
        initials.setdefault(key, sig)
        if initials[key] != sig:
            issues.append("cell %s/%s/%s initial diverges from common witness"
                          % (r["case"], r["seed"], r["method"]))
    print("cells:", len(rows), "registered_infeasible:", registered_infeasible,
          "truncated_ok:", truncated, "extras(多期合并):", extras, "issues:", len(issues))
    for i in issues[:50]:
        print(" -", i)
    print("P5 AUDIT", "PASS" if not issues else "FAIL")
    return len(issues)


if __name__ == "__main__":
    sys.exit(1 if main(sys.argv[1]) else 0)
