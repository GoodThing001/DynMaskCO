# -*- coding: utf-8 -*-
"""OR-Tools 公平重基线结果摘要（A-05/A-05b 修正后 vs C1 修正基线）。"""
import json

BASE = r"D:\PyCharm_\MASKCO-Main\C-VRP_Cold-chainVehicleRoutingProblem\results"
d = json.load(open(rf"{BASE}\a1_ext_c1_ortools\gate.json", encoding="utf-8"))


def fmt_u(u):
    if isinstance(u, dict) and 'mean' in u:
        return f"{u['mean']:.1f} [{u['ci_lo']:.1f},{u['ci_hi']:.1f}]"
    return str(u)


print("seeds:", d["seeds"])
for seed, blks in d["results"].items():
    for pn in ["p_c=0", "p_c=(5,10,15)", "p_c=(10,20,30)"]:
        b = blks.get(pn)
        if not b:
            continue
        print("==", seed, pn)
        for a, arm in b.get("arms", {}).items():
            print("  ", a, fmt_u(arm.get("utility_all_days")),
                  "| adj", arm.get("adjudicable"), "| feas", arm.get("hard_feasible_rate"))
        for k in ["ortools_accept_minus_cond_hist", "ortools_accept_minus_uncond_hist"]:
            s = b.get(k)
            if s:
                lo = f"[{s['ci_lo']:.1f},{s['ci_hi']:.1f}]" if s.get("ci_lo") is not None else "n/a"
                print("   ", k, s.get("mean"), lo, "adj", s.get("adjudicable"))
        rows = b.get("per_day") or []
        crash = sum(1 for r in rows if r.get("failures") == ["execution_crash"])
        print("   crash_days", crash, "/40 | meta", {k: v for k, v in (b.get("solver_meta") or {}).items()
                                                    if k in ("n_solves", "n_fail")})
