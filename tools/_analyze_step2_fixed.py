# -*- coding: utf-8 -*-
"""对比旧实现步骤 2 门（有 bug 的 SAA）与修正后 SAA 的 20260926 调试结果。"""
import json
import os
import sys

BASE = r"D:\PyCharm_\MASKCO-Main\C-VRP_Cold-chainVehicleRoutingProblem\results"
OLD = os.path.join(BASE, "a1_step2_gate", "gate.json")
NEW = os.path.join(BASE, "a1_step2_gate_fixed_20260926", "gate.json")


def load(path):
    if not os.path.exists(path):
        print(f"MISSING: {path}")
        sys.exit(1)
    return json.load(open(path, encoding="utf-8"))


def show(name, d):
    print(f"\n########## {name} (gate seed {d['seeds']['gate']}) ##########")
    for pname, b in d["gate"].items():
        print(f"=== {pname} ===")
        for a, arm in b["arms"].items():
            u = arm["utility_all_days"]
            print(f"  {a}: feas={arm['hard_feasible_rate']} fails={arm['fail_counts']} "
                  f"timeouts={arm['mean_timeouts']} | utility {u['mean']:.1f} "
                  f"[{u['ci_lo']:.1f},{u['ci_hi']:.1f}] | elapsed {arm['elapsed_total_s']:.0f}s")
        mc = b["main_comparison"]
        st = mc["cond_minus_uncond"]
        print(f"  MAIN cond-uncond: {st['mean']:.1f} [{st['ci_lo']:.1f},{st['ci_hi']:.1f}] "
              f"passed={mc['passed']} (mean>=20 & ci_lo>0 & feas==1)")


if __name__ == "__main__":
    show("OLD (buggy SAA)", load(OLD))
    show("NEW (fixed SAA)", load(NEW))
