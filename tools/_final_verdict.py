# -*- coding: utf-8 -*-
"""最终裁决：20260930/20261001 复验结果汇总。"""
import json

BASE = r"C:\AA_Proj".replace("C:\\AA_Proj", r"D:\PyCharm_\MASKCO-Main\C-VRP_Cold-chainVehicleRoutingProblem\results")

for seed in ["20260930", "20261001"]:
    d = json.load(open(rf"{BASE}\a1_step2_gate_verify_{seed}\gate.json", encoding="utf-8"))
    print(f"==== seed {seed} ====")
    for pn, b in d["gate"].items():
        if isinstance(b, dict) and "main_comparison" in b:
            mc = b["main_comparison"]
            st = mc["cond_minus_uncond"]
            feas = all(b["arms"][a]["hard_feasible_rate"] == 1.0
                       for a in ["cond_hist", "uncond_hist"])
            print(f"  {pn}: cond-uncond {st['mean']:.1f} [{st['ci_lo']:.1f},{st['ci_hi']:.1f}] "
                  f"passed={mc['passed']} feas={feas}")
    b = d["gate"]["p_c=(5,10,15)"]
    pd = b["per_day"]
    pos = sum(1 for a, c in zip(pd["cond_hist"], pd["uncond_hist"])
              if a["utility"] > c["utility"])
    viol = sum(1 for arm in pd for r in pd[arm] if not r["hard_feasible"])
    tmo = sum(1 for arm in pd for r in pd[arm] if r["timeouts"] > 0)
    print(f"  positive days {pos}/40, violation-day-rows {viol}, timeout-day-rows {tmo}")
    for a, arm in b["arms"].items():
        u = arm["utility_all_days"]
        print(f"    {a}: {u['mean']:.1f} feas={arm['hard_feasible_rate']} fails={arm['fail_counts']}")
