# -*- coding: utf-8 -*-
"""分析步骤 3 三臂 (random/explicit/pretrained) 比较结果，对照 DELTA_3=20 门槛。"""
import json
import sys

BASE = r"D:\PyCharm_\MASKCO-Main\C-VRP_Cold-chainVehicleRoutingProblem\results\a1_step3_dev"
ARMS = ["random", "explicit", "pretrained"]
DELTA_3 = 20.0

verdicts = {}
for arm_dir in ARMS:
    path = f"{BASE}\\{arm_dir}\\gate.json"
    try:
        d = json.load(open(path, encoding="utf-8"))
    except FileNotFoundError:
        print(f"=== {arm_dir}: MISSING ({path}) ===")
        continue
    except (json.JSONDecodeError, ValueError):
        print(f"=== {arm_dir}: EMPTY/INVALID ({path}) ===")
        continue
    print(f"\n########## ARM: {arm_dir} ##########")
    for seed, blocks in d["results"].items():
        for pname, b in blocks.items():
            print(f"=== seed {seed} | {pname} ===")
            for a, arm in b["arms"].items():
                u = arm["utility_all_days"]
                print(f"  {a}: feas={arm['hard_feasible_rate']} fails={arm['fail_counts']} "
                      f"timeouts={arm['mean_timeouts']} | utility {u['mean']:.1f} "
                      f"[{u['ci_lo']:.1f},{u['ci_hi']:.1f}] | elapsed {arm['elapsed_total_s']:.0f}s")
            mc = b["main_comparison"]
            st = mc["maskco_minus_cond_hist"]
            pos_days = "n/a"
            if "per_day" in b and "maskco_minus_cond_hist" in b.get("per_day", {}):
                pd = b["per_day"]["maskco_minus_cond_hist"]
                vals = [v for v in pd.values() if isinstance(v, (int, float))]
                pos_days = f"{sum(1 for v in vals if v > 0)}/{len(vals)}"
            print(f"  MAIN maskco-cond: {st['mean']:.1f} [{st['ci_lo']:.1f},{st['ci_hi']:.1f}] "
                  f"passed={mc['passed']} (mean>=20 & ci_lo>0) | pos_days={pos_days}")
            for k in ("maskco_minus_explicit_feat", "maskco_minus_uncond_hist"):
                s = b[k]
                print(f"  {k}: {s['mean']:.1f} [{s['ci_lo']:.1f},{s['ci_hi']:.1f}]")
            verdicts[f"{arm_dir}:{seed}:{pname}"] = (st["mean"], st["ci_lo"], st["ci_hi"], mc["passed"])

print("\n########## VERDICT ##########")
for k, (m, lo, hi, ok) in verdicts.items():
    status = "PASS" if ok else ("FAIL" if m < DELTA_3 or lo <= 0 else "MARGINAL")
    print(f"  {k}: {m:.1f} [{lo:.1f},{hi:.1f}] -> {status}")
