# -*- coding: utf-8 -*-
"""独立核查 A-v1 步骤 2 门判定：从逐日原始数据重算硬可行率/配对差/day-clustered bootstrap CI/判据。"""
import json
import numpy as np

GATE = r"D:\PyCharm_\MASKCO-Main\C-VRP_Cold-chainVehicleRoutingProblem\results\a1_step2_gate\gate.json"
DELTA_STAR = 20.0
SEED = 20260926

d = json.load(open(GATE, encoding="utf-8"))
print("config:", {k: v for k, v in d["config"].items() if k in ("train_instances", "gate_instances", "K", "time_limit")})
print("seeds:", d["seeds"])

rng = np.random.default_rng(SEED + 777)

for pname, block in d["gate"].items():
    arms = block["arms"]
    rows = block["per_day"]
    ok_all = {}
    for a in arms:
        r = rows[a]
        feas = [x["hard_feasible"] for x in r]
        tos = [x["timeouts"] for x in r]
        fails = sorted({f for x in r for f in x["failures"]})
        ok_all[a] = (all(feas), fails, sum(tos))
    cond = np.array([x["utility"] for x in rows["cond_hist"]], float)
    uncond = np.array([x["utility"] for x in rows["uncond_hist"]], float)
    diff = cond - uncond
    n = len(diff)
    boots = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(2000)])
    mean, lo, hi = float(diff.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    verdict = (ok_all["cond_hist"][0] and ok_all["uncond_hist"][0]
               and mean >= DELTA_STAR and lo > 0)
    print(f"\n[{pname}]")
    print(f"  cond  hard_feas={ok_all['cond_hist'][0]} fails={ok_all['cond_hist'][1]} timeouts={ok_all['cond_hist'][2]}")
    print(f"  uncond hard_feas={ok_all['uncond_hist'][0]} fails={ok_all['uncond_hist'][1]} timeouts={ok_all['uncond_hist'][2]}")
    print(f"  explicit hard_feas={ok_all['explicit_feat'][0]} fails={ok_all['explicit_feat'][1]} timeouts={ok_all['explicit_feat'][2]}")
    print(f"  independent: cond-uncond mean={mean:.2f} CI=[{lo:.2f}, {hi:.2f}] n={n} -> "
          f"{'PASS' if verdict else 'FAIL'}")
    print(f"  driver said: passed={block['main_comparison']['passed']}")
    assert verdict == block["main_comparison"]["passed"], "verdict mismatch!"
print("\nINDEPENDENT VERIFICATION: consistent with driver verdict")
