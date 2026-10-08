# -*- coding: utf-8 -*-
"""步骤 2 门判定逐日深度复核：检查逐日数据的异常/分布，不只汇总数。"""
import json
import numpy as np

GATE = r"D:\PyCharm_\MASKCO-Main\C-VRP_Cold-chainVehicleRoutingProblem\results\a1_step2_gate\gate.json"
d = json.load(open(GATE, encoding="utf-8"))

for pname, block in d["gate"].items():
    rows = block["per_day"]
    print(f"\n=== {pname} ===")
    for name in ("uncond_hist", "cond_hist", "explicit_feat"):
        r = rows[name]
        u = np.array([x["utility"] for x in r], float)
        served = np.array([x["n_served"] for x in r], float)
        rej = np.array([x["n_rejected"] for x in r], float)
        eng = np.array([x["energy_kwh"] for x in r], float)
        feas = all(x["hard_feasible"] for x in r)
        tos = sum(x["timeouts"] for x in r)
        fails = sorted({f for x in r for f in x["failures"]})
        print(f"  {name}: feas={feas} fails={fails} timeouts={tos} | "
              f"utility {u.mean():.1f}±{u.std():.1f} [min {u.min():.1f}, max {u.max():.1f}] | "
              f"served {served.mean():.1f}±{served.std():.1f} | rej {rej.mean():.1f} | "
              f"energy {eng.mean():.1f}±{eng.std():.1f} (B={d['budget']['B']:.1f})")
    # 配对差逐日分布（主比较）
    du = np.array([x["utility"] for x in rows["cond_hist"]], float) - \
         np.array([x["utility"] for x in rows["uncond_hist"]], float)
    print(f"  paired diff: {du.mean():.1f}±{du.std():.1f} "
          f"[{du.min():.1f}, {du.max():.1f}] | positive days: {(du > 0).sum()}/{len(du)}")
    # 预算使用率
    for name in ("uncond_hist", "cond_hist"):
        usage = np.array([x["energy_kwh"] / d["budget"]["B"] for x in rows[name]])
        print(f"  {name} budget usage: {usage.mean():.2%}±{usage.std():.2%}")
