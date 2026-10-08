# -*- coding: utf-8 -*-
"""对照 V2 三臂与 OR-Tools 接单的逐日分解（served/revenue/fuel/rejloss）。"""
import json

BASE = r"D:\PyCharm_\MASKCO-Main\C-VRP_Cold-chainVehicleRoutingProblem\results"
V2 = rf"{BASE}\a1_step2_gate_vote2_20260926\gate.json"
ORT = rf"{BASE}\a1_ext_v2_ortools\gate.json"


def rows(path, arm, pn):
    d = json.load(open(path, encoding="utf-8"))
    if "gate" in d:   # 步骤 2 驱动：gate[pname].per_day[arm]
        pd = d["gate"][pn].get("per_day")
        return pd.get(arm) if isinstance(pd, dict) else None
    else:             # 外部接单驱动：results[seed][pname].per_day = 求解器臂逐日行列表
        blk = d["results"]["20260926"][pn]
        return blk.get("per_day") if arm == "ortools_accept" else None


for pn in ["p_c=0", "p_c=(5,10,15)"]:
    print("==", pn)
    for path, arm in [(V2, "uncond_hist"), (V2, "cond_hist"), (ORT, "ortools_accept")]:
        rs = rows(path, arm, pn)
        if rs is None:
            print("  ", arm, "MISSING")
            continue

        def m(k):
            vals = [r.get(k) for r in rs]
            vals = [v for v in vals if v is not None]
            return round(sum(vals) / len(vals), 1) if vals else "n/a"
        print(f"  {arm}: served {m('n_served')} rejected {m('n_rejected')} "
              f"revenue {m('revenue')} fuel {m('fuel_cost')} rejloss {m('reject_loss')} "
              f"utility {m('utility')} energy {m('energy_kwh')}")
