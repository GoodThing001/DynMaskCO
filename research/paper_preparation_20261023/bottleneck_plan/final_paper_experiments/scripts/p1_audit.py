# -*- coding: utf-8 -*-
"""P1 端点审计：完整性、初解严格配对、计时与独立复算一致性。

将 P1 MASK/SLOT 372 端点与统一16法同 (case, seed) 端点的初解逐路线比对。
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
FP = HERE.parent
P1 = FP / "P1_original_baselines"
UNI = FP.parent / "unified_external_20261120"
EPS = 1e-6


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    p1dir = P1 / "results" / "out"
    rows = {}
    for f in sorted(p1dir.glob("*.json")):
        d = read(f)
        rows[(d["case"], d["seed"], d["method"])] = d
    issues = []
    for (c, s, m), d in rows.items():
        if d["status"] != "AUDITED" or not d["endpoint"]["independently_audited"]:
            issues.append(f"{c}/{s}/{m} not audited")
        if d["timing"]["budget_s"] != 60 or not 0 <= d["selected_s"] <= 60:
            issues.append(f"{c}/{s}/{m} timing")
        if abs(d["U"] - d["endpoint"]["net_benefit_U"]) > EPS or d["U"] < d["initial_U"] - EPS:
            issues.append(f"{c}/{s}/{m} objective")
        if len(d["cpu"]) != 1:
            issues.append(f"{c}/{s}/{m} cores {d['cpu']}")
    # 与统一16法初解比对
    unified_initial = {}
    for name in ("benchmark01", "prospective01", "retail01"):
        base = UNI / "results" / name / "out"
        for f in sorted(base.glob("*.json")):
            d = read(f)
            if d["method"] != "ORDER_SPLIT":
                continue
            unified_initial[(d["case"], d["seed"])] = (d["initial_routes"], d["initial_U"])
    mismatches = 0
    for (c, s, m), d in sorted(rows.items()):
        ref = unified_initial.get((c, s))
        if ref is None:
            issues.append(f"{c}/{s}/{m} no unified reference")
            continue
        if d["initial_routes"] != ref[0] or abs(d["initial_U"] - ref[1]) > EPS:
            mismatches += 1
            issues.append(f"{c}/{s}/{m} initial solution differs from unified")
    print(f"P1 endpoints: {len(rows)}; expected 372")
    print(f"initial-solution mismatches vs unified16: {mismatches}")
    print(f"issues: {len(issues)}")
    for i in issues[:20]:
        print(" -", i)
    report = dict(endpoints=len(rows), expected=372, issues=len(issues), initial_mismatches=mismatches)
    (P1 / "P1_审计报告.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
