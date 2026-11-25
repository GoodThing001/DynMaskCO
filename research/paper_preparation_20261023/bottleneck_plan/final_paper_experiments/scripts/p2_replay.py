# -*- coding: utf-8 -*-
"""P2 时点车队独立复算：240 终点 × 4 时点 + 终值，本地严格评价器重放。"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FP = HERE.parent
P2 = FP / "P2_retail_analysis"
RECENT = FP.parent / "recent5_20261120"
EXT = FP.parent.parent.parent / "C-VRP_Cold-chainVehicleRoutingProblem"
sys.path.insert(0, str(EXT / "tools"))
sys.path.insert(0, str(RECENT))
from recent_independent import original_model, independent_audit  # noqa: E402

CUTS = ("5", "15", "30", "60")


def main():
    folder = P2 / "results" / "out"
    files = sorted(folder.glob("*.json"))
    assert len(files) == 240, len(files)
    audit_count = 0
    max_error = 0.0
    issues = []
    for f in files:
        row = json.loads(f.read_text(encoding="utf-8"))
        case, seed, method = row["case"], row["seed"], row["method"]
        input_path = P2 / "inputs" / (case + ".json")
        model, first, u0 = original_model(input_path, case)
        if row["initial_routes"] != first or abs(row["initial_U"] - u0) > 1e-6:
            issues.append(f"{case}/{seed}/{method} initializer differs from frozen witness")
            continue
        # 终值复算
        routes = row["routes"]
        if sorted(v for r in routes for v in r) != list(range(1, 101)):
            issues.append(f"{case}/{seed}/{method} coverage")
            continue
        u = independent_audit(model, model.solution(routes), True)
        err = abs(u - row["U"])
        max_error = max(max_error, err)
        if err > 1e-6 or u < u0 - 1e-6:
            issues.append(f"{case}/{seed}/{method} final U mismatch {err}")
        audit_count += 1
        # 时点复算
        previous = -math.inf
        traj = row.get("trajectory", {})
        if set(traj) != set(CUTS):
            issues.append(f"{case}/{seed}/{method} trajectory keys {sorted(traj)}")
            continue
        for t in CUTS:
            sn = traj[t]
            if sn is None:
                issues.append(f"{case}/{seed}/{method} lost incumbent at {t}")
                continue
            tr = sn["routes"]
            if sorted(v for r in tr for v in r) != list(range(1, 101)) or not all(tr):
                issues.append(f"{case}/{seed}/{method} snapshot coverage {t}")
                continue
            v = independent_audit(model, model.solution(tr), True)
            e2 = abs(v - sn["U"])
            max_error = max(max_error, e2)
            if e2 > 1e-6 or v < previous - 1e-6 or not 0 <= sn["accepted_s"] <= int(t) or (t == "60" and abs(v - row["U"]) > 1e-6):
                issues.append(f"{case}/{seed}/{method} snapshot {t} mismatch {e2}")
            previous = v
            audit_count += 1
    print(f"P2 replay: {len(files)} endpoints, {audit_count} fleet audits, max_error={max_error:.3g}, issues={len(issues)}")
    for i in issues[:20]:
        print(" -", i)
    (P2 / "P2_独立复算报告.json").write_text(json.dumps(
        dict(endpoints=len(files), fleet_audits=audit_count, max_error=max_error, issues=issues[:200]),
        indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
