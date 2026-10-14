"""逐日对账两个 step-2 风格 gate（用于验证 v2 正式基线是否复现 c1rr2 的行为）。

用法：
    python CC_Compare/tools/compare_two_gates.py <gateA.json> <gateB.json>

前提（否则差异无意义）：两者用**同一版源码**、同一 `--train-instances` / `--gate-seed` /
`--K` / `--n-orders`。此时若逐日效用一致 ⇒ 同协议复现；差异应主要来自负载导致的
超时（同时报告两侧 `mean_timeouts`）。
"""
import json
import os
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    A, B = load(sys.argv[1]), load(sys.argv[2])
    for tag, d in (("A", A), ("B", B)):
        ident = d.get("identity") or {}
        cfg = d.get("config") or {}
        print("%s: %s" % (tag, os.path.basename(os.path.dirname(sys.argv[1 if tag == 'A' else 2]))))
        print("   train_instances=%s gate_instances=%s seed=%s K=%s workers=%s" % (
            cfg.get("train_instances"), cfg.get("gate_instances"),
            cfg.get("gate_seed"), cfg.get("K"), cfg.get("workers")))
        print("   data_sha256=%s contract=%s B=%s" % (
            str(ident.get("data_sha256"))[:16],
            str(ident.get("contract_sha256"))[:16], d.get("budget", {}).get("B")))
    same_data = (A.get("identity", {}).get("data_sha256")
                 == B.get("identity", {}).get("data_sha256"))
    print("\ndata_sha256 equal:", same_data,
          "(不同 ⇒ 逐日对账无意义)" if not same_data else "")
    if not same_data:
        return
    print()
    for pname in sorted(A.get("gate", {})):
        if pname not in B.get("gate", {}):
            continue
        for arm in sorted(A["gate"][pname]["arms"]):
            ra = A["gate"][pname]["per_day"][arm]
            rb = {r["i"]: r for r in B["gate"][pname]["per_day"][arm]}
            deltas = []
            for row in ra:
                j = rb.get(row["i"])
                if j is None or row.get("utility") is None or j.get("utility") is None:
                    continue
                deltas.append(row["utility"] - j["utility"])
            ta = A["gate"][pname]["arms"][arm].get("mean_timeouts")
            tb = B["gate"][pname]["arms"][arm].get("mean_timeouts")
            ua = A["gate"][pname]["arms"][arm]["utility_all_days"]["mean"]
            ub = B["gate"][pname]["arms"][arm]["utility_all_days"]["mean"]
            if deltas:
                mx = max(abs(x) for x in deltas)
                print("%-16s %-14s meanA=%9.2f meanB=%9.2f Δmean=%+8.2f  max|Δday|=%9.3f  timeouts A/B=%.2f/%.2f"
                      % (pname, arm, ua, ub, ua - ub, mx, ta or 0, tb or 0))
            else:
                print("%-16s %-14s 无可配对日" % (pname, arm))


if __name__ == "__main__":
    main()
