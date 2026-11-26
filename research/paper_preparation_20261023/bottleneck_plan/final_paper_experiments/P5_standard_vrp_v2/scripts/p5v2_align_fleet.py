# -*- coding: utf-8 -*-
"""开发面板 VRPTW 实例车队上限协议对齐（构造 K 保留为参考目标）。

问题：开发实例头部 NUMBER = 构造解车辆数（10/11/12），使任何构造式初解（K=11–17）
在容量/时窗均可行的情况下仍被判为"车队超限不可行"，五版本面板无法度量。

修正：把开发 VRPTW 头部 NUMBER 统一为 25（与 Solomon 协议一致：官方车队上限 25），
构造解 K 保留在 manifest 的 `k` 字段作为参考目标（不再作为硬界）。
只改 12 个开发实例（本地自建数据），官方 Solomon/Uchoa 数据不动。
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
TARGET = 25


def main():
    man_path = V2 / "development/manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    changed = []
    for row in man["instances"]:
        if row["kind"] != "vrptw":
            continue
        path = V2 / "development/instances" / row["file"]
        text = path.read_text(encoding="utf-8")
        m = re.search(r"NUMBER\s+CAPACITY\s*\n\s*(\d+)\s+(\d+)", text)
        if not m:
            raise SystemExit("header not found: " + str(path))
        old_n, cap = int(m.group(1)), int(m.group(2))
        if old_n == TARGET:
            continue
        new_text = text[:m.start()] + ("NUMBER CAPACITY\n%6d %6d" % (TARGET, cap)) + \
            text[m.end():]
        path.write_text(new_text, encoding="utf-8")
        row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        row["fleet_cap_note"] = ("header NUMBER %d -> %d（协议对齐；构造 K=%s 保留为参考目标）"
                                % (old_n, TARGET, row.get("k")))
        changed.append((row["name"], old_n, TARGET))
    if changed:
        man["fleet_cap_alignment"] = dict(
            date="2026-11-26", target=TARGET, changed=len(changed),
            reason="构造式初解 K>构造 K 时被误判车队超限；与 Solomon 官方上限 25 对齐",
            rows=[dict(case=c, before=a, after=b) for c, a, b in changed])
        man_path.write_text(json.dumps(man, ensure_ascii=False, indent=2), encoding="utf-8")
    print("changed", len(changed), [c[0] for c in changed])
    # 复核：12 个开发 VRPTW 初解应全部可行
    sys.path.insert(0, str(V2))
    from standard_vrp.core.standard_evaluator import VRPTWInstance
    from standard_vrp.adapters.objective import adapter_for
    from standard_vrp.solver.pool_search import nn_witness
    bad = []
    for row in man["instances"]:
        if row["kind"] != "vrptw":
            continue
        inst = VRPTWInstance.parse_solomon(V2 / "development/instances" / row["file"])
        ad = adapter_for(inst)
        w = [tuple(r) for r in nn_witness(inst)]
        ok, d, k = ad.evaluate(w)
        print("%-22s cap=%d K=%2d ok=%-5s D=%.2f (参考 K=%s)"
              % (row["name"], inst.max_vehicles, k, ok, d, row.get("k")))
        if not ok:
            bad.append(row["name"])
    print("infeasible witnesses:", bad)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
