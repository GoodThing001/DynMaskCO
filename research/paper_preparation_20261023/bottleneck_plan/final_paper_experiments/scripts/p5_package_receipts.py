# -*- coding: utf-8 -*-
"""P5/P4 收据打包：把 results_*/out/*.json 复制为 results_*/endpoints/*.json（交付清单不含 out/）。

去重：PyVRP 主批与续批重叠块按 (case, seed, method) 取先出现者。
"""
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
P5 = HERE.parent / "P5_standard_vrp"
P4 = HERE.parent / "P4_solomon_coldchain"
DIRS = ["results_smoke", "results_stage01", "results_stage02", "results_stage03",
        "results_vrptw_smoke", "results_vrptw_full", "results_external_pyvrp",
        "results_external_pyvrp3", "results_external_pyvrp_rest", "results_external_author"]


def main():
    for name in DIRS:
        base = P5 / name
        out = base / "out"
        if not out.exists():
            continue
        dst = base / "endpoints"
        dst.mkdir(exist_ok=True)
        n = 0
        for f in sorted(out.glob("*.json")):
            shutil.copy2(f, dst / f.name)
            n += 1
        print("%-28s -> endpoints %d" % (name, n))
    # PyVRP 多批合并（在 endpoints 层面去重后写入 combined）
    seen = {}
    for name in ("results_external_pyvrp", "results_external_pyvrp3", "results_external_pyvrp_rest"):
        d = P5 / name / "endpoints"
        if not d.exists():
            continue
        for f in sorted(d.glob("*.json")):
            r = json.loads(f.read_text(encoding="utf-8"))
            key = (r["case"], r["seed"], r["method"])
            seen.setdefault(key, r)
    if seen:
        combo = P5 / "results_external_pyvrp_combined" / "endpoints"
        combo.mkdir(parents=True, exist_ok=True)
        for (case, seed, method), r in seen.items():
            (combo / ("%s_%d_%s.json" % (case, seed, method))).write_text(
                json.dumps(r, ensure_ascii=False), encoding="utf-8")
        print("combined PyVRP endpoints:", len(seen))
    # P4（P4_solomon_coldchain/results 与 results_smoke 下的 out/）
    for base in (P4 / "results", P4 / "results_smoke"):
        out = base / "out"
        if not out.exists():
            continue
        dst = base / "endpoints"
        dst.mkdir(exist_ok=True)
        n = 0
        for f in sorted(out.glob("*.json")):
            shutil.copy2(f, dst / f.name)
            n += 1
        print("%-28s -> endpoints %d" % (base.name, n))


if __name__ == "__main__":
    main()
