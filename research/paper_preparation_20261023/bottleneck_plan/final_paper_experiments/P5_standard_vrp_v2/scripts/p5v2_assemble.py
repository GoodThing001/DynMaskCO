# -*- coding: utf-8 -*-
"""一键装配：抓取 → 独立复算 → 分析 → 图 → 汇总（正式面板 + 外部 + 诊断）。

用法：python p5v2_assemble.py [--no-fetch]
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
PY = sys.executable
STAGES = dict(cvrp=("/tmp/p5v2_formal/cvrp", "formal_results/cvrp"),
              vrptw=("/tmp/p5v2_formal/vrptw", "formal_results/vrptw"),
              norm=("/tmp/p5v2_formal/norm", "formal_results/norm"))
DIAG = ("/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5v2/diag01", "diagnostics/dev01")


def run(args, label):
    print("== " + label)
    proc = subprocess.run([PY, "-B"] + args, cwd=str(V2), capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    out = (proc.stdout or "").strip()
    if out:
        print(out[-3000:])
    if proc.returncode != 0:
        print("!! failed:", (proc.stderr or "")[-2000:])
    return proc.returncode


def main():
    do_fetch = "--no-fetch" not in sys.argv
    if do_fetch:
        for name, (remote, local) in STAGES.items():
            run(["scripts/p5v2_fetch_dir.py", remote, local], "fetch " + name)
        run(["scripts/p5v2_fetch_dir.py"] + list(DIAG), "fetch diag")
    for name, (remote, local) in STAGES.items():
        run(["scripts/p5v2_audit.py", local + "/phase.json", local, local + "/audit.json"],
            "audit " + name)
    run(["scripts/p5v2_audit.py", "external_results/pyvrp_vrptw/../pyvrp_vrptw/phase.json",
         "external_results/pyvrp_vrptw", "external_results/pyvrp_vrptw/audit_ext2.json"],
        "audit ext")
    run(["scripts/p5v2_formal_analysis.py", "formal_results/cvrp", "formal_results/vrptw",
         "formal_results/norm"], "formal analysis")
    run(["scripts/p5v2_ext_analysis.py"], "ext analysis")
    run(["scripts/p5v2_diag_analysis.py", "diagnostics/dev01"], "diag analysis")
    run(["scripts/p5v2_figures.py", "development_results/dev01_final"], "figures")
    # 汇总关键数字
    summary = {}
    for key, path in (("formal", "tables/analysis_formal.json"),
                      ("ext", "tables/analysis_external.json"),
                      ("diag", "tables/analysis_diag.json"),
                      ("dev", "development_results/dev01_final/analysis_dev.json")):
        p = V2 / path
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            summary[key] = data.get("report") or data.get("summary") or list(data)
    print("\n== summary keys")
    print(json.dumps(summary, ensure_ascii=False)[:2000])


if __name__ == "__main__":
    main()
