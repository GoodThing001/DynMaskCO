# -*- coding: utf-8 -*-
"""P5 定稿流水线：回收远端收据 → 独立审计 → 全量分析 → 表12–16 → 八图。

用法：python p5_finalize.py [recover|audit|analyze|all]
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
P5 = HERE.parent / "P5_standard_vrp"
PY = sys.executable

WORKSPACES = {
    "stage01": "results_stage01", "stage02": "results_stage02", "stage03": "results_stage03",
    "external_pyvrp3": "results_external_pyvrp3", "external_author2": "results_external_author",
}


def run(mod, *args):
    cmd = [PY, "-B", str(HERE / mod), *args]
    print(">>", " ".join(cmd[2:]))
    out = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    tail = [l for l in (out.stdout or "").splitlines() if l.strip()][-6:]
    for l in tail:
        print("   ", l)
    if out.returncode != 0:
        print("   STDERR:", (out.stderr or "")[-800:])
        raise SystemExit(1)


def recover():
    for ws, dst in WORKSPACES.items():
        run("p5a_recover.py", ws, dst.replace("results_", ""))


def audit():
    for dst in WORKSPACES.values():
        run("p5a_audit.py", str(P5 / dst))


def analyze():
    p5a = ",".join(str(P5 / d) for d in ("results_stage01", "results_stage02", "results_stage03",
                                          "results_external_pyvrp3", "results_external_author"))
    run("p5a_analysis.py", p5a, "analysis_p5a.json")
    p5b = ",".join(str(P5 / d) for d in ("results_vrptw_full", "results_external_author"))
    run("p5b_analysis.py", p5b, "analysis_p5b.json")


def tables():
    run("tables_p5.py")
    run("p5_figures.py")


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "all"
    if action in ("recover", "all"):
        recover()
    if action in ("audit", "all"):
        audit()
    if action in ("analyze", "all"):
        analyze()
    if action in ("tables", "all"):
        tables()
    print("P5 FINALIZE", action.upper(), "DONE")
