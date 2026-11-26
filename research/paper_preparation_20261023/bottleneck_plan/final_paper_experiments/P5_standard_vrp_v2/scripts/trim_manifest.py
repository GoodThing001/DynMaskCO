import ast
import shutil
from pathlib import Path

ROOT = Path(r"D:\PyCharm_\MASKCO-Main")
q = ROOT / "tools/git_deliver.py"
t = q.read_text(encoding="utf-8")
repl = [
    ('"development_results/*/endpoints/*.json", "development_results/*/out/*.json",',
     '"development_results/*/out/*.json",'),
    ('"diagnostics/*/status.json", "diagnostics/*/endpoints/*.json", "diagnostics/*/out/*.json",',
     '"diagnostics/*/status.json", "diagnostics/*/out/*.json",'),
    ('''                        "data/cvrplib/instances/*.vrp", "data/cvrplib/bks/*.sol",
                        "data/cvrplib/bks_derived.json",
                        "data/solomon/instances/*.vrp", "data/solomon/bks/*.sol",
                        "data/solomon_bks_derived.json"),''',
     '''                        "data/bks_derived.json", "data/solomon_bks_derived.json"),'''),
]
for old, new in repl:
    assert old in t, old[:60]
    t = t.replace(old, new, 1)
q.write_text(t, encoding="utf-8")
ast.parse(t)
print("manifest trimmed")

V2 = ROOT / ("research/paper_preparation_20261023/bottleneck_plan/final_paper_experiments/"
             "P5_standard_vrp_v2")
removed = 0
for root in ("development_results", "diagnostics"):
    for d in (V2 / root).rglob("endpoints"):
        shutil.rmtree(d)
        removed += 1
print("removed", removed, "endpoint mirror dirs")
