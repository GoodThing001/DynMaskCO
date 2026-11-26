# -*- coding: utf-8 -*-
"""在 tools/git_deliver.py 的 MANIFEST 中加入 P5-V2 交付组（只加不改旧组）。"""
from pathlib import Path

p = Path("tools/git_deliver.py")
t = p.read_text(encoding="utf-8")
marker = """        {
            "name": "p5_standard_vrp","""
group = '''        {
            "name": "p5_standard_vrp_v2",
            "note": "P5-V2 标准任务源码纠错/机制恢复/公平性审计/受控复验：报告、协议、"
                    "审计证据、V2 求解器与测试、开发/正式/外部/诊断端点、表 V2-1…V2-10、图 F01–F12",
            "root": "research/paper_preparation_20261023/bottleneck_plan/final_paper_experiments/P5_standard_vrp_v2",
            "include": ("*.md", "scripts/*.py", "standard_vrp/core/*.py", "standard_vrp/core/*.json",
                        "standard_vrp/solver/*.py", "standard_vrp/adapters/*.py",
                        "standard_vrp/tests/*.py",
                        "configs/*.json", "manifests/*.json", "audit/*.md", "audit/*.json",
                        "audit/*.csv",
                        "development/manifest.json", "development/instances/*.vrp",
                        "development_results/*/phase.json", "development_results/*/status.json",
                        "development_results/*/tables_dev.md",
                        "development_results/*/analysis_dev.json",
                        "development_results/*/audit_dev.json",
                        "development_results/*/out/*.json",
                        "formal_results/*/phase.json", "formal_results/*/status.json",
                        "formal_results/*/audit.json",
                        "formal_results/*/out/*.json",
                        "external_results/*/phase.json", "external_results/*/audit_ext.json",
                        "external_results/*/run*/**/*.json",
                        "diagnostics/*/status.json", "diagnostics/*/out/*.json",
                        "tables/*.csv", "tables/*.json", "tables/*.xlsx",
                        "tables/md/*.md", "tables/latex/*.tex",
                        "figures/*.png", "figures/*.pdf",
                        "data/cvrplib/instances/*.vrp", "data/cvrplib/bks/*.sol",
                        "data/cvrplib/bks_derived.json",
                        "data/solomon/instances/*.vrp", "data/solomon/bks/*.sol",
                        "data/solomon_bks_derived.json"),
        },
'''
assert marker in t, "p5 anchor not found"
if "p5_standard_vrp_v2" in t:
    raise SystemExit("group already present")
t = t.replace(marker, group + marker, 1)
p.write_text(t, encoding="utf-8")
import ast
ast.parse(t)
print("manifest group p5_standard_vrp_v2 added")
