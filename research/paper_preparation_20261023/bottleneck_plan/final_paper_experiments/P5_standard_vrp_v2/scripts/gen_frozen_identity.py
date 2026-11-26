# -*- coding: utf-8 -*-
"""生成 P5-V2 冻结身份与源码 SHA256 清单（manifests/frozen_identity.json）。

记录：起始 commit、冷链锁版源码与 P5-V1 源码的 SHA256、数据清单身份。
"""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(r"D:\PyCharm_\MASKCO-Main")
V2 = ROOT / "research/paper_preparation_20261023/bottleneck_plan/final_paper_experiments/P5_standard_vrp_v2"
LOCK = ROOT / "research/paper_preparation_20261023/bottleneck_plan/maskco_perishable_external_20261119/reserved_validation/solver"
V1 = ROOT / "research/paper_preparation_20261023/bottleneck_plan/final_paper_experiments/P5_standard_vrp"
FROZEN_PROFIT = ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_profit"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", errors="replace").stdout.strip()


def main():
    obj = {
        "start_commit": git("rev-parse", "HEAD"),
        "start_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "remote_branch_sha": git("rev-parse", "origin/codex/o0cc-cal-phys"),
        "main_branch_sha": git("rev-parse", "origin/main"),
        "frozen_coldchain_source": {},
        "p5_v1_source": {},
        "frozen_profit_source": {},
    }
    for p in sorted(LOCK.glob("*.py")) + sorted(LOCK.glob("*.cpp")) + sorted(LOCK.glob("*.json")):
        obj["frozen_coldchain_source"][p.name] = sha(p)
    for p in [V1 / "p5_worker.py", V1 / "p5_external_worker.py",
              V1 / "standard_vrp/core/standard_evaluator.py",
              V1 / "standard_vrp/core/standard_features.py",
              V1 / "standard_vrp/solver/standard_pool_search.py",
              V1 / "standard_vrp/solver/standard_classical.py",
              V1 / "P5_预登记.md"]:
        if p.exists():
            obj["p5_v1_source"][p.relative_to(V1).as_posix()] = sha(p)
    for p in sorted(FROZEN_PROFIT.glob("*.py")):
        obj["frozen_profit_source"][p.name] = sha(p)
    out = V2 / "manifests/frozen_identity.json"
    out.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved", out)
    print("start commit:", obj["start_commit"], obj["start_branch"])


if __name__ == "__main__":
    main()
