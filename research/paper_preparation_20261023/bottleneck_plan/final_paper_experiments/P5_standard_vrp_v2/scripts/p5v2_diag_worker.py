# -*- coding: utf-8 -*-
"""固定候选数量诊断的远端执行器：main(idx, "DIAG") → out/{idx}_DIAG.json。

复用 unified_runner（单核 taskset），内部调用 p5v2_fixed_candidates 的四种排序诊断。
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
P5V2 = Path("/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5v2")
sys.path.insert(0, str(P5V2))


def check(row, method, block, affinity, initial=None):
    if row["case"] != block["case"] or row["seed"] != block["seed"]:
        raise ValueError("diagnostic cell identity mismatch")
    if len(affinity) != 1:
        raise ValueError("single core required")
    if method != "DIAG":
        raise ValueError("unregistered diagnostic method")
    return True


def main(idx, method):
    cfg = json.loads((HERE / "phase.json").read_text(encoding="utf-8"))
    block = cfg["blocks"][int(idx)]
    if method != "DIAG" or method not in block["methods"]:
        raise ValueError("unregistered diagnostic method")
    out = HERE / "out"
    out.mkdir(exist_ok=True)
    dest = out / ("%s_DIAG.json" % idx)
    if dest.exists():
        return
    sys.path.insert(0, str(HERE))
    import importlib.util as iu
    spec = iu.spec_from_file_location("fixeddiag", str(HERE / "p5v2_fixed_candidates.py"))
    mod = iu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    tmp = out / ("%s_DIAG.tmp" % idx)
    try:
        mod.main(str(HERE / "instances"), str(HERE / "manifest.json"),
                 block["case"], block["seed"], str(tmp))
    except Exception:  # noqa: BLE001
        (out / ("%s_DIAG.error.txt" % idx)).write_text(traceback.format_exc(), encoding="utf-8")
        raise
    tmp.replace(dest)
    row = json.loads(dest.read_text(encoding="utf-8"))
    check(row, method, block, sorted(os.sched_getaffinity(0)))


if __name__ == "__main__":
    main(*sys.argv[1:])
