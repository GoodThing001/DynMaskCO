# -*- coding: utf-8 -*-
"""构建 P4 Solomon 冷链 18 法矩阵（9 例 × 3 种子 × 18 方法 = 486 端点）。"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "P4_solomon_coldchain"

CASES = ["c105", "c106", "c107", "c108", "c109", "c205", "c206", "c207", "c208"]
SEEDS = [67001, 67002, 67003]
METHODS = ["ORDER_SPLIT", "ALNS", "VNS", "NDS-Econ", "AttentionModel", "POMO", "DeepACO", "CaR",
           "PyVRP", "MVMoE", "CaDA", "RouteFinder", "RRNCO", "ORDER_SPLIT_D", "DIST", "DIST_SLOT",
           "MASK", "SLOT"]
INTERPRETERS = {
    "ORDER_SPLIT": "/home/hzeng/envs/MASKCO_env/bin/python",
    "ORDER_SPLIT_D": "/home/hzeng/envs/MASKCO_env/bin/python",
    "DIST": "/home/hzeng/envs/MASKCO_env/bin/python",
    "DIST_SLOT": "/home/hzeng/envs/MASKCO_env/bin/python",
    "MASK": "/home/hzeng/envs/MASKCO_env/bin/python",
    "SLOT": "/home/hzeng/envs/MASKCO_env/bin/python",
    "ALNS": "/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/comparison_v2_20261021/run02/envs/AttentionModel/bin/python",
    "VNS": "/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/comparison_v2_20261021/run02/envs/AttentionModel/bin/python",
    "NDS-Econ": "/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/comparison_v2_20261021/run02/envs/AttentionModel/bin/python",
    "CaR": "/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/comparison_v2_20261021/run02/envs/CaR/bin/python",
    "DeepACO": "/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/comparison_v2_20261021/run02/envs/DeepACO/bin/python",
    "POMO": "/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/comparison_v2_20261021/run02/envs/POMO/bin/python",
    "AttentionModel": "/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/comparison_v2_20261021/run02/envs/AttentionModel/bin/python",
    "RouteFinder": "/home/hzeng/envs/cc_compare/bin/python",
    "CaDA": "/home/hzeng/envs/cc_compare/bin/python",
    "MVMoE": "/home/hzeng/envs/cc_compare/bin/python",
    "RRNCO": "/home/hzeng/envs/cc_compare/bin/python",
    "PyVRP": "/home/hzeng/envs/MASKCO_env/bin/python",
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261127)
    blocks = []
    for c in CASES:
        for s in SEEDS:
            order = METHODS.copy()
            rng.shuffle(order)
            blocks.append(dict(case=c, seed=s, methods=order))
    assert len(blocks) == 27 and sum(len(b["methods"]) for b in blocks) == 486
    phase = dict(name="p4_solomon_coldchain", seconds=60, training_only=False,
                 methods=METHODS, interpreters=INTERPRETERS, blocks=blocks, max_cores=30,
                 schedule_note="P4：9 例 Solomon 冷链投影；种子 67001–67003；方法次序由固定种子 20261127 打乱")
    (OUT / "p4_phase.json").write_text(json.dumps(phase, ensure_ascii=False, indent=2), encoding="utf-8")
    print("P4 matrix:", len(blocks), "blocks,", sum(len(b["methods"]) for b in blocks), "cells")


if __name__ == "__main__":
    main()
