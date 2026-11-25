# -*- coding: utf-8 -*-
"""构建 P2 八城十种子轨迹矩阵（8 城 × 10 种子 × 3 方法 = 240 终点）。

预登记（2026-11-26，在任何新增端点查看之前固定）：
  - 保留前瞻种子 65001/65002/65003（与零售三批登记一致），新增 65004–65010；
  - 方法：ORDER_SPLIT、ORDER_SPLIT_D、ALNS；
  - 原三种子的零售端点未记录 5/15/30/60 秒轨迹，不能严格复用 → 十种子全部本批重跑，
    总计 240 个终点（含轨迹），原零售三批 384 端点保持冻结不动。
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent / "audit" / "batch_registries"
OUT = HERE.parent / "P2_retail_analysis"

CITIES = ["rw_zg170_p100", "rw_br100_p100", "rw_cp150_p100", "rw_ru120_p100",
          "rw_sg200_p100", "rw_me340_p100", "rw_ca240_p100", "rw_sf300_p100"]
SEEDS = [65001, 65002, 65003, 65004, 65005, 65006, 65007, 65008, 65009, 65010]
METHODS = ["ORDER_SPLIT", "ORDER_SPLIT_D", "ALNS"]

INTERPRETERS = {
    "ORDER_SPLIT": "/home/hzeng/envs/MASKCO_env/bin/python",
    "ORDER_SPLIT_D": "/home/hzeng/envs/MASKCO_env/bin/python",
    "ALNS": "/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/comparison_v2_20261021/run02/envs/AttentionModel/bin/python",
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    reg = json.loads((AUDIT / "retail01_heldout_sources.json").read_text(encoding="utf-8"))
    assert sorted(reg["cases"]) == sorted(CITIES)
    derived = {c: reg["derived_instances"][c] for c in CITIES}
    rng = np.random.default_rng(20261125)
    blocks = []
    pairs = [(c, s) for c in CITIES for s in SEEDS]
    rng.shuffle(pairs)
    order = METHODS.copy()
    rng.shuffle(order)
    for i, (c, s) in enumerate(pairs):
        blocks.append(dict(case=c, seed=s, methods=order[i % 3:] + order[:i % 3]))
    assert len(blocks) == 80 and sum(len(b["methods"]) for b in blocks) == 240
    phase = dict(seconds=60, training_only=False, name="p2_retail_trajectory", methods=METHODS,
                 max_cores=24, interpreters=INTERPRETERS, blocks=blocks,
                 schedule_note="P2 八城十种子：固定种子集 65001-65010；方法次序由固定种子 20261125 打乱")
    (OUT / "p2_phase.json").write_text(json.dumps(phase, ensure_ascii=False, indent=2), encoding="utf-8")
    registry = dict(status="PASS", cases=sorted(derived), derived_instances=derived,
                    note="RWVRP 八城冻结输入注册表（沿用 retail01 heldout_sources）")
    (OUT / "p2_heldout_sources.json").write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    print("P2 matrix:", len(blocks), "blocks, 240 cells, seeds", SEEDS)


if __name__ == "__main__":
    main()
