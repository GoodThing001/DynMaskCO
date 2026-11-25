# -*- coding: utf-8 -*-
"""构建 P3 HYBRID/HYBRID_D 矩阵与注册表（26 前瞻案例 × 3 种子 × 2 方法 = 156 端点）。"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent / "audit" / "batch_registries"
OUT = HERE.parent / "P3_factorial_ablation"

BATCHES = [
    ("prospective01", "prospective01_heldout_sources.json", [65001, 65002, 65003],
     lambda c: c.startswith("vrppdl_") or c.startswith("xtw_")),
    ("retail01", "retail01_heldout_sources.json", [65001, 65002, 65003], lambda c: c.startswith("rw_")),
]

INTERPRETERS = {
    "HYBRID": "/home/hzeng/envs/MASKCO_env/bin/python",
    "HYBRID_D": "/home/hzeng/envs/MASKCO_env/bin/python",
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    seeds_by_batch = {b: s for b, _, s, _ in BATCHES}
    derived = {}
    case_order = []
    for batch, registry_name, seeds, keep in BATCHES:
        reg = json.loads((AUDIT / registry_name).read_text(encoding="utf-8"))
        for case in sorted(reg["cases"]):
            if not keep(case):
                continue
            entry = reg["derived_instances"][case]
            derived[case] = dict(file="inputs/" + Path(entry["file"]).name,
                                 sha256=entry["sha256"], geometry_sha256=entry["geometry_sha256"])
            case_order.append((batch, case))
    rng = np.random.default_rng(20261124)
    blocks = []
    for batch, case in case_order:
        for seed in seeds_by_batch[batch]:
            order = ["HYBRID", "HYBRID_D"]
            rng.shuffle(order)
            blocks.append(dict(case=case, seed=seed, methods=order))
    assert len(derived) == 26, len(derived)
    assert len(blocks) == 78 and sum(len(b["methods"]) for b in blocks) == 156
    assert {(b["case"], b["seed"]) for b in blocks} == {(c, s) for b, c in case_order for s in seeds_by_batch[b]}
    phase = dict(name="p3_factorial", seconds=60, training_only=False,
                 methods=["HYBRID", "HYBRID_D"], interpreters=INTERPRETERS,
                 blocks=blocks, max_cores=24,
                 schedule_note="每区组两方法同物理核、同初解；方法次序由固定种子 20261124 打乱")
    (OUT / "p3_phase.json").write_text(json.dumps(phase, indent=2, ensure_ascii=False), encoding="utf-8")
    registry = dict(status="PASS", cases=sorted(derived), derived_instances=derived,
                    note="26 个前瞻案例（VRPPDL 10 + 新XTW 8 + RWVRP 8）的冻结输入注册表")
    (OUT / "p3_heldout_sources.json").write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
    from collections import Counter
    print("P3 matrix:", len(blocks), "blocks,", len(derived), "cases, 156 cells;",
          "batch:", Counter(b for b, _ in case_order))


if __name__ == "__main__":
    main()
