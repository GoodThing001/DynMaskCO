# -*- coding: utf-8 -*-
"""构建 P1 MASK/SLOT 统一实验矩阵与输入注册表（本地，只读冻结注册表）。"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent / "audit" / "batch_registries"
OUT = HERE.parent / "P1_original_baselines"

CASES = {
    # (case, seed) 列表按统一16法三批次顺序构造；方法与批次种子保持官方身份。
}
BATCHES = [
    ("benchmark01", "benchmark01_heldout_sources.json", [33001, 33002, 33003]),
    ("prospective01", "prospective01_heldout_sources.json", [65001, 65002, 65003]),
    ("retail01", "retail01_heldout_sources.json", [65001, 65002, 65003]),
]

INTERPRETERS = {
    "MASK": "/home/hzeng/envs/MASKCO_env/bin/python",
    "SLOT": "/home/hzeng/envs/MASKCO_env/bin/python",
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    seeds_by_batch = {b: s for b, _, s in BATCHES}
    derived = {}
    case_order = []
    for batch, registry_name, seeds in BATCHES:
        reg = json.loads((AUDIT / registry_name).read_text(encoding="utf-8"))
        cases = sorted(reg["cases"])
        for case in cases:
            if case in derived:
                raise ValueError("cross-batch case collision: " + case)
            entry = reg["derived_instances"][case]
            # 路径统一指向本批 inputs/ 目录（stage 脚本在远端按 SHA 复制）。
            derived[case] = dict(file="inputs/" + Path(entry["file"]).name,
                                 sha256=entry["sha256"], geometry_sha256=entry["geometry_sha256"])
            case_order.append((batch, case))
    rng = np.random.default_rng(20261123)
    blocks = []
    for batch, case in case_order:
        for seed in seeds_by_batch[batch]:
            order = ["MASK", "SLOT"]
            rng.shuffle(order)
            blocks.append(dict(case=case, seed=seed, methods=order))
    # 交叉核查：62 例、186 区组、372 方法槽位
    assert len(derived) == 62, len(derived)
    assert len(blocks) == 186, len(blocks)
    assert sum(len(b["methods"]) for b in blocks) == 372
    assert {(b["case"], b["seed"]) for b in blocks} == {(c, s) for b, c in case_order for s in seeds_by_batch[b]}
    phase = dict(name="p1_mask_slot", seconds=60, training_only=False,
                 methods=["MASK", "SLOT"], interpreters=INTERPRETERS,
                 blocks=blocks, max_cores=24,
                 schedule_note="每区组两方法同物理核、同初解；方法次序由固定种子 20261123 打乱")
    (OUT / "p1_phase.json").write_text(json.dumps(phase, indent=2, ensure_ascii=False), encoding="utf-8")
    registry = dict(status="PASS", cases=sorted(derived),
                    derived_instances=derived,
                    note="62 个统一16法案例的冻结输入注册表；来源为 benchmark01/prospective01/retail01 的 heldout_sources.json")
    (OUT / "p1_heldout_sources.json").write_text(json.dumps(registry, indent=2, ensure_ascii=False), encoding="utf-8")
    print("P1 matrix:", len(blocks), "blocks,", len(derived), "cases, 372 cells")
    from collections import Counter
    print("per-case seeds:", set(Counter((b["case"], b["seed"]) for b in blocks).values()))
    print("batch counts:", Counter(b for b, _ in case_order))


if __name__ == "__main__":
    main()
