# -*- coding: utf-8 -*-
"""生成 P5-B phase.json：Solomon 56 例 × 3 种子 × 8 法（内部 6 + ALNS/VNS），字典序 (K,D)。

- 外部 VRPTW 方法（PyVRP、MVMoE、CaDA、RouteFinder、RRNCO）另行批次（P5_预登记 §9：14/18）。
- smoke 模式：仅 C101（VRPTW 神经编码路径预检）；full 模式：56 例。
- 块内方法顺序轮转（抗顺序混淆）；种子 69001–69003。
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
METHODS = ["ORDER_SPLIT", "ORDER_SPLIT_D", "MASK", "SLOT", "DIST", "DIST_SLOT", "ALNS", "VNS"]
SEEDS = (69001, 69002, 69003)
PY = "/home/hzeng/envs/MASKCO_env/bin/python"


def rotate(seq, k):
    return seq[k % len(seq):] + seq[:k % len(seq)]


def solomon_names():
    d = HERE / "data/solomon/instances"
    return sorted(f.stem for f in d.glob("*.vrp"))


def main():
    names = solomon_names()
    assert len(names) == 56, len(names)
    for mode, subset, cores in (("smoke", ["C101"], 3), ("full", names, 12)):
        blocks = []
        for name in subset:
            for si, seed in enumerate(SEEDS):
                blocks.append(dict(case=name, seed=seed, objective="vrptw",
                                   methods=rotate(METHODS, si + (0 if mode == "smoke" else
                                                                 sorted(names).index(name)))))
        phase = dict(
            name="p5b_" + mode,
            seconds=60,
            training_only=False,
            methods=METHODS,
            interpreters={m: PY for m in METHODS},
            blocks=blocks,
            max_cores=cores,
            schedule_note="P5-B %s：Solomon %d 例 × 3 种子 × 8 法（字典序 K,D）；外部方法另行批次"
                          % ("冒烟" if mode == "smoke" else "正式", len(subset)))
        (HERE / ("phase_vrptw_" + mode + ".json")).write_text(
            json.dumps(phase, ensure_ascii=False, indent=2), encoding="utf-8")
        print(("phase_vrptw_" + mode + ".json"), "instances", len(subset),
              "blocks", len(blocks), "cells", len(blocks) * len(METHODS))


if __name__ == "__main__":
    main()
