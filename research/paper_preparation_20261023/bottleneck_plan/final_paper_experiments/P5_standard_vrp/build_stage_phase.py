# -*- coding: utf-8 -*-
"""生成 P5-A 正式三阶段 phase.json（内部 6 + ALNS/VNS 共 8 法，100 例 × 3 种子）。

方法支持矩阵见 P5_预登记.md §9：
- 本批运行 8 法（ORDER_SPLIT/ORDER_SPLIT_D/MASK/SLOT/DIST/DIST_SLOT/ALNS/VNS）——
  这些方法对所有 X 规模有效（内部掩码+窗口路径、经典方法无定长限制）。
- 外部方法（PyVRP、AM、POMO、DeepACO、CaR、MVMoE、CaDA、RouteFinder、RRNCO）单独批次运行：
  n=100（X-n101-k25）全部 9 项；n>100 仅 PyVRP。
- 各块内方法顺序按实例轮转（与 P4 相同的抗顺序混淆设计）。
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
METHODS = ["ORDER_SPLIT", "ORDER_SPLIT_D", "MASK", "SLOT", "DIST", "DIST_SLOT", "ALNS", "VNS"]
SEEDS = (68001, 68002, 68003)
PY = "/home/hzeng/envs/MASKCO_env/bin/python"
STAGES = [
    ("stage01", (100, 199), 12),
    ("stage02", (200, 499), 10),
    ("stage03", (500, 1000), 8),
]


def instance_list():
    d = HERE / "data/cvrplib/instances"
    rows = []
    for f in sorted(d.glob("X-n*.vrp")):
        m = re.search(r"X-n(\d+)-k(\d+)", f.name)
        if m:
            rows.append((int(m.group(1)) - 1, f.stem))
    return [name for _, name in sorted(rows)]


def rotate(seq, k):
    return seq[k % len(seq):] + seq[:k % len(seq)]


def build(instances, seed, offset):
    blocks = []
    for i, name in enumerate(instances):
        blocks.append(dict(
            case=name, seed=seed, objective="cvrp",
            methods=rotate(METHODS, offset + i)))
    return blocks


def main():
    names = instance_list()
    assert len(names) == 100, len(names)
    for stage_name, (lo, hi), max_cores in STAGES:
        subset = [n for n in names if lo <= int(re.search(r"X-n(\d+)-", n).group(1)) - 1 <= hi]
        blocks = []
        for si, seed in enumerate(SEEDS):
            blocks += build(subset, seed, si)
        phase = dict(
            name="p5a_" + stage_name,
            seconds=60,
            training_only=False,
            methods=METHODS,
            interpreters={m: PY for m in METHODS},
            blocks=blocks,
            max_cores=max_cores,
            schedule_note=("P5-A 内部8法批次 %s：%d 例（客户 %d–%d）× 3 种子 × 8 方法；"
                           "外部方法另行批次" % (stage_name, len(subset), lo, hi)))
        out = HERE / ("phase_" + stage_name + ".json")
        out.write_text(json.dumps(phase, ensure_ascii=False, indent=2), encoding="utf-8")
        print(out.name, "instances", len(subset), "blocks", len(blocks),
              "cells", len(blocks) * len(METHODS))


if __name__ == "__main__":
    main()
