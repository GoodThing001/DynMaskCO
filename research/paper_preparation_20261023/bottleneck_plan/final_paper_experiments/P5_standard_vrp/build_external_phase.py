# -*- coding: utf-8 -*-
"""生成 P5 外部方法批次 phase.json：
  - PyVRP：100 CVRP + 56 VRPTW × 3 种子 = 468 格；
  - 作者模型：CVRP n=100（X-n101-k25）8 法 × 3 种子 = 24 格 +
    VRPTW（56 Solomon）4 法（MVMoE/CaDA/RouteFinder/RRNCO）× 3 种子 = 672 格，共 696 格。
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = "/home/hzeng/envs/MASKCO_env/bin/python"
CVRP_AUTHORS = ["AttentionModel", "POMO", "DeepACO", "CaR", "MVMoE", "CaDA", "RouteFinder"]  # RRNCO 无 CVRP 权重（UNSUPPORTED）
VRPTW_AUTHORS = ["MVMoE", "CaDA", "RouteFinder", "RRNCO"]


def main():
    cvrp = sorted((p.stem for p in (HERE / "data/cvrplib/instances").glob("X-n*.vrp")),
                  key=lambda s: int(s.split("-n")[1].split("-")[0]))
    vrptw = sorted(p.stem for p in (HERE / "data/solomon/instances").glob("*.vrp"))
    assert len(cvrp) == 100 and len(vrptw) == 56
    blocks = []
    for name in cvrp:
        for seed in (68001, 68002, 68003):
            blocks.append(dict(case=name, seed=seed, objective="cvrp", methods=["PyVRP"]))
    for name in vrptw:
        for seed in (69001, 69002, 69003):
            blocks.append(dict(case=name, seed=seed, objective="vrptw", methods=["PyVRP"]))
    phase = dict(
        name="p5_external_pyvrp", seconds=60, training_only=False, methods=["PyVRP"],
        interpreters={"PyVRP": PY}, blocks=blocks, max_cores=12,
        schedule_note="P5 外部方法批次·PyVRP：100 CVRP + 56 VRPTW × 3 种子 = 468 格")
    out = HERE / "phase_external_pyvrp.json"
    out.write_text(json.dumps(phase, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out.name, "blocks", len(blocks), "cells", len(blocks))

    author_blocks = []
    for seed in (68001, 68002, 68003):
        author_blocks.append(dict(case="X-n101-k25", seed=seed, objective="cvrp",
                                  methods=list(CVRP_AUTHORS)))
    for name in vrptw:
        for seed in (69001, 69002, 69003):
            author_blocks.append(dict(case=name, seed=seed, objective="vrptw",
                                      methods=list(VRPTW_AUTHORS)))
    author_methods = sorted(set(CVRP_AUTHORS) | set(VRPTW_AUTHORS))
    interpreters = {m: PY for m in author_methods}
    interpreters["RRNCO"] = "/home/hzeng/envs/cc_compare/bin/python"  # rl4co 生态（rrnco 依赖）
    phase = dict(
        name="p5_external_author", seconds=60, training_only=False, methods=author_methods,
        interpreters=interpreters, blocks=author_blocks, max_cores=8,
        schedule_note="P5 外部方法批次·作者模型：CVRP n=100 8 法 + VRPTW 56 例 4 法 × 3 种子 = 696 格；"
                      "RRNCO 用 cc_compare 环境（rl4co），其余 MASKCO_env")
    out = HERE / "phase_external_author.json"
    out.write_text(json.dumps(phase, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out.name, "blocks", len(author_blocks), "cells", len(author_blocks) * 4)


if __name__ == "__main__":
    main()
