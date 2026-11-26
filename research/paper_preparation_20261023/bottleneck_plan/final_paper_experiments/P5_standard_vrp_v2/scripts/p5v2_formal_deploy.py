# -*- coding: utf-8 -*-
"""P5-V2 正式复验面板：phase 生成 + 部署 + 顺序启动。

面板组成（EXPERIMENT_PROTOCOL_V2 §4）：
  cvrp : Uchoa X 100 例 × 种子 68001/68002/68003 × 6 内部组合 = 1800 端点
  vrptw: Solomon 56 例 × 种子 69001/69002/69003 × 6 内部组合 = 1008 端点
  norm : 同上 156 例 × 3 种子 × STD_NORM = 468 端点
合计 3276 端点（另 240 开发面板 + 336 外部基线 = 3852）。

落地：远端 /tmp/p5v2_formal/<stage>（/home 仅剩 15MB，结果写根分区）。
用法：
  python p5v2_formal_deploy.py [build|deploy|launch|all]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V2 = HERE.parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

PY = "/home/hzeng/envs/MASKCO_env/bin/python"
P5 = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5"
P5V2 = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5v2"
FORMAL = "/tmp/p5v2_formal"
INTERNAL = ["ORDER_SPLIT", "ORDER_SPLIT_D", "MASK", "SLOT", "DIST", "DIST_SLOT"]
SEEDS = dict(cvrp=(68001, 68002, 68003), vrptw=(69001, 69002, 69003))
LANES = dict(cvrp=20, vrptw=14, norm=10)


def instances():
    local = V2 / "data"
    cvrp = sorted(p.stem for p in (local / "cvrplib/instances").glob("*.vrp"))
    vrptw = sorted(p.stem for p in (local / "solomon/instances").glob("*.vrp"))
    if len(cvrp) != 100 or len(vrptw) != 56:
        raise SystemExit("instance sets incomplete: %d/%d" % (len(cvrp), len(vrptw)))
    return cvrp, vrptw


def interleave(names):
    """按目标规模交错，避免尾部堆积大实例。"""
    def n_of(name):
        digits = "".join(ch if ch.isdigit() else " " for ch in name).split()
        return max((int(d) for d in digits), default=0)
    ordered = sorted(names, key=n_of)
    out, lo, hi = [], 0, len(ordered) - 1
    while lo <= hi:
        out.append(ordered[hi])
        if lo != hi:
            out.append(ordered[lo])
        lo += 1
        hi -= 1
    return out


def build():
    cvrp, vrptw = instances()
    phases = {}
    for kind, names, root in (("cvrp", cvrp, P5 + "/cvrplib/instances"),
                              ("vrptw", vrptw, P5 + "/solomon/instances")):
        blocks = []
        for name in interleave(names):
            for seed in SEEDS[kind]:
                methods = list(INTERNAL)
                order = seed % len(methods)
                methods = methods[order:] + methods[:order]
                blocks.append(dict(case=name, seed=seed, objective=kind,
                                   path=root + "/" + name + ".vrp", methods=methods))
        phases[kind] = dict(name="p5v2_formal_" + kind, seconds=60, training_only=False,
                            methods=INTERNAL, blocks=blocks, max_cores=LANES[kind],
                            interpreters={m: PY for m in INTERNAL},
                            schedule_note="正式复验：%d 例 × 3 种子 × 6 组合" % len(names))
    norm_blocks = []
    for kind, names, root in (("cvrp", cvrp, P5 + "/cvrplib/instances"),
                              ("vrptw", vrptw, P5 + "/solomon/instances")):
        for name in interleave(names):
            for seed in SEEDS[kind]:
                norm_blocks.append(dict(case=name, seed=seed, objective=kind,
                                        path=root + "/" + name + ".vrp",
                                        methods=["STD_NORM"]))
    phases["norm"] = dict(name="p5v2_formal_norm", seconds=60, training_only=False,
                          methods=["STD_NORM"], blocks=norm_blocks, max_cores=LANES["norm"],
                          interpreters={"STD_NORM": PY},
                          schedule_note="STD_NORM 迁移策略：156 例 × 3 种子")
    for stage, phase in phases.items():
        out = V2 / "configs" / ("phase_formal_%s.json" % stage)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(phase, ensure_ascii=False, indent=2), encoding="utf-8")
        print("%-6s blocks=%4d cells=%5d" % (stage, len(phase["blocks"]),
                                            len(phase["blocks"]) * len(phase["methods"])))
    total = sum(len(p["blocks"]) * len(p["methods"]) for p in phases.values())
    print("total cells", total, "core-hours %.1f" % (total * 60 / 3600))
    return phases


def deploy(phases):
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        with client.open_sftp() as s:
            def makedirs(path):
                parts = path.strip("/").split("/")
                cur = ""
                for part in parts:
                    cur += "/" + part
                    try:
                        s.mkdir(cur)
                    except OSError:
                        pass

            makedirs(P5V2)
            for f in sorted((V2 / "standard_vrp").rglob("*.py")):
                rel = f.relative_to(V2).as_posix()
                target = P5V2 + "/" + rel
                makedirs(target.rsplit("/", 1)[0])
                s.put(str(f), target)
            for stage, phase in phases.items():
                d = FORMAL + "/" + stage
                makedirs(d)
                s.put(str(V2 / "configs" / ("phase_formal_%s.json" % stage)), d + "/phase.json")
                s.put(str(HERE / "p5v2_worker.py"), d + "/unified_worker.py")
                s.put(str(HERE / "p5v2_worker.py"), d + "/future_dispatch.py")
            print("deployed")
    finally:
        client.close()


def launch(stages=("cvrp", "vrptw", "norm")):
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        for i, stage in enumerate(stages):
            d = FORMAL + "/" + stage
            cmd = ("mkdir -p " + d + " && cd " + d + " && "
                   "cp " + P5 + "/stage01/unified_runner.py . 2>/dev/null; "
                   "rm -rf out status.json runner.log; "
                   "setsid nohup " + PY + " -B unified_runner.py > runner.log 2>&1 < /dev/null & "
                   "sleep 3; ls -1 " + d + " | tr '\\n' ' '")
            print(stage, "->", transport.remote_exec(client, cmd))
            if i + 1 < len(stages):
                transport.remote_exec(client, "sleep 25")
    finally:
        client.close()


def main(what="all"):
    if what in ("build", "all"):
        phases = build()
    else:
        phases = {s: json.loads((V2 / "configs" / ("phase_formal_%s.json" % s))
                                .read_text(encoding="utf-8"))
                  for s in ("cvrp", "vrptw", "norm")}
    if what in ("deploy", "all"):
        deploy(phases)
    if what in ("launch", "all"):
        launch()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "all")
