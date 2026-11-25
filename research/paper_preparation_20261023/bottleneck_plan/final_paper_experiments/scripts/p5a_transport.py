# -*- coding: utf-8 -*-
"""构建 P5-A 冒烟矩阵（2 例 × 3 种子 × 6 本文方法 = 36 端点）并提交远端。"""
import json
import shlex
import sys
import tarfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

PYTHON = "/home/hzeng/envs/MASKCO_env/bin/python"
P5DIR = HERE.parent / "P5_standard_vrp"
REMOTE = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5"

INTERNAL = ["ORDER_SPLIT", "ORDER_SPLIT_D", "MASK", "SLOT", "DIST", "DIST_SLOT"]


def build_phase(smoke=True):
    cases = ["X-n101-k25", "X-n106-k14"] if smoke else []
    seeds = [68001, 68002, 68003]
    rng = np.random.default_rng(20261128)
    blocks = []
    for c in cases:
        for s in seeds:
            order = INTERNAL.copy()
            rng.shuffle(order)
            blocks.append(dict(case=c, seed=s, objective="cvrp", methods=order))
    phase = dict(name="p5a_smoke" if smoke else "p5a", seconds=60, training_only=False,
                 methods=INTERNAL,
                 interpreters={m: PYTHON for m in INTERNAL},
                 blocks=blocks, max_cores=12,
                 schedule_note="P5-A 冒烟：n=100（X-n101-k25）与窗口（X-n106-k14）双路径验证")
    (P5DIR / "p5a_smoke_phase.json").write_text(json.dumps(phase, ensure_ascii=False, indent=2), encoding="utf-8")
    return phase


def prepare():
    phase = build_phase()
    import ast
    for f in ("p5_worker.py",):
        ast.parse((P5DIR / f).read_text(encoding="utf-8"))
    with tarfile.open(P5DIR / "p5a_smoke.tar.gz", "w:gz") as z:
        z.add(P5DIR / "p5a_smoke_phase.json", arcname="phase.json")
        z.add(P5DIR / "p5_worker.py", arcname="unified_worker.py")
        z.add(P5DIR / "p5_worker.py", arcname="future_dispatch.py")
        z.add(HERE.parent / "P4_solomon_coldchain" / "unified_runner.py", arcname="unified_runner.py")
    print("prepared", len(phase["blocks"]), "blocks,", sum(len(b["methods"]) for b in phase["blocks"]), "cells")


def ship_standard():
    from p5_ship_standard import SRC, REMOTE as RR
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        with client.open_sftp() as s:
            def makedirs(path):
                try:
                    s.mkdir(path)
                except OSError:
                    pass
            makedirs(RR)
            for f in sorted(SRC.rglob("*.py")):
                rel = f.relative_to(SRC).as_posix()
                cur = RR
                for part in rel.split("/")[:-1]:
                    cur = cur + "/" + part
                    makedirs(cur)
                s.put(str(f), RR + "/" + rel)
        print("standard_vrp shipped")
    finally:
        client.close()


def submit():
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        with client.open_sftp() as s:
            s.put(str(P5DIR / "p5a_smoke.tar.gz"), REMOTE + "/dispatch.tar.gz")
            cmd = "cd " + REMOTE + " && rm -rf smoke01 && mkdir -p smoke01 && cd smoke01 && tar -xzf ../dispatch.tar.gz && " + \
                  PYTHON + " -B unified_runner.py > controller.log 2>&1; echo $? > exit_code"
            out = transport.remote_exec(client, "tmux new-session -d -s p5a_smoke " + shlex.quote(cmd))
            print("submitted:", out)
    finally:
        client.close()


def follow(wait=120):
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        with client.open_sftp() as s:
            while True:
                try:
                    with s.open(REMOTE + "/smoke01/exit_code") as f:
                        code = int(f.read())
                    break
                except FileNotFoundError:
                    try:
                        with s.open(REMOTE + "/smoke01/status.json") as f:
                            st = json.load(f)
                        print({k: st[k] for k in ("cells", "total_cells", "complete", "failed")}, flush=True)
                    except FileNotFoundError:
                        print("staging", flush=True)
                    time.sleep(wait)
            print("exit", code)
            if code:
                with s.open(REMOTE + "/smoke01/controller.log") as f:
                    print(f.read().decode("utf-8", "replace")[-3000:])
                return code
            out_dir = P5DIR / "results_smoke"
            out_dir.mkdir(exist_ok=True)
            import tarfile as tf
            transport.remote_exec(client, "cd " + REMOTE + "/smoke01 && tar -czf receipts.tar.gz --ignore-failed-read out status.json phase.json controller.log")
            s.get(REMOTE + "/smoke01/receipts.tar.gz", str(out_dir / "receipts.tar.gz"))
            with tf.open(out_dir / "receipts.tar.gz") as z:
                for m in z:
                    p = (out_dir / m.name).resolve()
                    if not p.is_relative_to(out_dir.resolve()):
                        raise ValueError(m.name)
                    if m.isdir():
                        p.mkdir(parents=True, exist_ok=True)
                    elif m.isfile():
                        p.write_bytes(z.extractfile(m).read())
            print("recovered", out_dir)
            return code
    finally:
        client.close()


if __name__ == "__main__":
    action = sys.argv[1]
    if action == "prepare":
        prepare()
    elif action == "submit":
        prepare()
        ship_standard()
        submit()
    elif action == "follow":
        follow()
    else:
        raise ValueError(action)
