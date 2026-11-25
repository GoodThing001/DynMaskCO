# -*- coding: utf-8 -*-
"""通用远端实验传输器：prepare/submit(含冒烟)/follow/recover。

prepare  本地构建 dispatch tarball（worker 以 unified_worker.py 与 future_dispatch.py 双份部署）。
submit   远端冻结源码身份核对 + 62 例输入 SHA 预检 + 单次 tmux 启动。
smoke    仅提交 2 个区组到 smoke 子目录，人工核验通过后再 submit 完整批次。
follow   轮询 exit_code/status.json。
recover  回收 receipts 到 results/<name>。
"""
import ast
import json
import shlex
import sys
import tarfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

PYTHON = "/home/hzeng/envs/MASKCO_env/bin/python"
PROJECT = "/home/hzeng/project/MASKCO-Main"

FROZEN = [
    ("_maskco_cross_dataset_20261119/compare.py",
     HERE.parent.parent / "maskco_cross_dataset_20261119/compare.py"),
    ("_maskco_distance_slot_20261119/worker.py",
     HERE.parent.parent / "maskco_distance_slot_20261119/worker.py"),
    ("_maskco_distance_slot_20261119/operators.py",
     HERE.parent.parent / "maskco_distance_slot_20261119/operators.py"),
]

BATCH_DIRS = {"benchmark01": "_maskco_unified_external_20261120/benchmark01",
              "prospective01": "_maskco_unified_external_20261120/prospective01",
              "retail01": "_maskco_unified_external_20261120/retail01"}


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def prepare(name, pkg_dir):
    phase_src = pkg_dir / "p1_phase.json"
    phase = read(phase_src)
    files = [("p1_phase.json", "phase.json"), ("p1_worker.py", "unified_worker.py"),
             ("p1_worker.py", "future_dispatch.py"), ("stage_p1.py", "stage_p1.py"),
             ("unified_runner.py", "unified_runner.py"), ("p1_heldout_sources.json", "heldout_sources.json")]
    for src, _ in files:
        if src.endswith(".py"):
            ast.parse((pkg_dir / src).read_text(encoding="utf-8"))
    with tarfile.open(pkg_dir / (name + ".tar.gz"), "w:gz") as tar:
        for src, arc in files:
            tar.add(pkg_dir / src, arcname=arc)
    print("prepared", name, len(phase["blocks"]), "blocks,", sum(len(b["methods"]) for b in phase["blocks"]), "cells")


def remote_python(client, code):
    return transport.remote_exec(client, PYTHON + " -B -c " + shlex.quote(code))


def frozen_identity(client, sftp):
    for remote_rel, local in FROZEN:
        with sftp.open(PROJECT + "/" + remote_rel, "rb") as f:
            if f.read() != local.read_bytes():
                raise ValueError("Frozen source differs at " + remote_rel)
    print("frozen identity OK")


def submit(name, pkg_dir, smoke=False):
    transport.record = lambda *a, **k: None
    phase = read(pkg_dir / "p1_phase.json")
    smoke_phase = None
    if smoke:
        blocks = phase["blocks"][:2]
        smoke_phase = dict(phase, name="smoke01", blocks=blocks)
        with tarfile.open(pkg_dir / "smoke01.tar.gz", "w:gz") as tar:
            tar.add(pkg_dir / "p1_worker.py", arcname="unified_worker.py")
            tar.add(pkg_dir / "p1_worker.py", arcname="future_dispatch.py")
            tar.add(pkg_dir / "stage_p1.py", arcname="stage_p1.py")
            tar.add(pkg_dir / "unified_runner.py", arcname="unified_runner.py")
            tar.add(pkg_dir / "p1_heldout_sources.json", arcname="heldout_sources.json")
            tmp = pkg_dir / "smoke_phase.json"
            tmp.write_text(json.dumps(smoke_phase, ensure_ascii=False, indent=2), encoding="utf-8")
            tar.add(tmp, arcname="phase.json")
        tmp.unlink()
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        with client.open_sftp() as s:
            frozen_identity(client, s)
            base = PROJECT + "/_maskco_final_paper_p1"
            try:
                s.mkdir(base)
            except OSError:
                pass
            target = base + ("/smoke01" if smoke else "/p1")
            transport.remote_exec(client, "rm -rf " + target + " && mkdir -p " + target)
            tarball = "smoke01.tar.gz" if smoke else name + ".tar.gz"
            s.put(str(pkg_dir / tarball), target + "/dispatch.tar.gz")
            command = "cd " + target + " && (" + PYTHON + " -B stage_p1.py && " + PYTHON + \
                      " -B unified_runner.py) > controller.log 2>&1; echo $? > exit_code"
            session = "maskco_final_paper_" + ("p1_smoke" if smoke else "p1")
            transport.remote_exec(client,
                                  f"cd {target} && tar -xzf dispatch.tar.gz && tmux new-session -d -s {session} {shlex.quote(command)}")
            print("submitted", "smoke01" if smoke else name)
    finally:
        client.close()


def follow(name, pkg_dir, smoke=False, wait=30):
    transport.record = lambda *a, **k: None
    target = PROJECT + "/_maskco_final_paper_p1/" + ("smoke01" if smoke else "p1")
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        with client.open_sftp() as s:
            while True:
                try:
                    with s.open(target + "/exit_code") as f:
                        code = int(f.read())
                    break
                except FileNotFoundError:
                    try:
                        with s.open(target + "/status.json") as f:
                            state = json.load(f)
                        print({k: state[k] for k in ("complete", "cells", "total_cells", "failed", "end")}, flush=True)
                    except FileNotFoundError:
                        print("staging/queued", flush=True)
                    time.sleep(wait)
            print("exit", code)
            if code:
                with s.open(target + "/controller.log") as f:
                    print(f.read().decode("utf-8", "replace")[-4000:])
                return code
            out = HERE.parent / "P1_original_baselines" / ("results_smoke" if smoke else "results")
            out.mkdir(parents=True, exist_ok=True)
            transport.remote_exec(client, "cd " + target +
                                  " && tar -czf receipts.tar.gz --ignore-failed-read out status.json phase.json controller.log resources.json exit_code")
            s.get(target + "/receipts.tar.gz", str(out / "receipts.tar.gz"))
            with tarfile.open(out / "receipts.tar.gz") as z:
                for m in z:
                    p = (out / m.name).resolve()
                    if not p.is_relative_to(out.resolve()):
                        raise ValueError(m.name)
                    if m.isdir():
                        p.mkdir(parents=True, exist_ok=True)
                    elif m.isfile():
                        p.write_bytes(z.extractfile(m).read())
            print("recovered", out)
            return code
    finally:
        client.close()


if __name__ == "__main__":
    action = sys.argv[1]
    pkg = HERE.parent / "P1_original_baselines"
    if action == "prepare":
        prepare("p1", pkg)
    elif action == "submit":
        submit("p1", pkg, smoke="--smoke" in sys.argv)
    elif action == "follow":
        follow("p1", pkg, smoke="--smoke" in sys.argv)
    else:
        raise ValueError(action)
