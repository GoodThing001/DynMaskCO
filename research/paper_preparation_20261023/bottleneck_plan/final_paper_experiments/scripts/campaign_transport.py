# -*- coding: utf-8 -*-
"""P2/P3 通用远端传输器：prepare/submit/follow/recover。"""
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
LOCAL_ROOT = HERE.parent  # final_paper_experiments（执行包根）
PROJECT_LOCAL = HERE.parent.parent  # bottleneck_plan（冻结源码本地对照）
SRC = LOCAL_ROOT / "audit" / "remote_sources"

SPECS = {
    "p2": dict(pkg="P2_retail_analysis", remote="_maskco_final_paper_p2", session="maskco_final_paper_p2",
               phase_src="p2_phase.json", stage_src="p2_stage.py", stage_arc="stage_p2.py",
               worker_src="trajectory_worker.py", registry_arc="p2_heldout_sources.json",
               frozen=[("_maskco_perishable_external_20261119/carryprobe01/external_common.py", "remote"),
                       ("_maskco_perishable_external_20261119/carryprobe01/pool_search.py", "remote"),
                       ("_maskco_perishable_external_20261119/carryprobe01/block_split.py", "remote")]),
    "p3": dict(pkg="P3_factorial_ablation", remote="_maskco_final_paper_p3", session="maskco_final_paper_p3",
               phase_src="p3_phase.json", stage_src="p3_stage.py", stage_arc="stage_p3.py",
               worker_src="p3_worker.py", registry_src="p3_heldout_sources.json", registry_arc="heldout_sources.json",
               worker_dual=True,
               frozen=[("_maskco_cross_dataset_20261119/compare.py", "project"),
                       ("_maskco_distance_slot_20261119/worker.py", "project"),
                       ("_maskco_distance_slot_20261119/operators.py", "project"),
                       ("_maskco_perishable_external_20261119/carryprobe01/pool_search.py", "remote"),
                       ("_maskco_perishable_external_20261119/carryprobe01/block_split.py", "remote"),
                       ("_maskco_perishable_external_20261119/carryprobe01/calibration.py", "remote"),
                       ("_maskco_perishable_external_20261119/carryprobe01/calibrated.json", "remote")]),
    "p4": dict(pkg="P4_solomon_coldchain", remote="_maskco_final_paper_p4", session="maskco_final_paper_p4",
               phase_src="p4_phase.json", stage_src="p4_stage.py", stage_arc="stage_p4.py",
               worker_src="p4_trajectory_worker.py", worker_arc="unified_worker.py",
               dispatch_src="p4_dispatch.py", dispatch_arc="future_dispatch.py",
               registry_src=None, registry_arc=None,
               frozen=[("_maskco_cross_dataset_20261119/compare.py", "project"),
                       ("_maskco_distance_slot_20261119/worker.py", "project"),
                       ("_maskco_distance_slot_20261119/operators.py", "project"),
                       ("_maskco_perishable_external_20261119/carryprobe01/external_common.py", "remote"),
                       ("_maskco_perishable_external_20261119/carryprobe01/pool_search.py", "remote"),
                       ("_maskco_perishable_external_20261119/carryprobe01/block_split.py", "remote")]),
}


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def frozen_local(remote_rel, kind):
    if kind == "remote":
        return SRC / remote_rel.replace("/", "__")
    base = remote_rel.split("/", 1)[0][1:]
    if base == "maskco_cross_dataset_20261119":
        return PROJECT_LOCAL / "maskco_cross_dataset_20261119" / Path(remote_rel).name
    if base == "maskco_distance_slot_20261119":
        return PROJECT_LOCAL / "maskco_distance_slot_20261119" / Path(remote_rel).name
    raise ValueError(base)


def frozen_identity(client, sftp, spec):
    for remote_rel, kind in spec["frozen"]:
        local = frozen_local(remote_rel, kind)
        if not local.is_file():
            print("skip frozen check (no local copy):", remote_rel)
            continue
        with sftp.open(PROJECT + "/" + remote_rel, "rb") as f:
            if f.read() != local.read_bytes():
                raise ValueError("Frozen source differs at " + remote_rel)
    print("frozen identity OK")


def build_tarball(spec, name, smoke):
    pkg = LOCAL_ROOT / spec["pkg"]
    phase = read(pkg / spec["phase_src"])
    for f in (spec["worker_src"], spec["stage_src"], "unified_runner.py", spec.get("dispatch_src", "unified_runner.py")):
        ast.parse((pkg / f).read_text(encoding="utf-8"))
    members = []
    if smoke:
        blocks = phase["blocks"][:1]
        sp = dict(phase, name="smoke01", blocks=blocks)
        tmp = pkg / "smoke_phase.json"
        tmp.write_text(json.dumps(sp, ensure_ascii=False, indent=2), encoding="utf-8")
        members.append((tmp, "phase.json"))
    else:
        members.append((pkg / spec["phase_src"], "phase.json"))
    members += [(pkg / spec["worker_src"], spec.get("worker_arc", "unified_worker.py")),
                (pkg / spec["stage_src"], spec["stage_arc"]),
                (pkg / "unified_runner.py", "unified_runner.py")]
    if spec.get("registry_src") or spec.get("registry_arc"):
        members.append((pkg / spec.get("registry_src", spec["registry_arc"]), spec["registry_arc"]))
    if spec.get("worker_dual"):
        members.append((pkg / spec["worker_src"], "future_dispatch.py"))
    if spec.get("dispatch_src"):
        members.append((pkg / spec["dispatch_src"], spec["dispatch_arc"]))
    out_name = "smoke01.tar.gz" if smoke else name + ".tar.gz"
    with tarfile.open(pkg / out_name, "w:gz") as z:
        for src, arc in members:
            z.add(src, arcname=arc)
    if smoke:
        tmp.unlink()
    print("prepared", out_name, len(phase["blocks"]), "blocks",
          sum(len(b["methods"]) for b in phase["blocks"]), "cells")


def submit(spec, name, smoke=False):
    transport.record = lambda *a, **k: None
    pkg = LOCAL_ROOT / spec["pkg"]
    build_tarball(spec, name, smoke)
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        with client.open_sftp() as s:
            frozen_identity(client, s, spec)
            base = PROJECT + "/" + spec["remote"]
            try:
                s.mkdir(base)
            except OSError:
                pass
            target = base + ("/smoke01" if smoke else "/" + name)
            transport.remote_exec(client, "rm -rf " + target + " && mkdir -p " + target)
            tarball = "smoke01.tar.gz" if smoke else name + ".tar.gz"
            s.put(str(pkg / tarball), target + "/dispatch.tar.gz")
            command = ("cd " + target + " && (" + PYTHON + " -B " + spec["stage_arc"] + " && " + PYTHON +
                       " -B unified_runner.py) > controller.log 2>&1; echo $? > exit_code")
            session = spec["session"] + ("_smoke" if smoke else "")
            transport.remote_exec(client, "cd " + target + " && tar -xzf dispatch.tar.gz && tmux new-session -d -s "
                                  + session + " " + shlex.quote(command))
            print("submitted", target)
    finally:
        client.close()


def follow(spec, name, smoke=False, wait=30):
    transport.record = lambda *a, **k: None
    target = PROJECT + "/" + spec["remote"] + ("/smoke01" if smoke else "/" + name)
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
            out = LOCAL_ROOT / spec["pkg"] / ("results_smoke" if smoke else "results")
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
    name = sys.argv[1]
    action = sys.argv[2]
    spec = SPECS[name]
    smoke = "--smoke" in sys.argv
    if action == "submit":
        submit(spec, name, smoke)
    elif action == "follow":
        follow(spec, name, smoke)
    else:
        raise ValueError(action)
