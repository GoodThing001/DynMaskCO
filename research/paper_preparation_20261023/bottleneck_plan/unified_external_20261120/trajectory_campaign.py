"""Prepare, run and retrieve the paired ten-seed timing study on the original host."""
import ast
from collections import defaultdict
import io
import json
from pathlib import Path
import random
import shlex
import sys
import tarfile
import time

HERE = Path(__file__).resolve().parent
NAME = "trajectory01"
METHODS = ("ORDER_SPLIT", "ORDER_SPLIT_D", "ALNS", "PyVRP")


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def prepare():
    matrix, meta = read(HERE/"matrix.json"), read(HERE/"metadata.json")
    grouped = defaultdict(lambda: defaultdict(list))
    for case in meta["primary"]:
        item=meta["cases"][case]
        grouped[item["dataset"]][item["geometry"]].append(case)
    chosen = []
    for groups in grouped.values():
        names = sorted(min(cases) for cases in groups.values())
        chosen.extend((names[0], names[len(names)//2], names[-1]))
    if len(set(chosen)) != 9: raise ValueError("Expected three geometries per source")
    rng=random.Random(65100)
    pairs=[(c,s) for c in sorted(chosen) for s in range(65101,65111)]
    rng.shuffle(pairs)
    order=list(METHODS)
    rng.shuffle(order)
    cfg=dict(seconds=60,training_only=False,name=NAME,methods=METHODS,max_cores=18,
             interpreters={m:matrix["interpreters"][m] for m in METHODS},
             blocks=[dict(case=c,seed=s,methods=order[i%4:]+order[:i%4]) for i,(c,s) in enumerate(pairs)])
    (HERE/"trajectory_matrix.json").write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding="utf-8")
    for p in ("trajectory_worker.py","stage_trajectory.py","unified_runner.py"):
        ast.parse((HERE/p).read_text(encoding="utf-8"))
    with tarfile.open(HERE/(NAME+".tar.gz"),"w:gz") as z:
        for p, name in (("trajectory_matrix.json","phase.json"),("trajectory_worker.py","unified_worker.py"),
                        ("stage_trajectory.py","stage_trajectory.py"),("unified_runner.py","unified_runner.py")):
            z.add(HERE/p,arcname=name)
    print("90 blocks, 360 solves, 1440 planned time samples", flush=True)


def remote(action):
    from unified_transport import connect, CAMPAIGN, PYTHON, transport
    target=CAMPAIGN+"/"+NAME
    with connect() as c, c.open_sftp() as s:
        if action=="submit":
            s.mkdir(target)
            s.put(str(HERE/(NAME+".tar.gz")), target+"/dispatch.tar.gz")
            command=(f"cd {target} && ({PYTHON} -B stage_trajectory.py && {PYTHON} -B unified_runner.py) "
                     "> controller.log 2>&1")
            print(transport.remote_exec(c,f"cd {target} && tar -xzf dispatch.tar.gz "
                                        f"&& tmux new-session -d -s maskco_unified_trajectory01 {shlex.quote(command)}"),flush=True)
            print("submitted trajectory01",flush=True)
            return
        while True:
            try:
                with s.open(target+"/status.json") as f:state=json.load(f)
            except FileNotFoundError:
                transport.remote_exec(c,"tmux has-session -t maskco_unified_trajectory01")
                if action=="status":
                    print("Queued for benchmark01",flush=True)
                    return
                time.sleep(30)
                continue
            print(json.dumps({k:state[k] for k in ("cores","cells","complete","failed","end")}),flush=True)
            if state["end"] or action=="status":break
            time.sleep(30)
        if state["end"] and action=="follow":
            out=HERE/"results"/NAME
            out.mkdir(parents=True,exist_ok=True)
            transport.remote_exec(c,f"cd {target} && tar -czf receipts.tar.gz out status.json controller.log phase.json")
            s.get(target+"/receipts.tar.gz",str(out/"receipts.tar.gz"))
            with tarfile.open(out/"receipts.tar.gz") as z:
                for item in z:
                    path=(out/item.name).resolve()
                    if not path.is_relative_to(out.resolve()):raise ValueError(item.name)
                    if item.isdir():path.mkdir(parents=True,exist_ok=True)
                    if item.isfile():path.write_bytes(z.extractfile(item).read())


if __name__=="__main__":
    if sys.argv[1]=="prepare":prepare()
    elif sys.argv[1] in ("submit","follow","status"):remote(sys.argv[1])
    else:raise ValueError(sys.argv[1])
