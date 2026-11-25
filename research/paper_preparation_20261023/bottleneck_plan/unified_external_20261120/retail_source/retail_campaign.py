"""Prepare, submit once, observe, recover and report the eight-city retail comparison."""
import ast
from pathlib import Path
import csv
import json
import random
import shlex
import sys
import tarfile
import tempfile
import time

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent
sys.path.insert(0,str(STUDY))
from stage_retail import PEERS,reservation
from unified_report import generate,read,write

NAME="retail01"
RESULTS=STUDY/"results"/NAME


def qualification():
    with tempfile.TemporaryDirectory() as temp:
        root=Path(temp)
        target=root/NAME
        for name in PEERS:
            (root/name).mkdir()
        affinity=list(range(16))
        def idle(limit,excluded):
            return [i for i in affinity if i not in excluded][:limit],16,96.
        for name,core in zip(PEERS[:2],(0,1)):
            write(root/name/"status.json",dict(end=None,cores=[core]))
        assert reservation(target,idle,affinity,8,4) is None
        write(root/PEERS[2]/"status.json",dict(end=None,cores=[2,3,4]))
        result=reservation(target,idle,affinity,8,4)
        assert set(result["selected"])==set(range(5,12))
        assert set(result["available_reserved_for_other_work"])==set(range(12,16))
        write(root/PEERS[0]/"status.json",dict(end="complete",cores=[0]))
        assert reservation(target,idle,affinity,8,4)["selected"]==[0,5,6,7,8,9,10,11]
        assert reservation(target,idle,affinity,8,15) is None


def prepare():
    from unified_plan import METHODS,EXTERNAL
    qualification()
    admitted=read(HERE/"heldout_admission.json")
    registry=read(HERE/"heldout_sources.json")
    cases=registry["cases"]
    if admitted["status"]!="PASS" or len(cases)!=8 or set(admitted["cases"])!=set(cases):
        raise ValueError("Retail cohort has not passed admission")
    ref=read(STUDY/"matrix.json")
    inputs=registry["derived_instances"]
    code=read(STUDY/"new_panel/matrix.json")["reused_code"]
    seeds=(65001,65002,65003)
    rng=random.Random(65200)
    order=METHODS.copy();rng.shuffle(order)
    pairs=[(c,s) for c in sorted(cases) for s in seeds];rng.shuffle(pairs)
    blocks=[dict(case=c,seed=s,methods=order[i%16:]+order[:i%16]) for i,(c,s) in enumerate(pairs)]
    if len(blocks)!=24 or not all(set(b["methods"])==set(METHODS) and len(b["methods"])==16 for b in blocks):
        raise ValueError("Incomplete sixteen-method matrix")
    cfg=dict(name=NAME,seconds=60,training_only=False,methods=METHODS,external_methods=EXTERNAL,
             interpreters=ref["interpreters"],max_cores=16,keep_free=4,exclude_active_stages=PEERS,
             reused_code=code,blocks=blocks)
    write(HERE/"matrix.json",cfg)
    write(HERE/"metadata.json",dict(cases=cases,primary=sorted(cases),panels={"RWVRP-p100":sorted(cases)}))
    entries={"phase.json":HERE/"matrix.json","unified_runner.py":STUDY/"unified_runner.py",
             "stage_retail.py":HERE/"stage_retail.py","heldout_admission.json":HERE/"heldout_admission.json",
             "heldout_sources.json":HERE/"heldout_sources.json",**{e["file"]:HERE/e["file"] for e in inputs.values()}}
    package=HERE/(NAME+".tar.gz")
    with tarfile.open(package,"w:gz") as out:
        for name,path in entries.items():
            if name.endswith(".py"):ast.parse(path.read_text(encoding="utf-8"))
            out.add(path,arcname=name)
    with tarfile.open(package) as bundle:
        if set(bundle.getnames())!=set(entries) or any(n.startswith("raw/") for n in bundle.getnames()):
            raise ValueError("Package scope differs from the paired inputs and dispatcher")
    write(HERE/"qualification.json",dict(status="PASS",cells=384,core_seconds=23040,blocks=24,cities=8,
        request="same staging host as previous study, project-specific remote directory, registered projections and scheduler only",
        no_data_tests=["missing peer reservation queues","active physical lanes disjoint","four physical lanes reserved",
                       "released lane returned","insufficient resources queue"],methods=METHODS,seeds=seeds,package_members=list(entries)))
    print("PASS eight admitted cities, 24 blocks, 384 endpoints, 23,040 search core-seconds; no-data resource fixtures passed")


def city_tables():
    from unified_report import ranks
    from statistics import fmean
    result=read(RESULTS/"analysis.json")
    audit=read(RESULTS/"independent_audit.json")
    if result["status"]!="COMPLETE" or audit["status"]!="PASS" or audit["endpoints"]!=384:
        raise ValueError("Original retail cohort has not completed independent replay")
    meta=read(HERE/"metadata.json")["cases"]
    with (RESULTS/"endpoints.csv").open(encoding="utf-8-sig",newline="") as stream:
        endpoints=list(csv.DictReader(stream))
    rows=[];panel=result["panels"]["RWVRP-p100"]
    best=panel["strongest_external"];methods=result["external_methods"]
    lines=["# 八座城市的完整方法与热图增量","",
           "同城先平均三颗配对种子。城市外部名次含十二个外部法和 MASKCO，十六法名次另含三项项目内参照。主要对照取八城等权净效益最高的外部法；城市内最高外部法另列。","",
           f"| 城市或区域 | 原作者实例 | 名次 /13 | 名次 /16 | 对八城最强 {best} ΔU | 对同规则距离 ΔU | 同城最强外部法 | 对它 ΔU |",
           "|---|---|---:|---:|---:|---:|---|---:|"]
    for case in sorted(meta):
        subset=[r for r in endpoints if r["case"]==case]
        if len(subset)!=48:raise ValueError(case)
        net={m:fmean(float(r["U"]) for r in subset if r["method"]==m) for m in result["methods"]}
        external_ranks=ranks({m:net[m] for m in ["ORDER_SPLIT",*methods]})
        local=max(methods,key=net.get);u=net["ORDER_SPLIT"]
        row=dict(case=case,city=meta[case]["city"],origin=meta[case]["origin"],rank13=external_ranks["ORDER_SPLIT"],
                 rank16=ranks(net)["ORDER_SPLIT"],net_U=u,panel_comparator=best,delta_panel_best=u-net[best],
                 delta_matched=u-net["ORDER_SPLIT_D"],city_comparator=local,delta_city_best=u-net[local])
        rows.append(row)
        lines.append(f"| {row['city']} | {row['origin']} | {row['rank13']} | {row['rank16']} | {row['delta_panel_best']:+.2f} | {row['delta_matched']:+.2f} | {local} | {row['delta_city_best']:+.2f} |")
    if abs(fmean(r["delta_panel_best"] for r in rows)-panel["contrasts"][best]["mean"])>1e-6:
        raise ValueError("City and source contrasts differ")
    with (RESULTS/"cities.csv").open("w",encoding="utf-8-sig",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines+=["","[逐城表](cities.csv) · [锁版十六法全表](RESULTS.md) · [作者场景与准入](../../retail_source/准入.md)"]
    (RESULTS/"CITY_RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    intro=RESULTS/"RESULTS.md"
    note="[八个城市的逐城名次和配对收益](CITY_RESULTS.md)"
    old=intro.read_text(encoding="utf-8")
    if note not in old:intro.write_text(old+"\n"+note+"\n",encoding="utf-8")


def remote(action):
    from unified_transport import CAMPAIGN, PYTHON, connect, transport
    target=CAMPAIGN+"/"+NAME
    session="maskco_unified_retail01"
    with connect() as client,client.open_sftp() as sftp:
        if action=="submit":
            if read(HERE/"qualification.json")["status"]!="PASS":
                raise ValueError("Missing local qualification")
            sftp.mkdir(target)
            sftp.put(str(HERE/(NAME+".tar.gz")),target+"/dispatch.tar.gz")
            run=f"cd {target} && ({PYTHON} -B stage_retail.py && {PYTHON} -B unified_runner.py) > controller.log 2>&1"
            command=f"cd {target} && tar -xzf dispatch.tar.gz && tmux new-session -d -s {session} "+shlex.quote(run)
            transport.remote_exec(client,command)
            print("Submitted once",target,flush=True)
            return
        while True:
            try:
                with sftp.open(target+"/status.json") as f:
                    state=json.load(f)
            except FileNotFoundError:
                transport.remote_exec(client,"tmux has-session -t "+session)
                if action=="status":
                    print("WAITING_PEER_RESERVATIONS",flush=True)
                    return
                time.sleep(45)
                continue
            info={k:state[k] for k in ("cores","complete","total","cells","total_cells","failed","end")}
            info["active_blocks"]=[v["block"] for v in state["active"].values()]
            print(info,flush=True)
            if state["end"] or action=="status":break
            time.sleep(45)
        if action in ("recover","follow"):
            if not state["end"]:
                raise ValueError("Original run has not finished")
            RESULTS.mkdir(parents=True,exist_ok=True)
            transport.remote_exec(client,f"cd {target} && tar -czf receipts.tar.gz out status.json resource_admission.json controller.log phase.json")
            sftp.get(target+"/receipts.tar.gz",str(RESULTS/"receipts.tar.gz"))
            with tarfile.open(RESULTS/"receipts.tar.gz") as tar:
                for item in tar:
                    path=(RESULTS/item.name).resolve()
                    if not path.is_relative_to(RESULTS.resolve()):
                        raise ValueError("Receipt outside result folder")
                    if item.isdir():path.mkdir(exist_ok=True,parents=True)
                    elif item.isfile():path.write_bytes(tar.extractfile(item).read())
            if state["failed"]:raise ValueError("Original failure receipts recovered")
            generate(HERE,NAME,STUDY/"results")
            city_tables()


if __name__=="__main__":
    action=sys.argv[1]
    if action=="prepare":prepare()
    elif action=="report":
        generate(HERE,NAME,STUDY/"results")
        city_tables()
    elif action=="cities":city_tables()
    elif action in ("submit","follow","status","recover"):remote(action)
    else:raise ValueError(action)
