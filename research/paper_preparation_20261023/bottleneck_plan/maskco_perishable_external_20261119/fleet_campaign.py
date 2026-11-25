"""Prepare and execute the preregistered development matrices on the shared research host."""
import argparse
import ast
import json
from pathlib import Path
import sys
import tarfile

HERE = Path(__file__).resolve().parent
PREV = HERE.parent/"maskco_heatmap_calibration_20261119"
sys.path.insert(0, str(PREV))
from phase_plan import paired, read
import transport_heat


ARMS = ["FLEET_PRICE", "FLEET_PRICE_D", "PREFIX_DISTANCE", "ORDER_SPLIT", "DIST", "DIST_SLOT"]
PHASES = ("fleetqual01", "fleetscreen01", "fleetdev01")
transport_heat.HERE = HERE
transport_heat.DEST = transport_heat.PROJECT + "/_maskco_perishable_external_20261119"
transport_heat.transport.LOCAL = HERE


def prepare():
    old = read(HERE/"mixscreen01.json")
    full = read(HERE/"mixdev01.json")
    admitted = read(HERE/"heldout_admission.json")
    cases = read(HERE/"heldout_sources.json")
    if (not read(HERE/"results/fleetpricing01/all.json")["passed"]
            or admitted["status"] != "PASS" or set(admitted["cases"]) != set(cases["cases"])):
        raise ValueError("Development inputs or fixed-work qualification incomplete")
    screen = {b["case"] for b in old["blocks"]}
    development = {b["case"] for b in full["blocks"]}
    if len(screen) != 11 or development != set(cases["cases"]) or len(development) != 27:
        raise ValueError("Historical development panel differs")
    short_case = "dk_data100_a"
    recipes = [
        ("fleetqual01", "software acceptance on one previously used development case", 16,
         [dict(case=short_case, seed=56801, arms=[a], seconds=60 if a in ("DIST","DIST_SLOT") else 16) for a in ARMS]),
        ("fleetscreen01", "paired development screening", 60, paired(screen, [56831,56832], ARMS)),
        ("fleetdev01", "complete paired development", 60, paired(development, [56851,56852,56853], ARMS))]
    for phase, purpose, seconds, blocks in recipes:
        subset = sorted({b["case"] for b in blocks})
        files = ["block_split.py", "calibrated.json", "calibration.py", "masked_repair.cpp", "split_order.cpp",
                 "pool_search.py", "runner.py", "external_common.py", "locked_method.json", "heldout_sources.json",
                 "heldout_admission.json", "fleet_timed_worker.py", "fleet_timed_pool.py", "fleet_pricing.py"]
        sources = {n: old["file_sources"][n] for n in files if n in old["file_sources"]}
        files += [cases["derived_instances"][c]["file"] for c in subset]
        for n in files:
            p = HERE/sources.get(n,n)
            if not p.is_file():raise ValueError(p)
            if n.endswith(".py"):ast.parse(p.read_text(encoding="utf8"))
        plan = dict(purpose=purpose, worker="fleet_timed_worker.py", training_only=False,
                    seconds=seconds, block_timeout_s=1000, files=files, file_sources=sources, blocks=blocks)
        dest = HERE/(phase+".json")
        if dest.exists():
            if read(dest) != plan:raise ValueError("Existing phase differs from the requested matrix")
        else:
            dest.write_text(json.dumps(plan,indent=2,ensure_ascii=False),encoding="utf8")
        print(phase,len(blocks),sum(len(b["arms"]) for b in blocks))


def qualify():
    q = HERE/"results"/"fleetqual01"
    state = read(q/"status.json")
    plan = read(HERE/"fleetqual01.json")
    if not state["end"] or state["complete"]!=6 or state["failed"]:
        raise ValueError("Incomplete technical run")
    reference = None
    for i,b in enumerate(plan["blocks"]):
        arm, = b["arms"]
        r = read(q/"out"/f"{i}.json")["arms"][arm]
        if reference is None: reference = r["initial_routes"]
        if (r["initial_routes"]!=reference or r["case"]!=b["case"] or r["seed"]!=b["seed"]
                or r["status"]!="AUDITED" or not r["endpoint"]["independently_audited"]
                or not 0<=r["selected_s"]<=b["seconds"] or r["iterations"]<1):
            raise ValueError("Technical endpoint or cutoff differs: "+arm)
        if arm in ARMS[:3]:
            p = r["prefix"]
            if (len(set(p["mask"]))!=40 or p["attempts"]<1 or p["common_columns"]>p["columns"]
                    or p["forwards"]!=int(arm=="FLEET_PRICE")
                    or p["completion_s"]>=b["seconds"] or p["unique_route_requests"]<1
                    or r["forwards"]<int(arm!="FLEET_PRICE_D")
                    or abs(p["accepted_U"]-(p["proposed_U"] if p["accepted"] else r["initial_U"]))>1e-6
                    or abs(r["initial_U"]-p["initial_U"])>1e-6 or r["U"]<p["accepted_U"]-1e-6):
                raise ValueError("Pricing, strict acceptance, or later search missing: "+arm)
    print("PASS six qualification arms")


def preflight():
    from host_access import connect, remote_python
    code="""import json, os, shutil, psutil
from pathlib import Path
from importlib.metadata import version
cpu=psutil.cpu_percent(interval=1,percpu=True)
aff=sorted(os.sched_getaffinity(0))
groups={}
for ln in aff:
 p=Path(f'/sys/devices/system/cpu/cpu{ln}/topology')
 identity=tuple((p/f).read_text().strip() for f in ['physical_package_id','core_id'])
 groups.setdefault(identity,[]).append(ln)
root=Path('/home/hzeng/project/MASKCO-Main')
print(json.dumps(dict(physical=len(groups),idle=sum(max(cpu[i] for i in g)<40 for g in groups.values()),
 ram_gib=psutil.virtual_memory().available/2**30,disk_gib=shutil.disk_usage(root).free/2**30,
 jax=version('jax'),scipy=version('scipy'),prior_controller_closed=json.loads((root/'_maskco_perishable_external_20261119/replicate01/status.json').read_text())['end'] is not None)))"""
    with connect() as c:data=json.loads(remote_python(c,code))
    (HERE/"fleet_host_resources.json").write_text(json.dumps(data,indent=2),encoding="utf8")
    if not data["idle"] or data["ram_gib"]<5 or data["disk_gib"]<1 or not data["prior_controller_closed"]:
        raise ValueError(data)
    print(json.dumps(data))


def check_sources(phase):
    if phase=="fleetqual01":return
    source_phase = "fleetqual01" if phase=="fleetscreen01" else "fleetscreen01"
    with tarfile.open(HERE/(source_phase+"_inputs.tar.gz")) as previous:
        plan = read(HERE/(phase+".json"))
        for name in plan["files"]:
            if name.startswith("inputs/"):continue
            expected = (HERE/plan["file_sources"].get(name,name)).read_bytes()
            if previous.extractfile(name).read()!=expected:
                raise ValueError("Code or metadata differs from earlier phase: "+name)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action",choices=["plan","probe","qualify","submit","status","recover"])
    cli.add_argument("phase",choices=PHASES,nargs="?")
    args=cli.parse_args()
    if args.action=="plan":prepare()
    elif args.action=="probe":preflight()
    elif args.action=="qualify":qualify()
    elif args.phase is None:cli.error("A registered phase is required")
    else:
        if args.action=="submit":
            if args.phase=="fleetscreen01":qualify()
            if args.phase=="fleetdev01":
                gate=read(HERE/"results"/"fleetscreen01"/"analysis.json")
                if gate["status"]!="COMPLETE" or not gate["passed"]:
                    raise ValueError("Development screening gate not passed")
            check_sources(args.phase)
        transport_heat.action(args.action,args.phase)
