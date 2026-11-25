"""One-shot locked three-source comparison, one process per method and seed."""
from contextlib import ExitStack
import datetime as dt
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time
from unittest.mock import patch

import numpy as np

HERE=Path(__file__).resolve().parent
OLD=HERE.parents[1]/"_maskco_cross_dataset_20261119"
sys.path.insert(0,str(OLD))
import compare as study

def read(path):
    return json.loads(path.read_text())

def install(arm,manifest,qualify):
    import pool_search
    is_new=arm in (manifest["candidate"],manifest["matched"])
    source_arm="SLOT" if is_new else arm
    prefix=study.REPEAT if source_arm in study.MASK_ARMS else study.RUN10
    sys.path.insert(0,str(prefix/study.EXT/"scripts/static_profit"))
    import profit_core as core
    base=core.ProfitTask
    if not qualify:
        meta=read(HERE/"heldout_sources.json")
        study.CFG.update(derived_instances=meta["derived_instances"],paths=dict(workspace=str(HERE)))
    access,core,source,locked=study.setup(source_arm)
    if not qualify:
        cases=meta["cases"]
        original=access.identity
        factory=core.ProfitTask
        if set(cases)!=set(meta["derived_instances"]):raise ValueError("Expected an external-only panel")
        def identity(case,fold,role):
            if fold==0 and role=="external_test" and case in meta["derived_instances"]:return (fold,role,case)
            return original(case,fold,role)
        class LockedTask(base):
            def __init__(self,case,*,fold,role):
                access.require_case(case,fold,role)
                self.c=core.load_config()
                if (fold,role)==(0,"external_test") and case in meta["derived_instances"]:
                    entry=meta["derived_instances"][case]
                    path=HERE/entry["file"]
                    raw=path.read_bytes()
                    if hashlib.sha256(raw).hexdigest()!=entry["sha256"]:raise ValueError("Benchmark projection differs")
                    data=json.loads(raw);rows=data["nodes"]
                    if data["case"]!=case or [v[0] for v in rows]!=list(range(101)):raise ValueError(case)
                    ins=dict(name=case,path=str(path),nodes=rows,sha256=entry["sha256"],geometry_sha256=entry["geometry_sha256"],
                             distance=[[math.dist(a[1:3],b[1:3]) for b in rows] for a in rows])
                else:raise ValueError("Unregistered locked test request")
                self.model=core.PublicationModel(ins,self.c,"main_Product1",.8)
                self.ins,self.fold,self.role=self.model.ins,fold,role
                self.condition=dict(profile="main_Product1",minimum_quality=.8,value_multiplier=1)
                self.route=lru_cache(maxsize=30000)(self.model.route)
        access.identity=identity
        def task(case,**kw):
            return LockedTask(case,**kw) if case in cases else factory(case,**kw)
        core.ProfitTask=task
    return source_arm,is_new,access,core,source,locked

def endpoint(arm,case,seed,cfg):
    qualify=cfg["training_only"]
    manifest=read(HERE/("qualification_method.json" if qualify else "locked_method.json"))
    technical=cfg.get("archived_case_acceptance",False)
    if technical and (case!="gh_c1_4_1_p100" or cfg["seconds"]!=4):raise ValueError("Qualification case differs")
    if not qualify and (manifest["development_decision"]!="COMPLETE_PASS" or cfg["seconds"]!=(4 if technical else 60) or case not in read(HERE/"heldout_sources.json")["cases"]):
        raise ValueError("A passing locked development recipe and prescribed case are required")
    admission=read(HERE/"heldout_admission.json")
    if not qualify and (admission["status"]!="PASS" or case not in admission["cases"]):
        raise ValueError("Common initializer not admitted")
    if qualify and case!="r101":raise ValueError("Qualification uses r101 only")
    role="train" if qualify else "external_test"
    key,is_new,access,core,source,locked=install(arm,manifest,qualify)
    import timed_search
    from run_dev72_clock import clock_type
    affinity=timed_search.single_cpu_guard()
    start=time.perf_counter()
    policy,_=study.load(key,access,core,source,locked,"c205" if key in ("MASK","SLOT") else case)
    setup_s=time.perf_counter()-start
    scope=access.Scope("locked-comparison-cell",frozenset([(0,role,case)]),frozenset(["initialize","infer","search"]))
    np.random.seed(seed);random.seed(seed)
    if "torch" in sys.modules:sys.modules["torch"].manual_seed(seed)
    seconds=cfg["seconds"]
    with access.restricted_scope(scope):
        if is_new:
            import pool_search
            out=pool_search.solve(pool_search.library(),core,locked,policy,case,role,seed,arm,seconds)
            out.update(cpu=affinity,setup_s=setup_s)
            return out
        start=time.perf_counter();utc=dt.datetime.now(dt.timezone.utc);cpu=time.process_time()
        task=core.ProfitTask(case,fold=0,role=role)
        clock=clock_type(timed_search.SearchClock)(task,start=start,utc=utc,seconds=seconds)
        clock.candidate_source=arm
        forwards=None
        if arm in ("DIST","DIST_SLOT"):
            forwards=0
            provider=locked.RepairProvider(arm,None,True)
            with ExitStack() as stack:
                if arm=="DIST_SLOT":locked.setup_operators(stack,provider,"SLOT")
                stack.enter_context(patch.object(timed_search,"SearchClock",lambda *args,**kwargs:clock))
                result=timed_search.guided_search(task,provider,"maskco_base",seconds=seconds,iterations=1000000,seed=seed,remove_count=10,topk=8)
        elif arm in ("VNS","ALNS"):
            forwards=0
            with patch.object(timed_search,"SearchClock",lambda *args,**kwargs:clock):
                result=timed_search.classical_search(task,arm.lower(),seconds=max(.01,clock.deadline-time.perf_counter()),iterations=1000000,seed=seed)
        elif arm=="NDS-Econ":
            from nds_economic import search
            result=search(policy,task,clock,seed)
        else:
            result=source.neural(*policy,task,clock)
        if result["failure"] or not result["selected_incumbent_in_budget"]:raise ValueError(str(result["failure"]))
        final=clock.events[-1]
        fresh=core.ProfitTask(case,fold=0,role=role)
        audit=fresh.evaluate(final["routes"],audit=True)
        if final["elapsed_s"]>seconds or abs(audit["net_benefit_U"]-result["final"]["net_benefit_U"])>1e-6:raise ValueError("Endpoint audit differs")
        counts=task.route.cache_info()
        return dict(case=case,role=role,seed=seed,method=arm,status="AUDITED",U=audit["net_benefit_U"],endpoint=audit,
                    initial_routes=clock.events[0]["routes"],initial_U=clock.events[0]["score"]["net_benefit_U"],routes=final["routes"],
                    selected_s=final["elapsed_s"],forwards=forwards,iterations=result["iterations"],setup_s=setup_s,cpu=affinity,
                    cpu_s=time.process_time()-cpu,timing=result["timing"],requests=counts.hits+counts.misses)

def main():
    cfg=read(HERE/"phase.json")
    i=int(sys.argv[1]);block=cfg["blocks"][i]
    if len(sys.argv)==3:
        arm=sys.argv[2]
        if arm not in block["arms"]:raise ValueError(arm)
        out=endpoint(arm,block["case"],block["seed"],cfg)
        (HERE/"out"/f"{i}_{arm}.cell.json").write_text(json.dumps(out,allow_nan=False))
        return
    arms={}
    for arm in block["arms"]:
        binding=study.CFG["bindings"].get(arm,study.CFG["bindings"]["SLOT"])
        subprocess.run([binding["python"],"-B",str(HERE/"independent.py"),str(i),arm],check=True)
        arms[arm]=read(HERE/"out"/f"{i}_{arm}.cell.json")
    if len({json.dumps(r["initial_routes"]) for r in arms.values()})!=1:raise ValueError("Unpaired initializer")
    (HERE/"out"/f"{i}.json").write_text(json.dumps(dict(case=block["case"],seed=block["seed"],arms=arms),allow_nan=False))
if __name__=="__main__":
    main()
