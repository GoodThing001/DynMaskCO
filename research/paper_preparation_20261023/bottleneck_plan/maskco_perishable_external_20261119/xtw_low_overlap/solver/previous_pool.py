"""Amortized MASKCO prediction over a moving pool of hidden customers."""
import ctypes
import datetime as dt
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
OLD = HERE.parents[1] / "_maskco_cross_dataset_20261119"
sys.path.insert(0, str(OLD))
import compare as study

F64 = np.ctypeslib.ndpointer(dtype=np.float64, flags="C_CONTIGUOUS")
I32 = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")
I64 = np.ctypeslib.ndpointer(dtype=np.int64, flags="C_CONTIGUOUS")

def library():
    so = HERE / "masked_repair.so"
    with (HERE / "build.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not so.exists():
            subprocess.run(["g++","-std=c++17","-O3","-fPIC","-shared",str(HERE/"masked_repair.cpp"),"-o",str(so)], check=True)
    lib = ctypes.CDLL(str(so))
    lib.route_score.argtypes = [F64,F64,F64,I32,ctypes.c_int]
    lib.route_score.restype = ctypes.c_double
    lib.repair_batch.argtypes = [F64,F64,F64,F64,I32,ctypes.c_int,I32,I32,ctypes.c_uint,ctypes.c_double,ctypes.c_double,ctypes.c_int,ctypes.c_int,ctypes.c_int,I64]
    lib.repair_batch.restype = None
    return lib

def pack(routes):
    buffer = np.full(201,-1,dtype=np.int32)
    seq = []
    for route in routes:
        if seq:
            seq.append(0)
        seq.extend(route)
    buffer[:len(seq)] = seq
    return buffer

def unpack(buffer):
    routes = [[]]
    for i in buffer:
        if i < 0:
            return tuple(tuple(r) for r in routes)
        if i == 0:
            routes.append([])
        else:
            routes[-1].append(int(i))
    raise ValueError("Incomplete route buffer")

def inputs(task):
    model = task.model
    return (np.ascontiguousarray(task.ins["nodes"],dtype=np.float64),
            np.ascontiguousarray(task.ins["distance"],dtype=np.float64),
            np.array([task.c["routing"]["speed_km_per_min"],model.rate,model.minimum,model.price,
                      task.c["money"]["distance_U_per_km"],model.cold,task.c["routing"]["capacity_kg"]],dtype=np.float64))

def audit_kernel(lib,task):
    nodes,dist,params = inputs(task)
    rng = np.random.default_rng(1241)
    checked = [tuple(r) for r in task.initial()]
    for r in list(checked):
        checked.extend((r[::-1], r[:max(1,len(r)//2)]))
    for _ in range(100):
        checked.append(tuple(map(int,rng.choice(np.arange(1,101),size=rng.integers(1,22),replace=False))))
    for order in checked:
        ref = task.route(order)
        score = lib.route_score(nodes,dist,params,np.asarray(order,np.int32),len(order))
        if np.isfinite(score) != ref.feasible or (ref.feasible and abs(score-ref.net_benefit)>1.e-6):
            raise ValueError(f"Native route score differs for {order}")
    return len(checked)

def conflict_pool(routes, group, heat):
    """Focus on masked customers whose present links rank behind many alternatives."""
    rank = np.argsort(np.argsort(-heat,axis=1,kind="stable"),axis=1,kind="stable")
    neighbours = {}
    for route in routes:
        for a,i,b in zip((0,)+route,route,route[1:]+(0,)):
            neighbours[i] = (a,b)
    ids = list(map(int,group))
    conflicts = sorted(ids,key=lambda i:max(rank[i,neighbours[i][0]],rank[i,neighbours[i][1]]),reverse=True)
    chosen = set(conflicts[:12])
    for i in conflicts[:12]:
        a,b = neighbours[i]
        for j in sorted(ids,key=lambda j:rank[i,j]+rank[j,i]):
            if j not in (i,a,b):
                chosen.add(j)
                break
    chosen.update(ids[:8])
    return np.array([i for i in ids if i in chosen],dtype=np.int32)

def solve(lib,core,locked,policy,case,role,seed,arm,seconds):
    import timed_search
    from run_dev72_clock import clock_type
    base = arm.removesuffix("_D")
    distance_only = arm.endswith("_D")
    if base not in ("POOL","RELATED","ADAPT","FOCUS","CALIB","ARCHIVE","CHOICE","REGRET","HYBRID"):
        raise ValueError(arm)
    if base=="ARCHIVE":
        from route_bank import RouteBank
    start = time.perf_counter()
    utc = dt.datetime.now(dt.timezone.utc)
    cpu = time.process_time()
    task = core.ProfitTask(case,fold=0,role=role)
    clock = clock_type(timed_search.SearchClock)(task,start=start,utc=utc,seconds=seconds)
    clock.candidate_source = arm
    first = clock.initialize()
    state = pack(first)
    best = state.copy()
    mask_rng = np.random.default_rng(seed)
    nodes,dist,params = inputs(task)
    negative_dist = np.ascontiguousarray(-dist)
    if base in ("CALIB","CHOICE","HYBRID"):
        from calibration import residual_edges
        alpha=float(json.loads((HERE/"calibrated.json").read_text())["alpha"])
    provider = locked.RepairProvider(arm,policy,distance_only)
    bank=None
    if base=="ARCHIVE":
        bank=RouteBank(task)
        bank.add(first)
    stats = np.zeros(3,dtype=np.int64)
    total = np.zeros_like(stats)
    epochs = 0
    rewards = np.zeros(2)
    uses = np.zeros(2,dtype=int)
    ranking_uses = np.zeros(3,dtype=int)
    regret_epochs = 0
    while seconds-clock.elapsed() > .004:
        epoch_start = time.perf_counter()
        before_value = clock.events[-1]["score"]["net_benefit_U"]
        if epochs % 4 == 0:
            state = best.copy()
        group = mask_rng.choice(np.arange(1,101),size=40,replace=False).astype(np.int32)
        batch_rng = np.random.default_rng([seed,epochs,19])
        use_distance=distance_only or (bank is not None and epochs%4==0)
        if use_distance:
            heat = negative_dist
        else:
            heat = np.ascontiguousarray(provider.heatmap(task,unpack(state),group,"maskco_base",epochs),dtype=np.float64)
        if base=="CALIB" and not distance_only:
            spatial,residual=residual_edges(dist,heat)
            heat=np.ascontiguousarray(spatial+alpha*residual)
        mode = int(base=="RELATED")
        if base in ("ADAPT","FOCUS","CALIB","ARCHIVE","CHOICE","HYBRID"):
            if epochs<2 or epochs%4==0:
                mode=int(np.argmin(uses))
            else:
                mode=int(np.argmax(rewards+.1/np.sqrt(1+uses)))
        uses[mode]+=1
        focused = conflict_pool(unpack(state),group,heat) if base in ("FOCUS","CALIB","ARCHIVE","CHOICE","HYBRID") and mode else group
        end = min(clock.deadline,epoch_start+1.6)
        regret_active = base=="REGRET"
        if base in ("CHOICE","HYBRID") and end-time.perf_counter()>.30:
            options=[negative_dist]
            if not distance_only:
                spatial,residual=residual_edges(dist,heat)
                options.extend([heat,np.ascontiguousarray(spatial+alpha*residual)])
            seed_trial=int(batch_rng.integers(2**32))
            proposals=[(ranking,g,h) for g in (range(2) if base=="HYBRID" else range(1)) for ranking,h in enumerate(options)]
            outcomes=[]
            for proposal,(ranking,g,h) in enumerate(proposals):
                allowance=min(.10,end-time.perf_counter()-.005)
                if allowance<=0:break
                trial=state.copy()
                trial_best=state.copy()
                cap=96 if base=="CHOICE" else 0
                duration=allowance if cap else min(allowance,.04)
                lib.repair_batch(nodes,dist,params,h,focused,len(focused),trial,trial_best,seed_trial,duration,1.5,mode,g,cap,stats)
                total+=stats
                if (cap and stats[0]!=cap) or (not cap and duration<.04):continue
                score=task.evaluate(unpack(trial_best))["net_benefit_U"]
                outcomes.append((score,-((proposal-epochs)%len(proposals)),ranking,g,trial,trial_best))
            if outcomes:
                score,_,ranking,regret_active,state,trial_best=max(outcomes,key=lambda x:x[:2])
                heat=options[ranking]
                ranking_uses[ranking]+=1
                regret_epochs+=int(regret_active)
                if score>clock.events[-1]["score"]["net_benefit_U"]+1e-7 and clock.offer(unpack(trial_best),int(total[0])):
                    best=trial_best.copy()
        while time.perf_counter()<end:
            allowance = min(0.10,end-time.perf_counter()-.002)
            if allowance <= 0:
                break
            previous = best.copy()
            temperature = 2.5 * max(.10,1-clock.elapsed()/seconds)
            lib.repair_batch(nodes,dist,params,heat,focused,len(focused),state,best,int(batch_rng.integers(2**32)),
                             allowance,temperature,mode,int(regret_active),0,stats)
            total += stats
            if not np.array_equal(previous,best):
                clock.offer(unpack(best),int(total[0]))
        gain = max(0.,clock.events[-1]["score"]["net_benefit_U"]-before_value)
        if bank is not None:
            bank.add(unpack(state))
            bank.add(unpack(best))
            if epochs%3==2:
                cover=bank.combine(unpack(best),clock.deadline)
                if cover is not None and clock.offer(cover,int(total[0])):
                    state=pack(cover)
                    best=state.copy()
        # Initial construction improvements are large for both mask policies.
        rewards[mode] = .65*rewards[mode]+.35*min(3.,np.log1p(gain))
        epochs += 1
    result = clock.finish(arm,int(total[0]))
    final = clock.events[-1]
    fresh = core.ProfitTask(case,fold=0,role=role)
    audit = fresh.evaluate(final["routes"],audit=True)
    if (not result["selected_incumbent_in_budget"] or final["elapsed_s"]>seconds or
            abs(audit["net_benefit_U"]-result["final"]["net_benefit_U"])>1e-6):
        raise ValueError("Independent route audit or cutoff mismatch")
    return dict(case=case,seed=seed,method=arm,role=role,status="AUDITED",U=audit["net_benefit_U"],endpoint=audit,routes=final["routes"],
                initial_U=clock.events[0]["score"]["net_benefit_U"],initial_routes=first,selected_s=final["elapsed_s"],
                epochs=epochs,ranking_counts=ranking_uses.tolist(),regret_epochs=regret_epochs,mask_mode_counts=uses.tolist(),iterations=int(total[0]),improvements=int(total[1]),native_route_requests=int(total[2]),
                archive_mips=bank.calls if bank else 0,archive_improvements=bank.new_covers if bank else 0,
                archive_s=bank.solver_s if bank else 0.,forwards=provider.forward_calls,forward_s=provider.forward_seconds,cpu_s=time.process_time()-cpu,timing=result["timing"])

def main(i):
    cfg = json.loads((HERE / "phase.json").read_text())
    block = cfg["blocks"][i]
    case,seed = block["case"],block["seed"]
    role = block.get("role","validation")
    lib=library()
    access,core,source,locked=study.setup("SLOT")
    from timed_search import single_cpu_guard
    affinity=single_cpu_guard()
    policy,scope=study.load("SLOT",access,core,source,locked,"c205" if role=="train" else case)
    if role=="train":
        scope=access.Scope("kernel-acceptance",frozenset([(0,"train",case)]),frozenset(["initialize","infer","search"]))
    records={}
    with access.restricted_scope(scope):
        checked=audit_kernel(lib,core.ProfitTask(case,fold=0,role=role)) if role=="train" else None
        for arm in block["arms"]:
            if arm in ("DIST","DIST_SLOT"):
                if role!="validation":raise ValueError("Development comparison requires a validation role")
                r=study.execute(arm,case,seed,False)
            else:
                r=solve(lib,core,locked,policy,case,role,seed,arm,cfg["seconds"])
                if role=="train" and (r["iterations"] < 1 or (not arm.endswith("_D") and r["forwards"] < 1)):
                    raise ValueError("Masked repair path did not execute")
            r["cpu"]=affinity
            records[arm]=r
    if len({json.dumps(r["initial_routes"]) for r in records.values()})!=1:
        raise ValueError("Initial routes disagree")
    (HERE/"out"/f"{i}.json").write_text(json.dumps(dict(case=case,seed=seed,arms=records,checked_routes=checked),allow_nan=False))

if __name__=="__main__":
    main(int(sys.argv[1]))
