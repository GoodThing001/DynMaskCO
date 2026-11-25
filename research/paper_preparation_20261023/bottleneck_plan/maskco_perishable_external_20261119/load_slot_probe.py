"""Record strict insertion gains and masked edge scores for capacity-conditioned ranking."""
from pathlib import Path
import json
import sys
import time

import numpy as np

import external_common as adapter
import pool_search as pool
from heat_placebo import remap

HERE = Path(__file__).resolve().parent


def edge_ranks(heat):
    values = heat.copy()
    np.fill_diagonal(values, -np.inf)
    return 1 + np.argsort(np.argsort(-values, axis=1, kind="stable"), axis=1, kind="stable")


def geometric_features(task, rows, distances, route, position, i, used_load):
    a = route[position-1] if position else 0
    b = route[position] if position < len(route) else 0
    detour = distances[a,i] + distances[i,b] - distances[a,b]
    width = (rows[i,5] - rows[i,4]) / 1440.
    demand = rows[i,3] / 20000.
    time_to_i = distances[a,i] / task.c["routing"]["speed_km_per_min"]
    time_to_b = distances[i,b] / task.c["routing"]["speed_km_per_min"]
    head_slack = (rows[i,5] - rows[a,4] - rows[a,6] - time_to_i) / 1440.
    tail_slack = (rows[b,5] - rows[i,4] - rows[i,6] - time_to_b) / 1440.
    load_after = used_load + demand
    return [detour, distances[a,i], distances[i,b], distances[0,i], demand, used_load,
            load_after**2, width, head_slack, tail_slack, min(head_slack, tail_slack),
            rows[a,3]/20000., rows[b,3]/20000., position/max(1,len(route)), float(a==0),float(b==0)]


def learned_features(ranks, distance_ranks, a, i, b, load, width):
    def signal(u, v):
        return np.clip(np.log(distance_ranks[u,v] / ranks[u,v]), -3., 3.)
    left, right, old = signal(a,i), signal(i,b), signal(a,b)
    change = left + right - old
    return [left,right,old,change*load,change*width,change*load**2]


def main(index):
    plan = json.loads((HERE/"phase.json").read_text())
    case = plan["blocks"][index]["case"]
    rep_count = plan["masks_per_state"]
    states = json.loads((HERE/"archived_states.json").read_text())[case]
    manifest = dict(candidate="ORDER_SPLIT", matched="ORDER_SPLIT_D")
    _,_,access,core,source,locked = adapter.install("ORDER_SPLIT",manifest,False)
    from timed_search import single_cpu_guard
    single_cpu_guard()
    policy,_ = pool.study.load("SLOT",access,core,source,locked,"c205")
    scope=access.Scope("load-conditioned-development",frozenset([(0,"external_test",case)]),frozenset(["initialize","infer","search"]))
    start=time.process_time()
    physical, actual, null, labels, lengths = [], [], [], [], []
    rng=np.random.default_rng(202611209)
    with access.restricted_scope(scope):
        task=core.ProfitTask(case,fold=0,role="external_test")
        nodes=np.asarray(task.ins["nodes"]); dist=np.asarray(task.ins["distance"])
        d_ranks=edge_ranks(-dist)
        provider=locked.RepairProvider("load-slot",policy,False)
        if tuple(map(tuple,states["initial"]))!=tuple(map(tuple,task.initial())):
            raise ValueError("Archived common start differs")
        for state_key in ("initial","terminal"):
            routes=tuple(map(tuple,states[state_key]))
            task.evaluate(routes,audit=True)
            for rep in range(rep_count):
                group=rng.choice(np.arange(1,101),size=40,replace=False)
                removed=set(map(int,rng.choice(group,10,replace=False)))
                predicted=provider.heatmap(task,routes,group,"maskco_base",rep)
                aligned,shuffled=remap(predicted,dist,routes,group,rng)
                real_ranks,shuffled_ranks=edge_ranks(aligned),edge_ranks(shuffled)
                remaining=[tuple(x for x in r if x not in removed) for r in routes]
                for i in sorted(removed):
                    slots=[]
                    for r in filter(None,remaining):
                        mass=nodes[list(r),3].sum()
                        if mass+nodes[i,3]>20000.+1.e-7:continue
                        parent=task.route(r)
                        if not parent.feasible:raise ValueError("Infeasible visible route")
                        for j,(a,b) in enumerate(zip((0,)+r,r+(0,))):
                            value=task.route(r[:j]+(i,)+r[j:])
                            x=geometric_features(task,nodes,dist,r,j,i,mass/20000.)
                            slots.append((x[0],a,b,x,value.net_benefit-parent.net_benefit if value.feasible else -np.inf))
                    slots.sort(key=lambda v:v[0])
                    anchor=max([task.route((i,)).net_benefit]+[v[-1] for v in slots[:6]])
                    lengths.append(len(slots[6:]))
                    for detour,a,b,x,gain in slots[6:]:
                        physical.append(x)
                        actual.append(learned_features(real_ranks,d_ranks,a,i,b,x[4]+x[5],x[7]))
                        null.append(learned_features(shuffled_ranks,d_ranks,a,i,b,x[4]+x[5],x[7]))
                        labels.append(max(0.,gain-anchor))
    if len(lengths)!=20*rep_count or provider.forward_calls!=2*rep_count:
        raise ValueError("Missing contexts")
    out=HERE/"out"
    np.savez_compressed(out/f"{index}.npz",lengths=np.asarray(lengths,np.int32),
                        physical=np.asarray(physical,np.float32),real=np.asarray(actual,np.float32),
                        shuffled=np.asarray(null,np.float32),gain=np.asarray(labels,np.float64))
    result=dict(case=case,contexts=len(lengths),slots=len(labels),forwards=provider.forward_calls,
                cpu_s=time.process_time()-start,route_requests=sum(task.route.cache_info()[:2]))
    (out/f"{index}.json").write_text(json.dumps(result,allow_nan=False))


if __name__=="__main__":
    main(int(sys.argv[1]))
