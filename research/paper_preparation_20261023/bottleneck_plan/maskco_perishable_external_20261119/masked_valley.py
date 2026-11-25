"""Follow paired feasible, temporarily inferior fleets with identical local operations."""
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter, process_time
import argparse
import json
import numpy as np
from directed_decoder_fit import HERE, read
from masked_order_bank import OUT as BANK, MASKS
from global_masked_split import chain, split_order
from prepare_panel import independent_audit
from remaining_edges_study import development_model
from view_frame_probe import shuffle_probabilities

OUT = HERE / "results/order_valley01"


def schedules(d, seed):
    nearest = np.argsort(d[1:, 1:], axis=1, kind="stable") + 1
    neighbors = {i: r[r != i][:12] for i,r in enumerate(nearest, 1)}
    rng = np.random.default_rng(seed)
    steps=[]
    for turn in range(4):
        pairs=[]
        for idx in range(128):
            i=int(rng.integers(1,101))
            j=int(neighbors[i][rng.integers(12)])
            pairs.append((idx % 4, i, j, int(rng.integers(2))))
        steps.append(pairs)
    return steps


def edits(a, b, ia, ib, op, side):
    pos=ib+side
    if op == 0:
        return a[:ia]+a[ia+1:], b[:pos]+a[ia:ia+1]+b[pos:]
    if op == 1:
        return a[:ia]+b[ib:ib+1]+a[ia+1:], b[:ib]+a[ia:ia+1]+b[ib+1:]
    if op == 2:
        return a[:ia]+b[ib:], b[:ib]+a[ia:]
    if ia+1 >= len(a):
        return a, b
    return a[:ia]+a[ia+2:], b[:pos]+a[ia:ia+2]+b[pos:]


def descend(fleet, steps, value):
    fleet=list(fleet)
    prices = [value(r) for r in fleet]
    count = 0
    def price(r):
        return value(r) if r else 0.
    for wave in steps:
        owner = {v:(j,p) for j,route in enumerate(fleet) for p,v in enumerate(route)}
        best, change = 1.e-7, None
        for op,i,j,side in wave:
            x,ia=owner[i]; y,ib=owner[j]
            if x==y:
                continue
            a,b=edits(fleet[x],fleet[y],ia,ib,op,side)
            if a==fleet[x] and b==fleet[y]:
                continue
            va,vb=price(a),price(b)
            delta=va+vb-prices[x]-prices[y]
            if delta>best:
                best,change=delta,(x,y,a,b,va,vb)
        if change is not None:
            x,y,a,b,va,vb=change
            fleet[x],fleet[y]=a,b
            prices[x],prices[y]=va,vb
            count+=1
    return tuple(r for r in fleet if r),sum(prices),count


def run_case(i):
    offer=read(BANK/f"{i}.json");case=offer["case"]
    cfg=read(HERE/"headcapture01.json")
    if cfg["blocks"][i]["case"]!=case:
        raise ValueError("Different bank case")
    data=read(HERE/"heldout_sources.json")
    nodes,d,model,value=development_model(case,data["derived_instances"][case])
    states=read(HERE/"archived_states.json")[case]
    capture=HERE/"results/headcapture01/out";scale=read(capture/f"{i}.json")["scale"]
    wall,cpu=perf_counter(),process_time()
    rows=[];audits=0;positions=0
    with np.load(capture/f"{i}.npz") as z:
        for row in offer["rows"]:
            k=row["mask_index"]
            original=tuple(map(tuple,states["initial" if k<6 else "terminal"]))
            rng=np.random.default_rng([56303,i,k])
            h=z["embed"][k]
            shuffle_probabilities(h@h.T*scale,d,original,z["mask"][k].astype(int),rng)
            order=chain(original,int(rng.integers(len(original))),d)
            base,*direct=row["U"]
            selected=sorted((n for n,u in enumerate(direct) if u<base-1.e-6),key=lambda n:(-direct[n],n))[:8]
            candidates=[(order,base)]+[(tuple(row["orders"][j]),direct[j]) for j in selected]
            schedule=schedules(d,[56497,i,k])
            final=[];accepted=[];fleets=[]
            for seq,old in candidates:
                current,fleet,_=split_order(seq,nodes[:,3],value)
                if abs(current-old)>1.e-6:raise ValueError("Starting split changed")
                result,u,n=descend(fleet,schedule,value)
                check=independent_audit(model,[model.route(r) for r in result],True)
                if abs(u-check)>1.e-6 or check<old-1.e-6:raise ValueError("Local search fleet differs")
                audits+=1
                positions+=sum(len(w) for w in schedule)
                final.append(check); accepted.append(n);fleets.append(result)
            rows.append(dict(k=k,selected=selected,U=final,accepted=accepted,fleets=fleets))
    doc=dict(case=case,rows=rows,audits=audits,attempts=positions,prices=value.cache_info().misses,
             cpu_s=process_time()-cpu,wall_s=perf_counter()-wall)
    (OUT/f"{i}.json").write_text(json.dumps(doc,allow_nan=False,separators=(",",":")),encoding="utf8")
    return i


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--case",type=int);args=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if args.case is not None:
        print(run_case(args.case))
        return
    missing=[i for i in range(27) if not (OUT/f"{i}.json").exists()]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for case in pool.map(run_case,missing):
            print(case,flush=True)


if __name__=="__main__":
    main()
