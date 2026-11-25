"""Compare equal-size incremental route lists and identify missing economic complements."""
from collections import defaultdict
from functools import lru_cache
from itertools import combinations
from statistics import mean
import json
import time
import numpy as np
from route_columns import BASE,DEST,HERE,ARMS,matrices,partition,assemble,read,core,independent_audit


def ranked_paths(nodes,score,mass,factor):
    table={}
    for start in nodes:
        beam=[((start,),mass[start],float(score[0,start]))]
        for length,cap in ((2,4),(3,6),(4,8)):
            pool=[]
            for seq,load,x in beam:
                for v in nodes:
                    if v not in seq and load+mass[v]<=20000.+1.e-7:
                        y=x+score[seq[-1],v]
                        pool.append((-(y+score[v,0]),seq+(v,),load+mass[v],y))
            pool.sort();beam=[(seq,load,x) for _,seq,load,x in pool[:cap*factor]]
            table[start,length]=[r for r,_,_ in beam]
    return table


def minimal(parts,base,winner,value,init):
    prices=defaultdict(lambda:-np.inf)
    ids=[i for r in parts for i in r]
    for r in base|set(parts)|{(i,) for i in ids}:
        key=tuple(sorted(r));prices[key]=max(prices[key],value(r))
    missing=[r for r in winner if value(r)>prices[tuple(sorted(r))]+1.e-6]
    for size in range(len(missing)+1):
        for combo in combinations(missing,size):
            result,_,_=partition(parts,base|set(combo),value)
            gain=sum(map(value,result))-init
            if gain>1.e-6:return dict(required=size,missing_all=missing,witness=combo,gain=gain)
    raise ValueError("Known improving solution missing after adding all its routes")


def main():
    start=time.process_time();meta=read(HERE/"heldout_sources.json");archives=read(HERE/"archived_states.json")
    groups=read(DEST/"regions.json");opportunity=read(DEST/"full_opportunity.json")["rows"]
    cases=sorted({r["case"] for r in groups});data,_=assemble();folds=read(BASE/"fit.json")["folds"]
    rows=[];explanations=[]
    for c in cases:
        e0=[e for e in data if e["case"]==c];k=folds[c]
        forward=dict(np.load(BASE/f"readout_fold{k}.npz"));back=dict(np.load(BASE/f"predecessor_fold{k}.npz"))
        nodes=np.array(read(HERE/meta["derived_instances"][c]["file"])["nodes"])
        dist=np.linalg.norm(nodes[:,None,1:3]-nodes[None,:,1:3],axis=2);mass=nodes[:,3]
        model=core.PublicationModel(dict(nodes=nodes.tolist(),distance=dist.tolist(),name=c),core.load_config(),"main_Product1",.8)
        @lru_cache(maxsize=80000)
        def value(r):
            a=model.route(r);return a.net_benefit if a.feasible else -np.inf
        for z in (r for r in groups if r["case"]==c):
            state=z["state"];rep=z["rep"];parent=list(map(tuple,archives[c][state]))
            e=e0[(6 if state=="terminal" else 0)+rep];masks=matrices(e,forward,back,rep)
            opt={r["group"]:r for r in opportunity if r["case"]==c and r["rep"]==rep} if state=="terminal" else {}
            variants={"neighbour":z["source_routes"]}
            if state=="terminal":variants["max_spare"]=opt["max_spare"]["parts"]
            for kind,region in variants.items():
                parts=list(map(tuple,region));outside=[r for r in parent if r not in parts]
                ids=sorted(i for r in parts for i in r);part_U=sum(map(value,parts))
                common=ranked_paths(ids,masks["distance"],mass,1)
                extra={arm:ranked_paths(ids,masks[arm],mass,2) for arm in ARMS}
                cols=set(r for chunk in common.values() for r in chunk)
                choices={arm:set(cols) for arm in ARMS}
                for key,orig in common.items():
                    eligible={arm:[r for r in extra[arm][key] if r not in orig] for arm in ARMS}
                    count=min(len(orig),*(len(v) for v in eligible.values()))
                    for arm in ARMS:choices[arm].update(eligible[arm][:count])
                if len(set(map(len,choices.values())))!=1:raise ValueError("Unmatched candidate counts")
                gains={}
                for arm,columns in choices.items():
                    won,_,_=partition(parts,columns,value)
                    u=independent_audit(model,[model.route(r) for r in outside+won],True)
                    if abs(u-z["original_U"]-(sum(map(value,won))-part_U))>1.e-6:raise ValueError("Whole fleet differs")
                    gains[arm]=max(0.,u-z["original_U"])
                rows.append(dict(case=c,state=state,rep=rep,kind=kind,gains=gains,total=len(choices["distance"]),base=len(cols)))
                if kind in opt and opt[kind]["gain"]>1.e-6:
                    ex=minimal(parts,cols,list(map(tuple,opt[kind]["winner"])),value,part_U)
                    explanations.append(dict(case=c,rep=rep,group=kind,**ex))
    if len(rows)!=30:raise ValueError("Incomplete panel")
    avg={};lines=["# 距离与神经候选联合配车","","每个起点和路线长度增补相同名额。数值是从各自完整候选池保留的全车队净收益 U。","",
        "| 起点和区域 | 距离加距离 | 距离加物理 | 距离加热图 | 距离加置乱 |","|---|---:|---:|---:|---:|"]
    for state,kind,label in (("initial","neighbour","初解，邻近"),("terminal","neighbour","强终态，邻近"),("terminal","max_spare","强终态，最大余量")):
        subset=[r for r in rows if (r["state"],r["kind"])==(state,kind)]
        avg[state,kind]={a:mean(r["gains"][a] for r in subset) for a in ARMS}
        lines.append("| "+label+" | "+" | ".join(f"{avg[state,kind][a]:.3f}" for a in ARMS)+" |")
    scores={c:{a:mean(r["gains"][a] for r in rows if r["case"]==c and r["kind"]=="neighbour") for a in ARMS} for c in cases}
    wins=sum(v["neural"]>v["distance"]+1.e-7 for v in scores.values())
    t=avg["terminal","neighbour"]
    passed=bool(t["neural"]>t["distance"]+1.e-7 and wins>=3 and all(mean(v["neural"]-v[a] for v in scores.values())>1.e-7 for a in ("physical","shuffled")))
    lines+=["",f"两种起点合并后，{wins}/5 个丹麦几何优于距离扩展。晋级门槛"+("通过。" if passed else "未通过。"),"",
            "| 完整池改善的区域 | 距离池缺失的入选路线 | 首次提高净效益最少补回条数 | 首次改善 U |","|---|---:|---:|---:|"]
    for r in explanations:lines.append(f"| {r['case'][-1]} / {r['group']} / {r['rep']+1} | {len(r['missing_all'])} | {r['required']} | {r['gain']:.3f} |")
    lines+=["","[候选与晋级条件](../../配货候选的互补性.md)"]
    (DEST/"complements.json").write_text(json.dumps(dict(rows=rows,attribution=explanations,passed=passed,wins=wins,cpu_s=time.process_time()-start),indent=2))
    (DEST/"路线互补.md").write_text("\n".join(lines)+"\n",encoding="utf8")
    print("\n".join(lines))
if __name__=="__main__":main()
