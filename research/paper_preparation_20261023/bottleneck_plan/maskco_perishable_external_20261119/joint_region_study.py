"""Paired comparison on preserved development states, with an attainable-mask diagnostic."""
from functools import lru_cache
from statistics import mean
from pathlib import Path
import json
import time
import numpy as np
from route_columns import HERE,BASE,DEST,assemble,read,core,matrices,independent_audit
from exchange_support import support
from joint_reinsertion import nominate,optimize

ARMS=("distance","physical","neural","shuffled","scattered")
OUT=HERE/"results/jointcapture01"

def study():
    cfg=read(HERE/"jointcapture01.json");status=read(OUT/"status.json")
    if status["complete"]!=5 or status["failed"] or not status["end"]:raise ValueError("Incomplete forwards")
    rows,_=assemble();meta=read(HERE/"heldout_sources.json");folds=read(BASE/"fit.json")["folds"]
    archive=read(HERE/"archived_states.json")
    targets={(r["case"],r["group"],r["rep"]):r for r in read(DEST/"full_opportunity.json")["rows"] if r["gain"]>1.e-6}
    records=[];oracles=[];start=time.perf_counter();cpu=time.process_time()
    for ci,b in enumerate(cfg["blocks"]):
        c=b["case"];reports=read(OUT/f"out/{ci}.json")
        if reports["case"]!=c or reports["max_logit_error"]>.002 or reports["contexts"]!=10:raise ValueError("Forward identity")
        x=dict(np.load(OUT/f"out/{ci}.npz"));ex=[e for e in rows if e["case"]==c]
        nodes=np.asarray(read(HERE/meta["derived_instances"][c]["file"])["nodes"])
        d=np.linalg.norm(nodes[:,None,1:3]-nodes[None,:,1:3],axis=2);mass=nodes[:,3]
        model=core.PublicationModel(dict(name=c,nodes=nodes.tolist(),distance=d.tolist()),core.load_config(),"main_Product1",.8)
        @lru_cache(maxsize=200000)
        def price(r):
            v=model.route(r);return v.net_benefit if v.feasible else -np.inf
        k=folds[c];fw=dict(np.load(BASE/f"readout_fold{k}.npz"));bw=dict(np.load(BASE/f"predecessor_fold{k}.npz"))
        for ri,z in enumerate(b["regions"]):
            group=z["group"];state=z["state"];rep=z["rep"];parts=list(map(tuple,z["parts"]))
            scattered=ex[rep+6*(state=="terminal")]
            targeted=dict(scattered,mask=x["mask"][ri].astype(int),phys=x["phys"][ri],h=x["embed"][ri])
            if list(targeted["mask"])!=z["mask"] or not {i for r in parts for i in r}<=set(z["mask"]):raise ValueError("Mismatched target")
            ranks=matrices(targeted,fw,bw,rep)
            ranks["scattered"]=matrices(scattered,fw,bw,rep)["neural"]
            routes=list(map(tuple,archive[c][state]))
            outside=[r for r in routes if r not in parts]
            def audit(chosen):
                return independent_audit(model,[model.route(r) for r in outside+list(map(tuple,chosen))],True)
            initial=audit(parts);pool={};arms={}
            noise=np.random.default_rng(871930+100*ci+ri).gumbel(size=101)*.3
            target=targets.get((c,group,rep)) if state=="terminal" else None
            def evaluate(mask):
                if mask not in pool:
                    tick=time.process_time();r=optimize(parts,mask,mass,price)
                    final=audit(r["chosen"])
                    if abs(final-initial-r["gain"])>1.e-6:raise ValueError("Independent full-fleet audit failed")
                    r.update(final_U=final,cpu_s=time.process_time()-tick)
                    if target:r["missing_for_target"]=len(support(parts,target["winner"],mask))
                    pool[mask]=r
                return pool[mask]
            for a in ARMS:
                requested=[nominate(parts,ranks[a],j) for j in (np.zeros(101),noise)]
                trials=[evaluate(mask) for mask in requested]
                arms[a]=dict(nominations=requested,gain=max(r["gain"] for r in trials))
            ncompared=len(pool)
            if target:
                if list(map(list,parts))!=target["parts"]:raise ValueError("Mismatched opportunity")
                mask=tuple(support(parts,target["winner"]))
                result=evaluate(mask)
                oracles.append(dict(case=c,group=group,rep=rep,known=target["gain"],**result))
            records.append(dict(case=c,group=group,state=state,rep=rep,unique_compared=ncompared,initial_U=initial,
                                arms=arms,solutions=list(pool.values())))
        print(c,len(records),flush=True)
    if len(records)!=50 or len(oracles)!=3:raise ValueError("Unpaired data")
    result=dict(regions=records,target_diagnostic=oracles,cpu_s=time.process_time()-cpu,wall_s=time.perf_counter()-start)
    (OUT/"regions.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf8")
    return result

def summarize(v):
    rows=v["regions"];near=[r for r in rows if r["group"]=="neighbour"]
    groups={"原邻近终态":[r for r in near if r["state"]=="terminal" and r["rep"]<2],
            "新增邻近锚点":[r for r in near if r["rep"]>=2],"全部邻近初解":[r for r in near if r["state"]=="initial"],
            "全部邻近终态":[r for r in near if r["state"]=="terminal"],"最大余量终态":[r for r in rows if r["group"]=="max_spare"]}
    def metric(rs):
        return {a:mean(r["arms"][a]["gain"] for r in rs) for a in ARMS}
    stats={n:metric(rs) for n,rs in groups.items()};geo={c:metric([r for r in near if r["case"]==c]) for c in sorted({r["case"] for r in near})}
    wins=sum(m["neural"]>m["distance"]+1.e-6 for m in geo.values())
    passed=(all(stats[n]["neural"]>stats[n]["distance"]+1.e-6 for n in ("原邻近终态","新增邻近锚点"))
            and wins>=3 and all(mean(m["neural"]-m[a] for m in geo.values())>1.e-6 for a in ("physical","shuffled")))
    trials=[s for r in rows for s in r["solutions"]];oracle=v["target_diagnostic"]
    n=sum(r["unique_compared"] for r in rows)
    lines=["# 整区掩码与多客同时换车","","五个丹麦开发几何，共 50 个区域。每种方法每区提名两次，表中为各自较优车队相对同一起点的净效益增量，单位 U。","",
        "| 区域 | 数量 | 同规则距离 | 纯物理 | MASKCO | 分层置乱 | 散布掩码 MASKCO | MASKCO 减距离 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name,m in stats.items():
        lines.append("| "+name+f" | {len(groups[name])} | "+" | ".join(f"{m[a]:.3f}" for a in ARMS)+f" | {m['neural']-m['distance']:+.3f} |")
    lines+=["",f"邻近区域热图在 {wins}/5 个几何超过距离。三来源开发门槛"+("通过。" if passed else "未通过。"),"",
            "| 已知终态换车 | 最小移动客户数 | 已知增量 U | 给定最小掩码后可实现 U | 求解剩余界差 U |","|---|---:|---:|---:|---:|"]
    for r in oracle:
        gap=r["gap_U"];s="未知" if gap is None else f"{gap:.3f}"
        lines.append(f"| {r['case'][-1]} / {r['group']} | {len(r['hidden'])} | {r['known']:.3f} | {r['gain']:.3f} | {s} |")
    optimal=sum(s["optimal"] for s in trials)
    lines+=["",f"五臂共 500 次掩码提名，区内去重后 {n} 次重构。连同已知目标诊断，共独立审计 {len(trials)} 个车队，其中 {optimal} 次求得候选池最优。单机求解及审计墙钟 {v['wall_s']:.1f} 秒、CPU {v['cpu_s']:.1f} 秒。","",
        "[逐区域与逐客户结果](regions.json) · [固定机制与晋级门槛](../../联合掩码与整组插回.md)"]
    (OUT/"analysis.json").write_text(json.dumps(dict(stats=stats,by_geometry=geo,wins=wins,passed=passed,requested=500,distinct=n,audits=len(trials),optimal=optimal),indent=2),encoding="utf8")
    (OUT/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf8");print("\n".join(lines))
if __name__=="__main__":summarize(study())
