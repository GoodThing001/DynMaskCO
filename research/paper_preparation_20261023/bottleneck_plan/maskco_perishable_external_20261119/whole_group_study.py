"""Evaluate full multi-source repartition on two Danish parent seeds with equal group budgets."""
from collections import defaultdict
from functools import lru_cache
from statistics import mean
import time,json
import numpy as np
from directed_decoder_fit import distance_score
from route_columns import HERE,BASE,read,matrices,core,independent_audit
from whole_group import candidate_groups,alternatives,group_scores,select_two,priced_cover,region_columns

OUT=HERE/"results/whole_group01"
ARMS=("distance","physical","neural","shuffled","spare","random")

def evaluate_case(ci):
    start=time.perf_counter();cpu=time.process_time()
    seeds=(("jointcapture01","archived_states.json",32001),("groupcapture01","group_parent_states.json",32003))
    c=read(HERE/"jointcapture01.json")["blocks"][ci]["case"];catalog=read(HERE/"heldout_sources.json")
    nodes=np.asarray(read(HERE/catalog["derived_instances"][c]["file"])["nodes"])
    d=np.linalg.norm(nodes[:,None,1:3]-nodes[None,:,1:3],axis=2);mass=nodes[:,3]
    model=core.PublicationModel(dict(name=c,nodes=nodes.tolist(),distance=d.tolist()),core.load_config(),"main_Product1",.8)
    @lru_cache(maxsize=180000)
    def value(r):
        v=model.route(r);return v.net_benefit if v.feasible else -np.inf
    k=read(BASE/"fit.json")["folds"][c];fw=dict(np.load(BASE/f"readout_fold{k}.npz"));bw=dict(np.load(BASE/f"predecessor_fold{k}.npz"))
    targets={(r["rep"],r["group"]):r for r in read(HERE/"results/columns01/full_opportunity.json")["rows"] if r["case"]==c and r["gain"]>1.e-6}
    cases=[];diagnostics=[]
    for phase,afile,seed in seeds:
        b=read(HERE/(phase+".json"))["blocks"][ci];arc=read(HERE/afile)[c];receipt=read(HERE/f"results/{phase}/out/{ci}.json")
        if receipt["case"]!=c or receipt["contexts"]!=len(b["regions"]) or receipt["max_logit_error"]>.002:raise ValueError("Masked forward identity")
        with np.load(HERE/f"results/{phase}/out/{ci}.npz") as src:
            x=src["embed"];phys=src["phys"];masks=src["mask"].astype(int)
        for ix,z in enumerate(b["regions"]):
            if z["rep"]>1:continue
            state=z["state"];rep=z["rep"];parts=list(map(tuple,z["parts"]));all_routes=list(map(tuple,arc[state]))
            if masks[ix].tolist()!=z["mask"] or set(i for r in parts for i in r)-set(masks[ix]):raise ValueError("Region-mask mismatch")
            ranks=matrices(dict(dist=distance_score(d),h=x[ix],phys=phys[ix],mask=masks[ix]),fw,bw,rep)
            groups=candidate_groups(parts);scores={}
            for a in ARMS[:4]:scores[a]=group_scores(groups,alternatives(parts,ranks[a],mass))
            scores["spare"]=[sum(1.-sum(mass[i] for i in parts[j])/20000. for j in g) for g in groups]
            scores["random"]=np.random.default_rng(400993+ci*50+ix+1000*(seed==32003)).random(len(groups))
            choice={a:select_two(groups,v) for a,v in scores.items()};nominations=set(g for arm in choice.values() for g in arm)
            target=targets.get((rep,z["group"])) if seed==32001 and state=="terminal" else None
            if target:
                changed=tuple(i for i,r in enumerate(parts) if list(r) not in target["winner"])
                if set(parts)!=set(map(tuple,target["parts"])):raise ValueError("Target region")
                nominations.add(changed)
            clients=sorted({v for g in nominations for j in g for v in parts[j]})
            columns=region_columns(clients,mass,value);out={}
            orig=independent_audit(model,[model.route(r) for r in all_routes],True)
            for g in sorted(nominations):
                selected=[parts[i] for i in g];outside=[r for r in all_routes if r not in selected]
                sol=priced_cover(selected,columns,value)
                u=independent_audit(model,[model.route(r) for r in outside+sol["routes"]],True)
                if abs(u-orig-sol["gain"])>1.e-6:raise ValueError("Group full-fleet audit")
                out[g]=sol
            arms={a:dict(selected=gs,gain=max(out[g]["gain"] for g in gs),upper=max(out[g]["gain"]+out[g]["gap_U"] for g in gs)) for a,gs in choice.items()}
            if target:
                oldsrc={i:j for j,r in enumerate(parts) for i in r}
                origins=max(len({oldsrc[i] for i in r}) for r in target["winner"])
                diagnostics.append(dict(case=c,group=z["group"],rep=rep,changed=changed,customers=sum(len(parts[i]) for i in changed),
                    max_origins=origins,known=target["gain"],achieved=out[changed]["gain"],gap_U=out[changed]["gap_U"],
                    covered={a:any(set(changed)<=set(g) for g in gs) for a,gs in choice.items()}))
            cases.append(dict(case=c,seed=seed,state=state,group=z["group"],rep=rep,parts=parts,available=len(groups),arms=arms,
                original_U=orig,solutions=[dict(group=g,**v) for g,v in out.items()]))
    if len(cases)!=8:raise ValueError("Missing Danish contexts")
    (OUT/f"case_{ci}.json").write_text(json.dumps(dict(rows=cases,diagnostics=diagnostics,cpu_s=time.process_time()-cpu,wall_s=time.perf_counter()-start),allow_nan=False),encoding="utf8")
    print(c,len(cases),flush=True)

def summarize():
    docs=[read(OUT/f"case_{i}.json") for i in range(5)];rows=[r for doc in docs for r in doc["rows"]]
    diagnostics=[r for doc in docs for r in doc["diagnostics"]];cells=defaultdict(list)
    for r in rows:
        cell=r["state"] if r["state"]=="initial" else "new_terminal" if r["seed"]==32003 else r["group"]
        cells[cell].append(r)
    if sorted(map(len,cells.values()))!=[10]*4 or len(diagnostics)!=3:raise ValueError("Incomplete contrasts")
    def avg(group,a):return mean(r["arms"][a]["gain"] for r in group)
    means={n:{a:avg(rs,a) for a in ARMS} for n,rs in cells.items()}
    use=[r for r in rows if r["group"]=="neighbour"]
    geo={c:{a:avg([r for r in use if r["case"]==c],a) for a in ARMS} for c in sorted({r["case"] for r in rows})}
    wins=sum(v["neural"]>v["distance"]+1.e-6 for v in geo.values())
    passed=all(means[n]["neural"]>means[n]["distance"]+1.e-6 for n in ("initial","neighbour","new_terminal")) and wins>=3 and all(mean(v["neural"]-v[a] for v in geo.values())>1.e-6 for a in ("physical","shuffled"))
    allsol=[s for r in rows for s in r["solutions"]];cpu=sum(doc["cpu_s"] for doc in docs)
    result=dict(means=means,geo=geo,wins=wins,passed=passed,audits=len(allsol),optimal=sum(s["optimal"] for s in allsol),
                max_gap_U=max(s["gap_U"] for s in allsol),cpu_s=cpu,diagnostics=diagnostics,wall_s_by_case=[doc["wall_s"] for doc in docs])
    (OUT/"analysis.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    lines=["# 允许一辆新车接收多个旧车的客户","","五个丹麦开发几何，每行十区。每臂提名至多两个完整车组，表中为入选车队相对同一起点的可行净增益 U。","",
           "| 起点和区域 | 距离 | 纯物理 | MASKCO | 置乱 | 空余容量 | 随机 | 热图减距离 |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for n,label in (("initial","初解"),("neighbour","32001 邻近终态"),("max_spare","32001 最大余量终态"),("new_terminal","32003 新终态")):
        m=means[n];lines.append("| "+label+" | "+" | ".join(f"{m[a]:.3f}" for a in ARMS)+f" | {m['neural']-m['distance']:+.3f} |")
    lines+=["",f"三个邻近组等权后，神经版在 {wins}/5 个几何优于距离。三来源开发门槛"+("通过。" if passed else "未通过。"),"",
           "| 已知终态改善 | 变动的旧车 / 客户 | 单辆新车最多来源数 | 已知净增 U | 给定完整车组后获得 U | 热图提名覆盖全组 |",
           "|---|---:|---:|---:|---:|---:|"]
    for t in diagnostics:
        lines.append(f"| {t['case'][-1]} | {len(t['changed'])} / {t['customers']} | {t['max_origins']} | {t['known']:.3f} | {t['achieved']:.3f} | "+("是" if t["covered"]["neural"] else "否")+" |")
    lines+=["",f"共独立复算 {len(allsol)} 个车队，{result['optimal']} 次证实候选池最优；最大剩余界差 {result['max_gap_U']:.3f} U。五个几何累计 CPU {cpu:.1f} 秒。","",
       "[逐几何统计](analysis.json) · [方案与门槛](../../整车组多来源重新配车.md)"]
    (OUT/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf8");print("\n".join(lines))
if __name__=="__main__":
    from concurrent.futures import ProcessPoolExecutor
    OUT.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=2) as pool:list(pool.map(evaluate_case,range(5)))
    summarize()
