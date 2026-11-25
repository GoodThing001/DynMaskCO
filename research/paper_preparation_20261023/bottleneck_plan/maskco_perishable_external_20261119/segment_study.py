"""Paired development evaluation of complete segment assignments under masked edge guidance."""
from functools import lru_cache
from statistics import mean
import json,time
import numpy as np
from segment_assignment import segment_patterns,assignments
from route_columns import HERE,BASE,assemble,read,core,independent_audit,matrices

ARMS=("distance","physical","neural","shuffled","kilometres")
OUT=HERE/"results/segment_assignment01"

def main():
    OUT.mkdir(exist_ok=True)
    start=time.perf_counter();cpu=time.process_time();examples,_=assemble()
    archive=read(HERE/"archived_states.json");catalog=read(HERE/"heldout_sources.json");folds=read(BASE/"fit.json")["folds"];rows=[]
    for ci,initial_block in enumerate(read(HERE/"jointcapture01.json")["blocks"]):
        c=initial_block["case"];ex=[e for e in examples if e["case"]==c];seen=set()
        nodes=np.asarray(read(HERE/catalog["derived_instances"][c]["file"])["nodes"])
        d=np.linalg.norm(nodes[:,None,1:3]-nodes[None,:,1:3],axis=2);mass=nodes[:,3]
        model=core.PublicationModel(dict(name=c,nodes=nodes.tolist(),distance=d.tolist()),core.load_config(),"main_Product1",.8)
        @lru_cache(maxsize=60000)
        def value(r):
            if not r:return 0.
            v=model.route(r);return v.net_benefit if v.feasible else -np.inf
        k=folds[c];fw=dict(np.load(BASE/f"readout_fold{k}.npz"));bw=dict(np.load(BASE/f"predecessor_fold{k}.npz"))
        for phase in ("jointcapture01","utilitycapture01"):
            b=read(HERE/(phase+".json"))["blocks"][ci]
            with np.load(HERE/"results"/phase/f"out/{ci}.npz") as f:
                embeds=f["embed"];physical=f["phys"];masks=f["mask"].astype(int)
            for ix,z in enumerate(b["regions"]):
                if z["group"]!="neighbour":continue
                state=z["state"];rep=z["rep"];parts=list(map(tuple,z["parts"]));key=(state,tuple(sorted(parts)))
                if key in seen:continue
                seen.add(key)
                if masks[ix].tolist()!=z["mask"] or {i for r in parts for i in r}-set(z["mask"]):raise ValueError("Region-mask mismatch")
                true_e=dict(ex[rep+6*(state=="terminal")],h=embeds[ix],phys=physical[ix],mask=masks[ix])
                ranks=matrices(true_e,fw,bw,rep);ranks["kilometres"]=-d
                table=np.stack([ranks[a] for a in ARMS])
                @lru_cache(maxsize=None)
                def measure(r):
                    if not r:return np.zeros(len(ARMS)+1)
                    return np.r_[value(r),table[:,[0,*r],[*r,0]].sum(axis=1)]
                routes=list(map(tuple,archive[c][state]));outside=[r for r in routes if r not in parts]
                initial=independent_audit(model,[model.route(r) for r in routes],True)
                patterns=segment_patterns(parts,771403+ci*384+rep+32*(state=="terminal"))
                options=[];audited=set()
                for j,pattern in enumerate(patterns):
                    gains,winner=assignments(parts,pattern,mass,measure)
                    selected=tuple(sorted(winner))
                    if selected not in audited:
                        u=independent_audit(model,[model.route(r) for r in outside+list(selected)],True)
                        if abs(u-initial-gains[0])>1.e-6:raise ValueError("Independent fleet audit failed")
                        audited.add(selected)
                    options.append(dict(pattern=pattern,gain=gains[0],surrogate=dict(zip(ARMS,gains[1:])),winner=winner))
                picked={a:sorted(range(len(options)),key=lambda j:-options[j]["surrogate"][a])[:4] for a in ARMS}
                picked["random"]=list(range(min(4,len(options))))
                best={a:max(options[j]["gain"] for j in ids) for a,ids in picked.items()}
                rows.append(dict(case=c,state=state,rep=rep,parts=parts,picked=picked,net=best,all=max(v["gain"] for v in options),
                    audits=len(audited),options=options,original_U=initial));measure.cache_clear()
        print(c,len(rows),flush=True)
    if len(rows)!=42:raise ValueError("Missing original route regions")
    (OUT/"rows.json").write_text(json.dumps(rows,allow_nan=False),encoding="utf8")
    elapsed=dict(cpu_s=time.process_time()-cpu,wall_s=time.perf_counter()-start)
    (OUT/"elapsed.json").write_text(json.dumps(elapsed),encoding="utf8")
    summarize(rows,elapsed["cpu_s"],elapsed["wall_s"])

def summarize(rows,cpu,wall):
    cases=sorted({r["case"] for r in rows});arms=list(ARMS)+["random","all"]
    def avg(group):return {a:mean((r["all"] if a=="all" else r["net"][a]) for r in group) for a in arms}
    src={s:{c:avg([r for r in rows if r["case"]==c and r["state"]==s]) for c in cases} for s in ("initial","terminal")}
    bystate={s:{a:mean(q[a] for q in figures.values()) for a in arms} for s,figures in src.items()}
    bycase={c:{a:mean(src[s][c][a] for s in src) for a in arms} for c in cases}
    wins=sum(bool(v["neural"]>v["distance"]+1.e-6) for v in bycase.values())
    passes=wins>=3 and all(v["neural"]>v["distance"]+1.e-6 for v in bystate.values()) and all(mean(v["neural"]-v[a] for v in bycase.values())>1.e-6 for a in ("physical","shuffled"))
    audit=sum(r["audits"] for r in rows);n=sum(len(r["options"]) for r in rows)
    result=dict(by_state=bystate,by_case=bycase,state_case=src,patterns=n,independent_audits=audit,passed=bool(passes),wins=wins,cpu_s=cpu,wall_s=wall)
    (OUT/"analysis.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    lines=["# 多车片段同步指派","","五个丹麦开发几何，42 个互异路线区域。每臂从同一批片段组合中选四个，再给每个组合求严格经济最优指派。每种起点先对同几何的路线区域取均值，单位 U。","",
        "| 起点 | 同规则距离 | 纯物理 | MASKCO | 置乱 | 公里距离 | 随机 | 全试 |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for s,m in bystate.items():lines.append("| "+s+" | "+" | ".join(f"{m[a]:.3f}" for a in arms)+" |")
    lines+=["",f"神经版在 {wins}/5 个几何比同规则距离带来更多净效益。三来源开发门槛"+("通过。" if passes else "未通过。"),"",
        f"累计 {n} 个片段组合，逐区域去掉重复胜出车队后独立审计 {audit} 个车队。"+(f"CPU {cpu:.1f} 秒，墙钟 {wall:.1f} 秒。" if cpu is not None else "阶段 CPU 与墙钟分项未留存。"),"",
        "[原始指派与分数](rows.json) · [候选设计及门槛](../../多车片段同步指派.md)"]
    (OUT/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf8");print("\n".join(lines))
if __name__=="__main__":main()
