"""Learn whether two customers on different incumbent routes belong to one economic tour."""
from collections import Counter,defaultdict
from functools import lru_cache
from pathlib import Path
from statistics import mean
import json,time
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from directed_decoder_fit import HERE,SOURCES,assemble,weighting,read

BASE=HERE/"results/headcapture01";OUT=HERE/"results/vehicle_partner01"

def owners(routes):
    o=np.full(101,-1,int)
    for j,r in enumerate(routes):
        if np.any(o[r]!=-1):raise ValueError("Repeated customer in teacher")
        o[r]=j
    if (o[1:]<0).any() or o[0]!=-1:raise ValueError("Missing customer in teacher")
    return o

def features(nodes,routes):
    routes=[list(r) for r in routes];owner=owners(routes)
    geo=nodes[:,1:3];d=np.linalg.norm(geo[:,None]-geo[None,:],axis=2)
    dr=d.copy();np.fill_diagonal(dr,np.inf);rank=np.argsort(np.argsort(dr,axis=1,kind="stable"),axis=1,kind="stable")
    mass=nodes[:,3]/20000.;times=nodes[:,4:7]/1440.
    loads=np.zeros(101);n=np.zeros(101);oldlen=np.zeros(101);cross=np.zeros_like(d)
    for r in routes:
        loads[r]=mass[r].sum();n[r]=len(r);cross[r,:]=d[r,:].min(axis=0)
        for i,a,b in zip(r,[0]+r[:-1],r[1:]+[0]):oldlen[i]=(d[i,a]+d[i,b])/2
    v=[]
    def pair(z):
        v.extend([(z[:,None]+z[None,:])*.5,np.abs(z[:,None]-z[None,:])])
    v.append(np.log1p(d))
    v.extend([np.log1p(rank+rank.T),np.abs(np.log1p(rank)-np.log1p(rank.T))])
    for x in (mass,times[:,0],times[:,1],times[:,1]-times[:,0],times[:,2],loads,n,np.log1p(oldlen),np.log1p(d[:,0])):
        pair(x)
    v.extend([np.minimum(times[:,1,None],times[None,:,1])-np.maximum(times[:,0,None],times[None,:,0]),
              np.log1p(np.minimum(cross,cross.T)),np.log1p(np.maximum(cross,cross.T))])
    eligible=(owner[:,None]!=owner[None,:])&(mass[:,None]+mass[None,:]<=1.+1.e-8)
    eligible[0]=False;eligible[:,0]=False
    return dict(physical=np.stack(v,axis=-1).astype(np.float32),eligible=eligible,dist=d)

def pair_inputs(e,ctx,projector,i,j):
    ph=ctx["physical"][i,j]
    active=np.zeros(101,bool);active[e["mask"]]=True
    ph=np.column_stack((ph,active[i].astype(float)+active[j]))
    h=e["h"];v=np.clip((h-projector["eh"])@projector["pca"],-6,6)
    raw=h@h.T*e["scale"]
    nh=np.column_stack((raw[i,j],v[i]*v[j],np.abs(v[i]-v[j])))
    return np.column_stack((ph,nh)).astype(np.float32),ph.shape[1]

def pair_truth(catalog):
    loads=read(HERE/"replicate01.json")["blocks"];teacher=defaultdict(list)
    for ix,b in enumerate(loads):
        c=b["case"];seed=b["seed"]
        if (c,seed) not in catalog:continue
        v=read(HERE/f"results/replicate01/out/{ix}.json")["arms"][catalog[c,seed]]
        if v["status"]!="AUDITED":raise ValueError("Teacher identity")
        o=owners(v["routes"]);teacher[c].append((seed,o[:,None]==o[None,:]))
    return {c:np.array([m for _,m in sorted(ms)],dtype=np.float32) for c,ms in teacher.items() if len(ms)==2}

def fit(x,y,w):
    w=np.array(w,dtype=float);w/=w.sum()
    x=np.array(x,dtype=float);mu=w@x;sd=np.maximum(np.sqrt(w@((x-mu)**2)),.001)
    x-=mu;x/=sd
    def fun(b):
        u=x@b[:-1]+b[-1];p=expit(u)
        loss=w@(np.logaddexp(0.,u)-y*u)+.0005*float(b[:-1]@b[:-1])
        res=w*(p-y)
        return loss,np.r_[res@x+.001*b[:-1],res.sum()]
    start=np.zeros(x.shape[1]+1);rate=np.clip(w@y,.0001,.9999);start[-1]=np.log(rate/(1.-rate))
    out=minimize(fun,start,jac=True,method="L-BFGS-B",options=dict(maxiter=180,ftol=1.e-10,gtol=1.e-7))
    if not out.success:raise ValueError(out.message)
    return mu,sd,out.x

def apply(x,pars):
    mu,sd,b=pars
    return ((x-mu)/sd)@b[:-1]+b[-1]

def difference_placebo(physical,neural,dist,seed):
    order=np.argsort(dist,kind="stable");rng=np.random.default_rng(seed);delta=neural-physical;p=physical.copy()
    for t in range(0,len(order),8):
        g=order[t:t+8];p[g]+=delta[rng.permutation(g)]
    return p

def assess(i,j,y,teacher,dist,logits):
    out={};unique=np.unique(i)
    for a,z in logits.items():
        loss=float(np.mean(np.logaddexp(0.,z)-y*z))
        hit=0.;seeds=np.zeros(2);usable=0
        for q in unique:
            idx=np.flatnonzero(i==q);order=np.argsort(dist[idx] if a=="distance" else -z[idx],kind="stable")[:8]
            pick=idx[order];hit+=y[pick].sum();seeds+=teacher[:,i[pick],j[pick]].sum(axis=1);usable+=int(y[idx].sum()>0)
        positives=float(y.sum());total=teacher[:,i,j].sum(axis=1)
        out[a]=dict(bce=loss,recall=hit/positives if positives else None,
                    seed_hits=seeds.astype(int).tolist(),seed_total=total.astype(int).tolist(),
                    queries=len(unique),active_queries=usable,partners=len(i))
    return out

def main():
    OUT.mkdir(exist_ok=True);wall=time.perf_counter();cpu=time.process_time()
    rows,catalog=assemble();parents=read(HERE/"archived_states.json");meta=read(HERE/"heldout_sources.json")
    teachers=pair_truth(catalog);folds=read(BASE/"fit.json")["folds"]
    data={};counter=Counter()
    for e in rows:
        c=e["case"];n=counter[c];counter[c]+=1;state="initial" if n<6 else "terminal"
        if (c,state) not in data:
            nodes=np.array(read(HERE/meta["derived_instances"][c]["file"])["nodes"])
            data[c,state]=features(nodes,parents[c][state])
        e.update(orig_state=state,rep=n%6)
    output=[]
    for k in range(5):
        train=[e for e in rows if folds[e["case"]]!=k];test=[e for e in rows if folds[e["case"]]==k]
        with np.load(BASE/f"readout_fold{k}.npz") as f:projector={n:f[n] for n in ("eh","pca")}
        xs=[];ys=[];ws=[]
        for ix,(e,weight) in enumerate(zip(train,weighting(train))):
            c=e["case"];ctx=data[c,e["orig_state"]];used,partners=np.where(ctx["eligible"][e["mask"]]);i=e["mask"][used]
            y=teachers[c][:,i,partners].mean(axis=0)
            rng=np.random.default_rng(966301+k*1000+ix);keep=(y>0)|(rng.integers(8,size=len(y))==0)
            x,nphys=pair_inputs(e,ctx,projector,i[keep],partners[keep]);xs.append(x);ys.append(y[keep])
            ws.append(weight*np.where(y[keep]>0,1.,8.)/len(y))
        x=np.concatenate(xs);y=np.concatenate(ys);w=np.concatenate(ws);del xs,ys,ws
        fitted={n:fit(x[:,:cols],y,w) for n,cols in (("distance",1),("physical",nphys),("neural",x.shape[1]))}
        np.savez_compressed(OUT/f"fold{k}.npz",**projector,**{a+"_"+n:v for a,ps in fitted.items() for n,v in zip(("mu","sd","b"),ps)})
        del x,y,w
        for ix,e in enumerate(test):
            c=e["case"];ctx=data[c,e["orig_state"]];used,j=np.where(ctx["eligible"][e["mask"]]);i=e["mask"][used]
            x,nphys=pair_inputs(e,ctx,projector,i,j);y=teachers[c][:,i,j].mean(axis=0);dist=ctx["dist"][i,j]
            logits={a:apply(x[:,:len(pars[0])],pars) for a,pars in fitted.items()}
            placebo=logits["physical"].copy()
            for q in np.unique(i):
                sub=i==q;placebo[sub]=difference_placebo(logits["physical"][sub],logits["neural"][sub],dist[sub],609031+k*100000+ix*128+q)
            logits["shuffled"]=placebo
            output.append(dict(case=c,src=e["src"],geo=e["geo"],state=e["orig_state"],rep=e["rep"],fold=k,
                     arms=assess(i,j,y,teachers[c],dist,logits)))
        print("fold",k,len(train),len(test),flush=True)
    if len(output)!=324:raise ValueError("Incomplete geometry holdout")
    (OUT/"contexts.json").write_text(json.dumps(output,allow_nan=False),encoding="utf8")
    elapsed=dict(cpu_s=time.process_time()-cpu,wall_s=time.perf_counter()-wall)
    (OUT/"elapsed.json").write_text(json.dumps(elapsed),encoding="utf8")
    summarize(output,elapsed["cpu_s"],elapsed["wall_s"])

def summarize(output,cpu,wall):
    result={};better=0;passed=True;arms=("distance","physical","neural","shuffled")
    lines=["# 预测原属不同车辆的客户是否会被重新配到同车","","同车概率用其他几何的两次求解标签拟合，查询限于实际遮蔽客户与其他车中容量兼容的伙伴。每格为二元交叉熵 / 前八同车伙伴召回率，前者越低、后者越高越好。","",
        "| 开发来源 | 公里距离 | 纯物理 | 加 MASKCO | 置乱 | 神经召回高于物理的几何 |","|---|---:|---:|---:|---:|---:|"]
    for s in SOURCES:
        arows=[r for r in output if r["src"]==s];geo={}
        for g in sorted({r["geo"] for r in arows}):
            cases={r["case"] for r in arows if r["geo"]==g}
            geo[g]={a:{key:mean(mean(r["arms"][a][key] for r in arows if r["case"]==c and r["arms"][a][key] is not None) for c in cases)
                            for key in ("bce","recall")} for a in arms}
        perf={a:{key:mean(v[a][key] for v in geo.values()) for key in ("bce","recall")} for a in arms}
        wins=sum(bool(v["neural"]["recall"]>v["physical"]["recall"]) for v in geo.values())
        lift=all(perf["neural"]["recall"]>perf[a]["recall"] for a in ("physical","shuffled"))
        better+=lift;passed=passed and perf["neural"]["bce"]<perf["physical"]["bce"] and perf["neural"]["recall"]>=perf["distance"]["recall"] and (s!=SOURCES[0] or lift)
        record=[r["arms"]["neural"] for r in arows]
        result[s]=dict(metrics=perf,by_geometry=geo,wins=wins,queries=sum(q["queries"] for q in record),query_with_target=sum(q["active_queries"] for q in record),
                        labelled_pairs=sum(q["partners"] for q in record),seed_hits=np.sum([q["seed_hits"] for q in record],axis=0).tolist(),seed_total=np.sum([q["seed_total"] for q in record],axis=0).tolist())
        lines.append("| "+s+" | "+" | ".join(f"{perf[a]['bce']:.4f} / {perf[a]['recall']:.2%}" for a in arms)+f" | {wins}/{len(geo)} |")
    passed=bool(passed and better>=2)
    (OUT/"analysis.json").write_text(json.dumps(dict(sources=result,passed=passed,cpu_s=cpu,wall_s=wall),indent=2))
    lines+=["",f"共有 {sum(v['queries'] for v in result.values()):,} 个查询，{sum(v['labelled_pairs'] for v in result.values()):,} 个容量兼容的有向客户对上下文。转入整车组经济评估的门槛"+("通过。" if passed else "未通过。"),"",
        (f"五折拟合及评价 CPU {cpu:.1f} 秒，墙钟 {wall:.1f} 秒。" if cpu is not None else "五折拟合的 CPU 与墙钟分项未留存。"),"",
        "[逐来源和逐几何指标](analysis.json) · [学习目标与门槛](../../跨车同车伙伴解码.md)"]
    (OUT/"RESULTS.md").write_text("\n".join(lines)+"\n",encoding="utf8");print("\n".join(lines))
if __name__=="__main__":main()
