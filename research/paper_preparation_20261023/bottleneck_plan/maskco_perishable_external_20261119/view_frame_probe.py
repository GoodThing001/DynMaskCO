"""Rotate only the two network input axes and compare fixed effort reconstruction."""
from pathlib import Path
import time,json,sys
import numpy as np
HERE=Path(__file__).resolve().parent

def view(batch,n):
    if not 0<=n<8:raise ValueError(n)
    f=batch["features"].copy();xy=batch["features"][...,:2]
    f[...,:2]=xy[...,::-1] if n&4 else xy
    if n&1:f[...,0]*=-1
    if n&2:f[...,1]*=-1
    return dict(batch,features=f)

def logprob(raw):
    z=np.asarray(raw,dtype=float);z=z-z.max(axis=1,keepdims=True)
    return z-np.log(np.exp(z).sum(axis=1,keepdims=True))

def blend(heats):return np.logaddexp.reduce(heats,axis=0)-np.log(len(heats))

def shuffle_probabilities(logp,dist,routes,mask,rng):
    active=np.zeros(len(logp),bool);active[mask]=True
    observed=np.zeros(logp.shape,bool)
    for r in routes:
        for i,j in zip((0,)+r,r+(0,)):
            if not active[i] and not active[j]:observed[i,j]=observed[j,i]=True
    result=logp.copy()
    for i in range(len(logp)):
        for state in (0,1,2,3):
            js=np.flatnonzero(active.astype(int)+2*observed[i]==state)
            js=js[(js!=0)&(js!=i)]
            seq=js[np.argsort(dist[i,js],kind="stable")]
            for start in range(0,len(seq),8):
                group=seq[start:start+8];result[i,group]=logp[i,rng.permutation(group)]
    return result

def main(ix):
    import external_common as h
    import pool_search as pool
    from block_split import bundle,library
    from paired_operator_probe import insertion,orders
    cfg=json.loads((HERE/"phase.json").read_text());case=cfg["blocks"][ix]["case"];archive=json.loads((HERE/"archived_states.json").read_text())[case]
    _,_,access,core,source,locked=h.install("ORDER_SPLIT",dict(candidate="ORDER_SPLIT",matched="ORDER_SPLIT_D"),False)
    from base_inputs import BaseInferenceInputs
    from timed_search import single_cpu_guard
    import profit_neural
    coreid=single_cpu_guard();policy,_=pool.study.load("SLOT",access,core,source,locked,"c205")
    scope=access.Scope("orthogonal-model-axes",frozenset([(0,"external_test",case)]),frozenset(["initialize","infer","search"]))
    cpu=time.process_time();wall=time.perf_counter();forward=0.;queries=0;results=[];error=0.;provider=locked.RepairProvider("paired-input",policy,False)
    with access.restricted_scope(scope):
        task=core.ProfitTask(case,fold=0,role="external_test");builder=BaseInferenceInputs(task)
        lib=pool.library();fn=library();inputs=pool.inputs(task);d=inputs[1];dist=np.ascontiguousarray(-d)
        for si,state in enumerate(("initial","terminal")):
            routes=tuple(map(tuple,archive[state]))
            if not si and routes!=tuple(map(tuple,task.initial())):raise ValueError("Different initial state")
            orig=task.evaluate(routes,audit=True)["net_benefit_U"]
            for k in range(cfg["groups_per_state"]):
                rng=np.random.default_rng(662901+ix*113+si*8+k)
                groups=list(dict.fromkeys(bundle(routes,d,int(j)) for j in rng.permutation(len(routes))))
                ids=next((g for g in groups if g),())
                members=[v for j in ids for v in routes[j]]
                extra=rng.choice(sorted(set(range(1,101))-set(members)),40-len(members),replace=False)
                mask=np.asarray(rng.permutation(members+extra.tolist()),np.int32)
                batch=builder.build(routes,mask);xy=batch["features"][0,:,:2]
                norm=np.linalg.norm(xy[:,None]-xy[None,:],axis=2);heats=[]
                for n in range(8):
                    transformed=view(batch,n);features=transformed["features"][0];coord=features[:,:2]
                    if not np.array_equal(features[:,2:],batch["features"][0,:,2:]) or not np.allclose(np.linalg.norm(coord[:,None]-coord[None,:],axis=2),norm,rtol=1.e-6,atol=1.e-7):raise ValueError("Not a model-space isometry")
                    tick=time.process_time();heats.append(logprob(profit_neural.predict(policy,transformed)));forward+=time.process_time()-tick;queries+=1
                if not results:
                    true=provider.heatmap(task,routes,mask,"maskco_base",0);off=~np.eye(101,dtype=bool)
                    error=float(np.max(np.abs(heats[0][off]-true[off])))
                    if error>.002:raise ValueError("Locked provider identity differs")
                ensemble=blend(heats);shuffled=shuffle_probabilities(ensemble,d,routes,mask,rng)
                arms={"v"+str(j):m for j,m in enumerate(heats)}
                arms.update(distance=dist,two=blend([heats[0],heats[3]]),eight=ensemble,shuffled=shuffled)
                out={}
                for a,rank in arms.items():
                    w=np.ascontiguousarray(rank,dtype=float);seed=43101+ix*97+si*5+k
                    u,requests,_=insertion(task,lib,inputs,routes,mask,w,seed,k%2,32)
                    split,work=orders(task,fn,inputs,routes,ids,w,seed+10000,24) if ids else (orig,0)
                    if min(u,split)<orig-1.e-6:raise ValueError("Original fleet lost")
                    out[a]=dict(gain=max(u,split)-orig,insertion=u-orig,order=split-orig,requests=requests+work)
                idx=(ix+3*k+4*si)%8
                top=np.argsort(-np.stack(heats)[:,mask,:],axis=2,kind="stable")[:,:,:8]
                overlap=np.mean([len(set(p)&set(q))/8 for other in top[1:] for p,q in zip(top[0],other)])
                results.append(dict(state=state,rep=k,group=ids,mask=mask.tolist(),original_U=orig,cycle="v"+str(idx),overlap=float(overlap),arms=out))
    (HERE/f"out/{ix}.json").write_text(json.dumps(dict(case=case,rows=results,forwards=queries,identity_checks=provider.forward_calls,
        identity_error=error,forward_cpu_s=forward,cpu_s=time.process_time()-cpu,wall_s=time.perf_counter()-wall,core=coreid),allow_nan=False))
if __name__=="__main__":main(int(sys.argv[1]))
