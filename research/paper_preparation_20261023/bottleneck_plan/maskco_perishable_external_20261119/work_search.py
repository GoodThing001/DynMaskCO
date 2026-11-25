"""Run the masked search on a deterministic sequence of work units until the external 60-second deadline."""
import datetime as dt
import json
from pathlib import Path
import sys
import time
import numpy as np
import pool_common as pool
from disagreement import dispute_group
from calibration import residual_edges
HERE=Path(__file__).resolve().parent
library=pool.library
study=pool.study
original_solve=pool.solve

def solve(lib,core,locked,policy,case,role,seed,arm,seconds):
    if arm.removesuffix("_D")!="WORKMASK":
        return original_solve(lib,core,locked,policy,case,role,seed,arm,seconds)
    import timed_search
    from run_dev72_clock import clock_type
    cap=json.loads((HERE/"work_caps.json").read_text())["greedy_regret_caps"]
    alpha=json.loads((HERE/"calibrated.json").read_text())["alpha"]
    only_distance=arm.endswith("_D")
    started=time.perf_counter();utc=dt.datetime.now(dt.timezone.utc);cpu=time.process_time()
    task=core.ProfitTask(case,fold=0,role=role)
    clock=clock_type(timed_search.SearchClock)(task,start=started,utc=utc,seconds=seconds)
    clock.candidate_source=arm
    initial=clock.initialize();state=pool.pack(initial);best=state.copy()
    nodes,dist,params=pool.inputs(task);distance=np.ascontiguousarray(-dist)
    provider=locked.RepairProvider(arm,policy,only_distance)
    rng=np.random.default_rng(seed);stats=np.zeros(3,dtype=np.int64);totals=stats.copy()
    rewards=np.zeros(2);uses=np.zeros(2,dtype=int);counts=np.zeros(3,dtype=int)
    epoch=regret_used=targeted=0;donor=None;trace=[]
    def allowance():return clock.deadline-time.perf_counter()-.003
    def offer(buf):
        nonlocal best
        if allowance()<=0:return
        routes=pool.unpack(buf)
        if task.evaluate(routes)["net_benefit_U"]>clock.events[-1]["score"]["net_benefit_U"]+1e-7 and clock.offer(routes,int(totals[0])):
            best=buf.copy()
    while allowance()>0:
        before=clock.events[-1]["score"]["net_benefit_U"]
        if epoch%4==0:state=best.copy()
        group=rng.choice(np.arange(1,101),40,replace=False).astype(np.int32)
        if donor is not None and epoch%2:
            group,_=dispute_group(pool.unpack(state),donor,rng);targeted+=1
        if only_distance:
            base=distance;options=[distance]
        else:
            base=np.ascontiguousarray(provider.heatmap(task,pool.unpack(state),group,"maskco_base",epoch),dtype=np.float64)
            space,res=residual_edges(dist,base)
            options=[distance,base,np.ascontiguousarray(space+alpha*res)]
        mode=int(np.argmin(uses) if epoch<2 or epoch%4==0 else np.argmax(rewards+.1/np.sqrt(1+uses)))
        uses[mode]+=1
        focused=pool.conflict_pool(pool.unpack(state),group,base) if mode else group
        search_rng=np.random.default_rng([seed,epoch,19]);trial_seed=int(search_rng.integers(2**32))
        choices=[(r,g,h) for g in range(2) for r,h in enumerate(options)]
        outcomes=[]
        for k,(rank,g,h) in enumerate(choices):
            time_left=allowance()
            if time_left<=0:break
            tmp=state.copy();peak=state.copy()
            lib.repair_batch(nodes,dist,params,h,focused,len(focused),tmp,peak,trial_seed,time_left,1.5,mode,g,cap[g],stats)
            totals+=stats
            if allowance()<=0:break
            value=task.evaluate(pool.unpack(peak))["net_benefit_U"]
            if value>clock.events[-1]["score"]["net_benefit_U"]+1.e-7:offer(peak)
            if stats[0]==cap[g]:outcomes.append((value,-((k-epoch)%len(choices)),rank,g,tmp,peak))
        if len(outcomes)!=len(choices):break
        outcomes.sort(key=lambda v:v[:2],reverse=True)
        _,_,rank,g,state,peak=outcomes[0]
        counts[rank]+=1;regret_used+=g
        identity=sorted(pool.unpack(peak))
        donor=next((pool.unpack(o[-1]) for o in outcomes[1:] if sorted(pool.unpack(o[-1]))!=identity),None)
        temp=2.5*max(.1,1.-epoch/38.)
        complete=True
        for j in range(24):
            remaining=allowance()
            if remaining<=0:
                complete=False;break
            record=best.copy()
            lib.repair_batch(nodes,dist,params,options[rank],focused,len(focused),state,record,int(search_rng.integers(2**32)),
                             remaining,temp,mode,g,cap[g],stats)
            totals+=stats
            if not np.array_equal(record,best):offer(record)
            if stats[0]!=cap[g]:
                complete=False;break
        if not complete:break
        gain=max(0.,clock.events[-1]["score"]["net_benefit_U"]-before)
        rewards[mode]=.65*rewards[mode]+.35*min(3.,np.log1p(gain))
        if role=="train" and epoch<3:trace.append([rank,g,int(totals[0]),pool.unpack(state),pool.unpack(best)])
        epoch+=1
    result=clock.finish(arm,int(totals[0]));final=clock.events[-1]
    fresh=core.ProfitTask(case,fold=0,role=role);check=fresh.evaluate(final["routes"],audit=True)
    if not result["selected_incumbent_in_budget"] or final["elapsed_s"]>seconds or abs(check["net_benefit_U"]-result["final"]["net_benefit_U"])>1e-6:
        raise ValueError("Economic audit or cutoff differs")
    return dict(case=case,seed=seed,method=arm,role=role,status="AUDITED",U=check["net_benefit_U"],endpoint=check,
        routes=final["routes"],initial_U=clock.events[0]["score"]["net_benefit_U"],initial_routes=initial,selected_s=final["elapsed_s"],
        epochs=epoch,ranking_counts=counts.tolist(),regret_epochs=regret_used,mask_mode_counts=uses.tolist(),targeted_masks=targeted,
        iterations=int(totals[0]),improvements=int(totals[1]),native_route_requests=int(totals[2]),forwards=provider.forward_calls,
        forward_s=provider.forward_seconds,cpu_s=time.process_time()-cpu,timing=result["timing"],first_three_epochs=trace)
def checked_solve(*args):
    out=solve(*args)
    if args[5]=="train" and args[7] in ("WORKMASK","WORKMASK_D"):
        second=solve(*args)
        if len(out["first_three_epochs"])!=3 or out["first_three_epochs"]!=second["first_three_epochs"]:
            raise ValueError("The fixed-work search prefix changed under the same seed")
        out["repeat_three_epochs_equal"]=True
        out["repeated_qualification_cpu_s"]=second["cpu_s"]
    return out
if __name__=="__main__":
    pool.solve=checked_solve
    pool.main(int(sys.argv[1]))
