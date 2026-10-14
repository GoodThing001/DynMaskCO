"""All 30 already observed different-adjacency failures: recorded actions only.

Never invokes reconstruct, a teacher/search, model, route_ok, or physical evaluator.
"""
import hashlib,json,math,os,sys,traceback
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'research/review_3080_20261010';D=R/'delivery';O=D/'family2_remaining_action_diagnosis_root_v1'
RAW=D/'multistep_insertable_family2_root_v1/downloaded/STATES.jsonl'
AUD=D/'multistep_insertable_family2_data_audit_root_v2/inspection_20261011T111137553714Z/INDEPENDENT_FAMILY2_DATA_ACCEPTANCE.json'
SRC=D/'multistep_insertable_family2_root_v1/downloaded/SOURCE_START.json'
EQ=R/'parent_v3/source_snapshot/C-VRP_Cold-chainVehicleRoutingProblem/scripts/research_20261010/delivery_equal_schedule.py'
BL=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem/scripts/delivery/delivery_baselines.py'
REG=D/'multistep_insertable_family2_root_v1/RESEARCH_REGISTRATION.json'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(n,v):
    with (O/n).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def compatible(routes,target):
    present={o for r in routes for o in r}
    return routes==[[o for o in r if o in present] for r in target]
def score(coords,edges,dscale,lscale,r,i,at):
    a=r[at-1] if at else 0;b=r[at] if at<len(r) else 0
    def dist(a,b):return math.sqrt((coords[a][0]-coords[b][0])**2+(coords[a][1]-coords[b][1])**2)
    d=dist(a,i)+dist(i,b)-dist(a,b)
    gain=int((min(a,i),max(a,i)) in edges)+int((min(i,b),max(i,b)) in edges)-(0 if a==b else int((min(a,b),max(a,b)) in edges))
    return dict(predecessor=a,successor=b,distance_delta=d,binary_gain=gain,score=d/dscale-gain/lscale)
def main():
    if sys.argv[1:] == ['--register']:
        O.mkdir(exist_ok=False)
        inputs={str(p):sha(p) for p in (RAW,AUD,SRC,EQ,BL,REG,Path(__file__))}
        actual=read(SRC)['files'];matches=[]
        for p in (EQ,BL):
            found=[(a,h) for a,h in actual.items() if a.replace('\\','/').endswith('/'+p.name)]
            assert len(found)==1 and sha(p)==found[0][1];matches.append(dict(local=str(p),actual_path=found[0][0],sha256=found[0][1]))
        assert sha(REG)=='562fc0304cb0202238e6618c0a20cf1602ac38d6763261d7d407743e02d21176'
        save('REGISTRATION.json',dict(registered_utc=datetime.now(timezone.utc).isoformat(),inputs=inputs,actual_source_matches=matches,
            selection='All existing audit summaries with oracle_exact=False and same_undirected_adjacency=False; expect30, no utility selection',
            method='Replay saved insertion anchors only. First failure of per-slot teacher subsequence; score chosen and unique teacher-compatible same-customer insertion using saved binary target. No legality/solver/physical/model calls.',
            score_arithmetic_abs_tolerance=1e-10,score_tolerance_is_new_diagnostic_not_original_acceptance_rule=True,
            causal_limit='An unchosen target action may be excluded by physical filtering; score comparison alone does not prove cause. Customer regret competitors are not recomputed.',
            family2_FAIL_unchanged=True,new_scientific_family=False,new_probe=False,new_training=False,stable_online_gain=False))
        print(json.dumps({'registered':True}));return 0
    import psutil
    psutil.Process().cpu_affinity([7]);reg=read(O/'REGISTRATION.json');assert all(sha(p)==h for p,h in reg['inputs'].items())
    a=read(AUD);assert a['raw_replay_failure_names']==[] and a['failed']==36
    keys={tuple(x['key']) for x in a['summaries'] if not x['oracle_exact'] and not x['oracle_same_undirected_adjacency']};assert len(keys)==30
    checks=[];records=[];pairs=[]
    def ck(n,v):checks.append(dict(name=n,pass_=bool(v)))
    with RAW.open(encoding='utf-8') as f:
        for line in f:
            x=json.loads(line);key=(x['instance_id'],x['step'])
            if key not in keys:continue
            q=x['request'];target=x['teacher']['routes'];coords=q['data']['coords'];removed=set(q['mask']);skeleton=[[o for o in r if o not in removed] for r in q['initial']]
            edges={(min(u,v),max(u,v)) for r in target for u,v in zip([0]+r,r+[0]) if u!=v}
            ds=[math.sqrt((p[0]-q_[0])**2+(p[1]-q_[1])**2) for j,p in enumerate(coords) for k,q_ in enumerate(coords) if j!=k]
            dscale=math.fsum(ds)/len(ds);density=(2*len(edges))/len(ds);lscale=math.sqrt(density*(1-density))
            ck(str(key)+':initial_teacher_subsequence',compatible(skeleton,target));first=None;errors=[]
            for j,event in enumerate(x['oracle']['insertions']):
                i,v,at=event['customer'],event['vehicle'],event['position'];pre=[r[:] for r in skeleton];sc=score(coords,edges,dscale,lscale,pre[v],i,at)
                diff=sc['score']-event['score'];errors.append(abs(diff));ck(str(key)+':saved_score:'+str(j),abs(diff)<=reg['score_arithmetic_abs_tolerance'])
                before=compatible(pre,target);skeleton[v].insert(at,i);after=compatible(skeleton,target)
                if before and not after and first is None:
                    tv=next(k for k,r in enumerate(target) if i in r);seen={o for r in pre for o in r};ta=sum(o in seen for o in target[tv][:target[tv].index(i)])
                    alt=score(coords,edges,dscale,lscale,pre[tv],i,ta);tr=[r[:] for r in pre];tr[tv].insert(ta,i)
                    ck(str(key)+':target_action_restores_subsequence',compatible(tr,target))
                    gap=alt['score']-sc['score'];group='chosen_strictly_lower_score' if gap>1e-10 else ('target_strictly_lower_score_filter_or_missing_context' if gap< -1e-10 else 'score_tie')
                    first=dict(key=list(key),insertion_index=j,customer=i,pre_routes=pre,chosen=dict(vehicle=v,position=at,**sc),teacher_compatible=dict(vehicle=tv,position=ta,**alt),
                        target_minus_chosen_score=gap,score_relation=group,wrong_vehicle=v!=tv,partial_teacher_adjacency_gain=alt['binary_gain'],
                        target_edge_blocked=q and x['partial'][alt['predecessor']][alt['successor']]>0,
                        teacher_action_feasibility_not_recomputed=True,regret_customer_selection_not_recomputed=True)
            ck(str(key)+':recorded_insertions_only',skeleton==x['oracle']['routes'])
            ck(str(key)+':first_divergence_found',first is not None)
            pairs.append(dict(instance_id=x['instance_id'],step=x['step'],request=q,partial=x['partial'],teacher_routes=target,oracle=x['oracle']))
            if first:first['max_saved_score_error']=max(errors,default=0.);records.append(first)
    ck('all30_recorded_keys',{tuple(x['key']) for x in records}==keys)
    for p,h in reg['inputs'].items():ck('unchanged:'+p,sha(p)==h)
    save('OBSERVED_ACTION_RECORDS.json',pairs)
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),checks=checks,checks_count=len(checks),failed=sum(not c['pass_'] for c in checks),
        records=records,counts=dict(first_divergence_step=dict(Counter(x['insertion_index'] for x in records)),score_relation=dict(Counter(x['score_relation'] for x in records)),wrong_vehicle=sum(x['wrong_vehicle'] for x in records),target_binary_gain=dict(Counter(x['partial_teacher_adjacency_gain'] for x in records))),
        max_score_error=max(x['max_saved_score_error'] for x in records),scientific_family3_started=False,family2_scientific_barrier_still_FAIL=True,
        production_decoder_modified=False,new_solver_probe_model_calls=0,causal_isolation_qualified=False,stable_online_gain=False)
    save('RESULT.json',result);save('RUN_EXIT.json',dict(exit_code=int(result['failed']>0),pid=os.getpid()))
    print(json.dumps({k:result[k] for k in ('checks_count','failed','counts','max_score_error')}));return int(result['failed']>0)
if __name__=='__main__':
    try:sys.exit(main())
    except Exception:
        if O.exists():save('FAILURE.json',dict(traceback=traceback.format_exc()))
        raise
