"""Observed first-divergence structural guard comparison, no reconstruct replay."""
import ast,hashlib,json,sys,traceback
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'research/review_3080_20261010/delivery';P=D/'family2_remaining_action_diagnosis_root_v1';O=D/'retained_edge_action_guard_root_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(n,v):
    with (O/n).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
    if sys.argv[1:]==['--register']:
        O.mkdir(exist_ok=False);inputs={str(p):sha(p) for p in (P/'RESULT.json',P/'OBSERVED_ACTION_RECORDS.json',P/'INDEPENDENT_REMAINING_ACTIONS.json',Path(__file__),Path(__file__).with_name('delivery_retained_edge_action_guard_v1.py'))}
        save('REGISTRATION.json',dict(registered_utc=datetime.now(timezone.utc).isoformat(),inputs=inputs,selection='All30 previously fixed first divergences; not utility-selected',
            scope='Compare original unconditional edge-skip with isolated complete undirected edge-union after one saved target-compatible insertion. Independent occurrence-list subtraction/addition; no reconstruct, feasibility/C0, solver, probe or model calls.',
            new_scientific_family=False,production_decoder_modified=False,scientific_family2_still_FAIL=True,stable_online_gain=False));print('{"registered":true}');return 0
    from delivery_retained_edge_action_guard_v1 import permits
    reg=read(O/'REGISTRATION.json');assert all(sha(p)==h for p,h in reg['inputs'].items());r=read(P/'RESULT.json');ind=read(P/'INDEPENDENT_REMAINING_ACTIONS.json');assert r['failed']==ind['failed']==0
    rows={tuple((x['instance_id'],x['step'])):x for x in read(P/'OBSERVED_ACTION_RECORDS.json')};checks=[];records=[]
    def ck(n,v):checks.append(dict(name=n,passed=bool(v)))
    # Independent edge occurrence ledger; does not use guard.edges.
    def occurrences(routes):
        count={}
        for route in routes:
            p=[0]+route+[0]
            for u,v in zip(p,p[1:]):
                if u!=v:
                    k=(min(u,v),max(u,v));count[k]=count.get(k,0)+1
        return count
    for x in r['records']:
        key=tuple(x['key']);row=rows[key];routes=x['pre_routes'];t=x['teacher_compatible'];i=x['customer'];v=t['vehicle'];at=t['position'];u=t['predecessor'];b=t['successor']
        required={(a,b) for a,rs in enumerate(row['partial']) for b,z in enumerate(rs) if a<b and z>0};before=occurrences(routes);trial=[z[:] for z in routes];trial[v].insert(at,i);after=occurrences(trial)
        ledger=before.copy();old=(min(u,b),max(u,b))
        if u!=b:ledger[old]-=1
        for a,c in ((u,i),(i,b)):
            if a!=c:k=(min(a,c),max(a,c));ledger[k]=ledger.get(k,0)+1
        ledger={e:n for e,n in ledger.items() if n>0};ck(str(key)+':independent_occurrence_transition',ledger==after)
        passed=permits(routes,i,v,at,required);expected=all(after.get(e,0)>0 for e in required)
        ck(str(key)+':guard_equals_full_occurrence_invariant',passed==expected);ck(str(key)+':target_action_all_retained_edges',passed)
        blocked=x['target_edge_blocked'];ck(str(key)+':original_edge_skip',blocked==(old in required))
        if blocked:
            ck(str(key)+':single_node_route',len(routes[v])==1);ck(str(key)+':doubled_depot_occurrence',0 in old and before[old]==2 and after[old]==1)
        ck(str(key)+':immutable_unmasked_subsequence',[[n for n in z if n not in set(row['request']['mask'])] for z in trial]==[[n for n in z if n not in set(row['request']['mask'])] for z in row['request']['initial']])
        records.append(dict(key=list(key),original_blocked=blocked,new_structural_guard_allows=passed,route_length=len(routes[v]),original_edge=list(old),before_occurrences=before.get(old,0),after_occurrences=after.get(old,0)))
    ck('all30',len(records)==30);ck('original_blocked26',sum(x['original_blocked'] for x in records)==26)
    ck('singleton_depot_allowed',permits([[4],[],[]],7,0,0,{(0,4)}))
    ck('singleton_depot_other_end_allowed',permits([[4],[],[]],7,0,1,{(0,4)}))
    ck('retained_internal_edge_rejected',not permits([[4,5],[]],7,0,1,{(4,5)}))
    ck('retained_depot_unique_end_rejected',not permits([[4,5],[]],7,0,0,{(0,4)}))
    ck('unretained_edge_allowed',permits([[4,5],[]],7,0,1,{(0,4),(0,5)}))
    for label,args in [('duplicate',([[4],[5]],4,0,0,set())),('slot',([[4]],7,1,0,set())),('position',([[4]],7,0,2,set()))]:
        rejected=False
        try:permits(*args)
        except ValueError:rejected=True
        ck('reject:'+label,rejected)
    for p,h in reg['inputs'].items():ck('unchanged:'+p,sha(p)==h)
    out=dict(created_utc=datetime.now(timezone.utc).isoformat(),checks=checks,checks_count=len(checks),failed=sum(not c['passed'] for c in checks),records=records,
        all26_skip_rejections_do_not_require_breaking_retained_edges=True,original_predicate_stricter_than_final_edge_union_for_observed_singletons=True,
        modified_production_decoder=False,reconstruct_probe_solver_model_calls=0,physical_feasibility_or_improved_final_outcome_qualified=False,
        independent_occurrence_ledger=True,scientific_family2_still_FAIL=True,new_scientific_family=False,stable_online_gain=False)
    save('RESULT.json',out);save('RUN_EXIT.json',dict(exit_code=int(out['failed']>0)));print(json.dumps(dict(checks_count=len(checks),failed=out['failed'],blocked=26)));return int(out['failed']>0)
if __name__=='__main__':
    try:sys.exit(main())
    except Exception:
        if O.exists():save('FAILURE.json',dict(traceback=traceback.format_exc()))
        raise
