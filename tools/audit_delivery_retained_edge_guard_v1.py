"""Separate process edge-occurrence evidence check; no guard/decoder execution."""
import hashlib,json,sys,os,traceback
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'research/review_3080_20261010/delivery';O=D/'retained_edge_action_guard_root_v1';P=D/'family2_remaining_action_diagnosis_root_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(n,v):
    with (O/n).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
    if sys.argv[1:]==['--register']:
        paths=(O/'REGISTRATION.json',O/'RESULT.json',O/'RUN_EXIT.json',P/'RESULT.json',P/'OBSERVED_ACTION_RECORDS.json',P/'INDEPENDENT_REMAINING_ACTIONS.json',Path(__file__))
        save('AUDIT_REGISTRATION.json',dict(registered_utc=datetime.now(timezone.utc).isoformat(),inputs={str(p):sha(p) for p in paths},scope='Direct ordered occurrence list, target slot/ordinal invariant and saved guard outcome; do not import guard/diagnostic/production; no feasibility or full decoder qualification'))
        print('{"registered":true}');return 0
    ar=read(O/'AUDIT_REGISTRATION.json');checks=[]
    def ck(n,v):checks.append(dict(name=n,passed=bool(v)))
    for p,h in ar['inputs'].items():ck('start:'+p,sha(p)==h)
    r=read(O/'RESULT.json');prior=read(P/'RESULT.json');rows={tuple((x['instance_id'],x['step'])):x for x in read(P/'OBSERVED_ACTION_RECORDS.json')};results={tuple(x['key']):x for x in r['records']}
    # Retain repeated occurrence entries, including both singleton depot edges.
    def ledger(routes):
        out=[]
        for v,route in enumerate(routes):
            for k in range(len(route)+1):
                a=0 if k==0 else route[k-1];b=0 if k==len(route) else route[k]
                if a!=b:out.append((v,k,min(a,b),max(a,b)))
        return out
    count=0
    for x in prior['records']:
        key=tuple(x['key']);row=rows[key];obs=results[key];t=x['teacher_compatible'];slot,at=t['vehicle'],t['position'];pre=x['pre_routes'];after=[v[:] for v in pre];after[slot]=pre[slot][:at]+[x['customer']]+pre[slot][at:]
        raw0=ledger(pre);raw1=ledger(after);required=[(a,b) for a,line in enumerate(row['partial']) for b,z in enumerate(line) if a<b and z==1];actual={(a,b) for _,_,a,b in raw1}
        ck(str(key)+':all_required_union_edges_after',all(e in actual for e in required));ck(str(key)+':saved_guard_acceptance',obs['new_structural_guard_allows'])
        target=row['teacher_routes'];location={n:(v,k) for v,rr in enumerate(target) for k,n in enumerate(rr)}
        ck(str(key)+':target_slot_and_monotone_ordinals',all(all(location[n][0]==v for n in rr) and all(location[a][1]<location[b][1] for a,b in zip(rr,rr[1:])) for v,rr in enumerate(after)))
        old=tuple(sorted((t['predecessor'],t['successor'])));blocked=old in required
        ck(str(key)+':actual_original_skip',blocked==obs['original_blocked']==x['target_edge_blocked'])
        if blocked:
            count+=1;beforeocc=[z for z in raw0 if z[2:]==old];afterocc=[z for z in raw1 if z[2:]==old]
            ck(str(key)+':two_depot_occurrences_same_slot',len(beforeocc)==2 and all(v==slot for v,_,_,_ in beforeocc) and pre[slot]==[old[1]] and old[0]==0)
            ck(str(key)+':remaining_opposite_depot_occurrence',len(afterocc)==1 and afterocc[0][0]==slot and obs['before_occurrences']==2 and obs['after_occurrences']==1)
        ck(str(key)+':immutable_unmasked_order_and_slot',[[n for n in rr if n not in row['request']['mask']] for rr in after]==[[n for n in rr if n not in row['request']['mask']] for rr in row['request']['initial']])
    ck('all30_and26',len(results)==len(rows)==30 and count==26);ck('original_checks0fail',r['failed']==0 and prior['failed']==0)
    reg=read(O/'REGISTRATION.json')
    for p,h in reg['inputs'].items():ck('original_bound_input:'+p,sha(p)==h)
    for p,h in ar['inputs'].items():ck('end:'+p,sha(p)==h)
    out=dict(created_utc=datetime.now(timezone.utc).isoformat(),checks=checks,checks_count=len(checks),failed=sum(not c['passed'] for c in checks),
        observed_singleton_excess_rejections=count,independent_occurrence_list=True,new_guard_not_imported=True,
        isolated_structural_guard_qualified=True,full_production_decoder_qualified=False,physical_feasibility_qualified=False,
        family2_FAIL_preserved=True,new_training_or_probe=False,stable_online_gain=False)
    save('INDEPENDENT_EDGE_OCCURRENCES.json',out);save('AUDIT_RUN_EXIT.json',dict(exit_code=int(out['failed']>0),pid=os.getpid()));print(json.dumps(dict(checks_count=len(checks),failed=out['failed'])));return int(out['failed']>0)
if __name__=='__main__':
    try:sys.exit(main())
    except Exception:save('AUDIT_FAILURE.json',dict(traceback=traceback.format_exc()));raise
