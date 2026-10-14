"""Direct independent all-action, original-coordinate and directed-rank replay."""
import os,sys
os.environ.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import json,math,hashlib,ctypes,traceback
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'research/review_3080_20261010/delivery';P=D/'family2_remaining_action_diagnosis_root_v1';O=D/'directed_action_chain_root_v1'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(n,v):
    with (O/n).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def edges(routes):
    return {(min(a,b),max(a,b)) for r in routes for a,b in zip([0]+r,r+[0]) if a!=b}
def main():
    if sys.argv[1:]==['--register']:
        paths=[O/n for n in ('REGISTRATION.json','RESULT.json','ACTION_CHAINS.jsonl','SOURCE_START.json','SOURCE_END.json','RUN_EXIT.json')]+[P/'OBSERVED_ACTION_RECORDS.json',Path(__file__),ROOT/'tools/pickup_loaded_closure_v1.py']
        save('AUDIT_REGISTRATION.json',dict(registered_utc=datetime.now(timezone.utc).isoformat(),inputs={str(p):sha(p) for p in paths},scope='Stream all30 NEW directed action chains; direct coordinate/capacity/TW/return, retained edges, teacher-sequence subset insertion score, every customer/option, saved-float tie/regret and transition. No new scorer/decoder/options/guard/original route_ok/matrices/C0/model imports or calls. Registered engineering1e-10 arithmetic only; exact score/slot/coverage comparisons.'))
        print('{"registered":true}');return 0
    import psutil
    from pickup_loaded_closure_v1 import capture
    psutil.Process().cpu_affinity([7]);ctypes.WinDLL(str(Path(os.environ['WINDIR'])/'System32/SHCore.dll'));list(psutil.process_iter(['pid']));capture()
    ar=read(O/'AUDIT_REGISTRATION.json');reg=read(O/'REGISTRATION.json');result=read(O/'RESULT.json');refs={tuple([x['instance_id'],x['step']]):x for x in read(P/'OBSERVED_ACTION_RECORDS.json')};contract=result['contract_payload']['delivery'];count=0;failures=[];positions=0;actions=0;maxerror=0.;seen=[]
    def ck(n,v):
        nonlocal count
        count+=1
        if not v:failures.append(n)
    for p,h in ar['inputs'].items():ck('input_start:'+p,sha(p)==h)
    start=capture(ar['inputs'].keys());save('AUDIT_SOURCE_START.json',start)
    with (O/'ACTION_CHAINS.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            out=json.loads(line);key=tuple(out['key']);seen.append(key);x=refs[key];req=x['request'];target=x['teacher_routes'];data=req['data'];coords=data['coords'];n=len(coords)
            dist=[[math.sqrt((coords[u][0]-coords[v][0])**2+(coords[u][1]-coords[v][1])**2) for v in range(n)] for u in range(n)];scale=max(1e-9,math.fsum(dist[u][v] for u in range(n) for v in range(n) if u!=v)/(n*(n-1)))
            required={(u,v) for u,r in enumerate(x['partial']) for v,b in enumerate(r) if u<v and b>0};mask=set(req['mask']);work=[[i for i in r if i not in mask] for r in req['initial']];remaining=sorted(mask);original_skeleton=[r[:] for r in work]
            ck(str(key)+':initial_target_compatible',original_skeleton==[[i for i in r if i not in mask] for r in target])
            for step,z in enumerate(out['trace']):
                pref=str(key)+':step:'+str(step);ck(pref+':before',z['before']==work);ck(pref+':remaining',z['remaining']==remaining);ck(pref+':every_customer', [c['customer'] for c in z['customers']]==remaining)
                for c in z['customers']:
                    i=c['customer'];expected=[(v,at) for v,r in enumerate(work) for at in range(len(r)+1)];ck(pref+':all_positions:'+str(i),[(r[1],r[2]) for r in c['rows']]==expected)
                    for idx,row in enumerate(c['rows']):
                        _,v,at,a,b,retained,legal,value,geom=row;r=work[v];route=r[:at]+[i]+r[at:];trial=[rr[:] for rr in work];trial[v]=route
                        pp=pref+':customer:'+str(i)+':option:'+str(idx)
                        ck(pp+':anchors',row[0]==i and a==(r[at-1] if at else 0) and b==(r[at] if at<len(r) else 0));ck(pp+':retained',retained==(required<=edges(trial)))
                        candidate_set=set(route);compat=route==[j for j in target[v] if j in candidate_set];ck(pp+':directed_slot_score',value==float(compat))
                        feasible=sum(data['demands'][j] for j in route)<=contract['capacity']+1e-9;clock=contract['loading_service_h'];prev=0
                        for j in route:
                            feasible=feasible and max(data['ready_time'][j],data['reveal_time'][j])<=1e-9
                            clock=max(clock+dist[prev][j]/contract['speed_kmph'],data['tw_start'][j]);feasible=feasible and clock<=data['tw_end'][j]+1e-9;clock+=data['service_time'][j];prev=j
                        feasible=feasible and clock+dist[prev][0]/contract['speed_kmph']<=contract['horizon_h']+1e-9
                        ck(pp+':local_route_filter',legal==feasible);error=abs(geom-(dist[a][i]+dist[i][b]-dist[a][b])/scale);maxerror=max(maxerror,error);ck(pp+':distance_tie_score',error<=reg['arithmetic_abs_tolerance']);positions+=1
                    eligible=sorted((r for r in c['rows'] if r[5] and r[6]),key=lambda r:(-r[7],r[8],r[1],r[2]));best=eligible[0] if eligible else None;regret=None if len(eligible)==1 else eligible[0][7]-eligible[1][7] if eligible else 0.
                    ck(pref+':best:'+str(i),best==c['best']);ck(pref+':regret:'+str(i),regret==c['regret'])
                feasible=[c for c in z['customers'] if c['best'] is not None]
                chosen=max(feasible,key=lambda c:(c['regret'] is None,0. if c['regret'] is None else c['regret'],-c['customer']))['best'] if feasible else None
                ck(pref+':chosen_exact',chosen==z['chosen'])
                if chosen is None:break
                i,v,at=chosen[:3];work[v].insert(at,i);remaining.remove(i);actions+=1
                ck(pref+':directed_prefix_all_slots',all(r==[j for j in target[v] if j in set(r)] for v,r in enumerate(work)));ck(pref+':all_unmasked_original_slots',[[j for j in r if j not in mask] for r in work]==original_skeleton)
            ck(str(key)+':final_exact',work==out['routes']==target);ck(str(key)+':complete',not remaining and out['remaining']==[] and out['status']=='complete');ck(str(key)+':final_retained',required<=edges(work));ck(str(key)+':component_time',0<out['wall_s']<reg['cooperative_component_limit_s'])
    ck('fixed30_once',len(seen)==30 and len(set(seen))==30 and set(seen)==set(refs));ck('750_actions',actions==750);ck('candidate_count',positions==sum(x['positions'] for x in result['metrics']))
    prod=read(O/'SOURCE_START.json');endprod=read(O/'SOURCE_END.json');ck('production_closed',prod['files']==endprod['files'] and prod['module_files']==endprod['module_files'])
    for p,h in prod['files'].items():ck('production_loaded:'+p,sha(p)==h)
    for p,h in ar['inputs'].items():ck('input_end:'+p,sha(p)==h)
    end=capture(ar['inputs'].keys());ck('independent_loaded_closed',start['files']==end['files'] and start['module_files']==end['module_files']);save('AUDIT_SOURCE_END.json',end)
    output=dict(created_utc=datetime.now(timezone.utc).isoformat(),checks_count=count,failed=len(failures),failure_names=failures,positions=positions,actions=actions,requests=len(seen),max_geometry_error=maxerror,ideal_fixture_only=True,no_new_component_reference=True,all_C0_or_model_calls=0,production_decoder_modified=False,learned_output_qualified=False,full_scientific_loaded_closure_qualified=False,stable_online_gain=False)
    save('INDEPENDENT_DIRECTED_ACTION_CHAIN.json',output);save('AUDIT_RUN_EXIT.json',dict(exit_code=int(bool(failures)),pid=os.getpid()));print(json.dumps(dict(checks=count,failed=len(failures),positions=positions,actions=actions,loaded_files=len(start['files']),module_maps=len(start['module_files']))));return int(bool(failures))
if __name__=='__main__':
    try:sys.exit(main())
    except Exception:
        save('AUDIT_FAILURE.json',dict(traceback=traceback.format_exc()));raise
