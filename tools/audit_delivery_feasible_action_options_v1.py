"""Separate direct arithmetic and all candidate union checks; no adapter import."""
import os,sys
os.environ.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import hashlib,json,math,ctypes,traceback
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'research/review_3080_20261010/delivery';P=D/'family2_remaining_action_diagnosis_root_v1';O=D/'feasible_action_options_root_v1'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(n,v):
    with (O/n).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def main():
    if sys.argv[1:]==['--register']:
        save('AUDIT_REGISTRATION.json',dict(registered_utc=datetime.now(timezone.utc).isoformat(),inputs={str(p):sha(p) for p in (O/'REGISTRATION.json',O/'RESULT.json',O/'ALL_ACTION_OPTIONS.json',O/'SOURCE_START.json',O/'SOURCE_END.json',O/'RUN_EXIT.json',P/'RESULT.json',P/'OBSERVED_ACTION_RECORDS.json',Path(__file__))},scope='All1525 saved candidates, pure coordinate capacity/TW/return and edge occurrence arithmetic; all30 outputs, scales, local tie selection; never import/call new options/guard or original route_ok/matrices/evaluator/solver/model'))
        print('{"registered":true}');return 0
    import psutil
    from pickup_loaded_closure_v1 import capture
    psutil.Process().cpu_affinity([7]);ctypes.WinDLL(str(Path(os.environ['WINDIR'])/'System32/SHCore.dll'));list(psutil.process_iter(['pid']));capture()
    ar=read(O/'AUDIT_REGISTRATION.json');reg=read(O/'REGISTRATION.json');result=read(O/'RESULT.json');outputs=read(O/'ALL_ACTION_OPTIONS.json');old=read(P/'RESULT.json');rows={tuple((x['instance_id'],x['step'])):x for x in read(P/'OBSERVED_ACTION_RECORDS.json')};refs={tuple(x['key']):x for x in old['records']};checks=[];details=[]
    def ck(n,v):checks.append(dict(name=n,passed=bool(v)))
    for p,h in ar['inputs'].items():ck('input_start:'+p,sha(p)==h)
    closure_start=capture(ar['inputs'].keys());save('AUDIT_SOURCE_START.json',closure_start)
    contract=result['contract_payload']['delivery'];tolerance=reg['arithmetic_absolute_tolerance'];maxscore=0.;candidate_total=0;targets=0;targetbest=0
    for out in outputs:
        key=tuple(out['key']);x=rows[key];ref=refs[key];data=x['request']['data'];coords=data['coords'];n=len(coords);base=ref['pre_routes'];i=ref['customer'];name=str(key)
        dist=[[math.sqrt((coords[u][0]-coords[v][0])**2+(coords[u][1]-coords[v][1])**2) for v in range(n)] for u in range(n)]
        scale=math.fsum(dist[u][v] for u in range(n) for v in range(n) if u!=v)/(n*(n-1));edges=set()
        for rr in x['teacher_routes']:
            prev=0
            for nxt in rr+[0]:
                if prev!=nxt:edges.add((min(prev,nxt),max(prev,nxt)))
                prev=nxt
        dens=2*len(edges)/(n*(n-1));ls=math.sqrt(dens*(1-dens))
        ck(name+':distance_scale',abs(scale-out['dscale'])<=tolerance);ck(name+':binary_scale',abs(ls-out['lscale'])<=tolerance)
        positions=[(v,at) for v,rr in enumerate(base) for at in range(len(rr)+1)];ck(name+':all_slot_positions_exact_order',[(c['vehicle'],c['position']) for c in out['rows']]==positions)
        required={(u,v) for u,line in enumerate(x['partial']) for v,bit in enumerate(line) if u<v and bit>0}
        for index,c in enumerate(out['rows']):
            v,at=c['vehicle'],c['position'];route=base[v][:at]+[i]+base[v][at:];allroutes=[rr[:] for rr in base];allroutes[v]=route
            union=set()
            for rr in allroutes:
                seq=[0]+rr+[0]
                for u,z in zip(seq,seq[1:]):
                    if u!=z:union.add((min(u,z),max(u,z)))
            retained=required<=union;legal=True;cap=sum(float(data['demands'][k]) for k in route);clock=contract['loading_service_h'];prev=0;min_tw=float('inf')
            if cap>contract['capacity']+1e-9:legal=False
            for k in route:
                if max(data['ready_time'][k],data['reveal_time'][k])>1e-9:legal=False
                clock=max(clock+dist[prev][k]/contract['speed_kmph'],data['tw_start'][k]);min_tw=min(min_tw,data['tw_end'][k]-clock)
                if clock>data['tw_end'][k]+1e-9:legal=False
                clock+=data['service_time'][k];prev=k
            return_clock=clock+dist[prev][0]/contract['speed_kmph']
            if return_clock>contract['horizon_h']+1e-9:legal=False
            a=base[v][at-1] if at else 0;b=base[v][at] if at<len(base[v]) else 0
            gain=int((min(a,i),max(a,i)) in edges)+int((min(i,b),max(i,b)) in edges)-(0 if a==b else int((min(a,b),max(a,b)) in edges))
            score=(dist[a][i]+dist[i][b]-dist[a][b])/scale-gain/ls;err=abs(score-c['score']);maxscore=max(maxscore,err)
            pref=name+':option:'+str(index)
            ck(pref+':customer',c['customer']==i);ck(pref+':anchors',c['predecessor']==a and c['successor']==b);ck(pref+':original_edge_skip',c['old_edge_skip']==(x['partial'][a][b]>0))
            ck(pref+':retained_all_union',c['retained_union_ok']==retained);ck(pref+':capacity_TW_return_filter',c['original_route_ok']==legal);ck(pref+':eligibility',c['eligible']==(retained and legal));ck(pref+':score_arithmetic',err<=tolerance)
            if c==out['target']:details.append(dict(key=list(key),capacity_margin=contract['capacity']-cap,min_tw_margin=min_tw,return_margin=contract['horizon_h']-return_clock,original_route_ok=legal,retained_union_ok=retained))
            candidate_total+=1
        best=min((c for c in out['rows'] if c['eligible']),key=lambda z:(z['score'],z['vehicle'],z['position']))
        ck(name+':saved_float_min_and_slot_tie_exact',best==out['best']);ck(name+':target_eligible',out['target']['eligible']);targets+=int(out['target']['eligible']);targetbest+=int(out['best']['vehicle']==out['target']['vehicle'] and out['best']['position']==out['target']['position'])
        ck(name+':component_time_only',0<out['wall_s']<reg['local_request_limit_s'])
    ck('all1525_and30',candidate_total==result['metrics']['candidate_positions']==1525 and len(outputs)==30 and targets==30);ck('only12_local_best_targets',targetbest==result['metrics']['new_local_best_is_target']==12)
    prodstart=read(O/'SOURCE_START.json');prodend=read(O/'SOURCE_END.json');ck('actual_production_loaded_identity',prodstart['files']==prodend['files'] and prodstart['module_files']==prodend['module_files'])
    for p,h in prodstart['files'].items():ck('actual_loaded_rechecked:'+p,sha(p)==h)
    for p,h in ar['inputs'].items():ck('input_end:'+p,sha(p)==h)
    closure_end=capture(ar['inputs'].keys());ck('independent_actual_loaded_identity',closure_start['files']==closure_end['files'] and closure_start['module_files']==closure_end['module_files']);save('AUDIT_SOURCE_END.json',closure_end)
    out=dict(created_utc=datetime.now(timezone.utc).isoformat(),checks=checks,checks_count=len(checks),failed=sum(not c['passed'] for c in checks),details=details,candidate_positions=candidate_total,targets_route_filter_and_union_qualified=targets,local_best_target_count=targetbest,max_score_arithmetic_error=maxscore,
        no_new_adapter_guard_original_filter_import=True,independent_route_coordinate_and_coverage_arithmetic=True,exact_float_bit_reproduction_claimed=False,
        local_feasible_options_scoped_qualified=True,full_C0_physics_qualified=False,full_decoder_or_learned_output_qualified=False,scientific_family2_still_FAIL=True,stable_online_gain=False)
    save('INDEPENDENT_FEASIBLE_ACTION_OPTIONS.json',out);save('AUDIT_RUN_EXIT.json',dict(exit_code=int(out['failed']>0),pid=os.getpid()));print(json.dumps(dict(checks_count=len(checks),failed=out['failed'],positions=candidate_total,max_score_error=maxscore,loaded_files=len(closure_start['files']))));return int(out['failed']>0)
if __name__=='__main__':
    try:sys.exit(main())
    except Exception:save('AUDIT_FAILURE.json',dict(traceback=traceback.format_exc()));raise
