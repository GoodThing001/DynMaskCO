"""New primitive enumeration on fixed already observed first-divergence states."""
import os,sys
os.environ.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import hashlib,json,time,traceback,ctypes,shutil
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';R=ROOT/'research/review_3080_20261010';D=R/'delivery';P=D/'family2_remaining_action_diagnosis_root_v1';G=D/'retained_edge_action_guard_root_v1';O=D/'feasible_action_options_root_v1'
sys.path.insert(0,str(E/'scripts/delivery'));sys.path.insert(0,str(E/'scripts/coldchain'))
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
        O.mkdir(exist_ok=False)
        inputs={str(p):sha(p) for p in (P/'RESULT.json',P/'OBSERVED_ACTION_RECORDS.json',G/'REGISTRATION.json',G/'INDEPENDENT_EDGE_OCCURRENCES.json',Path(__file__),ROOT/'tools/delivery_feasible_action_options_v1.py',ROOT/'tools/delivery_retained_edge_action_guard_v1.py',ROOT/'tools/pickup_loaded_closure_v1.py',E/'scripts/delivery/delivery_baselines.py',E/'scripts/delivery/delivery_contract.py',Path(sys.executable))}
        save('REGISTRATION.json',dict(registered_utc=datetime.now(timezone.utc).isoformat(),inputs=inputs,
            scope='All30 saved first divergent customer/skeleton inputs, one complete candidate enumeration per input, fixed guide1 teacher-binary labels only. Original matrices/route_ok and score grouping with changed retained union primitive. No customer regret/reconstruct/teacher/model/fullC0/finalcost calls.',
            expectation='No selection by cost or new outcomes; retain all routeslot/position candidates, failures, local best and target action; all30 included',
            local_request_limit_s=10,arithmetic_absolute_tolerance=1e-10,arithmetic_only_tolerance_not_scientific_gate=True,
            scientific_family2_still_FAIL=True,new_scientific_family=False,new_training=False,production_full_decoder_qualified=False,stable_online_gain=False))
        print('{"registered":true}');return 0
    import numpy as np,psutil
    from delivery_contract import DeliveryContract
    from delivery_feasible_action_options_v1 import options
    from pickup_loaded_closure_v1 import capture
    psutil.Process().cpu_affinity([7]);ctypes.WinDLL(str(Path(os.environ['WINDIR'])/'System32/SHCore.dll'));list(psutil.process_iter(['pid']))
    # Initialize existing read-only inventory dependencies before recorded work.
    np.array([1.,2.]).mean();np.std(np.array([0.,1.]));capture()
    reg=read(O/'REGISTRATION.json');assert all(sha(p)==h for p,h in reg['inputs'].items());source=capture(reg['inputs'].keys());save('SOURCE_START.json',source)
    result=read(P/'RESULT.json');rows={tuple((x['instance_id'],x['step'])):x for x in read(P/'OBSERVED_ACTION_RECORDS.json')};contract=DeliveryContract();checks=[];outputs=[]
    def ck(n,v):checks.append(dict(name=n,passed=bool(v)))
    for r in result['records']:
        key=tuple(r['key']);x=rows[key];n=len(x['request']['data']['coords']);binary=np.zeros((n,n),float)
        for route in x['teacher_routes']:
            for a,b in zip([0]+route,route+[0]):
                if a!=b:binary[a,b]=binary[b,a]=1.
        begin=time.perf_counter();out=options(x['request']['data'],r['pre_routes'],r['customer'],np.asarray(x['partial']),binary,contract);wall=time.perf_counter()-begin
        t=r['teacher_compatible'];ch=r['chosen'];target=next(z for z in out['rows'] if z['vehicle']==t['vehicle'] and z['position']==t['position']);chosen=next(z for z in out['rows'] if z['vehicle']==ch['vehicle'] and z['position']==ch['position'])
        ck(str(key)+':all_slot_positions',len(out['rows'])==sum(len(z)+1 for z in r['pre_routes']))
        ck(str(key)+':target_retained_union',target['retained_union_ok']);ck(str(key)+':chosen_still_eligible',chosen['eligible'])
        ck(str(key)+':old_skip_exact',target['old_edge_skip']==r['target_edge_blocked']);ck(str(key)+':within_local_cap',wall<reg['local_request_limit_s'])
        outputs.append(dict(key=list(key),customer=r['customer'],wall_s=wall,target=target,original_chosen=chosen,**out))
    ck('all30_once',len(outputs)==30 and {tuple(x['key']) for x in outputs}==set(rows))
    end=capture(reg['inputs'].keys());ck('actual_loaded_files_unchanged',source['files']==end['files'] and source['module_files']==end['module_files']);save('SOURCE_END.json',end)
    for p,h in reg['inputs'].items():ck('bound_end:'+p,sha(p)==h)
    save('ALL_ACTION_OPTIONS.json',outputs)
    metrics=dict(requests=len(outputs),candidate_positions=sum(len(x['rows']) for x in outputs),target_route_ok=sum(x['target']['original_route_ok'] for x in outputs),target_eligible=sum(x['target']['eligible'] for x in outputs),new_local_best_is_target=sum(x['best']['vehicle']==x['target']['vehicle'] and x['best']['position']==x['target']['position'] for x in outputs),walls_s=[x['wall_s'] for x in outputs],actual_loaded_files=len(source['files']),actual_module_maps=len(source['module_files']))
    out=dict(created_utc=datetime.now(timezone.utc).isoformat(),checks=checks,checks_count=len(checks),failed=sum(not c['passed'] for c in checks),metrics=metrics,contract_payload=contract.payload(),python=sys.version,numpy=np.__version__,cpu=psutil.Process().cpu_affinity(),
        original_route_ok_used=True,whole_decoder_model_or_probe_calls=0,full_C0_or_final_cost_calls=0,new_scientific_family=False,production_full_decoder_qualified=False,full_scientific_loaded_closure_qualified=False,stable_online_gain=False)
    save('RESULT.json',out);save('RUN_EXIT.json',dict(exit_code=int(out['failed']>0),pid=os.getpid()));print(json.dumps(dict(checks_count=len(checks),failed=out['failed'],metrics=metrics)));return int(out['failed']>0)
if __name__=='__main__':
    try:sys.exit(main())
    except Exception:
        if O.exists():save('FAILURE.json',dict(traceback=traceback.format_exc()))
        raise
