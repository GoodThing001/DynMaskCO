"""Prospective fixed30 ideal directed-slot score fixture; no old probe calls."""
import os,sys
os.environ.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import json,hashlib,ctypes,time,traceback
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';R=ROOT/'research/review_3080_20261010';D=R/'delivery';P=D/'family2_remaining_action_diagnosis_root_v1';O=D/'directed_action_chain_root_v1'
sys.path.insert(0,str(E/'scripts/delivery'));sys.path.insert(0,str(E/'scripts/coldchain'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(n,v):
    with (O/n).open('x',encoding='utf-8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False)
def fixture(final):
    # Candidate score is conditioned on current temporary anchors, not final arcs.
    membership={i:v for v,r in enumerate(final) for i in r}
    rank={i:at for r in final for at,i in enumerate(r)}
    def score(routes,rows):
        values=[]
        for x in rows:
            i=x['customer'];v=x['vehicle'];at=x['position']
            compatible=membership[i]==v and all(rank[j]<rank[i] for j in routes[v][:at]) and all(rank[j]>rank[i] for j in routes[v][at:])
            values.append(float(compatible))
        return values
    return score
def main():
    if sys.argv[1:]==['--register']:
        O.mkdir(exist_ok=False)
        paths=[P/'OBSERVED_ACTION_RECORDS.json',D/'feasible_action_options_root_v1/REGISTRATION.json',D/'feasible_action_options_root_v1/INDEPENDENT_FEASIBLE_ACTION_OPTIONS.json',Path(__file__),ROOT/'tools/delivery_directed_action_chain_v1.py',ROOT/'tools/delivery_feasible_action_options_v1.py',ROOT/'tools/delivery_retained_edge_action_guard_v1.py',ROOT/'tools/pickup_loaded_closure_v1.py',E/'scripts/delivery/delivery_baselines.py',E/'scripts/delivery/delivery_contract.py',Path(sys.executable)]
        save('REGISTRATION.json',dict(registered_utc=datetime.now(timezone.utc).isoformat(),inputs={str(p):sha(p) for p in paths},scope='All30 previously fixed observed full masks, one NEW isolated directed-slot action-chain call each; ideal fixture scalar1 iff inserting into current skeleton preserves final directed slot rank, scalar0 otherwise. All customers/options/rejections at every step retained. Score first, distance second, slot/position tie, regret2 score difference and singleton priority; lowest customer tie. No registered reconstruct/probe/teacher/solver/model/C0/finalcost calls.',cooperative_component_limit_s=10.,arithmetic_abs_tolerance=1e-10,ideal_fixture_only=True,no_learned_scorer=True,scientific_family2_still_FAIL=True,new_scientific_family=False,stable_online_gain=False,expected_requests=30,expected_masked_actions=750))
        print('{"registered":true}');return 0
    import numpy as np,psutil
    from pickup_loaded_closure_v1 import capture
    from delivery_contract import DeliveryContract
    from delivery_directed_action_chain_v1 import action_chain
    psutil.Process().cpu_affinity([7]);ctypes.WinDLL(str(Path(os.environ['WINDIR'])/'System32/SHCore.dll'));list(psutil.process_iter(['pid']));np.zeros((2,2)).mean();np.std(np.array([0.,1.]));capture()
    reg=read(O/'REGISTRATION.json');assert all(sha(p)==h for p,h in reg['inputs'].items());inputs=read(P/'OBSERVED_ACTION_RECORDS.json');contract=DeliveryContract();checks=[];metrics=[];source=capture(reg['inputs'].keys());save('SOURCE_START.json',source)
    def ck(n,v):checks.append(dict(name=n,passed=bool(v)))
    with (O/'ACTION_CHAINS.jsonl').open('x',encoding='utf-8') as stream:
        for x in inputs:
            req=x['request'];key=[x['instance_id'],x['step']];begin=time.perf_counter()
            out=action_chain(req['data'],req['initial'],req['mask'],x['partial'],contract,fixture(x['teacher_routes']),reg['cooperative_component_limit_s']);wall=time.perf_counter()-begin
            ck(str(key)+':completed',out['status']=='complete');ck(str(key)+':exact_directed_slots',out['routes']==x['teacher_routes']);ck(str(key)+':all_masked_once',len(out['trace'])==len(req['mask']) and {z['chosen'][0] for z in out['trace'] if z['chosen']}==set(req['mask']));ck(str(key)+':ideal_score_all_selected',all(z['chosen'] and z['chosen'][7]==1. for z in out['trace']));ck(str(key)+':component_cap',wall<reg['cooperative_component_limit_s'])
            metrics.append(dict(key=key,status=out['status'],exact=out['routes']==x['teacher_routes'],actions=len(out['trace']),positions=sum(len(c['rows']) for z in out['trace'] for c in z['customers']),wall_s=wall))
            stream.write(json.dumps(dict(key=key,wall_s=wall,**out),ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
    end=capture(reg['inputs'].keys());save('SOURCE_END.json',end);ck('loaded_identity',source['files']==end['files'] and source['module_files']==end['module_files']);ck('fixed30unique',len(metrics)==30 and len({tuple(x['key']) for x in metrics})==30)
    for p,h in reg['inputs'].items():ck('bound_end:'+p,sha(p)==h)
    result=dict(created_utc=datetime.now(timezone.utc).isoformat(),checks=checks,checks_count=len(checks),failed=sum(not c['passed'] for c in checks),metrics=metrics,contract_payload=contract.payload(),python=sys.version,numpy=np.__version__,cpu=psutil.Process().cpu_affinity(),ideal_fixture_only=True,production_decoder_modified=False,learned_output_qualified=False,full_C0_qualified=False,hard_request_budget_qualified=False,full_scientific_loaded_closure_qualified=False,new_scientific_family=False,stable_online_gain=False)
    save('RESULT.json',result);save('RUN_EXIT.json',dict(exit_code=int(result['failed']>0),pid=os.getpid()));print(json.dumps(dict(checks=result['checks_count'],failed=result['failed'],actions=sum(x['actions'] for x in metrics),positions=sum(x['positions'] for x in metrics),loaded_files=len(source['files']),module_maps=len(source['module_files']))));return int(result['failed']>0)
if __name__=='__main__':
    try:sys.exit(main())
    except Exception:
        if O.exists():save('FAILURE.json',dict(traceback=traceback.format_exc()))
        raise
