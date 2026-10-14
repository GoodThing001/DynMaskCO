"""Observed-fixture diagnostic: can deployed greedy insertion follow oracle edges?"""
import json,hashlib,time,sys,ctypes
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';D=ROOT/'research/review_3080_20261010/delivery';OUT=D/'insertable_teacher_oracle_probe_root_v1'
for sub in ('delivery','coldchain','research_20261010'):sys.path.insert(0,str(E/'scripts'/sub))
from delivery_equal_schedule import reconstruct
from delivery_maskco import partial_adjacency
from delivery_contract import DeliveryContract
from delivery_evaluator import evaluate
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    # Only this process; no server or team process touched.
    api=ctypes.WinDLL('kernel32',use_last_error=True);api.GetCurrentProcess.restype=ctypes.c_void_p;api.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
    if not api.SetProcessAffinityMask(api.GetCurrentProcess(),1<<7):raise ctypes.WinError(ctypes.get_last_error())
    api.GetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)];own=ctypes.c_size_t();system=ctypes.c_size_t();api.GetProcessAffinityMask(api.GetCurrentProcess(),ctypes.byref(own),ctypes.byref(system))
    if own.value!=1<<7:raise RuntimeError('Actual CPU7 qualification failed')
    OUT.mkdir(exist_ok=False);rawpath=D/'insertable_teacher_engineering_root_v3/RESULT.json';raw=json.loads(rawpath.read_text());paths=[E/'scripts/research_20261010/delivery_equal_schedule.py',E/'scripts/delivery/delivery_maskco.py',E/'scripts/delivery/delivery_baselines.py',E/'scripts/delivery/delivery_evaluator.py',E/'scripts/delivery/delivery_contract.py',Path(__file__)]
    hashes={str(p):sha(p) for p in paths}
    reg={'created_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':hashes,'input_sha256':sha(rawpath),'input':'two observed teacher fixtures plus all-retained boundaries',
      'actual_guidance_weight':1.,'oracle_logits':'1 on selected teacher undirected edges,0 elsewhere; standardized exactly by actual deployed decoder',
      'test_scope':'One finite oracle construction only; failure not proof that no logits can realize target; no training or gain measurement','new_research_data':False,'stable_online_gain':False}
    with (OUT/'REGISTRATION.json').open('x') as f:json.dump(reg,f,indent=2)
    records=[];checks=[]
    def ck(k,v):checks.append({'name':k,'pass':bool(v)})
    for r in raw['records']:
        if 'answer' not in r:continue
        q=r['request'];target=r['answer']['result']['routes'];n=len(q['data']['demands']);removed=set(q['masked_orders']);adj=np.asarray(q['partial'],np.float32);logits=partial_adjacency(target,set(),n)
        t=time.perf_counter();candidate,remaining,insertions,status=reconstruct(q['data'],q['initial'],DeliveryContract(),removed,adj,logits,1.,t+10.,time.perf_counter)
        report=evaluate(q['data'],candidate,DeliveryContract());teacher=evaluate(q['data'],target,DeliveryContract());finaladj=partial_adjacency(candidate,set(),n)
        name=str((r['instance_id'],r['mask_kind']));ck(name+':complete_actual_insertion',status=='complete' and not remaining);ck(name+':hard_complete_candidate',report['hard'] and report['complete']);ck(name+':retained_edges',np.all(finaladj[adj>0]>0));ck(name+':guide1_finite',np.all(np.isfinite(logits)))
        records.append({'instance':r['instance_id'],'mask':r['mask_kind'],'status':status,'remaining':remaining,'actual_decoder_insertions':insertions,'actual_candidate_routes':candidate,'actual_candidate_report':report,
          'selected_teacher_routes':target,'selected_teacher_report':teacher,'oracle_edge_target_reproduced':bool(np.array_equal(finaladj,logits)),'wall_s':time.perf_counter()-t,
          'candidate_minus_teacher_C0_cost':report['cost']-teacher['cost'],'engineering_not_MaskCO_model':True})
    ck('inputs_unchanged',sha(rawpath)==reg['input_sha256']);ck('source_bytes_unchanged',all(sha(p)==h for p,h in hashes.items()))
    failed=sum(not c['pass'] for c in checks);matched=all(r['oracle_edge_target_reproduced'] for r in records if r['mask']=='partial')
    out={'checks':checks,'checks_count':len(checks),'failed':failed,'records':records,'all_two_partial_teacher_edge_targets_reproduced_by_this_oracle':matched,
      'all_teacher_states_or_all_logits_reachability_proven':False,'learned_model_deployment_qualified':False,'stable_online_gain':False}
    with (OUT/'RESULT.json').open('x') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps({'checks':len(checks),'failed':failed,'finite_oracle_both_partial_targets_reproduced':matched,'records':[{k:r[k] for k in ('instance','mask','oracle_edge_target_reproduced','candidate_minus_teacher_C0_cost')} for r in records]}));return int(failed>0)
if __name__=='__main__':raise SystemExit(main())
