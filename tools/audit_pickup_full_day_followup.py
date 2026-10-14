"""Reexecute typed trace replay; inspect actual full-day request records."""
from dataclasses import is_dataclass
import hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem'
P=ROOT/'research/review_3080_20261010/pickup/actual_masked_fleet_followup_v1'
sys.path.insert(0,str(P.parent));import public_fleet as pf
pf.bootstrap(E/'scripts')
from audit_io import decode
from pickup_trace_runner import replay
import coldchain_contract,coldchain_state,strict_online_env
CLASSES={c.__module__+'.'+c.__name__:c for m in (coldchain_contract,coldchain_state,strict_online_env)
         for c in vars(m).values() if isinstance(c,type) and is_dataclass(c)}
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    checks=[];summaries=[]
    def ck(n,p):checks.append({'name':n,'pass':bool(p)})
    result=json.loads((P/'full_day_qa_v1/RESULT.json').read_text())
    for item in result['rows']+[{'arm':'cond_hist'}]:
        name=item['arm'];folder=P/'actual_cond_hist_root_v1' if name=='cond_hist' else P/'full_day_qa_v1'
        raw=folder/('FULLRAW.json' if name=='cond_hist' else name+'_fullraw.json')
        value=decode(json.loads(raw.read_text(encoding='utf-8')),CLASSES)
        trace=folder/(name+'_trace.jsonl');rr=replay(trace)
        ck(name+':independent_trace_replay',rr['all_evaluations_equal'] is True and rr['days_replayed']==[0])
        if name!='cond_hist':ck(name+':actual_raw_byte_identity',sha(raw)==item['fullraw_sha256'])
        logs=value['admissions'];ck(name+':all80_admissions',len(logs)==80 and len({x['order'] for x in logs})==80)
        walls=[];installed=0;proposed=0
        for i,r in enumerate(logs):
            ck(name+':request'+str(i)+':same10s_deadline',abs(r['deadline']-r['started']-10)<1e-8)
            ck(name+':request'+str(i)+':finite_elapsed',np.isfinite(r['wall_seconds']) and r['wall_seconds']>=0)
            if r['wall_seconds']>10:ck(name+':request'+str(i)+':late_fallback',r['fallback'] is True and r['masked_installed'] is False)
            walls.append(r['wall_seconds']);installed+=int(r['masked_installed']);proposed+=int(r['masked_proposal'] is not None)
        summaries.append({'arm':name,'evaluation':value['evaluation'],'admissions':len(logs),'max_wall_s':max(walls),
                          'timeouts':sum(r['base_response'].get('timeout') is True for r in logs),'model_proposals':proposed,'model_installed':installed,
                          'trace_replayed_independently':True})
    probe=P/'full_day_qa_v1/dtype_probe';z=np.load(probe/'same_input_outputs.npz')
    ck('dtype_npz_numeric_record_present',len(z.files)>0)
    timeout=result['hard_base_timeout_regression']
    ck('actual_baseline_sleep_killed_reaped',timeout['timeout'] is True and timeout['worker_reaped'] is True and timeout['fallback'] is True)
    report={'checks':checks,'n_checks':len(checks),'n_failures':sum(not r['pass'] for r in checks),'rows':summaries,
        'qualification_failures':[r['arm'] for r in summaries if r['evaluation']['hard_feasible'] is not True],
        'uses_synthetic_history_three_days':True,'original_history_pool_qualified':False,
        'all_arms_full_original_K10_SAA':False,'all_including_parent_projection_certification_hard_kill_proven':False,
        'independent_physical_implementation':False,'formal_online_qualified':False,'stable_gain_established':False}
    with (P/'ROOT_INDEPENDENT_FULL_DAY_AUDIT.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps({k:report[k] for k in ('n_checks','n_failures','qualification_failures','rows')},ensure_ascii=True));return int(report['n_failures']>0)
if __name__=='__main__':sys.exit(main())
