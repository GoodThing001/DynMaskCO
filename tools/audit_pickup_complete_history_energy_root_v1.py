"""Independently replay both engineering days; retain hard failures and scope."""
import hashlib,json,math,sys
from dataclasses import is_dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem'
P=ROOT/'research/review_3080_20261010/pickup';OUT=P/'complete_history_energy_qa_root_v1'
sys.path.insert(0,str(P));import public_fleet as pf
pf.bootstrap(E/'scripts')
import coldchain_contract,coldchain_state,strict_online_env
from audit_io import decode
from pickup_trace_runner import replay
CLASSES={c.__module__+'.'+c.__name__:c for m in (coldchain_contract,coldchain_state,strict_online_env)
    for c in vars(m).values() if isinstance(c,type) and is_dataclass(c)}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    reg=json.loads((OUT/'REGISTRATION.json').read_text());checks=[];rows=[];values={}
    def ck(n,v):checks.append({'name':n,'pass':bool(v)})
    ck('two_bounded_runs_finished0',json.loads((OUT/'RUN_EXIT.json').read_text())['complete'] is True)
    for arm in ('original_ledger','complete_ledger'):
        folder=OUT/arm;raw=decode(json.loads((folder/'FULLRAW.json').read_text(encoding='utf-8')),CLASSES)
        values[arm]=raw;logs=raw['admissions'];rr=replay(folder/'cond_hist_trace.jsonl')
        ck(arm+':full_actual_terminal_replay',rr['all_evaluations_equal'] is True and rr['days_replayed']==[0])
        ck(arm+':80_unique_actual_requests',len(logs)==80 and len({r['order'] for r in logs})==80)
        ck(arm+':actual200_day_identity',raw['registration']['history_pickle_sha256']==reg['history_sha256'])
        ck(arm+':corrected_arm_identity',raw['registration']['ledger_corrected']==(arm=='complete_ledger'))
        ck(arm+':worker_program_identity',raw['registration']['adapter_sha256']==reg['worker_sha256']==sha(OUT/'complete_history_worker.py'))
        for i,r in enumerate(logs):
            prefix=arm+':'+str(i)+':'
            ck(prefix+'finite_nonnegative_wall',math.isfinite(r['wall_seconds']) and r['wall_seconds']>=0)
            ck(prefix+'same10s_deadline',abs(r['deadline']-r['started']-10)<1e-8)
            ck(prefix+'actual_common_historical_identity',r['source_history_sha256']==raw['registration']['history_sha256'])
            ck(prefix+'accepted_has_hard_public_certificate',not r['accepted'] or r['base_response'].get('certificate',{}).get('qualified') is True)
            if r['wall_seconds']>10:ck(prefix+'late_fallback_retained',r['fallback'] is True and r['accepted'] is False)
        rows.append(dict(arm=arm,evaluation=raw['evaluation'],requests=len(logs),
            accepted=sum(r['accepted'] for r in logs),rejected=sum(not r['accepted'] for r in logs),
            max_request_wall_seconds=max(r['wall_seconds'] for r in logs),
            worker_timeouts=sum(r['base_response'].get('timeout') is True for r in logs),
            whole_raw_sha256=sha(folder/'FULLRAW.json'),independent_terminal_replay=True))
    original,corrected=(values[x] for x in ('original_ledger','complete_ledger'))
    decisions_equal=[(r['order'],r['accepted']) for r in original['admissions']]==[(r['order'],r['accepted']) for r in corrected['admissions']]
    ledger=json.loads((P/'shadow_energy_correction_root_v1/ACTUAL_LEDGER_CORRECTION_QA.json').read_text())
    affected=[r for r in ledger['events'] if r['omitted_already_consumed_kwh']>1e-8]
    last_admission_clock=max(r['clock'] for r in original['admissions'])
    # These raw events are from the earlier uncond engineering day, not this new policy.
    ck('previous_observed_ledger_failures_after_last_reveal',all(r['clock']>last_admission_clock for r in affected))
    result={'checks':checks,'checks_count':len(checks),'failed':sum(not r['pass'] for r in checks),'rows':rows,
        'same80_request_decisions':decisions_equal,'last_actual_admission_clock':last_admission_clock,
        'previous_ledger_omission_clocks':[r['clock'] for r in affected],
        'observed_ledger_bug_does_not_show_admission_gain_in_this_fixture':True,
        'original200_history_and_complete_actual_conditional_SAA':True,
        'qualified_all_five_arms':False,'parent_projection_and_installation_atomic_hard_kill_qualified':False,
        'full_branch_count_evidence':'accepted requests have worker assertion of20 completed branches; raw count per rejected request is not exported',
        'hard_qualification_failures':[r['arm'] for r in rows if r['evaluation']['hard_feasible'] is not True],
        'fresh_days':0,'independent_physical_implementation':False,'formal_online_qualified':False,'stable_online_gain':False}
    with (OUT/'INDEPENDENT_ROOT_COMPLETE_HISTORY_ENERGY_QA.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({k:result[k] for k in ('checks_count','failed','rows','same80_request_decisions','last_actual_admission_clock','hard_qualification_failures')}))
    return int(result['failed']>0)
if __name__=='__main__':sys.exit(main())
