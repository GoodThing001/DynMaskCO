"""Prospective fixed index8 local loop-base reuse registration."""
import ast,hashlib,json,datetime
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    a=(HERE/'run.py').read_text(encoding='utf-8').replace('from lazy_base_distance import LazyBaseDistanceSAA','from energy_complete_saa import EnergyCompleteSAA').replace('class Observer(LazyBaseDistanceSAA):','class Observer(EnergyCompleteSAA):')
    a=a.replace("save('SOURCE_START.json'", "save('AUDIT_SOURCE_START.json'").replace("save('SOURCE_END.json'", "save('AUDIT_SOURCE_END.json'")
    i=a.index("    np.save(HERE/'DISTANCE_MATRIX.npy'");j=a.index('    end_closure=seal.closure(extra);',i)
    a=a[:i]+a[j:]
    i=a.index('    result=dict(')
    a=a[:i]+'''    expected=read(HERE/'RESULT.json');check_before=len(checks)
    for path,h in reg['sources'].items():ck('registered_source:'+path,seal.sha(path)==h)
    ck('all_feasibility_call_sequence_and_values_exact',expected['feasible_log']==actor.feasible_log)
    ck('all_original_insert_choices_and_private_route_states_exact',expected['events']==events)
    ck('all_original_physical_transition_outputs_exact',expected['cold_transition_outputs']==transitions)
    ck('original_terminal_energy_hex',expected['terminal_energy_hex']==(None if energy is None else float(energy).hex()))
    ck('original_terminal_distance_hex',expected['terminal_distance_hex']==(None if distance is None else float(distance).hex()))
    filtered=[];score_reference=[];skipped=0;computed=0
    for step in range(4):
        op='insert_'+str(step);before=events[step]['before'];trials=[]
        for v,s in before.items():
            for pos in range(s['frozen'],len(s['route'])+1):trials.append((int(v),pos,s,s['route'][:pos]+[sequence[step]]+s['route'][pos:]))
        feasible=[r for r in actor.feasible_log if r['operation']==op];dist=[r for r in actor.distance_log if r['operation']==op]
        ck('all_original_candidate_positions:'+str(step),len(trials)==len(feasible) and all(f['cur']==s['cur'] and f['route']==trial for (_,_,s,trial),f in zip(trials,feasible)))
        good=[entry for entry,f in zip(trials,feasible) if f['feasible']];ck('original_two_distance_calls_per_good_trial:'+str(step),len(dist)==2*len(good));seen=set()
        for n,(vid,pos,s,trial) in enumerate(good):
            t,b=dist[2*n:2*n+2];ck('original_first_trial_then_base:'+op+':'+str(n),t['route']==trial and b['route']==s['route'] and t['cur']==b['cur']==s['cur'])
            value=np.float64(float.fromhex(t['value_hex']))-np.float64(float.fromhex(b['value_hex']))
            score_reference.append(dict(order=int(sequence[step]),vehicle=vid,position=pos,trial_hex=t['value_hex'],base_hex=b['value_hex'],score_hex=float(value).hex(),score_type=type(value).__module__+'.'+type(value).__qualname__))
            filtered.append(t)
            if vid not in seen:filtered.append(b);seen.add(vid);computed+=1
            else:skipped+=1
    filtered.extend(r for r in actor.distance_log if not r['operation'].startswith('insert_'))
    ck('only_repeated_same_loop_base_distance_calls_removed',expected['distance_log']==filtered)
    ck('all_original_candidate_scores_hex_and_types_exact',expected['score_log']==score_reference)
    ck('proved_skipped_base_calls',expected['base_reuses']==skipped and expected['base_computations']==computed and len(actor.distance_log)-len(expected['distance_log'])==skipped)
    matrix=np.load(HERE/'DISTANCE_MATRIX.npy',allow_pickle=False)
    for i,r in enumerate(expected['distance_log']):
        total=0.;cur=r['cur']
        for o in r['route']:total+=float(matrix[cur,o]);cur=o
        total+=float(matrix[cur,0]);ck('independent_ordered_double_sum:'+str(i),total.hex()==r['value_hex'])
    result=dict(checks_count=len(checks),failed=sum(not c['pass_'] for c in checks),failure_names=[c['name'] for c in checks if not c['pass_']],checks=checks,original_incr_eval=False,original_reference_distance_calls=len(actor.distance_log),candidate_distance_calls=len(expected['distance_log']),proved_removed_base_calls=skipped,original_candidates=len(actor.feasible_log),scored_feasible_candidates=len(score_reference),complete_original_C0_transition_count=len(transitions),actual_source_files=len(start_closure['files']),actual_source_maps=len(start_closure['modules']),actual_loaded_start_end=seal.same(start_closure,end_closure),reference_operation_time=finished-started,full_request_budget_qualified=False,full_scientific_loaded_closure=False,formal_strong_dynamic_qualified=False,stable_online_gain=False)
    save('INDEPENDENT_AUDIT.json',result);save('AUDIT_RUN_EXIT.json',dict(exit_code=int(result['failed']>0)));print(json.dumps({k:v for k,v in result.items() if k!='checks'}));return int(result['failed']>0)
if __name__=='__main__':raise SystemExit(main())
'''
    ast.parse(a);(HERE/'audit.py').write_text(a,encoding='utf-8',newline='\n')
    paths=[HERE/n for n in ('lazy_base_distance.py','run.py','audit.py','prepare.py','supervise.py')]+[ROOT/'tools/engineering_stage_closure_v1.py',ROOT/'tools/qualify_pickup_joint_policy_transaction_root_v3.py',ROOT/'C-VRP_Cold-chainVehicleRoutingProblem/scripts/evaluation/scenario_saa.py',HERE.parent/'shadow_energy_correction_root_v1/energy_complete_saa.py',HERE.parent/'independent_rolling_incremental_engineering_root_v1/SIMULATION_CONTEXT.json',HERE.parent/'shared_history_qualification_root_v1/COMMON_HISTORY.pkl']
    for p in paths:
        if p.suffix=='.py':ast.parse(p.read_text(encoding='utf-8'))
    reg=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources={str(p):sha(p) for p in paths},goal='Only loop-scoped lazy reuse of unchanged original _insert_u base route distance; first good trial remains first, then base; no delta reassociation or global route cache',scenario_index=8,order_selection='Fixed first four original reveal-sorted scenario8 IDs, original first eligible archived public packet and common K10 seed202610130901; no cost or label selection',independent_reference='Fresh original EnergyCompleteSAA(incr_eval=False), original insert4/local_search4/complete_u; no new helper imported',exact_gates='Feasible call sequence, choices/private route states, every scored candidate scorehex/type, original trial-before-base call order after deleting only same-loop repeated base, complete C0 physical transition outputs and terminal energy/distance; full loaded py/pyc/native/source identity',normal_runs=1,reference_runs=1,supervision_cap_seconds=60,resource=dict(python='3.12.7',numpy='1.26.4',cpu_affinity=[7],no_model_min_available_ram_gib=.5),input_and_matrix_preparation_outside_local_component=True,final_matrix_and_parameters_guard_inside_operation_time=True,K=10,seed=202610130901,incr_eval=False,local_passes=4,full_K10_request_or_online_day_executed=False,full_request_budget_qualified=False,worker_budget_retained_for_future_requests=8,cleanup_reserve_retained=2,scientific_data_generated=False,formal_strong_dynamic_qualified=False,stable_online_gain=False,paper=dict(title='PyVRP: a high-performance VRP solver package',doi='10.1287/ijoc.2023.0055',url='https://arxiv.org/html/2403.13795v2',read_scope='Parent read4.3.2 and5.3',use='Unchanged route-evaluation reuse inspiration only, not PyVRP implementation or coldchain equivalence'))
    with (HERE/'REGISTRATION.json').open('x',encoding='utf-8') as f:json.dump(reg,f,indent=2)
    print(json.dumps(dict(registration_sha256=sha(HERE/'REGISTRATION.json'))))
if __name__=='__main__':main()
