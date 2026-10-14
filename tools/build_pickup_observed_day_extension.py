"""Create exclusive continuation tools; never edit qualified prefix sources."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
tools=ROOT/'tools'
s=(tools/'qualify_pickup_live_prefix_v1.py').read_text(encoding='utf-8')
s=s.replace('First 14 reveal callbacks of an already observed day, actual fleet/owner execution.','Continue the sealed post-commit first14 boundary to original observed-day termination.')
s=s.replace("OUT=P/'live_event_prefix_root_v1';CALLS=14","OUT=P/'observed_day_extension_root_v1';PREFIX=P/'live_event_prefix_root_v1'")
s=s.replace("'qualify_pickup_live_prefix_v1.py','audit_pickup_live_prefix_v1.py'","'qualify_pickup_observed_day_extension_v1.py','audit_pickup_observed_day_extension_v1.py'")
s=s.replace("[fixture.TRACE,P/'undecided_public_input_root_v2/RESULT.json',old.HISTORY", "[PREFIX/'RESULT.json',PREFIX/'REGISTRATION.json',PREFIX/'INDEPENDENT_LIVE_PREFIX.json',PREFIX/'INDEPENDENT_MODE_INITIALIZATION_RECOVERY_R2.json',fixture.TRACE,P/'undecided_public_input_root_v2/RESULT.json',old.HISTORY")
s=s.replace('def run_prefix(packet,dataset,w,records,events,context):','def run_extension(packet,dataset,w,records,events,context):')
start=s.index('            if self.callback_count>=CALLS:')
end=s.index('            return value',start)
s=s[:start]+s[end:]
start=s.index("    env=Env(copy.deepcopy(dataset)")
end=s.index('    return records,events,context',start)
s=s[:start]+'''    _,decode=old.codecs();initial=decode(read(PREFIX/'RESULT.json'))['final']
    rp.restore_state(copy.deepcopy(initial['state']))
    env=Env(copy.deepcopy(dataset),packet['capacity'],packet['tw_speed'],packet['snapshot']['num_vehicles'],rp,packet['contract'],packet['booking_horizon']);env.callback_count=14
    fleet=copy.deepcopy(initial['fleet']);traces=copy.deepcopy(initial['traces']);served=initial['served'].copy()
    env.event_id=initial['event_id'];env.replan_count=initial['replan_count'];env.force_suffix={}
    clock=initial['clock'];env.context=(0,clock,fleet,traces,served)
    context['initial_post_commit_digest']=old.pf.digest(dict(fleet=fleet,traces=traces,served=served,state=rp.export_state(),clock=clock,event_id=env.event_id,replan_count=env.replan_count))
    all_customers=[int(j) for j in range(1,env.num_nodes) if env.demands[0,j]>0]
    reveal_events=sorted(set(float(env.reveal_time[0,j]) for j in all_customers if env.reveal_time[0,j]>0))
    reveal_idx=sum(t<=clock+1e-6 for t in reveal_events)
    # The sealed input is AFTER plan_and_commit: do not execute that boundary twice.
    env._run_loop(0,clock,reveal_idx,fleet,traces,served,all_customers,reveal_events,float(env.tw_end[0].max()))
    context.update(clock=env.context[1],event_id=env.event_id,replan_count=env.replan_count,fleet=copy.deepcopy(fleet),traces=copy.deepcopy(traces),served=served.copy(),state=copy.deepcopy(rp.export_state()),visible=all_customers)
''' +s[end:]
s=s.replace("prefix_rule='first 14 actual reveal callbacks, stop after their original plan_and_commit; no outcome-selected states',callbacks=CALLS,", "continuation_rule='sealed first14 post-plan_and_commit boundary, original _run_loop until termination; no prefix replay, no outcome-selected states',prefix_result_sha256=old.sha(PREFIX/'RESULT.json'),supervision_cap_s=900.,")
s=s.replace("fault_total_cap_s=.2,fault_reserve_s=.08,", "new_fault_injection=False,")
s=s.replace('run_prefix(packet,dataset,w,records,events,final)','run_extension(packet,dataset,w,records,events,final)')
s=s.replace("ck('registered_first14_callbacks',len(records)==14 and len(set(r['callback'] for r in records))==14)","ck('registered_continuation_only',len(records)>0 and min(r['callback'] for r in records)==15 and len(set(r['callback'] for r in records))==len(records))")
s=s.replace("ck('physical_events_advanced',len(events)>14", "ck('physical_events_advanced',len(events)>len(records)")
s=s.replace("r['transport']['sequence']==r['callback']", "r['transport']['sequence']==r['callback']-14")
start=s.index('        # One new resident input-decode cancellation boundary')
end=s.index("        ck('selected_sources_end'",start)
s=s[:start]+'''        from coldchain_evaluator_a1 import evaluate_trace_a1
        final_ev=evaluate_trace_a1(final['traces'],{k:v[0] for k,v in dataset.items()},packet['contract'],set(final['state']['accepted']),set(final['state']['rejected']),packet['budget'],packet['penalties'])
        ck('actual_full_day_original_C0',final_ev['hard_feasible'] and not final_ev['failures'])
        ck('all_actual_accepted_served',set(np.flatnonzero(final['served']))-{0}==set(final['state']['accepted']))
        ck('all_observed_decided_at_day_end',set(final['visible'])==set(final['state']['accepted'])|set(final['state']['rejected']))
        ck('actual_used_vehicles_returned',all(t.return_arrival is not None for t in final['traces'] if t.services))
        ck('actual_terminal_vehicle_states',all(v.status in ('idle','closed') for v in final['fleet']))
''' +s[end:]
s=s.replace('worker_pid=pid,fault=fault,','worker_pid=pid,original_final_evaluation=final_ev,')
s=s.replace('continuous_prefix_only=True,full_day_completed=False,','continuous_prefix_only=False,full_day_completed=True,first14_replayed=False,')
s=s.replace(",fault_wall=fault['wall_s']", ",full_day_completed=True")
with (tools/'qualify_pickup_observed_day_extension_v1.py').open('x',encoding='utf-8') as f:f.write(s)
a=(tools/'audit_pickup_live_prefix_v1.py').read_text(encoding='utf-8')
a=a.replace("OUT=P/'live_event_prefix_root_v1'","OUT=P/'observed_day_extension_root_v1';PREFIX=P/'live_event_prefix_root_v1'")
start=a.index('            if self.callbacks>=14:')
end=a.index('            return value',start)
a=a[:start]+a[end:]
start=a.index('    env=Env(copy.deepcopy(dataset)')
end=a.index('    return decisions,events,final',start)
a=a[:start]+'''    _,decode=old.codecs();initial=decode(read(PREFIX/'RESULT.json'))['final'];rp.restore_state(copy.deepcopy(initial['state']))
    env=Env(copy.deepcopy(dataset),template['capacity'],template['tw_speed'],template['snapshot']['num_vehicles'],rp,template['contract'],template['booking_horizon']);env.callbacks=14
    fleet=copy.deepcopy(initial['fleet']);traces=copy.deepcopy(initial['traces']);served=initial['served'].copy();clock=initial['clock']
    env.event_id=initial['event_id'];env.replan_count=initial['replan_count'];env.force_suffix={}
    all_customers=[int(j) for j in range(1,env.num_nodes) if env.demands[0,j]>0]
    reveal_events=sorted(set(float(env.reveal_time[0,j]) for j in all_customers if env.reveal_time[0,j]>0))
    env._run_loop(0,clock,sum(t<=clock+1e-6 for t in reveal_events),fleet,traces,served,all_customers,reveal_events,float(env.tw_end[0].max()))
    final.update(clock=events[-1]['clock'],event_id=env.event_id,replan_count=env.replan_count,fleet=fleet,traces=traces,served=served,state=rp.export_state())
''' +a[end:]
a=a.replace("ck('producer_completed_new_prefix',raw['failed']==0 and len(raw['records'])==14 and raw['continuous_prefix_only'] and not raw['full_day_completed'])", "ck('producer_completed_registered_extension',raw['failed']==0 and len(raw['records'])>0 and not raw['continuous_prefix_only'] and raw['full_day_completed'] and not raw['first14_replayed'])")
a=a.replace("ck('independent_original_callbacks',len(decisions)==14)", "ck('independent_original_callbacks',len(decisions)==len(raw['records']) and decisions[0]['callback']==15)")
a=a.replace("got['transport']['sequence']==got['callback']", "got['transport']['sequence']==got['callback']-14")
a=a.replace("==14 and len((OUT/'PREPARED_RESPONSES.jsonl').read_text(encoding='utf-8').splitlines())==14", "==len(decisions) and len((OUT/'PREPARED_RESPONSES.jsonl').read_text(encoding='utf-8').splitlines())==len(decisions)")
start=a.index("    ck('new_fault_no_publication_and_reserved_cleanup'")
end=a.index('    try:psutil.Process',start)
a=a[:start]+'''    from coldchain_evaluator_a1 import evaluate_trace_a1
    ev=evaluate_trace_a1(final['traces'],{k:v[0] for k,v in dataset.items()},first['contract'],set(final['state']['accepted']),set(final['state']['rejected']),first['budget'],first['penalties'])
    for k,v in ev.items():ck('actual_day_original_evaluation:'+k,old.pf.digest(v)==old.pf.digest(raw['original_final_evaluation'][k]))
    ck('original_complete_hard_and_coverage',ev['hard_feasible'] and not ev['failures'] and set(np.flatnonzero(final['served']))-{0}==set(final['state']['accepted']))
    ck('original_actual_used_return',all(t.return_arrival is not None for t in final['traces'] if t.services))
    ck('original_day_no_pending',set(np.flatnonzero(dataset['demands'][0]>0))==set(final['state']['accepted'])|set(final['state']['rejected']))
''' +a[end:]
a=a.replace("not raw['full_day_completed'] and not raw['parent_preemptible']", "raw['full_day_completed'] and not raw['parent_preemptible']")
a=a.replace('original_policy_prefix_execution=True','original_policy_extension_execution=True,first14_replayed=False')
a=a.replace("INDEPENDENT_LIVE_PREFIX.json","INDEPENDENT_OBSERVED_DAY_EXTENSION.json")
with (tools/'audit_pickup_observed_day_extension_v1.py').open('x',encoding='utf-8') as f:f.write(a)
print('exclusive continuation tools created')
