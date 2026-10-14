"""Separate-process direct admission/branch/trace replay; no adapter calls."""
import copy,json,pickle,sys,time
from pathlib import Path
import qualify_pickup_joint_policy_transaction_root_v3 as old
import numpy as np
import psutil
from strict_online_env import StrictOnlineEnv
from recourse_snapshot import restore_recourse_snapshot,snapshot_state_hash
from run_exp_rule_control import UncondDynamicReplanner
from energy_complete_saa import EnergyCompleteSAA
from scenario_saa import CondHistoricalSampler
from coldchain_evaluator_a1 import evaluate_trace_a1
from joint_install import state_dict,digest
ROOT=old.ROOT;P=old.P;OUT=P/'pending_admission_transaction_root_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def main():
    psutil.Process().cpu_affinity([7]);_,decode=old.codecs();raw=decode(read(OUT/'RESULT.json'));reg=read(OUT/'REGISTRATION.json')
    source_inputs=read(P/'undecided_public_input_root_v2/RESULT.json');checks=[];summary=[]
    def ck(n,v):checks.append(dict(name=n,pass_=bool(v)))
    ck('registered_complete_new_cases',raw['failed']==0 and [r['name'] for r in raw['records']]==reg['input_names'])
    ck('pinned_environment',np.__version__=='1.26.4')
    for p,h in reg['sources'].items():ck('selected_identity:'+p,old.sha(p)==h)
    with old.HISTORY.open('rb') as f:history=pickle.load(f)
    for row in raw['records']:
        name=row['name'];packet=row['packet'];o=packet['admission_input']['current_order'];response=row['response'];current=response['scoring_packet']
        state=packet['snapshot']['replanner_state'];a=set(state['accepted']);r=set(state['rejected']);known=set(int(x) for x in packet['snapshot']['customer_universe']);pending=known-a-r
        ck(name+':input_current_undecided',o in pending and o not in a|r)
        if name=='derived_two_pending':
            origin=decode(next(x['packet'] for x in source_inputs['records'] if x['kind']=='derived_old_state' and x['index']==13 and x['variant']==0))
            removed=min(origin['snapshot']['replanner_state']['rejected']);expected=copy.deepcopy(origin)
            expected['snapshot']['replanner_state']['rejected']=sorted(set(expected['snapshot']['replanner_state']['rejected'])-{removed})
            expected['admission_input']['pending']=sorted(set(expected['admission_input']['pending'])|{removed})
            expected['snapshot']['state_hash']=snapshot_state_hash(expected['snapshot'])
        else:
            if name=='sealed_first_reveal':origin=next(x for x in source_inputs['records'] if x['kind']=='actual_first_reveal_hook' and x['variant']==0)
            else:origin=next(x for x in source_inputs['records'] if x['kind']=='derived_old_state' and x['index']==int(name.split(':')[1]) and x['variant']==0)
            expected=decode(origin['packet'])
        ck(name+':exact_declared_input',old.pf.digest(expected)==old.pf.digest(packet))
        rp=UncondDynamicReplanner(budget=packet['budget'],capacity=packet['capacity'],booking_horizon=packet['booking_horizon'],contract=packet['contract'],cooling_share=0.,hist=history['history_times_classes'],k=10)
        rp.restore_state(copy.deepcopy(state));env=StrictOnlineEnv(packet['dataset'],packet['capacity'],packet['tw_speed'],packet['snapshot']['num_vehicles'],rp,packet['contract'],packet['booking_horizon'])
        env.replan_count=packet['snapshot']['replan_count'];fleet,traces,served=restore_recourse_snapshot(packet['snapshot'])
        rp.on_reveal(env,0,packet['snapshot']['clock'],fleet,served,[o]);replay=rp.export_state()
        plans={str(v):[int(j) for j in replay['plan'].get(v,replay['plan'].get(str(v),[])) if not served[j]] for v in range(len(fleet))}
        for v in fleet:
            if v.status=='committed' and v.committed_next not in (None,0):plans[str(v.vehicle_id)]=[int(v.committed_next)]+[j for j in plans[str(v.vehicle_id)] if j!=v.committed_next]
        replay['plan']=plans
        ck(name+':actual_original_admission',old.pf.digest(replay)==old.pf.digest(response['base_response']['state']))
        ck(name+':only_current_transition',set(replay['accepted'])-{o}==a and set(replay['rejected'])-{o}==r)
        remaining=pending-{o};ck(name+':explicit_remaining_partition',set(current['admission_input']['pending'])==remaining and not remaining&(set(replay['accepted'])|set(replay['rejected'])))
        ck(name+':physical_snapshot_unchanged',all(old.pf.digest(packet['snapshot'][k])==old.pf.digest(current['snapshot'][k]) for k in packet['snapshot'] if k not in ('state_hash','replanner_state')))
        ck(name+':state_hash',snapshot_state_hash(current['snapshot'])==current['snapshot']['state_hash'])
        for k in packet['dataset']:ck(name+':retained_observed_features:'+k,np.array_equal(packet['dataset'][k],current['dataset'][k]))
        # Original full physical closing run; only full-day coverage refuses pending.
        ev,tr=old.pf.evaluate_known(current,plans);stored=response['selected_certificate']['evaluation']
        ck(name+':full_original_C0_closing_trace',old.pf.digest(tr)==old.pf.digest(response['selected_raw']))
        ck(name+':original_coverage_boundary',ev['failures']==(['unaccounted_order'] if remaining else []))
        for k,v in ev.items():
            if k not in ('hard_feasible','failures'):ck(name+':full_trace_field:'+k,old.pf.digest(v)==old.pf.digest(stored[k]))
        ck(name+':provisional_hard_with_explicit_pending',stored['hard_feasible'] and stored['failures']==[] and stored['pending_count']==len(remaining) and stored['provisional_not_full_day_utility'])
        ck(name+':pending_no_physical_service',not remaining&{s.node for t in tr for s in t.services})
        expected_loss=sum(packet['penalties'][int(packet['dataset']['temp_class'][0,j])] for j in replay['rejected'])
        ck(name+':no_pending_rejection_loss',stored['reject_loss']==expected_loss)
        # Independently assemble original SAA class, sampler and all 20 branches.
        srp=EnergyCompleteSAA(budget=current['budget'],capacity=current['capacity'],booking_horizon=current['booking_horizon'],contract=current['contract'],cooling_share=0.,hist=history['history_times_classes'],k=10,sampler=CondHistoricalSampler(history['history'],n_neighbors=10),reject_penalty={0:5.,1:10.,2:15.},K=10,time_limit=10.,arm_seed=7001,anytime_vote=False,shadow_mode='greedy',energy_pricing='marginal',future_policy='density',standby_orders=0)
        srp.restore_state(copy.deepcopy(replay));senv=StrictOnlineEnv(current['dataset'],current['capacity'],current['tw_speed'],current['snapshot']['num_vehicles'],srp,current['contract'],current['booking_horizon'])
        sfleet,_,sserved=restore_recourse_snapshot(current['snapshot']);clock=current['snapshot']['clock'];srp._sync_id_map(senv,0,clock)
        ssnap=srp._build_snapshot(senv,0,clock,sfleet,sserved);scenarios=srp.sampler.sample(ssnap,np.random.default_rng(reg['common_seed']),10)
        saved=response['common_scores'];ck(name+':actual_observed_sampler_prefix',old.pf.digest(ssnap)==old.pf.digest(saved['observed_snapshot']))
        ck(name+':all_pending_remains_observed',{srp._id_map[j] for j in remaining}<=set(x.oid for x in ssnap.orders) and not {srp._id_map[j] for j in remaining}&(set(ssnap.accepted)|set(ssnap.rejected)))
        ck(name+':common_fixed_scenario_identity',old.pf.digest(scenarios)==saved['scenario_sha256'] and old.pf.digest(scenarios)==old.pf.digest(saved['scenarios']))
        space,orders=srp._build_space(senv,0,scenarios);ids={id(v):space.n_real+j for j,v in enumerate(orders)}
        for label,plan in {'own_incumbent':plans,'candidate':response['proposal']}.items():
            st=srp._sim_states(senv,0,clock,sfleet,sserved,{int(k):v for k,v in plan.items()})
            vals=[float(srp._sim_scenario(senv,0,clock,st,[ids[id(j)] for j in scen],space)) for scen in scenarios]
            c=saved['candidate_scores'][label]
            for j,(v,w) in enumerate(zip(vals,c['branch_values'])):ck(name+':'+label+':branch:'+str(j),v.hex()==w.hex())
            ck(name+':'+label+':all10mean_grouped',len(vals)==len(c['branch_values'])==10 and float(np.mean(vals)).hex()==c['mean_shadow_utility'].hex())
        ck(name+':no_route_gain_or_model',response['proposal']==plans and not response['accepted_route'] and not response['formal_learned_controls'])
        ck(name+':joint_compiled_export',state_dict(response['compiled_state'])==row['after']==replay and digest(row['after'])==response['compiled_state_sha256'] and row['installed'])
        ck(name+':normal_parent_decode_commit_cleanup_elapsed',row['started']<=response['worker_finished']<=row['received']<=row['committed']<=row['finished']<row['started']+10.)
        ck(name+':worker_sources_start_end',response['source_end']==row['ready']['sources'])
        try:psutil.Process(row['worker_pid']);gone=False
        except psutil.NoSuchProcess:gone=True
        ck(name+':own_worker_reaped',gone)
        summary.append(dict(name=name,current_order=o,accepted=o in replay['accepted'],remaining_pending=sorted(remaining),wall_s=row['wall_s'],provisional_utility=stored['utility'],online_gain=False))
    ck('new_semantic_boundaries',len(raw['boundaries'])==5 and all(b['error'] for b in raw['boundaries']))
    ck('explicit_scope_not_upgraded',not raw['full_live_10s_qualified'] and not raw['continuous_live_event_qualified'] and not raw['formal_learned_controls'] and not raw['full_loaded_dependency_closure_qualified'] and not raw['stable_online_gain'])
    result=dict(checks=checks,checks_count=len(checks),failed=sum(not c['pass_'] for c in checks),summaries=summary,
                producer_sha256=old.sha(OUT/'RESULT.json'),registration_sha256=old.sha(OUT/'REGISTRATION.json'),
                independent_admission_and_SAA_assembly=True,shared_physical_evaluator=True,full_live_request_qualified=False,stable_online_gain=False)
    with (OUT/'INDEPENDENT_PENDING_TRANSACTION.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(dict(checks=len(checks),failed=result['failed'],summaries=summary)));return int(result['failed']>0)
if __name__=='__main__':raise SystemExit(main())
