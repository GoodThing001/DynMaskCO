
import copy,math,time
import numpy as np
def score_partial_common(packet,plans,history,deadline,seed=202610111301):
    import public_fleet as pf
    from scenario_saa import CondHistoricalSampler
    from energy_complete_saa import EnergyCompleteSAA
    from strict_online_env import StrictOnlineEnv
    from recourse_snapshot import restore_recourse_snapshot
    from pickup_partial_validation_v1 import validate_partial
    from pickup_pending_transaction_v1 import certificate
    validate_partial(packet)
    rp=EnergyCompleteSAA(budget=packet['budget'],capacity=packet['capacity'],booking_horizon=packet['booking_horizon'],
      contract=packet['contract'],cooling_share=0.,hist=history['history_times_classes'],k=10,
      sampler=CondHistoricalSampler(history['history'],n_neighbors=10),reject_penalty={0:5.,1:10.,2:15.},
      K=10,time_limit=10.,arm_seed=7001,anytime_vote=False,shadow_mode='greedy',energy_pricing='marginal',future_policy='density',standby_orders=0)
    state=copy.deepcopy(packet['snapshot']['replanner_state']);rp.restore_state(state)
    env=StrictOnlineEnv(packet['dataset'],packet['capacity'],packet['tw_speed'],packet['snapshot']['num_vehicles'],rp,packet['contract'],packet['booking_horizon'])
    fleet,traces,served=restore_recourse_snapshot(packet['snapshot']);clock=float(packet['snapshot']['clock'])
    rp._sync_id_map(env,0,clock)
    snapshot=rp._build_snapshot(env,0,clock,fleet,served)
    # Draw once from public prefix and original known historical pool. No candidate
    # resource, arm label or private current-day future affects the demand law.
    scenarios=rp.sampler.sample(snapshot,np.random.default_rng(seed),10)
    scenario_hash=pf.digest(scenarios)
    space,orders=rp._build_space(env,0,scenarios);index={id(o):space.n_real+j for j,o in enumerate(orders)}
    indices=[[index[id(o)] for o in scen] for scen in scenarios]
    result={}
    for label,plan in plans.items():
        if time.perf_counter()>=deadline:raise TimeoutError('No partial candidate comparison')
        cert,_=certificate(packet,plan)
        if cert['qualified'] is not True:
            result[label]={'qualified':False,'certificate':cert,'scenario_sha256':scenario_hash,'branch_values':[]};continue
        shadow=rp._sim_states(env,0,clock,fleet,served,{int(k):list(v) for k,v in plan.items()})
        values=[]
        for ids in indices:
            if time.perf_counter()>=deadline:raise TimeoutError('No partial K10 candidate acceptance')
            value=float(rp._sim_scenario(env,0,clock,shadow,ids,space))
            if not math.isfinite(value):raise ValueError('Nonfinite complete scenario')
            values.append(value)
        if time.perf_counter()>=deadline:raise TimeoutError('Common candidate certification/submission too late')
        result[label]={'qualified':True,'certificate':cert,'scenario_sha256':scenario_hash,
          'branch_values':values,'mean_shadow_utility':float(np.mean(values)),'complete_scenarios':10}
    return {'observed_snapshot':snapshot,'pending_not_rejected':list(packet['admission_input']['pending']),'scenario_sha256':scenario_hash,'scenarios':scenarios,'candidate_scores':result,'sampling_once':True,
      'shadow_scores_are_not_realized_online_utility':True,'all_five_actual_deployment_qualified':False}
