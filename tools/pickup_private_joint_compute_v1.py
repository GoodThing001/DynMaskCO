"""Same registered engineering algorithms, now from private projection within worker.
Old scientific/engineering trees remain unmodified. No live simulator claim.
"""
from qualify_pickup_joint_policy_transaction_root_v3 import *
import hard_admission as hd
from common_score import score_common
from strict_online_env import StrictOnlineEnv
from recourse_snapshot import snapshot_state_hash
from rolling_optimizer import RollingOptimizeReplanner

def compute(projected,order,deadline,phase,manifest):
    old=projected;clock=old['snapshot']['clock']
    from recourse_snapshot import restore_recourse_snapshot
    fleet,traces,served=restore_recourse_snapshot(old['snapshot'])
    q={'order':order,'mode':'own_incumbent','request_id':str(order),'deadline':deadline}
    with HISTORY.open('rb') as f:history=pickle.load(f)
    config={'name':'own_uncond_dynamic','budget':old['budget'],'capacity':old['capacity'],'booking_horizon':old['booking_horizon'],'contract':old['contract'],'cooling_share':0.,'history':history['history_times_classes']}
    steps={}
    external_phase=phase
    def phase(name):
        external_phase(name);steps[name]=time.perf_counter()
    phase('admission')
    base=hd.decide(config,projected,q['order'],E/'scripts',deadline)
    if base['qualified'] is not True:raise ValueError('Actual admission branch not certified')
    current=copy.deepcopy(projected);current['plan']=base['plan'];current['snapshot']['replanner_state']=copy.deepcopy(base['state'])
    current['snapshot']['replanner_state']['status_inst']=old['snapshot']['replanner_state']['status_inst']
    current['ownership']={str(o):int(v) for v,r in current['plan'].items() for o in r}
    prior_snapshot_hash=projected['snapshot']['state_hash']
    current['snapshot']['state_hash']=snapshot_state_hash(current['snapshot'])
    phase('proposal')
    proposal=copy.deepcopy(current['plan'])
    if q['mode']=='strong_rolling':
        rolling=RollingOptimizeReplanner(budget=current['budget'],capacity=current['capacity'],booking_horizon=current['booking_horizon'],
            contract=current['contract'],cooling_share=0.,hist=history['history_times_classes'],k=10,reject_penalty=current['penalties'],time_limit=max(0.,deadline-time.perf_counter()))
        rolling.restore_state(copy.deepcopy(current['snapshot']['replanner_state']))
        candidate_env=StrictOnlineEnv(current['dataset'],current['capacity'],current['tw_speed'],current['snapshot']['num_vehicles'],rolling,current['contract'],current['booking_horizon'])
        plans,accepted,rejected,proxy=rolling._optimize(candidate_env,0,clock,fleet,served,set())
        if accepted or rejected:raise ValueError('Fixed-admissions route proposal changed ownership policy')
        proposal=hd.canonical(current,plans)
    elif q['mode']!='own_incumbent':raise ValueError('No unregistered proposer')
    phase('scoring')
    scores=score_common(current,{'own_incumbent':current['plan'],'candidate':proposal},history,deadline,seed=202610111301)
    own,candidate=scores['candidate_scores']['own_incumbent'],scores['candidate_scores']['candidate']
    if not own['qualified']:raise ValueError('Actual admitted incumbent not hard')
    accepted_route=candidate['qualified'] and candidate['mean_shadow_utility']>own['mean_shadow_utility']+1e-10
    selected=proposal if accepted_route else current['plan']
    phase('certification');certificate,raw=pf.certificate(current,selected)
    phase('compile');state=copy.deepcopy(base['state']);state['plan']=selected;state['status_inst']=old['snapshot']['replanner_state']['status_inst']
    compiled=compile_state(state);end={p:sha(p) for p in manifest};done=time.perf_counter()
    result={'kind':'response','request_id':q['request_id'],'packet_digest':pf.digest(projected),'deadline':deadline,
        'certified':certificate['qualified'] is True and done<deadline,'source_qualified':end==manifest,'source_end':end,
        'compiled_state':compiled,'compiled_state_sha256':digest(state_dict(compiled)),'projected_packet':projected,
        'base_response':base,'scoring_packet':current,'proposal':proposal,'common_scores':scores,'selected_certificate':certificate,
        'accepted_route':accepted_route,'prior_snapshot_hash':prior_snapshot_hash,'derived_snapshot_hash':current['snapshot']['state_hash'],'worker_finished':done,'phase_started':steps,'restored_private_extraction_included':True}
    return result
