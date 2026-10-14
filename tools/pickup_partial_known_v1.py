import numpy as np
def evaluate_partial_known(packet,plan):
    """Actual observed cargo/thermal state, all fleet energy, known-only closing.

    Forced completion immediately returns after the known accepted routes. This
    is a declared surrogate, not a prediction of future SAA's waiting/admissions.
    """
    from strict_online_env import StrictOnlineEnv,Replanner
    from recourse_snapshot import clone_snapshot,snapshot_state_hash
    from pickup_partial_trace_v1 import evaluate_partial_trace
    contract=packet['contract'];contract.validate()
    if contract.contract_hash!=packet['contract_hash']: raise ValueError('contract identity mismatch')
    speed=contract.units.speed_kmph/contract.units.distance_km_per_unit
    if abs(speed-packet['tw_speed'])>1e-12:raise ValueError('C0 unit speed mismatches physical env')
    snap=clone_snapshot(packet['snapshot'])
    class KnownOwned(Replanner):
        def plan(self,env,inst_idx,clock,vehicles,served_mask,visible_ids,replan_ids=None):
            for v in vehicles:
                if v.status in ('idle','ready') and (replan_ids is None or v.vehicle_id in replan_ids):
                    tail=[o for o in plan[str(v.vehicle_id)] if not served_mask[o]]
                    v.mutable_suffix=tail+[0]
    # Install pending known routes without changing the committed destination.
    width=max([len(v) for v in plan.values()]+[1])+1
    tails=np.full((snap['num_vehicles'],width),-1,np.int32);lens=[]
    for vid,status in enumerate(snap['vehicle_status']):
        cn=int(snap['committed_next'][vid]) if status=='committed' else None
        tail=[o for o in plan[str(vid)] if not snap['served_mask'][o] and o!=cn]
        if status in ('returning','closed'): tail=[]
        if status=='committed': tail=tail+[0]
        tails[vid,:len(tail)]=tail;lens.append(len(tail))
    snap['mutable_suffix']=tails;snap['mutable_suffix_len']=np.array(lens,np.int32)
    snap['state_hash']=snapshot_state_hash(snap)
    env=StrictOnlineEnv(packet['dataset'],packet['capacity'],packet['tw_speed'],
        snap['num_vehicles'],KnownOwned(),contract,packet['booking_horizon'])
    tr,_=env.run_resumed(snap)
    state=snap['replanner_state']
    ev=evaluate_partial_trace(tr,{k:v[0] for k,v in packet['dataset'].items()},contract,
        set(state['accepted']),set(state['rejected']),packet['budget'],packet['penalties'],set(packet['admission_input']['pending']))
    return ev,tr

