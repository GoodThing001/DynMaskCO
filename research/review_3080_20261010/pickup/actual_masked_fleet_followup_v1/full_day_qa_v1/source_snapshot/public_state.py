import numpy as np
from public_fleet import *
def export_public(env,i,clock,event,reveal,fleet,traces,served,universe):
    """Construct an actor input without unknown attributes/counts or full-day IDs.

    Original visible IDs are retained as public identities. Array width derives
    only from their maximum, not from the total hidden order count.
    """
    from recourse_snapshot import capture_recourse_snapshot,snapshot_state_hash
    rp=env.replanner
    known=[int(o) for o in universe if env.reveal_time[i,o]<=clock+1e-6]
    visible=set(known)
    if float(clock)<0: raise ValueError('nonnegative observed decision clock required')
    n=max(known,default=0)+1
    ds={}
    # Only explicit observed data fields; do not forward derived private matrices.
    fields=('coords','demands','temp_class','tw_start','tw_end','service_time','reveal_time','initial_quality')
    for key in fields:
        if key not in env.dataset: continue
        value=env.dataset[key]
        shape=(1,n)+value.shape[2:]
        dst=np.zeros(shape,dtype=value.dtype)
        dst[0,0]=value[i,0]
        for o in known: dst[0,o]=value[i,o]
        ds[key]=dst
    # capture records no unknown feature arrays, then explicitly project masks
    # and universe. Replanner state has only previously accepted/rejected IDs.
    snap=capture_recourse_snapshot(env,i,clock,event,reveal,fleet,traces,served,known)
    snap['instance_id']=0
    snap['num_nodes']=n
    snap['served_mask']=np.asarray(served[:n],bool).copy()
    snap['visible_mask']=np.zeros(n,bool);snap['visible_mask'][[0]+known]=True
    snap['customer_universe']=np.asarray(known,np.int32)
    snap['force_suffix']=[]
    state=snap['replanner_state']
    if state is None or state.get('version')!='c0-reserve-v1':
        raise ValueError('supported accepted-ownership state required')
    state['status_inst']=0
    for key in ('accepted','rejected'):
        if not set(state[key])<=visible: raise ValueError('private or unknown policy ID')
    if set(state['accepted'])|set(state['rejected']) != visible:
        raise ValueError('every revealed order must already have admission decision')
    plans={}
    ownership={}
    for v in fleet:
        route=[int(o) for o in rp._plan.get(v.vehicle_id,[]) if not served[o]]
        if v.status=='committed' and v.committed_next not in (None,0):
            nxt=int(v.committed_next)
            route=[nxt]+[o for o in route if o!=nxt]
        if not set(route)<=visible: raise ValueError('unknown suffix forbidden')
        for o in route:
            if o in ownership and ownership[o]!=v.vehicle_id: raise ValueError('duplicate ownership')
            ownership[o]=v.vehicle_id
        plans[str(v.vehicle_id)]=route
    outstanding=set(state['accepted'])-set(np.flatnonzero(snap['served_mask']))
    if set(ownership)!=outstanding: raise ValueError('accepted outstanding orders must have exact owner')
    state['plan']=plans
    snap['state_hash']=snapshot_state_hash(snap)
    physical=[]
    for v in fleet:
        origin=np.asarray(env.coords[i,v.current_node],float)
        target=v.committed_next if v.status in ('committed','returning') else None
        if target is not None:
            arrival=v.committed_arrive if v.status=='committed' else v.return_finish
            duration=float(arrival)-float(v.ready_time)
            progress=1. if duration<=0 else float(np.clip((clock-v.ready_time)/duration,0.,1.))
            pos=origin+progress*(np.asarray(env.coords[i,target],float)-origin)
        else:pos=origin
        physical.append(dict(vehicle_id=int(v.vehicle_id),position_xy=pos.tolist(),
            position_semantics='constant-speed interpolation of immutable committed leg',
            energy_kwh=0. if v.coldchain_state is None else float(v.coldchain_state.cumulative_energy_kwh)))
    return dict(schema='pickup-public-midcourse-v1',dataset=ds,snapshot=snap,
        plan=plans,ownership={str(o):v for o,v in ownership.items()},
        observed_vehicle_physics=physical,
        budget=float(rp.budget),capacity=float(env.capacity),tw_speed=float(env.tw_speed),
        booking_horizon=float(env.booking_horizon),contract=env.coldchain_contract,
        contract_hash=env.coldchain_contract.contract_hash,
        weight_hashes={},weight_scope='no learned model in this engineering actor',
        penalties={0:5.,1:10.,2:15.})
