"""Engineering undecided-order projector. No production controller or scoring adaptation.\nFrozen source SHA256 9e0d014ea3123b8dc8b6a7cc56531667446d870fbfcbaa0a3603fd90302a0291\n"""
import numpy as np
from public_fleet import *
def export_undecided(env,i,clock,event,reveal,fleet,traces,served,universe,current_order):
    """Construct an actor input without unknown attributes/counts or full-day IDs.

    Original visible IDs are retained as public identities. Array width derives
    only from their maximum, not from the total hidden order count.
    """
    from recourse_snapshot import capture_recourse_snapshot,snapshot_state_hash
    rp=env.replanner
    known=sorted(set(int(o) for o in universe if env.demands[i,o]>0 and env.reveal_time[i,o]<=clock+1e-6))
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
    accepted=set(state['accepted']);rejected=set(state['rejected'])
    pending=visible-accepted-rejected
    if accepted & rejected: raise ValueError('overlapping admission decisions')
    if type(current_order) is not int or current_order not in pending or served[current_order]:
        raise ValueError('current order must be genuinely undecided and unserved')
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
    packet=dict(schema='pickup-public-undecided-v1',dataset=ds,snapshot=snap,
        admission_input=dict(current_order=current_order,pending=sorted(pending),
            semantics='pending observed orders have no accept/reject decision or rejection penalty'),
        plan=plans,ownership={str(o):v for o,v in ownership.items()},
        observed_vehicle_physics=physical,
        budget=float(rp.budget),capacity=float(env.capacity),tw_speed=float(env.tw_speed),
        booking_horizon=float(env.booking_horizon),contract=env.coldchain_contract,
        contract_hash=env.coldchain_contract.contract_hash,
        weight_hashes={},weight_scope='no learned model in this engineering actor',
        penalties={0:5.,1:10.,2:15.})

    validate_undecided(packet)
    return packet

def validate_undecided(packet):
    """Separate input contract; old decided-input validator remains unchanged."""
    from recourse_snapshot import snapshot_state_hash
    if packet['schema']!='pickup-public-undecided-v1': raise ValueError('schema')
    snap=packet['snapshot'];ds=packet['dataset'];state=snap['replanner_state']
    known=set(int(o) for o in snap['customer_universe'])
    allowed={'coords','demands','temp_class','tw_start','tw_end','service_time','reveal_time','initial_quality'}
    if set(ds)-allowed: raise ValueError('private dataset channel')
    if set(np.flatnonzero(ds['demands'][0]>0))!=known: raise ValueError('known universe')
    if snap['num_nodes']!=max(known,default=0)+1: raise ValueError('public width')
    if any(ds['reveal_time'][0,o]>snap['clock']+1e-6 for o in known): raise ValueError('unrevealed order')
    expected=np.zeros(snap['num_nodes'],bool);expected[[0]+sorted(known)]=True
    if not np.array_equal(expected,snap['visible_mask']): raise ValueError('visibility')
    a=set(state['accepted']);r=set(state['rejected']);p=set(packet['admission_input']['pending'])
    if a&r or a&p or r&p or a|r|p!=known: raise ValueError('three-way admission partition')
    if len(p)!=len(packet['admission_input']['pending']): raise ValueError('duplicate pending ID')
    current=packet['admission_input']['current_order']
    if type(current) is not int or current not in p or snap['served_mask'][current]: raise ValueError('undecided current order')
    served=set(np.flatnonzero(snap['served_mask']))-{0}
    if not served<=a: raise ValueError('served order must be accepted')
    if state['version']!='c0-reserve-v1' or state['status_inst']!=0: raise ValueError('policy state identity')
    plan=packet['plan'];flat=[o for route in plan.values() for o in route]
    if set(plan)!={str(i) for i in range(snap['num_vehicles'])}: raise ValueError('fleet keys')
    if any(type(o) is not int for o in flat) or len(flat)!=len(set(flat)) or set(flat)!=a-served:
        raise ValueError('accepted outstanding ownership')
    if state['plan']!=plan or packet['ownership']!={str(o):int(v) for v,route in plan.items() for o in route}:
        raise ValueError('plan/ownership mismatch')
    for v,status in enumerate(snap['vehicle_status']):
        cn=int(snap['committed_next'][v])
        if status=='committed' and cn>0 and (not plan[str(v)] or plan[str(v)][0]!=cn):
            raise ValueError('immutable committed destination')
    if snapshot_state_hash(snap)!=snap['state_hash']: raise ValueError('snapshot digest')
    if packet['contract'].contract_hash!=packet['contract_hash']: raise ValueError('contract identity')
    if abs(packet['contract'].units.speed_kmph/packet['contract'].units.distance_km_per_unit-packet['tw_speed'])>1e-12:
        raise ValueError('C0 unit speed')
    return True
