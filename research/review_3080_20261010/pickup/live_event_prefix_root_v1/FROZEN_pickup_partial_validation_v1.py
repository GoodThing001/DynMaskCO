import numpy as np
def validate_partial(packet):
    """Separate input contract; old decided-input validator remains unchanged."""
    from recourse_snapshot import snapshot_state_hash
    if packet['schema']!='pickup-public-partial-admissions-v1': raise ValueError('schema')
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
    if type(current) is not int or current not in a|r or snap['served_mask'][current]: raise ValueError('current decision missing')
    if packet['admission_input']['current_decision'] != ('accepted' if current in a else 'rejected'): raise ValueError('decision label')
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
