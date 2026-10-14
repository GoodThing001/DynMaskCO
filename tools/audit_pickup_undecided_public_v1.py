"""Independent field reconstruction; does not call the new projector/validator."""
import copy,json,sys,hashlib
from pathlib import Path
import qualify_pickup_private_public_transport_v2 as fixture
import numpy as np
import public_fleet as pf
from recourse_snapshot import snapshot_state_hash
ROOT=fixture.ROOT;P=fixture.P;OUT=P/'undecided_public_input_root_v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    import psutil
    psutil.Process().cpu_affinity([7]);encode,decode=fixture.codec()
    raw=json.loads((OUT/'RESULT.json').read_text(encoding='utf-8'));reg=json.loads((OUT/'REGISTRATION.json').read_text(encoding='utf-8'))
    checks=[]
    def ck(n,v):checks.append(dict(name=n,pass_=bool(v)))
    for path,h in reg['sources'].items():ck('source:'+path,sha(path)==h)
    ck('producer_exit',json.loads((OUT/'RUN_EXIT.json').read_text())['exit_code']==0 and raw['failed']==0)
    packets=[]
    for j,record in enumerate(raw['records']):
        packet=decode(record['packet']);s=packet['snapshot'];a=set(s['replanner_state']['accepted']);r=set(s['replanner_state']['rejected'])
        o=record['order'];pending=set(packet['admission_input']['pending']);tag=str(j)
        # Independently restore the same declared private data, not projector output.
        row,env,fleet,traces,served=fixture.state(record.get('index',0),record['variant'])
        clock=s['clock'];ds=env.dataset
        known={c for c in range(1,env.num_nodes) if ds['demands'][0,c]>0 and ds['reveal_time'][0,c]<=clock+1e-6}
        ck(tag+':observed_universe',known==set(s['customer_universe']))
        ck(tag+':width',s['num_nodes']==max(known)+1)
        ck(tag+':partition',not(a&r or a&pending or r&pending) and a|r|pending==known)
        ck(tag+':actual_current_pending',o in pending and o not in a|r)
        expected_visible=np.zeros(s['num_nodes'],bool);expected_visible[[0]+sorted(known)]=True
        ck(tag+':visible_mask',np.array_equal(expected_visible,s['visible_mask']))
        for name,x in packet['dataset'].items():
            expected=np.zeros_like(x);expected[0,[0]+sorted(known)]=ds[name][0,[0]+sorted(known)]
            ck(tag+':field:'+name,np.array_equal(expected,x))
        ck(tag+':hash',snapshot_state_hash(s)==s['state_hash'])
        ck(tag+':private_nonmutation',record['before']==record['after'])
        ck(tag+':pending_unowned',not pending&set(int(x) for x in packet['ownership']))
        if record['kind']=='derived_old_state':
            old=row['public_request'];st=old['snapshot']['replanner_state']
            ck(tag+':derived_accepted',a==set(st['accepted']))
            ck(tag+':derived_rejected',r==set(st['rejected'])-{o})
            ck(tag+':original_plan',packet['plan']==old['plan'] and packet['ownership']==old['ownership'])
            for key,x in old['snapshot'].items():
                if key not in ('replanner_state','state_hash'):
                    ck(tag+':snapshot_field:'+key,pf.digest(x)==pf.digest(s[key]))
        else:
            ck(tag+':no_initial_decisions',not a and not r and pending==known)
            ck(tag+':initial_actual_hook',record['fleet_created_by_run'] and record['admission_calls']==record['plan_calls']==0)
            ck(tag+':zero_clock_event',s['clock']==0 and s['event_id']==0 and s['replan_count']==0)
            ck(tag+':initial_fleet',all(x=='idle' for x in s['vehicle_status']) and all(x==0 for x in s['vehicle_node']) and not packet['ownership'])
            ck(tag+':initial_served',list(np.flatnonzero(s['served_mask']))==[0])
            ck(tag+':initial_no_execution',all(x==0 for x in s['services_len']) and all(x==0 for x in s['served_route_len']))
        packets.append(packet)
    for j in (0,2,4,6):ck('pair:'+str(j),pf.digest(packets[j])==pf.digest(packets[j+1]))
    ck('nine_adversarial_boundaries',len(raw['boundaries'])==9 and all(x['error'] for x in raw['boundaries']))
    ck('registered_bytes_end',all(sha(p)==h for p,h in reg['sources'].items()))
    result=dict(checks=checks,checks_count=len(checks),failed=sum(not x['pass_'] for x in checks),
                producer_sha256=sha(OUT/'RESULT.json'),registration_sha256=sha(OUT/'REGISTRATION.json'),
                independent_projector_called=False,independent_admission_solver_called=False,
                actual_initial_hook_only=True,continuous_event_ownership_qualified=False,
                full_live_request_qualified=False,stable_online_gain=False)
    with (OUT/'INDEPENDENT_UNDECIDED_PUBLIC_FIELDS.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(dict(checks=len(checks),failed=result['failed'])));return int(result['failed']>0)
if __name__=='__main__':raise SystemExit(main())
