"""Fresh process audit; no new controller or run module imported as reference."""
import os,sys,json,copy,pickle,hashlib,time,ctypes,math
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];sys.path.insert(0,str(ROOT/'tools'))
os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import qualify_pickup_joint_policy_transaction_root_v3 as old
import numpy as np,psutil
from rolling_optimizer import RollingOptimizeReplanner
from strict_online_env import StrictOnlineEnv
from recourse_snapshot import restore_recourse_snapshot
from scenario_saa import UncondHistoricalSampler
from energy_complete_saa import EnergyCompleteSAA
from coldchain_evaluator_a1 import evaluate_trace_a1
def read(n):return json.loads((HERE/n).read_text(encoding='utf-8'))
def save(n,x):
    with (HERE/n).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def main():
    psutil.Process().cpu_affinity([7]);ctypes.WinDLL('SHCore.dll');psutil.Process().memory_maps();encode,decode=old.codecs()
    reg=read('REGISTRATION.json');receipt=read('RUN_RECEIPT.json');raw=decode(read('COMPONENT_RESULT.json'));p=raw['packet'];r=raw['result'];checks=[]
    def ck(name,value):checks.append(dict(name=name,pass_=bool(value)))
    for path,h in reg['sources'].items():ck('registered_source:'+path,old.sha(path)==h)
    origin=decode(json.loads(Path(next(path for path in reg['sources'] if path.endswith('undecided_public_input_root_v2\\RESULT.json'))).read_text(encoding='utf-8'))['records'][reg['chosen']['position']]['packet'])
    ck('fixed_metadata_input',old.pf.digest(origin)==old.pf.digest(p));ck('current_undecided',p['admission_input']['current_order'] not in set(p['snapshot']['replanner_state']['accepted'])|set(p['snapshot']['replanner_state']['rejected']))
    ck('independent_admission_ownership',r['independent_admission_owner'] and not r['own_uncond_dynamic_called']);ck('component_not_live_science',not r['formal_strong_dynamic_qualified'] and not r['full_request_budget_qualified'] and not r['stable_online_gain'])
    ck('input_unchanged',r['input_unchanged']);ck('producer_cleanly_reaped',receipt['worker_returncode']==0 and receipt['worker_reaped']);ck('outer_elapsed_under10',receipt['elapsed']<10.);ck('compute_predeadline',raw['finished']<raw['deadline'])
    start=read('SOURCE_START.json');end=read('SOURCE_END.json');ck('actual_loaded_source_start_end',start==end)
    for path,h in end.items():ck('loaded_source_byte:'+path,old.sha(path)==h)
    rp=RollingOptimizeReplanner(p['budget'],p['capacity'],p['booking_horizon'],p['contract'],0.,p['penalties'],10.)
    rp.restore_state(copy.deepcopy(p['snapshot']['replanner_state']));env=StrictOnlineEnv(p['dataset'],p['capacity'],p['tw_speed'],p['snapshot']['num_vehicles'],rp,p['contract'],p['booking_horizon']);fleet,_,served=restore_recourse_snapshot(p['snapshot']);clock=p['snapshot']['clock'];o=p['admission_input']['current_order']
    def states(plan):return rp._start_states(env,0,fleet,served,clock,{int(v):r for v,r in plan.items()})
    def feasible(plan):return all(rp._route_feasible(env,0,s['cur'],s['cur_time'],s['load'],s['route']) for s in states(plan).values())
    def distance(plan):return rp._total_distance(env,0,states(plan))
    def moves(plan):
        active=[str(v) for v,s in enumerate(p['snapshot']['vehicle_status']) if s in ('idle','ready','committed')]
        frozen={v:int(p['snapshot']['vehicle_status'][int(v)]=='committed' and int(p['snapshot']['committed_next'][int(v)])>0) for v in active}
        items=[(v,i,x) for v in active for i,x in enumerate(plan[v]) if i>=frozen[v]];keys=set();out=[]
        for v,i,x in items:
            for w in active:
                for j in range(frozen[w],len(plan[w])+1):
                    if v==w and j in (i,i+1):continue
                    q={k:list(vals) for k,vals in plan.items()};del q[v][i];q[w].insert(j-int(v==w and j>i),x);key=tuple(tuple(q[k]) for k in sorted(q))
                    if key not in keys:keys.add(key);out.append(q)
        for a,(v,i,x) in enumerate(items):
            for w,j,y in items[a+1:]:
                q={k:list(vals) for k,vals in plan.items()};q[v][i]=y;q[w][j]=x;key=tuple(tuple(q[k]) for k in sorted(q))
                if key not in keys:keys.add(key);out.append(q)
        return out
    expected_initial={'reject':[p['plan']],'accept':[]}
    for v,s in enumerate(p['snapshot']['vehicle_status']):
        if s not in ('idle','ready','committed'):continue
        frozen=int(s=='committed' and int(p['snapshot']['committed_next'][v])>0)
        for j in range(frozen,len(p['plan'][str(v)])+1):
            q=copy.deepcopy(p['plan']);q[str(v)].insert(j,o);expected_initial['accept'].append(q)
    best={}
    for decision,rows in r['candidates'].items():
        ck(decision+':all_initial_plans',[x['input_plan'] for x in rows]==expected_initial[decision]);best_row=None
        for row in rows:
            prefix=decision+':'+str(row['index']);ck(prefix+':local_feasibility',row['local_feasible']==feasible(row['input_plan']))
            if not row['local_feasible']:continue
            current=row['input_plan'];ck(prefix+':fixedpass_cap',len(row['recourse'])<=4)
            for step in row['recourse']:
                ck(prefix+':before:'+str(step['iteration']),current==step['before']);possible=moves(current);ck(prefix+':all_neighborhood:'+str(step['iteration']),possible==[x['plan'] for x in step['trials']]);best_plan=current;best_d=distance(current)
                for j,trial in enumerate(step['trials']):
                    ok=feasible(trial['plan']);ck(prefix+':local:'+str(step['iteration'])+':'+str(j),ok==trial['local_feasible'])
                    if ok:
                        d=distance(trial['plan']);ck(prefix+':distance:'+str(step['iteration'])+':'+str(j),d.hex()==float(trial['distance']).hex())
                        if d<best_d-1e-9:best_plan=trial['plan'];best_d=d
                ck(prefix+':best_improvement:'+str(step['iteration']),best_plan==step['after']);current=best_plan
            ck(prefix+':final_plan',current==row['plan']);bp=row['packet'];st=bp['snapshot']['replanner_state']
            ck(prefix+':current_only_partition',set(st['accepted'])-{o}==set(p['snapshot']['replanner_state']['accepted']) and set(st['rejected'])-{o}==set(p['snapshot']['replanner_state']['rejected']) and not set(st['accepted'])&set(st['rejected']))
            ck(prefix+':no_remaining_pending',not bp['admission_input']['pending']);flat=[x for rr in row['plan'].values() for x in rr];ck(prefix+':once_accepted_outstanding',len(flat)==len(set(flat)) and set(flat)==set(st['accepted'])-set(np.flatnonzero(served)))
            for v,s in enumerate(p['snapshot']['vehicle_status']):
                cn=int(p['snapshot']['committed_next'][v])
                if s=='committed' and cn>0:ck(prefix+':frozen:'+str(v),row['plan'][str(v)][0]==cn)
            # Direct original whole observed-universe evaluator, not the new
            # partial-admission adapter. Both branches resolve this sole pending.
            ev=evaluate_trace_a1(row['raw'],{k:v[0] for k,v in p['dataset'].items()},p['contract'],set(st['accepted']),set(st['rejected']),p['budget'],p['penalties'])
            for key,val in ev.items():ck(prefix+':C0:'+key,old.pf.digest(val)==old.pf.digest(row['certificate']['evaluation'][key]))
            ck(prefix+':hard_qualification',ev['hard_feasible']==row['qualified'])
            actual_service={x.node for trace in row['raw'] for x in trace.services};ck(prefix+':reject_not_served',decision!='reject' or o not in actual_service)
            ck(prefix+':current_reject_exact_once',ev['reject_loss']==sum(p['penalties'][int(p['dataset']['temp_class'][0,x])] for x in st['rejected']))
            if row['qualified'] and (best_row is None or ev['utility']>best_row['certificate']['evaluation']['utility']):best_row=row
        best[decision]=best_row;ck(decision+':chosen_candidate',r['selected'][decision]==(None if best_row is None else best_row['index']))
    with old.HISTORY.open('rb') as f:history=pickle.load(f)
    shadow=EnergyCompleteSAA(budget=p['budget'],capacity=p['capacity'],booking_horizon=p['booking_horizon'],contract=p['contract'],cooling_share=0.,hist=history['history_times_classes'],k=10,sampler=UncondHistoricalSampler(history['history']),reject_penalty=p['penalties'],K=10,time_limit=10.,arm_seed=7001,anytime_vote=False,shadow_mode='ls',energy_pricing='marginal',future_policy='reveal',standby_orders=0)
    shadow.restore_state(copy.deepcopy(p['snapshot']['replanner_state']));env.replanner=shadow;shadow._sync_id_map(env,0,clock);observed=shadow._build_snapshot(env,0,clock,fleet,served);scenarios=shadow.sampler.sample(observed,np.random.default_rng(reg['seed']),10)
    ck('unconditional_history_sampling_exact',old.pf.digest(scenarios)==r['scenario_sha256']==old.pf.digest(r['scenarios']));ck('observed_snapshot_exact',old.pf.digest(observed)==old.pf.digest(r['observed_snapshot']))
    space,orders=shadow._build_space(env,0,scenarios);ids={id(x):space.n_real+j for j,x in enumerate(orders)};indices=[[ids[id(x)] for x in ss] for ss in scenarios];ck('common_ids_exact',indices==r['scenario_indices'])
    means={}
    for decision in ('reject','accept'):
        row=best[decision]
        if row is None:ck(decision+':infeasible_no_vote',not r['scores'][decision]['qualified']);continue
        shadow.restore_state(copy.deepcopy(row['packet']['snapshot']['replanner_state']));st=shadow._sim_states(env,0,clock,fleet,served,{int(v):v0 for v,v0 in row['plan'].items()});values=[]
        ck(decision+':unavailable_ledger',old.pf.digest(shadow.unavailable_ledger)==old.pf.digest(r['scores'][decision]['unavailable_ledger']));ck(decision+':branch_state',old.pf.digest(shadow.export_state())==old.pf.digest(r['scores'][decision]['branch_admission_state']))
        for j,si in enumerate(indices):
            value=float(shadow._sim_scenario(env,0,clock,st,si,space))-(p['penalties'][int(p['dataset']['temp_class'][0,o])] if decision=='reject' else 0.)
            values.append(value);ck(decision+':complete_branch:'+str(j),value.hex()==float(r['scores'][decision]['branch_values'][j]).hex())
        means[decision]=float(np.mean(values));ck(decision+':mean_exact',means[decision].hex()==float(r['scores'][decision]['mean_shadow_utility']).hex())
    expected='accept' if 'accept' in means and means['accept']>means['reject'] else 'reject';ck('strict_accept_tie_reject',expected==r['choice']);ck('selected_output_exact',old.pf.digest(best[expected]['packet'])==old.pf.digest(r['output_packet']))
    source_bad=[c['name'] for c in checks if not c['pass_'] and ('source' in c['name'] or 'byte' in c['name'])];raw_bad=[c['name'] for c in checks if not c['pass_'] and c['name'] not in source_bad]
    result=dict(checks=checks,checks_count=len(checks),failed=sum(not c['pass_'] for c in checks),raw_replay_failure_names=raw_bad,registered_source_failure_names=source_bad,decision=r['choice'],candidate_counts={k:len(v) for k,v in r['candidates'].items()},qualified_candidate_counts={k:sum(x['qualified'] for x in v) for k,v in r['candidates'].items()},mean_shadow_utility=means,formal_strong_dynamic_qualified=False,full_request_budget_qualified=False,full_scientific_loaded_closure=False,stable_online_gain=False)
    save('INDEPENDENT_AUDIT.json',result);save('AUDIT_RUN_EXIT.json',dict(exit_code=int(result['failed']>0)));print(json.dumps({k:v for k,v in result.items() if k!='checks'}));return int(result['failed']>0)
if __name__=='__main__':raise SystemExit(main())
