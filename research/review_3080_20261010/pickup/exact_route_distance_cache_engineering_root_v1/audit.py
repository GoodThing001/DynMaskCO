"""Single new local action component, not another K10 utility request."""
import os,sys,json,pickle,copy,time,hashlib,ctypes,io
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];sys.path.insert(0,str(ROOT/'tools'))
os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import qualify_pickup_joint_policy_transaction_root_v3 as old
import psutil,numpy as np
from strict_online_env import StrictOnlineEnv
from recourse_snapshot import restore_recourse_snapshot
from scenario_saa import UncondHistoricalSampler
import scenario_saa
from energy_complete_saa import EnergyCompleteSAA
BASE=HERE.parent/'independent_rolling_incremental_engineering_root_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,x):
    with (HERE/name).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def closure():
    paths={Path(__file__).resolve(),Path(sys.executable).resolve()}
    for m in list(sys.modules.values()):
        p=getattr(m,'__file__',None)
        if p and Path(p).is_file():paths.add(Path(p).resolve())
    for m in psutil.Process().memory_maps():
        p=Path(m.path)
        if p.is_file():paths.add(p.resolve())
    return {str(p):sha(p) for p in sorted(paths)}
def brief(st):return {str(v):dict(route=[int(o) for o in s['route']],cur=int(s['cur']),cur_time=float(s['cur_time']).hex(),load=float(s['load']).hex(),frozen=int(s.get('frozen',0))) for v,s in st.items()}
class Observer(EnergyCompleteSAA):
    def _route_dist_u(self,space,cur,route):
        value=super()._route_dist_u(space,cur,route)
        self.distance_log.append(dict(cur=int(cur),route=[int(o) for o in route],value_hex=float(value).hex(),type=type(value).__module__+'.'+type(value).__qualname__))
        return value
    def _route_feasible_u(self,space,env,cur,cur_time,load,route,deadline):
        value=super()._route_feasible_u(space,env,cur,cur_time,load,route,deadline)
        self.feasible_log.append(dict(cur=int(cur),cur_time_hex=float(cur_time).hex(),load_hex=float(load).hex(),route=[int(o) for o in route],deadline_hex=float(deadline).hex(),feasible=bool(value)))
        return value
def main():
    psutil.Process().cpu_affinity([7]);ctypes.WinDLL('SHCore.dll');psutil.Process().memory_maps();encode,decode=old.codecs();old.pf.digest({'prime':'hash imports'});np.savez(io.BytesIO(),prime=np.array([0]))
    reg=read(HERE/'REGISTRATION.json');ctx=decode(read(BASE/'SIMULATION_CONTEXT.json'));p=ctx['packet'];prior=p['snapshot']['replanner_state'];clock=p['snapshot']['clock'];fleet,_,served=restore_recourse_snapshot(p['snapshot'])
    with old.HISTORY.open('rb') as f:history=pickle.load(f)
    actor=Observer(budget=p['budget'],capacity=p['capacity'],booking_horizon=p['booking_horizon'],contract=p['contract'],cooling_share=0.,hist=history['history_times_classes'],k=10,sampler=UncondHistoricalSampler(history['history']),reject_penalty=p['penalties'],K=10,time_limit=10.,arm_seed=7001,anytime_vote=False,shadow_mode='ls',energy_pricing='marginal',future_policy='reveal',standby_orders=0,incr_eval=False)
    actor.restore_state(copy.deepcopy(prior));env=StrictOnlineEnv(p['dataset'],p['capacity'],p['tw_speed'],p['snapshot']['num_vehicles'],actor,p['contract'],p['booking_horizon']);actor._sync_id_map(env,0,clock);observed=actor._build_snapshot(env,0,clock,fleet,served)
    scenarios=actor.sampler.sample(observed,np.random.default_rng(202610130901),10);space,orders=actor._build_space(env,0,scenarios);lookup={id(x):space.n_real+j for j,x in enumerate(orders)};ids=[[lookup[id(x)] for x in s] for s in scenarios]
    row=next(r for r in ctx['candidates']['reject'] if r['index']==ctx['selected']['reject']);actor.restore_state(copy.deepcopy(row['packet']['snapshot']['replanner_state']));states=actor._sim_states(env,0,clock,fleet,served,{int(v):r for v,r in row['plan'].items()})
    actor.distance_log=[];actor.feasible_log=[];sequence=sorted(ids[7],key=lambda o:space.reveal[o])[:4];source_start=closure();save('AUDIT_SOURCE_START.json',source_start)
    checks=[];events=[];transitions=[]
    def ck(n,v):checks.append(dict(name=n,pass_=bool(v)))
    original_transition=scenario_saa.transition_segment
    def physical(*args,**kwargs):
        result=original_transition(*args,**kwargs);transitions.append(encode(result));return result
    scenario_saa.transition_segment=physical
    started=time.perf_counter();deadline=float(env.tw_end[0,0])
    for o in sequence:
        before=brief(states);accepted=actor._insert_u(space,env,o,states,deadline);events.append(dict(kind='insert',order=int(o),accepted=bool(accepted),before=before,after=brief(states)))
    before=brief(states);actor._local_search_u(space,env,states,deadline,max_passes=4);events.append(dict(kind='local_search',max_passes=4,before=before,after=brief(states)))
    energy,distance=actor._complete_u(space,env,0,clock,states,actor.booking_horizon);finished=time.perf_counter();scenario_saa.transition_segment=original_transition
    ck('original_incr_eval_false',not actor.incr_eval);ck('same_original_common_scenarios',old.pf.digest(scenarios)==ctx['scenario_sha256'] and ids==ctx['indices']);ck('fixed_next_unexecuted_index7',reg['scenario_index']==7 and sequence==sorted(ids[7],key=lambda o:space.reveal[o])[:4]);ck('physical_completion_available',energy is not None and distance is not None)
    expected=read(HERE/'RESULT.json');checks=[]
    def ck(n,v):checks.append(dict(name=n,pass_=bool(v)))
    for p,h in reg['sources'].items():ck('registered_source:'+p,sha(p)==h)
    ck('original_independent_sequence',expected['sequence']==[int(o) for o in sequence])
    ck('all_recorded_local_candidates_feasibility_exact',expected['feasible_log']==actor.feasible_log)
    ck('all_recorded_distance_order_and_return_type_exact',expected['distance_log']==actor.distance_log)
    ck('all_insertion_and_four_pass_final_trajectories_exact',expected['events']==events)
    ck('all_original_C0_transition_outputs_exact',expected['cold_transition_outputs']==transitions)
    ck('full_terminal_energy_hex',expected['terminal_energy_hex']==(None if energy is None else float(energy).hex()))
    ck('full_terminal_distance_hex',expected['terminal_distance_hex']==(None if distance is None else float(distance).hex()))
    matrix=np.load(HERE/'DISTANCE_MATRIX.npy',allow_pickle=False)
    for i,row in enumerate(expected['distance_log']):
        total=0.;cur=row['cur']
        for o in row['route']:total+=float(matrix[cur,o]);cur=o
        total+=float(matrix[cur,0]);ck('independent_ordered_scalar_sum:'+str(i),total.hex()==row['value_hex'])
    source_end=closure();save('AUDIT_SOURCE_END.json',source_end);ck('actual_loaded_source_start_end',source_start==source_end)
    result=dict(checks_count=len(checks),failed=sum(not x['pass_'] for x in checks),failure_names=[x['name'] for x in checks if not x['pass_']],checks=checks,reference_incr_eval=False,reference_elapsed=finished-started,recorded_local_candidate_calls=len(actor.feasible_log),recorded_distance_calls=len(actor.distance_log),actual_cold_transition_count=len(transitions),actual_loaded_files=len(source_start),start_end_exact=source_start==source_end,full_request_budget_qualified=False,full_scientific_loaded_closure=False,stable_online_gain=False)
    save('INDEPENDENT_AUDIT.json',result);save('AUDIT_RUN_EXIT.json',dict(exit_code=int(result['failed']>0)));print(json.dumps({k:v for k,v in result.items() if k!='checks'}));return int(result['failed']>0)
if __name__=='__main__':raise SystemExit(main())
