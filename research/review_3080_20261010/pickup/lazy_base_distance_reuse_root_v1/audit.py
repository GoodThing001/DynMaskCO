"""One fixed new four-order local component; no full-day/K10 request."""
import os,sys,json,pickle,copy,time,hashlib,ctypes,io,encodings.gbk
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3];sys.path.insert(0,str(ROOT/'tools'))
os.environ.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
import qualify_pickup_joint_policy_transaction_root_v3 as old
import engineering_stage_closure_v1 as seal
import numpy as np,psutil
from strict_online_env import StrictOnlineEnv
from recourse_snapshot import restore_recourse_snapshot
from scenario_saa import UncondHistoricalSampler
import scenario_saa
from energy_complete_saa import EnergyCompleteSAA
BASE=HERE.parent/'independent_rolling_incremental_engineering_root_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(n,x):seal.save(HERE/n,x)
def brief(states):return {str(v):dict(route=[int(o) for o in s['route']],cur=int(s['cur']),cur_time=float(s['cur_time']).hex(),load=float(s['load']).hex(),frozen=int(s.get('frozen',0))) for v,s in states.items()}
class Observer(EnergyCompleteSAA):
    def _route_dist_u(self,space,cur,route):
        value=super()._route_dist_u(space,cur,route)
        self.distance_log.append(dict(operation=self.operation,cur=int(cur),route=[int(o) for o in route],value_hex=float(value).hex(),type=type(value).__module__+'.'+type(value).__qualname__))
        return value
    def _route_feasible_u(self,space,env,cur,cur_time,load,route,deadline):
        value=super()._route_feasible_u(space,env,cur,cur_time,load,route,deadline)
        self.feasible_log.append(dict(operation=self.operation,cur=int(cur),cur_time_hex=float(cur_time).hex(),load_hex=float(load).hex(),route=[int(o) for o in route],deadline_hex=float(deadline).hex(),feasible=bool(value)))
        return value
def main():
    seal.initialize();ctypes.WinDLL('kernel.appcore.dll');encode,decode=old.codecs();old.pf.digest({'prime':'digest dependencies'});np.savez(io.BytesIO(),prime=np.array([0]))
    if np.__version__!='1.26.4' or sys.version_info[:3]!=(3,12,7):raise ValueError('Registered runtime mismatch')
    if psutil.virtual_memory().available<.5*2**30:raise ValueError('No-model resource gate failed before stage')
    reg=read(HERE/'REGISTRATION.json');ctx=decode(read(BASE/'SIMULATION_CONTEXT.json'));p=ctx['packet'];initial_param_digest=old.pf.digest(p);prior=p['snapshot']['replanner_state'];clock=p['snapshot']['clock'];fleet,_,served=restore_recourse_snapshot(p['snapshot'])
    with old.HISTORY.open('rb') as f:history=pickle.load(f)
    actor=Observer(budget=p['budget'],capacity=p['capacity'],booking_horizon=p['booking_horizon'],contract=p['contract'],cooling_share=0.,hist=history['history_times_classes'],k=10,sampler=UncondHistoricalSampler(history['history']),reject_penalty=p['penalties'],K=10,time_limit=10.,arm_seed=7001,anytime_vote=False,shadow_mode='ls',energy_pricing='marginal',future_policy='reveal',standby_orders=0,incr_eval=False)
    actor.restore_state(copy.deepcopy(prior));env=StrictOnlineEnv(p['dataset'],p['capacity'],p['tw_speed'],p['snapshot']['num_vehicles'],actor,p['contract'],p['booking_horizon']);actor._sync_id_map(env,0,clock);observed=actor._build_snapshot(env,0,clock,fleet,served)
    scenarios=actor.sampler.sample(observed,np.random.default_rng(202610130901),10);space,orders=actor._build_space(env,0,scenarios);lookup={id(x):space.n_real+j for j,x in enumerate(orders)};ids=[[lookup[id(x)] for x in s] for s in scenarios]
    row=next(r for r in ctx['candidates']['reject'] if r['index']==ctx['selected']['reject']);actor.restore_state(copy.deepcopy(row['packet']['snapshot']['replanner_state']));states=actor._sim_states(env,0,clock,fleet,served,{int(v):r for v,r in row['plan'].items()})
    space.D=np.array(space.D,copy=True,order='C');space.D.flags.writeable=False;matrix_sha=hashlib.sha256(space.D.tobytes()).hexdigest()
    actor.base_computations=0;actor.base_reuses=0;actor.score_log=[];actor.distance_log=[];actor.feasible_log=[];sequence=sorted(ids[8],key=lambda o:space.reveal[o])[:4]
    extra=tuple(Path(x) for x in reg['sources']);start_closure=seal.closure(extra);save('AUDIT_SOURCE_START.json',start_closure)
    events=[];transitions=[];checks=[]
    def ck(n,v):checks.append(dict(name=n,pass_=bool(v)))
    original_transition=scenario_saa.transition_segment
    def physical(*args,**kwargs):
        out=original_transition(*args,**kwargs);transitions.append(encode(out));return out
    scenario_saa.transition_segment=physical
    started=time.perf_counter();deadline=float(env.tw_end[0,0])
    for j,o in enumerate(sequence):
        actor.operation='insert_'+str(j);before=brief(states);ok=actor._insert_u(space,env,o,states,deadline);events.append(dict(kind='insert',order=int(o),accepted=bool(ok),before=before,after=brief(states)))
    actor.operation='local_search_4';before=brief(states);actor._local_search_u(space,env,states,deadline,max_passes=4);events.append(dict(kind='local_search',max_passes=4,before=before,after=brief(states)))
    actor.operation='physical_completion';energy,distance=actor._complete_u(space,env,0,clock,states,actor.booking_horizon)
    ck('original_owned_matrix_still_readonly_and_exact',not space.D.flags.writeable and hashlib.sha256(space.D.tobytes()).hexdigest()==matrix_sha)
    ck('all_original_public_parameters_unchanged',old.pf.digest(p)==initial_param_digest)
    finished=time.perf_counter();scenario_saa.transition_segment=original_transition
    ck('fixed_scenario8_first4',sequence==sorted(ids[reg['scenario_index']],key=lambda o:space.reveal[o])[:4])
    ck('full_common_scenarios_and_ids_original',old.pf.digest(scenarios)==ctx['scenario_sha256'] and ids==ctx['indices'])
    ck('original_incr_eval_false',not actor.incr_eval);ck('original_physical_completion_available',energy is not None and distance is not None)
    end_closure=seal.closure(extra);save('AUDIT_SOURCE_END.json',end_closure);ck('actual_loaded_files_and_maps_start_end',seal.same(start_closure,end_closure))
    expected=read(HERE/'RESULT.json');check_before=len(checks)
    for path,h in reg['sources'].items():ck('registered_source:'+path,seal.sha(path)==h)
    ck('all_feasibility_call_sequence_and_values_exact',expected['feasible_log']==actor.feasible_log)
    ck('all_original_insert_choices_and_private_route_states_exact',expected['events']==events)
    ck('all_original_physical_transition_outputs_exact',expected['cold_transition_outputs']==transitions)
    ck('original_terminal_energy_hex',expected['terminal_energy_hex']==(None if energy is None else float(energy).hex()))
    ck('original_terminal_distance_hex',expected['terminal_distance_hex']==(None if distance is None else float(distance).hex()))
    filtered=[];score_reference=[];skipped=0;computed=0
    for step in range(4):
        op='insert_'+str(step);before=events[step]['before'];trials=[]
        for v,s in before.items():
            for pos in range(s['frozen'],len(s['route'])+1):trials.append((int(v),pos,s,s['route'][:pos]+[sequence[step]]+s['route'][pos:]))
        feasible=[r for r in actor.feasible_log if r['operation']==op];dist=[r for r in actor.distance_log if r['operation']==op]
        ck('all_original_candidate_positions:'+str(step),len(trials)==len(feasible) and all(f['cur']==s['cur'] and f['route']==trial for (_,_,s,trial),f in zip(trials,feasible)))
        good=[entry for entry,f in zip(trials,feasible) if f['feasible']];ck('original_two_distance_calls_per_good_trial:'+str(step),len(dist)==2*len(good));seen=set()
        for n,(vid,pos,s,trial) in enumerate(good):
            t,b=dist[2*n:2*n+2];ck('original_first_trial_then_base:'+op+':'+str(n),t['route']==trial and b['route']==s['route'] and t['cur']==b['cur']==s['cur'])
            value=np.float64(float.fromhex(t['value_hex']))-np.float64(float.fromhex(b['value_hex']))
            score_reference.append(dict(order=int(sequence[step]),vehicle=vid,position=pos,trial_hex=t['value_hex'],base_hex=b['value_hex'],score_hex=float(value).hex(),score_type=type(value).__module__+'.'+type(value).__qualname__))
            filtered.append(t)
            if vid not in seen:filtered.append(b);seen.add(vid);computed+=1
            else:skipped+=1
    filtered.extend(r for r in actor.distance_log if not r['operation'].startswith('insert_'))
    ck('only_repeated_same_loop_base_distance_calls_removed',expected['distance_log']==filtered)
    ck('all_original_candidate_scores_hex_and_types_exact',expected['score_log']==score_reference)
    ck('proved_skipped_base_calls',expected['base_reuses']==skipped and expected['base_computations']==computed and len(actor.distance_log)-len(expected['distance_log'])==skipped)
    matrix=np.load(HERE/'DISTANCE_MATRIX.npy',allow_pickle=False)
    for i,r in enumerate(expected['distance_log']):
        total=0.;cur=r['cur']
        for o in r['route']:total+=float(matrix[cur,o]);cur=o
        total+=float(matrix[cur,0]);ck('independent_ordered_double_sum:'+str(i),total.hex()==r['value_hex'])
    result=dict(checks_count=len(checks),failed=sum(not c['pass_'] for c in checks),failure_names=[c['name'] for c in checks if not c['pass_']],checks=checks,original_incr_eval=False,original_reference_distance_calls=len(actor.distance_log),candidate_distance_calls=len(expected['distance_log']),proved_removed_base_calls=skipped,original_candidates=len(actor.feasible_log),scored_feasible_candidates=len(score_reference),complete_original_C0_transition_count=len(transitions),actual_source_files=len(start_closure['files']),actual_source_maps=len(start_closure['modules']),actual_loaded_start_end=seal.same(start_closure,end_closure),reference_operation_time=finished-started,full_request_budget_qualified=False,full_scientific_loaded_closure=False,formal_strong_dynamic_qualified=False,stable_online_gain=False)
    save('INDEPENDENT_AUDIT.json',result);save('AUDIT_RUN_EXIT.json',dict(exit_code=int(result['failed']>0)));print(json.dumps({k:v for k,v in result.items() if k!='checks'}));return int(result['failed']>0)
if __name__=='__main__':raise SystemExit(main())
