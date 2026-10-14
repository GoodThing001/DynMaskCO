"""Reconstruct original200-day prior, verify lineage and public common K10 draws."""
import hashlib,json,pickle,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem'
P=ROOT/'research/review_3080_20261010/pickup';OUT=P/'shared_history_qualification_root_v1'
for sub in ('evaluation','coldchain','simulation','training','correction_engineering'):sys.path.insert(0,str(E/'scripts'/sub))
sys.path.insert(0,str(P));sys.path.insert(0,str(P/'actual_masked_fleet_followup_v1'))
import public_fleet as pf
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def main():
    from run_exp_reserve import generate_dataset
    from scenario_saa import (build_history,build_history_times_classes,make_c0_contract_v2,
        CondHistoricalSampler,SaaReplanner)
    from i3_pool_lineage import pool_lineage
    from strict_online_env import StrictOnlineEnv
    from recourse_snapshot import restore_recourse_snapshot
    from audit_io import encode
    card=E/'results/i3_registered_1008_v5/maskco/config.json'
    packetfile=P/'public_certificate_qa_v2/public_packet.pkl'
    OUT.mkdir(exist_ok=False)
    sources=pf.source_hashes(E/'scripts');sources[str(Path(__file__).resolve())]=sha(Path(__file__))
    write(OUT/'REGISTRATION.json',{'scope':'original historical data reconstruction and known packet K10 engineering only',
        'registered_utc':datetime.now(timezone.utc).isoformat(),'history_seed':20260925,'days':200,'n_order_slots':200,
        'excluded_days':[],'expected_card_sha256':sha(card),'public_packet_sha256':sha(packetfile),
        'common_scenario_seed':202610111301,'K':10,'new_evaluation_days':0,'source_start':sources,
        'no_final_seed_execution':True,'formal_online_qualified':False})
    data=generate_dataset(200,200,20260925)
    actual=pool_lineage(data,train_seed=20260925,excluded_days=[])
    expected=json.loads(card.read_text(encoding='utf-8'))['history_pool_lineage']
    history=build_history(data);tc=build_history_times_classes(data)
    np.savez_compressed(OUT/'HISTORICAL_DATA.npz',**data)
    with (OUT/'COMMON_HISTORY.pkl').open('xb') as f:pickle.dump({'data':data,'history':history,'history_times_classes':tc,'lineage':actual},f,protocol=4)
    with packetfile.open('rb') as f:packet=pickle.load(f)
    pf.validate_public(packet)
    c=make_c0_contract_v2()
    rp=SaaReplanner(budget=packet['budget'],capacity=packet['capacity'],booking_horizon=packet['booking_horizon'],
        contract=c,cooling_share=0.,hist=tc,k=10,sampler=CondHistoricalSampler(history,n_neighbors=10),
        reject_penalty={0:5.,1:10.,2:15.},K=10,time_limit=10.,arm_seed=202610111301,
        shadow_mode='greedy',energy_pricing='marginal',future_policy='density',standby_orders=0,
        incr_eval=False,anytime_vote=False,overage_price=None)
    rp.restore_state(packet['snapshot']['replanner_state'])
    env=StrictOnlineEnv(packet['dataset'],packet['capacity'],packet['tw_speed'],packet['snapshot']['num_vehicles'],rp,c,packet['booking_horizon'])
    vehicles,traces,served=restore_recourse_snapshot(packet['snapshot'])
    clock=float(packet['snapshot']['clock']);rp._sync_id_map(env,0,clock)
    snap=rp._build_snapshot(env,0,clock,vehicles,served)
    shared=rp.sampler.sample(snap,np.random.default_rng(202610111301),10)
    draws={name:rp.sampler.sample(snap,np.random.default_rng(202610111301),10) for name in ('strong_rolling','masked_pretrained','reconstruction_control','random','cond_hist')}
    drawhashes={name:pf.digest(v) for name,v in draws.items()}
    write(OUT/'SHARED_K10_SCENARIOS.json',encode({'snapshot':snap,'scenarios':shared,'hashes':drawhashes}))
    checks=[]
    def ck(n,v):checks.append({'name':n,'pass':bool(v)})
    ck('actual_history_exact_original_I3_lineage',actual==expected)
    ck('all200_history_days',len(history)==len(tc)==200 and actual['included_days']==list(range(200)))
    ck('registered_source_end',all(sha(Path(p))==v for p,v in sources.items()))
    ck('public_current_packet_only',all(x.reveal<=clock+1e-6 for x in snap.orders))
    ck('exactK10',len(shared)==10)
    ck('all_future_scenarios_after_clock',all(o.reveal>clock+1e-6 for day in shared for o in day))
    ck('all_five_arm_common_random_scenarios',len(set(drawhashes.values()))==1)
    for d,(orders,(times,classes)) in enumerate(zip(history,tc)):
        ck('day'+str(d)+':sameactualorders',len(orders)==len(times)==len(classes))
        ck('day'+str(d)+':times',[o.reveal for o in orders]==times.tolist())
        ck('day'+str(d)+':classes',[o.temp_class for o in orders]==classes.tolist())
    result={'checks':checks,'checks_count':len(checks),'failed':sum(not x['pass'] for x in checks),'actual_lineage':actual,'expected_lineage':expected,
        'actual_historical_archive_sha256':sha(OUT/'HISTORICAL_DATA.npz'),'actual_history_pickle_sha256':sha(OUT/'COMMON_HISTORY.pkl'),
        'commonK10_scenarios_sha256':sha(OUT/'SHARED_K10_SCENARIOS.json'),
        'original_history_identity_qualified':actual==expected,'common_future_draws_qualified':len(set(drawhashes.values()))==1,
        'all_arm_full_SAA_scoring_implemented':False,'all_in_atomic_budget_qualified':False,
        'new_evaluation_days':0,'stable_core_online_gain':False}
    write(OUT/'RESULT.json',result)
    print(json.dumps({k:result[k] for k in ('checks_count','failed','original_history_identity_qualified','common_future_draws_qualified')}))
    return int(result['failed']>0)
if __name__=='__main__':sys.exit(main())
