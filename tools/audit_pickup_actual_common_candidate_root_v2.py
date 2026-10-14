"""Recompute actual prototype candidate certificates, paired means and identity."""
import hashlib,json,math,sys
from dataclasses import is_dataclass
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem'
P=ROOT/'research/review_3080_20261010/pickup';OUT=P/'actual_common_candidate_root_v2'
sys.path.insert(0,str(P));import public_fleet as pf;pf.bootstrap(E/'scripts')
import coldchain_contract,coldchain_state,strict_online_env,scenario_saa
from audit_io import decode
CLASSES={c.__module__+'.'+c.__name__:c for m in (coldchain_contract,coldchain_state,strict_online_env,scenario_saa)
 for c in vars(m).values() if isinstance(c,type) and is_dataclass(c)}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    reg=json.loads((OUT/'REGISTRATION.json').read_text());r=decode(json.loads((OUT/'RESULT.json').read_text()),CLASSES);packet=r['public_packet']
    checks=[]
    def ck(n,v):checks.append({'name':n,'pass':bool(v)})
    ck('registration_bound_to_actual_package',r['registration_sha256']==sha(OUT/'REGISTRATION.json'))
    mapping={'launcher_sha256':ROOT/'tools/qualify_pickup_actual_common_candidate_root_v2.py','common_score_sha256':P/'common_candidate_saa_root_v2/common_score.py',
      'masked_fleet_sha256':P/'actual_masked_fleet_engineering_v1/masked_fleet.py','history_sha256':P/'shared_history_qualification_root_v1/COMMON_HISTORY.pkl',
      'checkpoint_sha256':ROOT/'MASKCO_code/ckpts/cvrp100.ckpt','energy_complete_saa_sha256':P/'shadow_energy_correction_root_v1/energy_complete_saa.py',
      'rolling_source_sha256':E/'scripts/evaluation/rolling_optimizer.py','public_fleet_sha256':P/'public_fleet.py','fixture_sha256':P/'common_candidate_saa_root_v2/RESULT.json'}
    for key,p in mapping.items():ck('actual_registered_bytes:'+key,reg[key]==sha(p))
    ck('actual_domain_has_at_least_two_mutable',len(pf.family(packet)[1])>=2)
    ck('exact_five_prototypes',len(r['records'])==5 and [x['mode'] for x in r['records']]==reg['modes'])
    summaries={};common_hash=None;common_inc=None;providers={}
    for row in r['records']:
        mode=row['mode'];a=row.get('actual',{});ready=row['ready']
        ck(mode+':complete_normal_request',a.get('qualified') is True and row['parent_on_time'])
        ck(mode+':all_parent_transmission_inside10',0<=row['wall_s']<=10 and row['started']<=row['received']<=row['deadline'])
        ck(mode+':all_actual_cpu7',ready['actual_cpu_affinity']==a['actual_cpu_affinity']==[7])
        ck(mode+':worker_clocks_inside_parent_cap',row['started']<=a['worker_started']<=a['worker_completed']<=row['received'])
        ck(mode+':packet_exact_public_identity',a['packet_sha256']==pf.digest(packet))
        ck(mode+':sealed_actual_source_start_end',a['source_end']==ready['source_start'] and all(sha(p)==h for p,h in a['source_end'].items()))
        ck(mode+':actual_checkpoint_bytes',a['checkpoint_end_sha256']==ready['checkpoint_sha256']==reg['checkpoint_sha256'])
        ck(mode+':real_full_original_history',ready['history_sha256']==reg['history_sha256'])
        ck(mode+':delayed10ms_request_no_reply',row.get('fault',{}).get('reply_kind')=='timeout')
        ck(mode+':fault_original_incumbent_only_and_reaped',row['fault']['own_incumbent_fallback'] and row['fault']['late_reply_applied'] is False and row['worker_reaped'])
        scores=a['common_scores'];ck(mode+':actual10draws',len(scores['scenarios'])==10 and pf.digest(scores['scenarios'])==scores['scenario_sha256'])
        for k,scene in enumerate(scores['scenarios']):ck(mode+':causal_historical_future'+str(k),all(o.reveal>packet['snapshot']['clock']+1e-6 for o in scene))
        if common_hash is None:common_hash=scores['scenario_sha256']
        ck(mode+':same10draws_all_proposals',scores['scenario_sha256']==common_hash)
        inc=scores['candidate_scores']['own_incumbent'];cand=scores['candidate_scores']['actual_candidate']
        for name,s,plan in [('incumbent',inc,packet['plan']),('candidate',cand,a['raw_candidate'])]:
            cert,_=pf.certificate(packet,plan)
            ck(mode+':'+name+':all10_complete',s['qualified'] and s['complete_scenarios']==len(s['branch_values'])==10)
            ck(mode+':'+name+':actual_complete_mean',math.isclose(np.mean(s['branch_values']),s['mean_shadow_utility'],abs_tol=1e-10))
            ck(mode+':'+name+':independent_known_certificate',cert['qualified'] and pf.digest(cert)==pf.digest(s['certificate']))
        if common_inc is None:common_inc=inc['branch_values']
        ck(mode+':incumbent_same10values_across_workers',inc['branch_values']==common_inc)
        improves=cand['mean_shadow_utility']>inc['mean_shadow_utility']+1e-10
        ck(mode+':common_SAA_selection_recomputed',a['accepted_on_common_SAA'] is improves and a['selected_plan']==(a['raw_candidate'] if improves else packet['plan']))
        selected,_=pf.certificate(packet,a['selected_plan']);ck(mode+':selected_full_C0_hard',selected['qualified'] and pf.digest(selected)==pf.digest(a['selected_certificate']))
        if mode in ('pretrained','random','discard'):
            providers[mode]=ready['provider_identity'];proposal=a['proposal']
            ck(mode+':full_actual_model_called',proposal['decoder_calls']==proposal['encoder_calls']==1 and providers[mode]['actual_param_leaves']==234)
            ck(mode+':retained_edges_and_committed_preserved',proposal['retained_edges_preserved'] is True)
            boundary=row['ineligible_domain_boundary'];b=boundary['response']
            ck(mode+':ineligible_boundary_before10',boundary['reply_kind']=='response' and boundary['started']<=boundary['received']<=boundary['deadline'])
            ck(mode+':ineligible_no_false_model_call',b['qualified'] and b['proposal']['kind']=='ineligible_incumbent_fallback' and b['proposal']['decoder_calls']==0)
            ck(mode+':ineligible_own_plan_preserved',b['accepted_on_common_SAA'] is False and b['selected_plan']==b['raw_candidate'])
        summaries[mode]={'candidate_changed':a['raw_candidate']!=packet['plan'],'accepted_on_common_SAA':improves,
            'paired_shadow_delta':cand['mean_shadow_utility']-inc['mean_shadow_utility'],'actual_request_s':row['wall_s'],
            'prototype_route_hash':pf.digest(a['raw_candidate']),'selected_route_hash':pf.digest(a['selected_plan'])}
    ck('actual_pretrained_discard_same_param_identity',providers['pretrained']['actual_param_hash']==providers['discard']['actual_param_hash'])
    ck('actual_random_different_params_same_architecture',providers['random']['random_same_architecture'] is True and providers['random']['random_seed']==42 and providers['random']['actual_param_hash']!=providers['pretrained']['actual_param_hash'])
    failed=sum(not c['pass'] for c in checks)
    report={'checks':checks,'checks_count':len(checks),'failed':failed,'actual_prototype_candidate_scoring_qualified':failed==0,
      'recorded_raw_sha256':sha(OUT/'RESULT.json'),'summaries':summaries,'independent_draws_verified':10,
      'normal_request_budget_s':10,'fault_budget_s':.01,'fault_is_engineering_only':True,
      'all_five_formal_deployment_qualified':False,'same_init_trained_reconstruction_control_present':False,
      'atomic_public_projection_live_installation_qualified':False,'realized_online_utility_gain_measured':False,'research_new_days':0,'stable_online_gain':False,
      'limits':['One observed public state; not a full admission/day experiment','Original3D pretrained/random/discard are prototype roles; no task-trained same-init CE attribution arm',
        'Common shadow scoring does not establish realized online gain','Worker cap includes actual proposal/scenarios/certification/identity/transmission; parent projection and live installation remain unqualified',
        'Physical certificate/thermal equations shared with production, not alternate independently calibrated physics']}
    with (OUT/'INDEPENDENT_ROOT_ACTUAL_CANDIDATE_QA.json').open('x') as f:json.dump(report,f,indent=2,allow_nan=False)
    print(json.dumps({'checks':len(checks),'failed':failed,'summaries':summaries,'formal_deployment_qualified':False}));return int(failed>0)
if __name__=='__main__':raise SystemExit(main())
