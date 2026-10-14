"""Read-only independent recorded-input, complete-score and C0 replay.

Shared domain evaluators are explicitly retained; this is not an independently
implemented physics model or a scientific gain test.
"""
import copy,hashlib,json,sys,time
from pathlib import Path
from dataclasses import is_dataclass
import numpy as np,psutil
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';P=ROOT/'research/review_3080_20261010/pickup'
OUT=P/'joint_policy_transaction_root_v3'
for folder in (P,P/'actual_masked_fleet_engineering_v1',P/'actual_masked_fleet_followup_v1',P/'common_candidate_saa_root_v2',P/'shadow_energy_correction_root_v1',P/'atomic_plan_installation_root_v2',OUT):sys.path.insert(0,str(folder))
import public_fleet as pf
pf.bootstrap(E/'scripts')
import coldchain_contract,coldchain_state,strict_online_env,scenario_saa,joint_install
from audit_io import decode
from joint_install import state_dict,digest
from recourse_snapshot import snapshot_state_hash
from common_score import score_common
import hard_admission as hd
CLASSES={c.__module__+'.'+c.__name__:c for m in (coldchain_contract,coldchain_state,strict_online_env,scenario_saa,joint_install) for c in vars(m).values() if isinstance(c,type) and is_dataclass(c)}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    import pickle
    psutil.Process().cpu_affinity([7]);checks=[];summaries=[]
    def ck(n,v):checks.append({'name':n,'pass':bool(v)})
    reg=json.loads((OUT/'REGISTRATION.json').read_text(encoding='utf-8'))
    raw=decode(json.loads((OUT/'RESULT.json').read_text(encoding='utf-8')),CLASSES)
    inp=P/'complete_history_energy_qa_root_v1/complete_ledger/FULLRAW.json'
    original=decode(json.loads(inp.read_text(encoding='utf-8')),CLASSES)
    hp=P/'shared_history_qualification_root_v1/COMMON_HISTORY.pkl'
    with hp.open('rb') as f:history=pickle.load(f)
    ck('actual_numpy',np.__version__==raw['ready']['numpy']==reg['numpy']=='1.26.4')
    ck('unchanged_observed_data',sha(inp)==reg['fixture_sha256'])
    ck('unchanged_original_history',sha(hp)==reg['history_sha256']=='44064ff788b7e322294a23e6ac6de39da9f399dec326b70af7a0f7439895d3c6')
    ck('bound_registration',sha(OUT/'REGISTRATION.json')==raw['registration_sha256'])
    ck('actual_registered_new_source',sha(ROOT/'tools/qualify_pickup_joint_policy_transaction_root_v3.py')==reg['source_sha256'] and sha(OUT/'joint_install.py')==reg['installer_sha256'])
    for p,h in raw['ready']['source_start'].items():ck('actual_source_runtime:'+p,sha(p)==h)
    expected=[(j,m) for j in reg['fixture_indices'] for m in reg['modes']]
    ck('all_six_unique_expected_requests',[(r['fixture_index'],r['mode']) for r in raw['records']]==expected and len(expected)==6)
    for r in raw['records']:
        label=str((r['fixture_index'],r['mode']));a=r['response'];row=original['admissions'][r['fixture_index']];old=r['packet'];current=a['scoring_packet']
        ck(label+':recorded_original_request',pf.digest(old)==pf.digest(row['public_request']) and r['order']==row['order'])
        ck(label+':original_snapshot_identity',old['snapshot']['state_hash']==snapshot_state_hash(old['snapshot']))
        ck(label+':derived_snapshot_identity',current['snapshot']['state_hash']==snapshot_state_hash(current['snapshot'])==a['derived_snapshot_hash'])
        ck(label+':projected_original_admission_state',a['projected_packet']['snapshot']['replanner_state']==r['before'])
        ck(label+':no_physical_state_changed_by_admission',all(pf.digest(old['snapshot'][k])==pf.digest(current['snapshot'][k]) for k in old['snapshot'] if k not in ('state_hash','replanner_state')))
        config={'name':'own_uncond_dynamic','budget':old['budget'],'capacity':old['capacity'],'booking_horizon':old['booking_horizon'],'contract':old['contract'],'cooling_share':0.,'history':history['history_times_classes']}
        replay=hd.decide(config,a['projected_packet'],r['order'],E/'scripts',time.perf_counter()+30.)
        ck(label+':independent_actual_admission_replay',replay['qualified'] and pf.digest(replay['state'])==pf.digest(a['base_response']['state']) and pf.digest(replay['certificate'])==pf.digest(a['base_response']['certificate']))
        scores=score_common(current,{'own_incumbent':current['plan'],'candidate':a['proposal']},history,time.perf_counter()+30.,seed=reg['common_scenario_seed'])
        ck(label+':one_original_pool_common_future_identity',scores['scenario_sha256']==a['common_scores']['scenario_sha256'] and pf.digest(scores['scenarios'])==pf.digest(a['common_scores']['scenarios']))
        for name,v in scores['candidate_scores'].items():
            stored=a['common_scores']['candidate_scores'][name]
            ck(label+':complete_branch_exact:'+name,v['qualified']==stored['qualified'] and v['branch_values']==stored['branch_values'])
            if v['qualified']:
                ck(label+':independent_arithmetic_mean:'+name,len(v['branch_values'])==10 and abs(sum(v['branch_values'])/10.-stored['mean_shadow_utility'])<1e-10)
        own,cand=scores['candidate_scores']['own_incumbent'],scores['candidate_scores']['candidate']
        choose=cand['qualified'] and cand['mean_shadow_utility']>own['mean_shadow_utility']+1e-10
        expected_plan=a['proposal'] if choose else current['plan']
        ck(label+':registered_selection',choose==a['accepted_route'] and pf.digest(expected_plan)==pf.digest(r['after']['plan']))
        cert,_=pf.certificate(current,expected_plan)
        ck(label+':independent_complete_C0_replay',cert['qualified'] and pf.digest(cert)==pf.digest(a['selected_certificate']))
        ck(label+':compiled_bundle_matches_actual_export',r['after']==state_dict(a['compiled_state']) and digest(r['after'])==a['compiled_state_sha256'])
        ck(label+':preserved_prior_commitments',set(r['before']['accepted'])<=set(r['after']['accepted']) and set(r['after']['accepted'])-set(r['before']['accepted'])<= {r['order']} and set(r['before']['rejected'])-{r['order']}==set(r['after']['rejected'])-{r['order']})
        ck(label+':actual_all_stages_before_single_deadline',r['started']<=a['worker_finished']<=r['received']<=r['finished']<a['deadline'] and abs(a['deadline']-r['started']-10)<1e-8 and set(a['phase_started'])=={'projection','admission','proposal','scoring','certification','compile'})
        ck(label+':source_end_identity',a['source_qualified'] and a['source_end']==raw['ready']['source_start'])
        ck(label+':actual_publication',r['installed'] and r['reason']=='published')
        summaries.append({'fixture_index':r['fixture_index'],'mode':r['mode'],'order':r['order'],'accepted':r['order'] in r['after']['accepted'],'route_changed':choose,'wall_s':r['wall_s'],'realized_online_gain':False})
    expected_faults=['changed_admissions','changed_plan','changed_fleet','wrong_request','changed_prepared_state','changed_prior_acceptance','expired','replayed_token']+['delay:'+x for x in ('projection','admission','proposal','scoring','certification','compile')]
    ck('all14_faults_retained',[r['kind'] for r in raw['faults']]==expected_faults)
    for r in raw['faults']:
        ck('no_invalid_publication:'+r['kind'],not r['installed'] and (r['old_policy_unchanged'] and r['worker_reaped'] if r['kind'].startswith('delay:') else r['state_unchanged']))
    remaining=[]
    for p in psutil.process_iter(['cmdline']):
        try:
            cmd=p.info['cmdline'] or []
            if '--worker' in cmd and any('qualify_pickup_joint_policy_transaction_root_v' in a for a in cmd):remaining.append(p.pid)
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    ck('owned_worker_reaped',not remaining)
    ck('honest_scope',not raw['whole_live_10s_qualified'] and not raw['stable_online_gain'] and not reg['all_formal_arms'] and not reg['live_raw_public_extraction_included'] and not reg['parent_serialization_is_interruptible'])
    failed=sum(not c['pass'] for c in checks)
    out={'checks':checks,'checks_count':len(checks),'failed':failed,'summaries':summaries,'raw_sha256':sha(OUT/'RESULT.json'),'joint_observed_policy_transaction_qualified':failed==0,'full_live_request_qualified':False,'formal_five_arms_qualified':False,'stable_online_gain':False,'scope':'Independent stored-input and actual-source reread, rerun admission and complete common-scenario/C0 evaluation; shared physical and sampler implementations. No learned proposals or new science. Requested delay stages may not be reached: only outer cancellation verified.'}
    with (OUT/'INDEPENDENT_JOINT_TRANSACTION_REPLAY.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2,allow_nan=False)
    print(json.dumps({'checks':len(checks),'failed':failed,'summaries':summaries}));return int(failed>0)
if __name__=='__main__':raise SystemExit(main())
