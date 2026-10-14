"""Independent new-process replay of integrated prepared results.
Shared sampler/physical evaluators, not an independent physical model.
"""
import ast,json,pickle,sys,time
from pathlib import Path
import qualify_pickup_joint_policy_transaction_root_v3 as old
from recourse_snapshot import snapshot_state_hash
from common_score import score_common
from joint_install import state_dict,digest
import hard_admission as hd
import psutil,numpy as np
ROOT=old.ROOT;P=old.P;OUT=P/'private_joint_integration_root_v1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def main():
    psutil.Process().cpu_affinity([7]);_,decode=old.codecs();raw=decode(read(OUT/'RESULT.json'));reg=read(OUT/'REGISTRATION.json')
    original=decode(read(old.INPUT));checks=[];summaries=[]
    def ck(n,v):checks.append({'name':n,'pass':bool(v)})
    ck('runtime_complete',raw['failed']==0 and len(raw['records'])==3 and len(raw['faults'])==8)
    ck('registered_original_environment',np.__version__=='1.26.4')
    for section in ('inputs','tools'):
        for p,h in reg[section].items():ck('current_identity:'+p,old.sha(p)==h)
    with old.HISTORY.open('rb') as f:history=pickle.load(f)
    for r in raw['records']:
        a=r['response'];packet=r['packet'];current=a['scoring_packet'];prefix=str(r['index'])+':'
        ck(prefix+'stored_original',old.pf.digest(packet)==old.pf.digest(original['admissions'][r['index']]['public_request']))
        ck(prefix+'true_private_projection_matches_original',old.pf.digest(a['projected_packet'])==old.pf.digest(packet))
        ck(prefix+'derived_snapshot',snapshot_state_hash(current['snapshot'])==current['snapshot']['state_hash'])
        ck(prefix+'physical_immutable',all(old.pf.digest(packet['snapshot'][k])==old.pf.digest(current['snapshot'][k]) for k in packet['snapshot'] if k not in ('state_hash','replanner_state')))
        config={'name':'own_uncond_dynamic','budget':packet['budget'],'capacity':packet['capacity'],'booking_horizon':packet['booking_horizon'],'contract':packet['contract'],'cooling_share':0.,'history':history['history_times_classes']}
        admission=hd.decide(config,a['projected_packet'],r['order'],old.E/'scripts',time.perf_counter()+30.)
        ck(prefix+'independent_admission',admission['qualified'] and old.pf.digest(admission['state'])==old.pf.digest(a['base_response']['state']) and old.pf.digest(admission['certificate'])==old.pf.digest(a['base_response']['certificate']))
        scores=score_common(current,{'own_incumbent':current['plan'],'candidate':a['proposal']},history,time.perf_counter()+30.,seed=reg['common_seed'])
        ck(prefix+'common_pool_scenario_identity',scores['scenario_sha256']==a['common_scores']['scenario_sha256'] and old.pf.digest(scores['scenarios'])==old.pf.digest(a['common_scores']['scenarios']))
        for name,c in scores['candidate_scores'].items():
            saved=a['common_scores']['candidate_scores'][name]
            ck(prefix+name+':all10_full_branches',c['qualified'] and c['branch_values']==saved['branch_values'] and len(c['branch_values'])==10)
            ck(prefix+name+':mean',sum(c['branch_values'])/10==saved['mean_shadow_utility'])
        own=scores['candidate_scores']['own_incumbent'];cand=scores['candidate_scores']['candidate']
        choose=cand['qualified'] and cand['mean_shadow_utility']>own['mean_shadow_utility']+1e-10
        plan=a['proposal'] if choose else current['plan'];cert,_=old.pf.certificate(current,plan)
        ck(prefix+'selection_and_complete_C0',choose==a['accepted_route'] and old.pf.digest(plan)==old.pf.digest(r['after']['plan']) and cert['qualified'] and old.pf.digest(cert)==old.pf.digest(a['selected_certificate']))
        ck(prefix+'actual_joint_export',r['installed'] and r['reason']=='published' and state_dict(a['compiled_state'])==r['after'] and digest(r['after'])==a['compiled_state_sha256'])
        ck(prefix+'preserved_acceptance',set(r['before']['accepted'])<=set(r['after']['accepted']) and set(r['after']['accepted'])-set(r['before']['accepted'])<={r['order']})
        ck(prefix+'original_deadline_decode_commit_cleanup',r['started']<=a['worker_finished']<=r['decoded']<=r['committed']<=r['finished']<r['started']+10.)
        ck(prefix+'stages',r['transport']['stages']==reg['phases'])
        ck(prefix+'actual_source_end',a['source_end']==r['ready']['sources'])
        for p,h in r['ready']['sources'].items():ck(prefix+'current_actual_source:'+p,old.sha(p)==h)
        summaries.append({'index':r['index'],'accepted':r['order'] in r['after']['accepted'],'route_changed':choose,'wall_s':r['wall_s'],'online_gain':False})
    ck('actual_reject_path',summaries[-1]['accepted'] is False)
    ck('faults_exact',[x['phase'] for x in raw['faults']]==reg['phases'])
    for f in raw['faults']:
        ck(f['phase']+':actual_phase_and_reap',f['phase'] in f['response']['stages'] and f['response']['kind']=='cancelled' and f['response']['worker_reaped'] and not f['publication_attempted'])
    source=ROOT/'tools/pickup_shared_joint_transport_v1.py';tree=ast.parse(source.read_text(encoding='utf-8'));ask=next(x for x in ast.walk(tree) if isinstance(x,ast.FunctionDef) and x.name=='ask')
    calls=[ast.unparse(x.func) for x in ast.walk(ask) if isinstance(x,ast.Call)]
    ck('no_timed_pipe_or_JSON',not any(x in ('self.read_frame','available','os.read','json.loads','json.dumps') for x in calls))
    left=[]
    for p in psutil.process_iter(['cmdline']):
        try:
            c=p.info['cmdline'] or []
            if '--worker' in c and any('qualify_pickup_private_joint_v1.py' in x for x in c):left.append(p.pid)
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    ck('owned_workers_gone',not left)
    ck('honest_scope',not raw['whole_live_10s_qualified'] and not raw['parent_decode_preemptible'] and not reg['live_event_ownership_integrated'] and not raw['stable_online_gain'] and not raw['full_loaded_dependency_closure_qualified'])
    result={'checks':checks,'checks_count':len(checks),'failed':sum(not x['pass'] for x in checks),'summaries':summaries,'input_sha256':old.sha(OUT/'RESULT.json'),'registered_sha256':old.sha(OUT/'REGISTRATION.json'),'whole_live_request_qualified':False,'stable_online_gain':False,'scope':'Independent stored private-derived public input/source recheck and actual admission/K10/C0 replay; shares domain evaluators. No new science or all formal arms.'}
    with (OUT/'INDEPENDENT_PRIVATE_JOINT_REPLAY.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({'checks':len(checks),'failed':result['failed'],'summaries':summaries}));return int(result['failed']>0)
if __name__=='__main__':raise SystemExit(main())
