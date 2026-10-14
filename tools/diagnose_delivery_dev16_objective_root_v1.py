"""Post-hoc, all-row arithmetic diagnosis. No new experiment or selected model."""
import hashlib,json,math
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'research/review_3080_20261010/delivery'
IN=D/'remote_fresh_development_root_v1/downloaded';OUT=D/'dev16_mechanism_diagnosis_root_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    OUT.mkdir(exist_ok=False)
    a=json.loads((IN/'INDEPENDENT_DEVELOPMENT_ADJUDICATION.json').read_text());assert a['failed']==0 and a['independent_acceptance']
    raw=json.loads((IN/'RESULT.json').read_text());rows={(r['instance_id'],r['arm']):r for r in raw['records']};arms=list(a['mean_costs'])
    checks=[]
    def ck(n,v):checks.append({'name':n,'pass':bool(v)})
    ck('all96_unique_rows',len(rows)==96)
    summary={};trajectory={}
    for arm in arms:
        rr=[rows[i,arm] for i in range(16)]
        fields={k:float(np.mean([r['final_report'][k] for r in rr])) for k in ('cost','distance_km','energy_kwh','quality_loss_value')}
        ck(arm+':same_accepted_cost_mean',abs(fields['cost']-a['mean_costs'][arm])<1e-10)
        for i,r in enumerate(rr):
            f=r['final_report'];ck(arm+':cost_decomposition'+str(i),abs(f['cost']-f['distance_km']-.1*f['energy_kwh']-f['quality_loss_value'])<1e-10)
            ck(arm+':no_filtered_timeout'+str(i),r['timeout'] is False and f['hard'] is True)
        fields['mean_actual_parent_seconds']=float(np.mean([r['wall_s'] for r in rr]));summary[arm]=fields
        if arm=='gls':continue
        tt=[]
        for step in range(8):
            ss=[r['actual_result']['attempts'][step] for r in rr]
            tt.append({'step':step,'accepted_count':sum(s['accepted'] for s in ss),
              'mean_candidate_cost':float(np.mean([s['candidate_report']['cost'] for s in ss])),
              'mean_selected_after_cost':float(np.mean([s['cost_after'] for s in ss]))})
        trajectory[arm]=tt
    pair=summary['paired_objective'];gaps={}
    for control in ('reconstruction_control','nonlearning','gls','pretrained','random'):
        c=summary[control];gap=pair['cost']-c['cost']
        terms={'distance':pair['distance_km']-c['distance_km'],'weighted_energy':.1*(pair['energy_kwh']-c['energy_kwh']),
               'quality':pair['quality_loss_value']-c['quality_loss_value']}
        ck(control+':same_cohort_gap_decomposition',abs(sum(terms.values())-gap)<1e-10)
        gains=[];first=[]
        for i in range(16):
            p=rows[i,'paired_objective'];r=rows[i,control]
            gains.append(r['final_report']['cost']-p['final_report']['cost'])
            if control!='gls':
                ps=p['actual_result']['attempts'][0];rs=r['actual_result']['attempts'][0]
                ck(control+':first_same_incumbent_and_mask'+str(i),ps['incumbent_before']==rs['incumbent_before'] and ps['mask_sha256']==rs['mask_sha256'] and ps['adjacency_sha256']==rs['adjacency_sha256'])
                first.append(ps['candidate_report']['cost']-rs['candidate_report']['cost'])
        gaps[control]={'paired_minus_control_mean_cost':gap,'cost_gap_terms':terms,'distance_fraction_of_mean_gap':terms['distance']/gap if abs(gap)>1e-12 else None,
          'paired_final_wins':sum(x>1e-10 for x in gains),'paired_final_ties':sum(abs(x)<=1e-10 for x in gains),'paired_final_losses':sum(x< -1e-10 for x in gains),
          'first_common_context_paired_minus_control_mean_cost':float(np.mean(first)) if first else None,
          'first_common_context_paired_better_count':sum(x< -1e-10 for x in first) if first else None,
          'later_contexts_are_policy_dependent_not_same_mask_state':True}
    loss_summary={}
    for arm in ('paired_objective','reconstruction_control'):
        path=D/'remote_finite_research_root_v2/training_downloaded'/(arm+'_STEPS.jsonl')
        steps=[json.loads(s) for s in path.read_text().splitlines()]
        ck(arm+':all128_fixed_steps',len(steps)==128 and [s['step'] for s in steps]==list(range(1,129)))
        loss_summary[arm]={'source_sha256':sha(path),'mean_CE_loss':float(np.mean([s['CE_loss'] for s in steps])),
          'mean_preference_loss':float(np.mean([s['preference_loss'] for s in steps])),
          'not_a_holdout_or_deployment_metric':True}
    failed=sum(not c['pass'] for c in checks)
    result={'created_utc':datetime.now(timezone.utc).isoformat(),'scope':'post-hoc diagnosis of all16 already adjudicated snapshots; no new numeric experiment',
      'source_result_sha256':sha(IN/'RESULT.json'),'independent_adjudication_sha256':sha(IN/'INDEPENDENT_DEVELOPMENT_ADJUDICATION.json'),
      'checks':checks,'checks_count':len(checks),'failed':failed,'mean_components':summary,'matched_cohort_gaps':gaps,'all8_step_trajectories':trajectory,
      'training_loss_descriptive_only':loss_summary,'observations':['All96 finals and all640 recorded reconstruction candidates independently qualified; no timeout explanation in this panel',
        'Preference objective loses to the same-init CE control in the actual deployed decoder; parameter update alone does not prove utility alignment'],
      'hypotheses_not_proven_causes':['Full-adjacency surrogate preference is not direct supervision of the constrained insertion action cascade',
        'Training labels use four masks of initial routes, deployment follows eight changing incumbent states; possible state-distribution mismatch'],
      'mathematical_caution':'Row-normalization cancels when candidate and incumbent have equal per-row edge counts; do not claim log-softmax versus raw logits alone proves a bug',
      'confirmation_decision':'Not started; resource decision after observed development disadvantage, not a preregistered numeric stopping-rule claim',
      'no_selected_checkpoint':True,'no_hyperparameter_change':True,'global_impossibility_proven':False,'stable_MaskCO_gain':False,
      'literature':[{'title':'Learning to Perform Local Rewriting for Combinatorial Optimization','year':2019,'url':'https://proceedings.neurips.cc/paper/2019/file/131f383b434fdf48079bff1e44e2d9a5-Paper.pdf','support':'Section4.2 rewards the actual task cost change of a rewriting transition; supports testing action-aligned objectives, not a positive result here'},
        {'title':'Learning to Delegate for Large-scale Vehicle Routing','year':2021,'url':'https://papers.neurips.cc/paper_files/paper/2021/file/dc9fa5f217a1e57b8a6adeb065560b38-Paper.pdf','support':'Learning subproblem selection with a fixed solver is a distinct method role requiring explicit attribution; it is not automatically evidence for MaskCO reconstruction'}]}
    with (OUT/'DIAGNOSIS.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({'checks':len(checks),'failed':failed,'gaps':gaps,'stable_MaskCO_gain':False}));return int(failed>0)
if __name__=='__main__':raise SystemExit(main())
