"""Read-only logical/identity audit; does not rerun or rewrite old experiments."""
import argparse
import hashlib
import json
from pathlib import Path


def audit(root,out):
    root=Path(root);base=root/'results/research_20261010';source={}
    def read(path):
        raw=path.read_bytes();source[str(path)]=hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    names=['tail_census','budget_headroom','residual_census','rejection_attribution','clairvoyant_gap','budget_sweep']
    p={n:read(base/('pickup_'+n+'_v1')/'result.json') for n in names}
    tail_rows=read(base/'pickup_tail_census_v1/per_instance.json')
    gate=read(root/'results/a1_formal_10s/gate.json')
    i3=read(root/'results/i3_online_recovery_1009_v1/online_20260926/gate.json')
    for n in names:
        path=root/'scripts/research_20261010'/('pickup_'+n+'.py')
        source[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    for n in ['PROGRAM_CLOSEOUT.md','ROUND_20261010_GUIDE_DOSE.md']:
        path=base/n;source[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    arm=gate['results'][gate['main_penalty']]['arms']['uncond_dynamic']
    usage=arm['mean_budget_usage'];observed=p['budget_headroom']['observed'];B=gate['budget']['B']
    result=dict(schema='pickup-20-round-read-only-audit-v1',
        conclusion='impossibility_and_no_policy_headroom_not_established',old_sources_and_results_unchanged=True,
        source_sha256=source,
        tail_census=dict(n=p['tail_census']['n_instances'],distance=p['tail_census']['distance_headroom'],
            utility_withheld=p['tail_census']['utility_verdict_withheld_because'],
            cost_share_instance0=tail_rows[0]['evaluator_distance_km']/tail_rows[0]['evaluator_coldchain_cost'],
            distance_only_ratio_product=p['tail_census']['distance_headroom']['mean_relative_gain']*
                tail_rows[0]['evaluator_distance_km']/tail_rows[0]['evaluator_coldchain_cost'],
            product_is_utility_upper_bound=False),
        budget_mixed_arm_bug=dict(recorded_usage=observed['budget_utilisation'],correct_uncond_usage=usage,
            recorded_rejected=observed['rejected'],correct_uncond_rejected=arm['mean_n_rejected'],
            corrected_energy_mean=usage*B,corrected_fixed_average_extrapolation=1/usage-1,
            extrapolation_is_proven_upper_bound=False),
        runtime_points=dict(tail=p['tail_census']['protocol'],historical_a1=gate['config'],
            historical_a1_B=B,current_i3=i3['config'],current_i3_B=i3['budget']['B'],same_runtime_point=False),
        rejection=dict(totals=p['rejection_attribution']['totals'],
            short_circuit_threshold_hides_counterfactual_feasibility=True,
            no_fleet_failure_implies_no_strategy_headroom=False),
        clairvoyant=dict(compared_quantity=p['clairvoyant_gap']['protocol']['compared_quantity'],
            paired=p['clairvoyant_gap']['served_count_comparison'],greedy_is_optimal_oracle=False,
            utility_compared=False,excludes_better_causal_policy=False),
        budget_sweep=dict(n=p['budget_sweep']['n_instances'],rows=p['budget_sweep']['rows'],
            establishes_policy_budget_response=True,excludes_fixed_budget_policy_improvement=False),
        scientific_positive_established=False,MaskCO_positive_established=False)
    out=Path(out);out.mkdir(parents=True,exist_ok=False)
    (out/'EVIDENCE_AUDIT.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();r=audit(a.root,a.out)
    print(json.dumps({'conclusion':r['conclusion'],'source_count':len(r['source_sha256']),
                     'mixed_arm_bug':r['budget_mixed_arm_bug']}))
