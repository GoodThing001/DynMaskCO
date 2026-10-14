"""Read-only raw numeric audit. No optimizer/model executes; fresh output only."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paired(rows, control, other, seed, field='final_cost'):
    ids = [r['instance_id'] for r in rows]
    if any(type(i) is not int for i in ids) or len(ids)!=len(set(ids)):
        raise ValueError('Wrong-type or duplicate instance key')
    # Keep actual saved row order to reproduce the recorded paired bootstrap.
    values = np.array([(r['arms'][control][field]-r['arms'][other][field])/
                        r['arms'][control][field] for r in rows],dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError('Nonfinite numeric rows')
    indices = np.random.default_rng(seed).integers(0,len(values),size=(2000,len(values)))
    return {'n':len(rows),'ids':ids,'control':control,'other':other,
        'mean_relative_gain':float(values.mean()),
        'descriptive_95_CI':np.percentile(values[indices].mean(axis=1),[2.5,97.5]).tolist(),
        'better':int(sum(values>0)),'worse':int(sum(values<0)),'tie':int(sum(values==0)),
        'all_saved_rows_retained':True,'not_equivalence_test':True}


def main(args):
    source = Path(args.source).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True,exist_ok=False)
    hashes = {}
    def read(name):
        path = source/name
        hashes[name] = sha(path)
        return json.loads(path.read_text(encoding='utf-8'))
    def rows(name, expected):
        path = source/name
        hashes[name] = sha(path)
        result = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
        ids = [r['instance_id'] for r in result]
        if len(ids)!=len(expected) or sorted(ids)!=expected or any(type(i)is not int for i in ids):
            raise ValueError('Exact registered cohort mismatch: '+name)
        return result
    cohorts = []
    stats = {}
    for name, file, ids, seed, dataset in (
        ('ANYTIME_dev','delivery_anytime_v1/dev_progress.jsonl',list(range(32)),2026101133,2026101002),
        ('ANYTIME_confirmation','delivery_anytime_v1/confirmation_progress.jsonl',list(range(1000,1032)),2026101133,2026101013),
        ('ANY50_primary_confirmation','delivery_any50_confirm_v1/progress.jsonl',list(range(2000,2032)),2026101151,2026101141)):
        rr = rows(file,ids)
        stats[name] = paired(rr,'LNS','ANY50',seed)
        stats[name]['arm_means'] = {a:float(np.mean([r['arms'][a]['final_cost'] for r in rr])) for a in rr[0]['arms']}
        stats[name]['actual_routes_saved'] = all('routes' in a for r in rr for a in r['arms'].values())
        stats[name]['actual_hard_value_saved'] = all('hard' in a or 'report' in a for r in rr for a in r['arms'].values())
        stats[name]['analysis_role'] = 'predeclared_primary_confirmation' if 'primary' in name else 'posthoc_secondary_comparison'
        cohorts.append({'name':name,'dataset_seed':dataset,'ids':ids})
    posthoc = read('delivery_anytime_v1/posthoc_any50_comparison.json')
    confirm = read('delivery_any50_confirm_v1/result.json')
    recomputation_checks=[]
    for name, recorded in [('ANYTIME_dev',posthoc['dev']['ANY50_over_LNS']),
        ('ANYTIME_confirmation',posthoc['confirmation']['ANY50_over_LNS']),
        ('ANY50_primary_confirmation',confirm['primary_ANY50_over_LNS'])]:
        calc=stats[name]
        error=max(abs(calc['mean_relative_gain']-recorded['mean_relative_gain']),
                  max(abs(a-b) for a,b in zip(calc['descriptive_95_CI'],recorded['descriptive_95_CI'])))
        recomputation_checks.append({'comparison':name,'max_abs_error':error,'agrees_1e_12':error<1e-12})
    gls=read('delivery_gls_v1/result.json')
    rr=rows('delivery_gls_v1/progress.jsonl',list(range(32)))
    stats['GLS_over_ANY50']=paired(rr,'ANY50','GLS',2026101211)
    stats['server_ANY50_over_LNS_same_dev_instances']=paired(rr,'LNS','ANY50',2026101211)
    stats['GLS_over_ANY50']['fallback_ids']=[r['instance_id'] for r in rr if r['arms']['GLS'].get('timeout')]
    stats['GLS_over_ANY50']['better_count_despite_negative_mean']=stats['GLS_over_ANY50']['better']
    subset=[r for r in rr if not r['arms']['GLS'].get('timeout')]
    stats['GLS_no_fallback_sensitivity_ONLY']=paired(subset,'ANY50','GLS',2026101211)
    stats['GLS_no_fallback_sensitivity_ONLY']['all_saved_rows_retained']=False
    stats['GLS_no_fallback_sensitivity_ONLY']['formal32_retained_elsewhere']=True
    stats['GLS_no_fallback_sensitivity_ONLY']['diagnostic_excluded_id']=9
    stats['GLS_no_fallback_sensitivity_ONLY']['not_formal_adjudication']='Fallback retained in formal32; this31-row statistic diagnoses influence only'
    for name,key in [('GLS_over_ANY50','GLS_over_ANY50'),('server_ANY50_over_LNS_same_dev_instances','ANY50_over_LNS')]:
        rec=gls['gains'][key]; calc=stats[name]
        error=max(abs(calc['mean_relative_gain']-rec['mean_relative_gain']),max(abs(a-b) for a,b in zip(calc['descriptive_95_CI'],rec['descriptive_95_CI'])))
        recomputation_checks.append({'comparison':name,'max_abs_error':error,'agrees_1e_12':error<1e-12})
    destroy=read('delivery_destroy2_v1/result.json')
    rr=rows('delivery_destroy2_v1/dev_progress.jsonl',list(range(32)))
    stats['distance_cluster_related_over_uniform']=paired(rr,'random','related',2026101241)
    decomp=read('delivery_decomp_v1/result.json')
    dr=decomp['rows']
    if [r['instance_id'] for r in dr]!=list(range(12)):
        raise ValueError('Actual decomp12 subset mismatch')
    decomposition={'actual_n':len(dr),'actual_ids':[r['instance_id'] for r in dr],
        'scope':'new wall-clock runs on first12 previously seen development instances; not original96 endpoint replay',
        'registered_full_ids':decomp['protocol']['instance_ids'],'actual_fraction_of_registered':len(dr)/len(decomp['protocol']['instance_ids']),
        'all_component_sums_agree':all(abs(r['cost']-sum(r[f] for f in ['distance_component','energy_component','quality_component']))<1e-9 for r in dr),
        'component_means':{},'CIs':{},'ratio_of_mean_shares':{}}
    for field in ['cost','distance_component','energy_component','quality_component','duty_time_difference_h']:
        v=np.array([r[field] for r in dr]); ind=np.random.default_rng(2026101251).integers(0,len(v),size=(2000,len(v)))
        decomposition['component_means'][field]=float(v.mean())
        decomposition['CIs'][field]=np.percentile(v[ind].mean(axis=1),[2.5,97.5]).tolist()
    for field in ['distance_component','energy_component','quality_component']:
        decomposition['ratio_of_mean_shares'][field]=decomposition['component_means'][field]/decomposition['component_means']['cost']
    recheck=read('delivery_recheck_v1/result.json')
    decomposition['recheck_actual_n']=recheck['n_instances']
    decomposition['recheck_max_distance_gap']=recheck['max_distance_gap_km']
    inventory={}
    for directory in ['delivery_anytime_v1','delivery_any50_confirm_v1','delivery_gls_v1','delivery_destroy2_v1','delivery_decomp_v1','delivery_recheck_v1']:
        inventory[directory]=[str(f.relative_to(source/directory)).replace('\\','/') for f in (source/directory).rglob('*') if f.is_file()]
    result={'version':'DELIVERY-20-ROUND-AUDIT-v1','read_only_old_artifacts':True,
        'no_numeric_optimization_or_model_run':True,'source_root':str(source),
        'cohorts':cohorts,'disjoint_identity_keys':len({(c['dataset_seed'],i) for c in cohorts for i in c['ids']})==96,
        'statistics':stats,'recomputation_checks':recomputation_checks,'decomposition':decomposition,
        'local_file_inventory':inventory,'input_sha256':hashes,
        'source_files_unchanged_end':all(sha(source/name)==value for name,value in hashes.items()),
        'numeric_bootstrap_recomputation_qualified':all(x['agrees_1e_12'] for x in recomputation_checks),
        'original_route_replay_qualified':False,'stable_maskco_gain_established':False,
        'limits':['No original route files in saved official confirmation rows; runtime self-evaluate cannot substitute independent replay.',
            'Three disjoint numeric cohorts include two posthoc secondary comparisons and one specified-primary cohort.',
            'Same10s ceiling with H8 uses0.3s versus ANY50 nearly10s; comparison estimates schedule/opportunity effect, not equal consumed compute.',
            'related is deterministic seed-nearest-k distance clustering, not Shaw1998 randomized iterative related removal.',
            '12-subset decomposition ratio has uncertain near-zero denominator; cannot establish global92.09% mechanism.',
            'GLS CI crossing zero cannot establish equivalence/noninferiority or rule out either superiority direction.']}
    with (out/'RECOMPUTATION.json').open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2,ensure_ascii=False,allow_nan=False)
    print(json.dumps({'checks':recomputation_checks,'routes_qualified':False,'output':str(out)},indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--source',required=True); ap.add_argument('--out',required=True)
    main(ap.parse_args())
