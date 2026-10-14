"""Independent arithmetic on saved rows; no solver rerun or historical writes."""
import hashlib
import json
from pathlib import Path
import numpy as np

E = Path(__file__).resolve().parents[1] / 'C-VRP_Cold-chainVehicleRoutingProblem'
OLD = E / 'results/research_20261010'
OUT = E / 'results/research_20261010_continuation_v2/parent_pairing_audit'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def pairing(folder, key, a, b, seed):
    root = OLD / folder
    source = root / 'progress.jsonl'
    rows = [json.loads(s) for s in source.read_text(encoding='utf-8').splitlines() if s.strip()]
    ids = [r['instance_id'] for r in rows]
    assert len(rows) == 32 and len(set(ids)) == 32
    x = np.array([(r['arms'][a]['final_cost']-r['arms'][b]['final_cost']) / r['arms'][a]['final_cost'] for r in rows])
    assert np.isfinite(x).all()
    boot = np.random.default_rng(seed).integers(0, len(x), (2000, len(x)))
    means = x[boot].mean(axis=1)
    ci = np.quantile(means, [.025, .975])
    result = json.loads((root / 'result.json').read_text(encoding='utf-8'))
    reported = result
    for k in key:
        reported = reported[k]
    assert abs(x.mean() - reported['mean_relative_gain']) < 1e-12
    assert np.allclose(ci, reported['descriptive_95_CI'], atol=1e-12, rtol=0)
    alpha = .05 / (1 * (1 + 1)) / 2 / (2 * 3)
    widened = np.quantile(means, [alpha / 2, 1-alpha / 2])
    return {
        'source_sha256': sha(source), 'result_sha256': sha(root / 'result.json'),
        'n': len(rows), 'unique_instance_ids': ids,
        'mean_relative_gain': float(x.mean()), 'descriptive_95_CI': ci.tolist(),
        'reported_arithmetic_matches': True,
        'reference_family1_alpha_two_contrasts_three_cohorts': alpha,
        'sensitivity_interval_not_formal_confirmation': widened.tolist(),
        'bootstrap_tail_expected_count': 2000 * alpha / 2,
        'minimum_tail_precision_25_met': 2000 * alpha / 2 >= 25,
        'raw_routes_in_saved_rows': all('routes' in r['arms'][a] and 'routes' in r['arms'][b] for r in rows),
        'online': False, 'learned': False,
        'stable_maskco_online_qualification': False,
        'negative_cases_retained': int((x < 0).sum()),
        'status_distribution': {arm: {str(s): sum(r['arms'][arm].get('status') == s for r in rows)
            for s in sorted({r['arms'][arm].get('status') for r in rows}, key=str)} for arm in (a,b)},
        'largest_negative_rows': [
            {'instance_id': rows[int(i)]['instance_id'], 'relative_gain': float(x[i]),
             'control_cost': rows[int(i)]['arms'][a]['final_cost'],
             'other_cost': rows[int(i)]['arms'][b]['final_cost'],
             'other_status': rows[int(i)]['arms'][b].get('status')}
             for i in np.argsort(x)[:3]]
    }

def main():
    OUT.mkdir(exist_ok=False)
    r = {
        'version': 'parent-independent-round20-arithmetic-v1',
        'ANY50_confirmation_vs_LNS': pairing('delivery_any50_confirm_v1', ['primary_ANY50_over_LNS'], 'LNS', 'ANY50', 2026101151),
        'ANY50_server_dev_vs_LNS': pairing('delivery_gls_v1', ['gains', 'ANY50_over_LNS'], 'LNS', 'ANY50', 2026101211),
        'GLS_vs_ANY50': pairing('delivery_gls_v1', ['gains', 'GLS_over_ANY50'], 'ANY50', 'GLS', 2026101211),
        'interpretation': 'Arithmetic consistency passes; stored rows omit routes, historical source/trajectory eligibility and parent stable-online criteria are not established.',
        'stable_gain_established': False, 'pickup_impossibility_established': False,
        'learning_impossibility_established': False,
    }
    with (OUT / 'AUDIT.json').open('x', encoding='utf-8') as f:
        json.dump(r, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k:{'mean':v['mean_relative_gain'],'CI':v['descriptive_95_CI'],
                         'routes':v['raw_routes_in_saved_rows'],'tail_count':v['bootstrap_tail_expected_count']}
                      for k,v in r.items() if isinstance(v,dict)},ensure_ascii=True))

if __name__ == '__main__':
    main()
