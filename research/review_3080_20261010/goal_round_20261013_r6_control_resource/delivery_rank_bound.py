"""Read-only, binary-rational ranking bounds for the failed compiled-score gate."""
import hashlib
import json
from fractions import Fraction
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / 'delivery/compiled_head_identity_root_v1/downloaded'

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rational(value):
    q = Fraction.from_float(float(value))
    return {'numerator': q.numerator, 'denominator': q.denominator}

def main():
    reg = json.loads((HERE / 'DELIVERY_RANK_BOUND_REGISTRATION.json').read_text(encoding='utf-8'))
    assert reg['stage'] == 'saved_compiled_score_rank_robustness_bound_v1'
    original = json.loads((PRIOR / 'RESULT.json').read_text(encoding='utf-8'))
    rows = []
    for rec in original['records']:
        path = PRIOR / rec['output_file']
        assert sha(path) == rec['output_sha256']
        with np.load(path, allow_pickle=False) as z:
            assert z.files == ['scores']
            scores = z['scores']
        assert scores.dtype == np.float32 and scores.ndim == 1 and len(scores) == rec['actions']
        assert np.isfinite(scores).all()
        order = np.argsort(-scores.astype(np.float64),kind='stable')
        ranked = [Fraction.from_float(float(scores[i])) for i in order]
        gaps = [a-b for a,b in zip(ranked,ranked[1:])]
        E = Fraction.from_float(float(rec['max_abs_score_difference']))
        vulnerable = [int(i) for i,g in enumerate(gaps) if g<=2*E]
        rows.append(dict(index=rec['index'],arm=rec['arm'],step=rec['step'],scores=len(scores),
            output_sha256=rec['output_sha256'],error=rational(float(E)),
            top_index=int(order[0]),runner_up_index=int(order[1]),top_gap=rational(gaps[0]),
            top_index_bound_robust=bool(gaps[0]>2*E),
            adjacent_rank_pairs_with_gap_le_2error=len(vulnerable),
            exact_adjacent_ties=sum(g==0 for g in gaps),
            smallest_adjacent_gap=rational(min(gaps))))
    assert len(rows)==34 and original['exact_equal_count']==0
    out=dict(stage=reg['stage'],rows=rows,records=34,
        top_index_robust_count=sum(r['top_index_bound_robust'] for r in rows),
        vulnerable_adjacent_pairs_total=sum(r['adjacent_rank_pairs_with_gap_le_2error'] for r in rows),
        exact_ties_total=sum(r['exact_adjacent_ties'] for r in rows),
        bound_means='Top-index robustness only under recorded per-vector max error; not complete insertion/regret/action-chain identity',
        strict_output_identity_qualified=False,formal_controls_qualified=False,full_request_budget_qualified=False,
        scientific_family3_started=False,stable_online_gain=False)
    with (HERE / 'DELIVERY_RANK_BOUND_RESULT.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2)

if __name__=='__main__': main()
