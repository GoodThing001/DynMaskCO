"""Cheap engineering checks; no 76M load, gradient or scientific data generation."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import numpy as np
os.environ['JAX_PLATFORMS'] = 'cpu'
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'actual_checkpoint_engineering_v1'))
from features_objective import UNITS, make_features, objective_terms, pair_label
from actual_full_step_qa import sources, sha, dump


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    for name in ('root', 'upstream', 'registration', 'out'):
        ap.add_argument('--' + name, type=Path, required=True)
    a = ap.parse_args()
    reg = json.loads(a.registration.read_text(encoding='utf-8'))
    if sources(a.root, a.upstream) != reg['sources']:
        raise ValueError('Sources changed')
    for name, expected in reg['programs'].items():
        if sha(HERE / name) != expected:
            raise ValueError('Programs changed')
    split = a.root / reg['teacher_split']
    for name, expected in reg['teacher_files'].items():
        if sha(split / name) != expected:
            raise ValueError('Old teacher changed')
    sys.path.insert(0, str(a.upstream))
    for folder in ('delivery', 'coldchain', 'research_20261010'):
        sys.path.insert(0, str(a.root / 'scripts' / folder))
    from delivery_equal_schedule import generate
    from delivery_contract import DeliveryContract
    from delivery_evaluator import evaluate
    from modules.functional import coord_normalize
    from modules.MultiHeadAttention import attention_fn
    import jax.numpy as jnp
    identity = json.loads((split / 'TRAIN_SPLIT_IDENTITY.json').read_text(encoding='utf-8'))
    teacher = next(json.loads(s) for s in (split / 'samples.jsonl').open(encoding='utf-8')
                   if json.loads(s)['outcome'] == 'certified')
    data, _ = generate(identity['train_seed'], teacher['instance'])
    contract = DeliveryContract()
    coords = np.asarray(coord_normalize(jnp.asarray(data['coords'], jnp.float32)[None]))[0]
    x = make_features(data, contract, coords, UNITS, contract.identity())
    target_report = evaluate(data, teacher['candidate_routes'], contract)
    before_report = evaluate(data, teacher['incumbent_routes_before'], contract)
    label = pair_label(before_report, target_report)
    if label != 1 or x.shape != (1, 51, 10):
        raise ValueError('Old actual fixture not qualified')
    rejected = {}
    unready = deepcopy(data)
    unready['ready_time'][1] = .1
    initial = deepcopy(data)
    initial['initial_quality'] = [100.] * len(data['demands'])
    cases = {'unready': (unready, UNITS, contract.identity()),
             'initial_quality_override': (initial, UNITS, contract.identity()),
             'wrong_units': (data, dict(UNITS, times='min'), contract.identity()),
             'wrong_identity': (data, UNITS, '0'*64)}
    for name, (d, units, h) in cases.items():
        try:
            make_features(d, contract, coords, units, h)
        except ValueError:
            rejected[name] = True
        else:
            rejected[name] = False
    if not all(rejected.values()):
        raise ValueError('Declared data semantics not fail-closed')
    logits = jnp.array([[[0., 1., 2.], [1., 0., 3.], [2., 3., 0.]]])
    target = jnp.array([[[0., 1., 1.], [1., 0., 1.], [1., 1., 0.]]])
    alternative = jnp.array([[[0., 2., 0.], [2., 0., 0.], [0., 0., 2.]]])
    ce, comp0 = objective_terms(logits, target, alternative, 1, 0.)
    paired, comp1 = objective_terms(logits, target, alternative, 1, .2)
    tie, _ = objective_terms(logits, target, alternative, 0, .2)
    objective_difference_correct = abs(float(paired-ce)-.2*float(comp1[1])) < 1e-6
    ce_exact_same = bool(np.array_equal(np.asarray(comp0[0]), np.asarray(comp1[0])))
    tie_equals_ce = bool(np.array_equal(np.asarray(tie), np.asarray(ce)))
    q = jnp.array([[[[3., 1.]], [[1., 2.]]]])
    k = q
    v = jnp.array([[[[1., 0.]], [[0., 2.]]]])
    output_none = np.asarray(attention_fn(q=q, k=k, v=v, softcap=None))
    output_one = np.asarray(attention_fn(q=q, k=k, v=v, softcap=1.))
    # Existing original checkpoint has softcap=None; ignoring finite softcap is
    # a real implementation limitation but not changed by this mapping.
    attention_audit = {'original_checkpoint_softcap': reg['original_model_config']['softcap'],
        'actual_kernel_ignores_supplied_softcap': bool(np.array_equal(output_none, output_one)),
        'original_none_requires_no_tanh_softcap': reg['original_model_config']['softcap'] is None,
        'paper_operator_equivalence_qualified': False,
        'source': 'upstream/modules/MultiHeadAttention.py attention_fn del softcap; TransformerLayer default binds it'}
    out = {'stage': 'cheap_old_fixture_engineering_only', 'registration_sha256': sha(a.registration),
        'shape': list(x.shape), 'features': x.tolist(), 'contract_hash': contract.identity(),
        'teacher_report': target_report, 'incumbent_report': before_report,
        'pair_sign': label, 'rejections': rejected, 'CE_exact_same': ce_exact_same,
        'objective_difference_correct': objective_difference_correct, 'tie_equals_CE': tie_equals_ce,
        'attention_audit': attention_audit, 'full_model_loaded': False, 'new_scientific_data': False,
        'gain_claim': False, 'engineering_pass': all(rejected.values()) and ce_exact_same
            and objective_difference_correct and tie_equals_ce}
    dump(a.out, out)
    if not out['engineering_pass']:
        raise ValueError('Cheap objective engineering failed')
    print(json.dumps({'engineering_pass': out['engineering_pass'], 'rejected_cases': len(rejected)}))
