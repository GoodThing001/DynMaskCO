"""Independently align raw days and reproduce A3 means/CI on downloaded artifacts."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CHAIN = ROOT / 'results/a3_repair_chain_1007_2250'
def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def vectors(rows):
    assert len(rows) == 40 and sorted(row['i'] for row in rows) == list(range(40))
    assert all(type(row['i']) is int and row['hard_feasible'] is True and
               row['utility'] is not None and np.isfinite(row['utility']) for row in rows)
    indexed = {row['i']: row['utility'] for row in rows}
    return np.array([indexed[i] for i in range(40)], dtype=float)

def verify_stats(a, b, seed, expected):
    differences = a - b
    rng = np.random.default_rng(int(seed) + 777)
    draws = differences[rng.integers(0, 40, size=(2000, 40))].mean(axis=1)
    stats = (float(differences.mean()), *map(float, np.quantile(draws, [.025, .975])))
    assert np.allclose(stats, [expected['mean'], expected['ci_lo'], expected['ci_hi']], rtol=0, atol=1e-10)
    assert expected['n_paired'] == 40 and expected['formal_adjudication'] and expected['identity_verified']
    return dict(zip(('mean', 'ci_lo', 'ci_hi'), stats))

answer = read(CHAIN / 'A3_SAME_SOURCE_ADJUDICATION.json')
assert answer['formal_adjudicable'] and answer['n_batches'] == 9 and not answer['partial']
manifest = read(sys.argv[1]) if len(sys.argv) > 1 else None
artifacts, pairs, rows = {}, {}, {}
pair_fields = ('shared_source_sha256', 'contract_sha256', 'data_sha256', 'data_meta_sha256',
               'budget_B', 'cooling_share', 'shared_config')
for name, batch in answer['per_batch'].items():
    seed, arm = batch['seed'], batch['arm']
    path = ROOT / 'results' / name / 'gate.json'
    base_path = ROOT / batch['baseline_gate']
    doc, base = read(path), read(base_path)
    for artifact in (path, path.with_name('run.exit'), base_path, base_path.with_name('run.exit')):
        rel = artifact.relative_to(ROOT).as_posix()
        artifacts[rel] = sha(artifact)
        if manifest:
            assert artifacts[rel] == manifest['a3_artifact_hashes'][rel]
    assert 'RUN_EXIT=0' in path.with_name('run.exit').read_text().splitlines()
    assert {'RUN_EXIT=0', 'SEAL_EXIT=0', 'DONE=1'} <= set(base_path.with_name('run.exit').read_text().splitlines())
    ident, other = doc['results'][seed]['_identity'], base['identity']
    assert set(doc['results']) == {seed} and doc['seeds']['ran'] == [int(seed)]
    assert all(ident[field] and ident[field] == other[field] for field in pair_fields)
    assert ident['source_stable'] and other['source_stable']
    assert ident['source_sha256'] == ident['source_end_sha256']
    assert other['source_sha256'] == other['source_end_sha256']
    for seed_type in ('train', 'gate'):
        assert ident['seeds'][seed_type] == other['seeds'][seed_type]
    if manifest:
        assert manifest['a3_actual_weight_hashes'][ident['model_bin']] == ident['model_bin_sha256']
    for tier in ('p_c=0', 'p_c=(5,10,15)', 'p_c=(10,20,30)'):
        av = vectors(doc['results'][seed][tier]['per_day']['maskco'])
        bv = vectors(base['gate'][tier]['per_day']['cond_hist'])
        key = str((arm, seed, tier))
        pairs[key] = verify_stats(av, bv, seed, answer['comparison_table'][key])
        rows[(arm, seed, tier)] = av
ablations = {}
for key, expected in answer['ablations'].items():
    seed, tier, name = ast.literal_eval(key)
    arms = ('B', 'D') if name == 'B_minus_D' else ('C', 'B')
    ablations[key] = verify_stats(rows[(arms[0], seed, tier)], rows[(arms[1], seed, tier)], seed, expected)
assert len(pairs) == 27 and len(ablations) == 18

diagnostic_path = ROOT / 'results/i3_teacher_risk_1008_v1/engineering/report.json'
diagnostic = read(diagnostic_path)
cfg = read(diagnostic_path.parents[1] / 'engineering_config.json')
assert diagnostic['config_sha256'] == sha(diagnostic_path.parents[1] / 'engineering_config.json')
assert diagnostic['source_stable'] and diagnostic['source_start'] == diagnostic['source_end']
spec = importlib.util.spec_from_file_location('diagnosis', ROOT / 'tools_remote/i3_teacher_risk_diagnostic_1008.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
for row in diagnostic['rows']:
    for arm, values in row['arms'].items():
        delta, margins = row['teacher_delta'], values['margins']
        # Recompute independently from raw teacher/margins.
        enabled = bool(row['decision_eligible'] and row['future_support'])
        if enabled:
            x = np.array(margins, dtype=float)
            risk = min(abs(delta), 100) * np.mean(np.logaddexp(0., -np.sign(delta) * x / 20))
            regret = abs(delta) * np.mean((x > 0) != (delta > 0))
            assert np.isclose(risk, values['preference_risk']) and np.isclose(regret, values['eligible_teacher_regret'])
        else:
            assert margins is None and not values['evaluated'] and values['preference_risk'] == values['eligible_teacher_regret'] == 0
summary = module.aggregate(diagnostic['rows'], cfg['evaluation_keys'],
    bootstrap_samples=cfg['bootstrap_samples'], bootstrap_seed=cfg['bootstrap_seed'])
assert all(summary[key] == diagnostic[key] for key in summary)
output = {'a3_pairs_verified': len(pairs), 'a3_ablations_verified': len(ablations),
    'all_days_exact_and_hard_feasible': True, 'paired_identity_verified': True,
    'server_artifact_bytes_verified': manifest is not None, 'actual_weights_checked_on_server': manifest is not None,
    'artifact_sha256': artifacts, 'pairs': pairs, 'ablations': ablations,
    'teacher_diagnostic_raw_recomputed': True, 'teacher_diagnostic_engineering_only': True,
    'diagnostic_single_day_ci_is_not_inferential_evidence': True,
    'maskco_online_gain_established': False,
    'limitations': ['Original model and new baseline were not run concurrently; wallclock load remains a limitation',
                   'C-B changes training and inference together; not an isolated iteration contribution',
                   'Diagnostic engineering sample is 3 states of 1 day, not a formal gain or unseen-day validation']}
out = CHAIN / 'A3_RAW_LOCAL_AUDIT_1008.json'
out.write_text(json.dumps(output, indent=2, allow_nan=False), encoding='utf-8')
print(json.dumps({'pairs': len(pairs), 'ablations': len(ablations), 'diagnostic_raw_recomputed': True,
                  'out': str(out)}))
