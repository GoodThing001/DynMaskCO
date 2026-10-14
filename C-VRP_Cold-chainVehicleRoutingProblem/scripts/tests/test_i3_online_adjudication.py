"""Independent decision-gate tests: complete day blocks and fail-closed identity.

These fixtures exercise adjudication only; their positive utilities are synthetic
and must never be reported as method evidence.
"""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'evaluation'))
from diag_i3_online_adjudicate import adjudicate, paired, sha


def put(path, value):
    path.write_text(json.dumps(value, allow_nan=False), encoding='utf-8')
    return sha(path)


def rejects(call):
    try:
        answer = call()
    except (ValueError, KeyError, TypeError):
        return
    assert not answer['formal_adjudicable'], 'Malformed identity acquired formal qualification'


def fixture(root):
    root = Path(root)
    source = root / 'src'
    source.mkdir()
    source.joinpath('runner.py').write_text('# synthetic fixture\n', encoding='utf-8')
    source_hashes = {'runner.py': sha(source / 'runner.py')}
    models = {}
    for name in ('maskco', 'warmstart', 'reconstruction_control'):
        folder = root / name
        folder.mkdir()
        weight = folder / 'model.bin'
        weight.write_bytes(name.encode('ascii'))
        card = folder / 'config.json'
        put(card, {'qualification': 'formal_development', 'training_complete': True,
            'i3_training': {'source_stable': True, 'checkpoint_reload_verified': True,
                            'model_bin_sha256': sha(weight)},
            'history_pool_lineage': {'fixture': True}})
        models[name] = {'path': str(weight), 'sha256': sha(weight),
                        'card': str(card), 'card_sha256': sha(card)}
    protocol = root / 'protocol.json'
    put(protocol, {'predeclared': True, 'bootstrap': {'samples': 2000, 'seed_offset': 777, 'unit': 'day'},
                   'online_days': 40, 'delta_3': 20, 'development_seeds': [20260926, 20260930, 20261001],
                   'main_comparison': 'I3_minus_cond_hist',
                   'online_parameters': {'dev_instances': 40}})
    config = root / 'online_config.json'
    cfg = {'qualification': 'formal_development', 'models': models,
        'history_pool_lineage': {'fixture': True}, 'online_parameters': {'dev_instances': 40},
        'expected_identity': {'seeds': {'train': 20260925, 'gate': 20260926,
            'final_test_reserved': [20260927, 20260928]}}, 'frozen_extension': str(source)}
    digest = put(config, cfg)
    rows = {name: [{'i': i, 'utility': float((50 if name == 'maskco' else 0) + (i % 4)),
                    'hard_feasible': True, 'timeouts': i % 2} for i in range(40)]
            for name in ('maskco', 'cond_hist', 'warmstart', 'reconstruction_control')}
    adapter = {'config_sha256': digest, 'qualification': 'formal_development'}
    identity = {**cfg['expected_identity'], 'source_stable': True,
                'source_sha256': source_hashes, 'source_end_sha256': source_hashes,
                'model_adapter': adapter}
    doc = {'model_adapter': adapter, 'config': cfg['online_parameters'], 'seeds': {'ran': [20260926]},
           'results': {'20260926': {'_identity': identity, 'p_c=(5,10,15)': {
               'per_day': rows, 'main_comparison': {'formal_adjudication': True}}}}}
    gate = root / 'gate.json'
    put(gate, doc)
    root.joinpath('run.exit').write_text('RUN_EXIT=0\n', encoding='utf-8')
    return gate, config, protocol, doc


def test():
    left = [{'i': i, 'utility': 100. + i, 'hard_feasible': True} for i in range(40)]
    right = [{'i': i, 'utility': float(i), 'hard_feasible': True} for i in range(40)]
    correct = paired(left, list(reversed(right)), 20260926, 40)
    assert correct == {'mean': 100., 'ci_lo': 100., 'ci_hi': 100., 'n_paired': 40, 'both_hard': True}
    for malformed in (right[:-1], right[:-1] + [right[0]],
                      [{**r, 'utility': None if r['i'] == 0 else r['utility']} for r in right],
                      [{**r, 'utility': np.inf if r['i'] == 0 else r['utility']} for r in right]):
        try:
            paired(left, malformed, 20260926, 40)
        except ValueError:
            pass
        else:
            raise AssertionError('Missing/duplicate/invalid day was filtered or accepted')
    assert not paired(left, [{**r, 'hard_feasible': r['i'] != 0} for r in right], 20260926, 40)['both_hard']
    with tempfile.TemporaryDirectory() as temp:
        gate, config, protocol, doc = fixture(temp)
        def run():
            return adjudicate(gate, config, sha(config), protocol, sha(protocol))
        result = run()
        assert result['formal_adjudicable'] and result['by_seed']['20260926']['formal_passed']
        assert result['by_seed']['20260926']['decision_objective_attribution_passed']
        for mutate in (
            lambda d: d['seeds'].update(ran=[]),
            lambda d: d['seeds'].update(ran=[20260926, 20260926]),
            lambda d: (d['seeds'].update(ran=[20260930]),
                       d['results'].update({'20260930': d['results'].pop('20260926')})),
            lambda d: d['results']['20260926']['_identity'].update(source_sha256={}, source_end_sha256={}),
            lambda d: d['results']['20260926']['_identity'].update(source_stable=False),
            lambda d: d['results']['20260926']['p_c=(5,10,15)']['main_comparison'].update(formal_adjudication=False),
        ):
            malformed = deepcopy(doc)
            mutate(malformed)
            put(gate, malformed)
            rejects(run)
        put(gate, doc)
        Path(temp, 'run.exit').unlink()
        rejects(run)
        Path(temp, 'run.exit').write_text('RUN_EXIT=0\n', encoding='utf-8')
        cfg = json.loads(config.read_text())
        cfg['qualification'] = 'engineering'
        put(config, cfg)
        doc['model_adapter']['config_sha256'] = sha(config)
        doc['results']['20260926']['_identity']['model_adapter'] = doc['model_adapter']
        put(gate, doc)
        rejects(run)
    print('ALL PASS: synthetic positive fixture, day pairing, invalid days/hard flags, empty/duplicate/wrong seeds, empty/changed source, exits and engineering qualification')


if __name__ == '__main__':
    test()
