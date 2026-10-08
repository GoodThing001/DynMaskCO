# -*- coding: utf-8 -*-
"""A1 new arms must be covered by the independent gate adjudicator."""
import hashlib
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'evaluation'))
from diag_adjudicate import adjudicate  # noqa: E402


def _rows(offset=0.0):
    return [{'i': i, 'utility': 100.0 + i + offset,
             'hard_feasible': True, 'timeouts': 0} for i in range(2)]


def _gate(v_path):
    ident = {'source_sha256': {'s': 'a' * 64},
             'source_end_sha256': {'s': 'a' * 64}, 'source_stable': True,
             'shared_source_sha256': {'s': 'a' * 64},
             'contract_sha256': 'b' * 64, 'data_sha256': 'c' * 64,
             'data_meta_sha256': 'd' * 64, 'budget_B': 100.0,
             'cooling_share': 4.0,
             'shared_config': {'time_limit': 10.0, 'capacity': 50.0,
                               'num_vehicles': 15, 'n_orders': 200},
             'seeds': {'train': 20260925, 'gate': 20260926},
             'a1': {'v_ckpt_sha256': hashlib.sha256(v_path.read_bytes()).hexdigest(),
                    'h': 2.0, 'arms': ['a1_consensus', 'a1_rollout']}}
    cfg = {'gate_instances': 2, 'time_limit': 10.0, 'K': 10,
           'capacity': 50.0, 'num_vehicles': 15, 'n_orders': 200,
           'arms': 'uncond_hist,cond_hist,explicit_feat,a1_consensus,a1_rollout',
           'a1_v_ckpt': str(v_path), 'a1_h': 2.0}
    block = {'per_day': {'uncond_hist': _rows(-10.0),
                         'cond_hist': _rows(), 'explicit_feat': _rows(),
                         'a1_consensus': _rows(5.0), 'a1_rollout': _rows(-20.0)}}
    return {'config': cfg, 'identity': ident,
            'seeds': {'gate': 20260926}, 'budget': {'B': 100.0},
            'gate': {t: block for t in ('p_c=0', 'p_c=(5,10,15)',
                                         'p_c=(10,20,30)')}}


def test_a1_extra_arms():
    with tempfile.TemporaryDirectory() as td:
        v = Path(td) / 'v.npz'
        v.write_bytes(b'frozen-model')
        d = _gate(v)
        good = adjudicate(d, expected=2)
        assert good['formal_adjudicable'], good['problems']
        assert good['extra_arm_comparisons']['p_c=(5,10,15)'][
            'a1_consensus_minus_cond_hist']['mean'] == 5.0
        d['gate']['p_c=0']['per_day']['a1_consensus'] = _rows(5.0)[:1]
        bad = adjudicate(d, expected=2)
        assert not bad['formal_adjudicable']
        assert 'day_incomplete_p_c=0_a1_consensus' in bad['problems']
        d = _gate(v)
        v.write_bytes(b'tampered')
        bad_v = adjudicate(d, expected=2)
        assert 'a1_v_weight_hash_mismatch' in bad_v['problems']
    print('A1 independent arm adjudication PASS')


if __name__ == '__main__':
    test_a1_extra_arms()
