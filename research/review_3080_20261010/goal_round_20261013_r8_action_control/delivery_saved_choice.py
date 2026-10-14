"""Frozen-input semantic diagnostic. Never alters the compiled-head strict gate."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'delivery'
SPLIT = BASE / 'score_split_timed_execution_root_v1' / 'downloaded'
COMP = BASE / 'compiled_head_identity_root_v1' / 'downloaded'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def choose(saved, values):
    at = 0
    candidates = []
    for customer in saved['customers']:
        eligible = []
        for row in customer['rows']:
            score = float(values[at])
            if row[5] and row[6]:
                eligible.append((score, float(row[8]), int(row[1]), int(row[2]), at, row[:5]))
            at += 1
        eligible.sort(key=lambda x: (-x[0], x[1], x[2], x[3]))
        if eligible:
            regret = None if len(eligible) == 1 else eligible[0][0] - eligible[1][0]
            candidates.append((regret, int(customer['customer']), eligible[0]))
    assert at == len(values)
    if not candidates:
        return None, None
    winner = max(candidates, key=lambda x: (x[0] is None, 0.0 if x[0] is None else x[0], -x[1]))
    return winner[2][4], winner[0]

def main():
    reg = json.loads((HERE/'REGISTRATION.json').read_text(encoding='utf-8'))
    assert reg['stage'] == 'dual_track_saved_action_choice_and_independent_admission_control_v1'
    manifest = json.loads((COMP/'SAVED_HEAD_INPUTS.json').read_text(encoding='utf-8'))
    original = json.loads((COMP/'RESULT.json').read_text(encoding='utf-8'))['records']
    assert len(manifest) == len(original) == 34
    records = []
    for m, r in zip(manifest, original):
        assert m['arm'] == r['arm'] and m['step'] == r['step']
        folder = 'pretrained_initial_fault0_worker' if r['arm'] == 'pretrained_initial' else 'random888_initial_fault0_worker'
        npz = SPLIT/folder/'request'/('%02d.npz' % r['step'])
        js = SPLIT/folder/'request'/('%02d.json' % r['step'])
        out = COMP/r['output_file']
        assert sha(npz) == m['sha256'] == r['input_sha256'] and sha(out) == r['output_sha256']
        saved = json.loads(js.read_text(encoding='utf-8'))
        with np.load(npz, allow_pickle=False) as z:
            eager = z['scores'].copy()
            actions = z['actions'].copy()
        with np.load(out, allow_pickle=False) as z:
            compiled = z['scores'].copy()
        assert eager.dtype == compiled.dtype == np.float32 and np.array_equal(eager.shape, compiled.shape)
        assert actions.shape == (len(eager), 5)
        assert len(saved['customers']) == len(saved['remaining'])
        e_idx, e_reg = choose(saved, eager)
        c_idx, c_reg = choose(saved, compiled)
        assert e_idx is not None and saved['chosen'][:5] == actions[e_idx].tolist()
        assert saved['chosen'][7] == float(eager[e_idx])
        before = saved['before']
        after = [list(x) for x in before]
        ca = actions[c_idx].tolist()
        after[ca[1]].insert(ca[2], ca[0])
        records.append(dict(index=r['index'], arm=r['arm'], step=r['step'], input_sha256=sha(npz),
            saved_request_sha256=sha(js), compiled_output_sha256=sha(out), actions=len(eager),
            eager_index=e_idx, compiled_index=c_idx, eager_action=actions[e_idx].tolist(),
            compiled_action=ca, eager_regret=e_reg, compiled_regret=c_reg,
            same_choice=e_idx == c_idx, compiled_after=after,
            saved_after_same=after == saved['after']))
    result = dict(stage=reg['stage'], records=records, count=len(records),
        same_choice_count=sum(x['same_choice'] for x in records),
        same_after_count=sum(x['saved_after_same'] for x in records),
        original_strict_bitwise_gate_pass=False, trained_model_qualified=False,
        complete_request_budget_qualified=False, scientific_family3_started=False,
        stable_online_gain=False)
    with (HERE/'DELIVERY_SAVED_CHOICE_RESULT.json').open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2,allow_nan=False)

if __name__ == '__main__': main()
