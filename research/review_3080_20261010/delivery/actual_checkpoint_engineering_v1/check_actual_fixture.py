"""Cheap exact old teacher/C0 replay qualification; no neural training."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from actual_full_step_qa import dump, sha, sources


def check(root, upstream, registration, out):
    reg = json.loads(registration.read_text(encoding='utf-8'))
    if sources(root, upstream) != reg['sources']:
        raise ValueError('Registered actual source identity changed')
    for folder in ('delivery', 'coldchain', 'research_20261010'):
        sys.path.insert(0, str(root / 'scripts' / folder))
    from delivery_equal_schedule import generate
    from delivery_maskco import partial_adjacency
    from delivery_contract import DeliveryContract
    from delivery_evaluator import evaluate
    split = root / reg['teacher_split']
    for name, expected in reg['teacher_files'].items():
        if sha(split / name) != expected:
            raise ValueError('Actual original teacher file changed')
    identity = json.loads((split / 'TRAIN_SPLIT_IDENTITY.json').read_text(encoding='utf-8'))
    teacher = next(json.loads(line) for line in (split / 'samples.jsonl').open(encoding='utf-8')
                   if json.loads(line)['outcome'] == 'certified')
    data, _ = generate(identity['train_seed'], teacher['instance'])
    contract = DeliveryContract()
    target = teacher['candidate_routes']
    report = evaluate(data, target, contract)
    observed = partial_adjacency(teacher['incumbent_routes_before'], set(teacher['mask']), len(data['demands']))
    preserved = bool(np.all(partial_adjacency(target, set(), len(data['demands']))[observed > 0] > 0))
    broken = [list(r) for r in target]
    broken[0] = broken[0][1:]
    negative_missing_customer = evaluate(data, broken, contract)
    q = {'stage': 'old_teacher_fixture_replay_only_no_model_execution', 'actual_nodes': len(data['demands']),
         'teacher_source_instance': teacher['instance'], 'teacher_hash': sha(split / 'samples.jsonl'),
         'contract_sha256': contract.identity(), 'complete': report['complete'], 'hard': report['hard'],
         'retained_edges': int(np.count_nonzero(np.triu(observed))), 'retained_edges_preserved': preserved,
         'original_stored_cost': teacher['cost_after_candidate'], 'independent_replayed_cost': report['cost'],
         'cost_error': abs(report['cost'] - teacher['cost_after_candidate']),
         'negative_missing_customer_triggers': negative_missing_customer['hard'] is False,
         'new_research_data': False, 'full_model_training_executed': False, 'gain_claim': False,
         'teacher_report': report}
    q['engineering_fixture_pass'] = q['complete'] is True and q['hard'] is True and preserved and q['cost_error'] < 1e-8 and q['negative_missing_customer_triggers']
    dump(out, q)
    print(json.dumps({k: q[k] for k in ('engineering_fixture_pass', 'actual_nodes', 'retained_edges', 'cost_error')}))
    if not q['engineering_fixture_pass']:
        raise RuntimeError('Actual old fixture/C0 qualification failed')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--upstream', type=Path, required=True)
    ap.add_argument('--registration', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    check(a.root, a.upstream, a.registration, a.out)
