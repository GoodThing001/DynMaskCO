"""Complete retained-state teacher generator; no registration bypass switch."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from finite_common import sha, digest, dump, classify_pair, verify_research_gate, check_dependency_identity


def collect(root, out, seed, ids, mask_seed, contract, registration_sha):
    for folder in ('delivery', 'coldchain', 'research_20261010'):
        sys.path.insert(0, str(root / 'scripts' / folder))
    from delivery_equal_schedule import generate, reconstruct, make_schedule
    from delivery_maskco import partial_adjacency
    from delivery_evaluator import evaluate
    out.mkdir(parents=True, exist_ok=False)
    summaries = []
    with (out / 'TEACHER_STATES.jsonl').open('x', encoding='utf-8') as stream:
        for instance_id in ids:
            start = time.perf_counter()
            overall_deadline = start + 10.
            data, incumbent = generate(seed, instance_id)
            initial = evaluate(data, incumbent, contract, deadline=overall_deadline)
            # Never replace or resample an initial policy/data hard failure.
            if initial['hard'] is not True:
                row = {'instance_id': instance_id, 'data': data, 'incumbent': incumbent,
                       'initial_report': initial, 'initial_hard_failure': True}
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
                raise ValueError('Initial hard failure retained; scientific dataset blocked')
            masks = make_schedule(51, 4, mask_seed, instance_id, fraction=.5)
            rows = []
            for ordinal, mask in enumerate(masks):
                step_start = time.perf_counter()
                deadline = min(overall_deadline, step_start + 2.5)
                partial = partial_adjacency(incumbent, set(mask), 51)
                candidate, remaining, insertions, status = reconstruct(data, incumbent, contract,
                    set(mask), partial, None, 0., deadline, time.perf_counter)
                # Whole-route certification under same teacher time ceiling.
                report = evaluate(data, candidate, contract, deadline=deadline)
                late = time.perf_counter() >= deadline
                retained = partial_adjacency(candidate, set(), 51)
                preserved = bool(np.all(retained[partial > 0] > 0))
                if not preserved:
                    status = 'retained_edge_failure'
                row = {'instance_id': instance_id, 'mask_ordinal': ordinal, 'data': data,
                    'data_sha256': digest(data), 'incumbent_routes': deepcopy(incumbent),
                    'initial_report': initial, 'mask': mask, 'partial_adjacency': partial.tolist(),
                    'candidate_routes': candidate, 'candidate_report': report, 'remaining': remaining,
                    'insertions': insertions, 'status': status, 'late': late,
                    'retained_edges_preserved': preserved, 'wall_s': time.perf_counter()-step_start,
                    'teacher_proposal_objective': 'distance_regret2',
                    'teacher_label_objective': 'full_C0_contract_cost', 'registration_sha256': registration_sha}
                row.update(classify_pair(initial, report, candidate, incumbent, status, late))
                # Reuse the within-budget full certification. A late candidate
                # never becomes a successful teacher through after-budget retry.
                ce_report = report if row['CE_target_choice'] == 'candidate' else initial
                row['CE_target_certified_report'] = ce_report
                row['CE_target_full_hard'] = ce_report['hard'] is True and ce_report['complete'] is True
                if not row['CE_target_full_hard']:
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                    raise ValueError('No certified full reconstruction target')
                # Store every state, including invalid/negative/small outcomes.
                row['state_key'] = [instance_id, ordinal]
                row['state_sha256'] = digest(row)
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
                rows.append(row)
            summaries.append({'instance_id': instance_id, 'data_sha256': digest(data),
                'states': 4, 'wall_s': time.perf_counter()-start,
                'categories': {k: sum(r['category']==k for r in rows) for k in ('positive','negative','small','invalid')}})
    result = {'stage': 'teacher_generation_only', 'instances': summaries, 'expected_ids': ids,
        'actual_state_count': 4*len(ids), 'all_states_retained': True, 'dataset_seed': seed, 'mask_seed': mask_seed,
        'teacher_archive_sha256': sha(out / 'TEACHER_STATES.jsonl'),
        'registration_sha256': registration_sha, 'contract_hash': contract.identity(),
        'engineering_pass': all(s['states']==4 for s in summaries),
        'teacher_optimality_claim': False, 'dynamic_online_claim': False}
    dump(out / 'TEACHER_SUMMARY.json', result)
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    for key in ('root', 'upstream', 'registration', 'out'):
        ap.add_argument('--' + key, type=Path, required=True)
    ap.add_argument('--mode', choices=('engineering', 'research'), required=True)
    ap.add_argument('--teacher-gate', type=Path)
    ap.add_argument('--objective-gate', type=Path)
    ap.add_argument('--budget-gate', type=Path)
    a = ap.parse_args()
    try:
        reg = json.loads(a.registration.read_text(encoding='utf-8'))
        check_dependency_identity(reg, a.root, a.upstream)
        for folder in ('delivery', 'coldchain'):
            sys.path.insert(0, str(a.root / 'scripts' / folder))
        from delivery_contract import DeliveryContract
        contract = DeliveryContract()
        if contract.identity() != reg['contract_hash']:
            raise ValueError('Contract mismatch')
        if a.mode == 'research':
            verify_research_gate(reg, {'teacher_engineering': a.teacher_gate,
                'actual_10D_objective': a.objective_gate, 'deployment_10s_budget': a.budget_gate})
            seed, mask_seed, ids = reg['seeds']['train'], reg['seeds']['teacher_mask'], reg['train_instance_ids']
            if len(ids)!=64 or any(type(i)is not int or i<0 for i in ids) or len(set(ids))!=64:
                raise ValueError('Exactly64 unique registered actual integer ids required')
        else:
            if reg.get('status') != 'REGISTERED_ENGINEERING_ONLY' or reg.get('engineering_seed') != 202610110071:
                raise ValueError('Not registered engineering inputs')
            seed, mask_seed, ids = 202610110071, 202610110072, [900000,900001]
        result = collect(a.root, a.out, seed, ids, mask_seed, contract, sha(a.registration))
        check_dependency_identity(reg, a.root, a.upstream)
        dump(a.out / 'SOURCE_END.json', {'dependency_sources': reg['dependency_sources'],
             'actual_rechecked_unchanged': True})
        print(json.dumps({'engineering_pass': result['engineering_pass'], 'states': result['actual_state_count']}))
    except Exception as error:
        if a.out.is_dir() and not (a.out / 'FAILURE.json').exists():
            dump(a.out / 'FAILURE.json', {'exception': repr(error), 'traceback': traceback.format_exc(),
                 'pass': False, 'all_existing_records_preserved': True})
        raise
