"""Fresh D-DESTROY-EQ8-v1: destroy information with actual decode-cost control.

Eight opportunities, not equal measured seconds. P/R score incumbent edges only
to select removed customers; Pdiscard executes P and discards its outputs before
uniform removal. Identical pure-distance regret repair/certified acceptance.
No old ANY50 rows are consumed or replaced. No online or stable gain claim.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from delivery_equal_schedule import (digest, dump_new, generate, parameter_digest,
    reconstruct, DeliveryContract, evaluate, MaskCORouteProvider, partial_adjacency)

REGISTERED = {'version': 'D-DESTROY-EQ8-v1', 'dataset_seed': 202610101701,
    'mask_seed': 202610101702, 'bootstrap_seed': 202610101703,
    'arm_order_seed': 202610101704, 'bootstrap_repetitions': 2000,
    'random_model_seed': 42, 'steps': 8, 'mask_fraction': .5,
    'time_limit_s': 10., 'gate_threshold': .01,
    'scope': 'destroy_information_fixed_schedule_snapshot_no_online_gain',
    'arms': ['Pguided', 'Rguided', 'Pdiscard'],
    'instance_ids': list(range(17000, 17032)),
    'guide_role': 'destroy_only_pure_distance_regret_repair',
    'all_failures_retained': True, 'claims_equal_consumed_seconds': False,
    'claims_online_gain': False, 'claims_stable_gain': False,
    'checkpoint_sha256': '03615b08a55125a8fe4a6dad1df9ab592b56f7ed9d21b54d2562070b3c6d38db'}


def verify_protocol(p):
    for key, expected in REGISTERED.items():
        got = p.get(key)
        if type(got) is not type(expected) or got != expected:
            raise ValueError('Registered field changed or wrong type: ' + key)
    if any(type(i) is not int for i in p['instance_ids']):
        raise ValueError('Instance identifiers must be actual integers')
    if p.get('stage') not in ('synthetic_engineering_qa', 'fresh_mechanism_pilot'):
        raise ValueError('Declare a permitted stage')
    if p['stage'] == 'fresh_mechanism_pilot' and not isinstance(p.get('synthetic_qa_artifact_sha256'), str):
        raise ValueError('Freeze actual synthetic QA artifact before research data')


def select_guided(logits, incumbent, nodes, size):
    """Lowest-score actual incumbent edges; deterministic endpoint/tie ordering.

No Shaw claim: no relatedness or selected-anchor sampling is used here.
"""
    valid = ~np.eye(nodes, dtype=bool)
    if logits.shape != (nodes, nodes) or not np.all(np.isfinite(logits[valid])):
        raise ValueError('Actual finite off-diagonal logits required')
    ranked = []
    for route in incumbent:
        chain = [0] + route + [0]
        ranked.extend((float(logits[a, b]), a, b)
                      for a, b in zip(chain, chain[1:]) if a != b)
    ranked.sort(key=lambda row: (row[0], row[1], row[2]))
    removed = set()
    for _, a, b in ranked:
        for customer in (a, b):
            if customer and len(removed) < size:
                removed.add(customer)
    if len(removed) != size:
        raise ValueError('Guide did not select exact registered cardinality')
    return sorted(removed)


def run_arm(instance, initial_routes, contract, arm, provider,
            seed, instance_id, clock=time.perf_counter, time_limit_s=10.):
    if arm not in REGISTERED['arms'] or provider is None:
        raise ValueError('All three arms must perform actual model inference')
    start = clock()
    deadline = start + time_limit_s
    initial = evaluate(instance, initial_routes, contract, deadline=deadline, clock=clock)
    if not initial['hard']:
        raise ValueError('Initial route qualification failed; never resample')
    incumbent, report = deepcopy(initial_routes), initial
    feature_start = clock()
    features = provider.features(instance, contract.capacity)
    if hasattr(features, 'block_until_ready'):
        features.block_until_ready()
    feature_s = clock() - feature_start
    nodes = len(instance['demands'])
    size = math.ceil(.5 * (nodes - 1))
    attempts = []
    for step in range(1, 9):
        began = clock()
        row = {'step': step, 'incumbent_before': deepcopy(incumbent),
            'cost_before': report['cost'], 'decode_calls': 0,
            'candidate_complete': False, 'candidate_hard': False,
            'accepted': False, 'retained_edges_preserved': False,
            'candidate_routes': None, 'candidate_report': None,
            'decode_s': 0., 'repair_s': 0., 'certification_s': 0.,
            'removed': [], 'remaining': [], 'insertions': [],
            'status': 'not_started_deadline'}
        if clock() < deadline:
            full = partial_adjacency(incumbent, set(), nodes)
            row['decode_input_adjacency_sha256'] = hashlib.sha256(full.tobytes()).hexdigest()
            dstart = clock()
            logits = provider.logits(features, full)
            row.update(decode_calls=1, decode_s=clock()-dstart,
                logits_sha256=hashlib.sha256(np.ascontiguousarray(logits).tobytes()).hexdigest(),
                logits_shape=list(logits.shape), status='decoded')
            valid = ~np.eye(nodes, dtype=bool)
            if logits.shape != (nodes, nodes) or not np.all(np.isfinite(logits[valid])):
                raise ValueError('Actual model logits invalid')
            if arm == 'Pdiscard':
                # Decode result cannot affect RNG, mask, insertion, ranking or acceptance.
                rng = np.random.default_rng(np.random.SeedSequence([seed, instance_id, step]))
                mask = sorted(map(int, rng.choice(np.arange(1, nodes), size, replace=False)))
            else:
                mask = select_guided(logits, incumbent, nodes, size)
            row.update(removed=mask, mask_sha256=digest(mask), guide_outputs_used=(arm != 'Pdiscard'))
            adj = partial_adjacency(incumbent, set(mask), nodes)
            row.update(adjacency_sha256=hashlib.sha256(adj.tobytes()).hexdigest(),
                retained_edges=np.argwhere(np.triu(adj, 1) > 0).tolist())
            rstart = clock()
            candidate, remaining, insertions, status = reconstruct(
                instance, incumbent, contract, set(mask), adj, None, 0., deadline, clock)
            row.update(candidate_routes=candidate, remaining=remaining,
                insertions=insertions, status=status, repair_s=clock()-rstart)
            if not remaining:
                row['candidate_complete'] = True
                restored = partial_adjacency(candidate, set(), nodes)
                row['retained_edges_preserved'] = bool(np.all(restored[adj > 0] > 0))
                if not row['retained_edges_preserved']:
                    raise RuntimeError('An original retained edge changed')
                cstart = clock()
                checked = evaluate(instance, candidate, contract, deadline=deadline, clock=clock)
                row.update(candidate_report=checked, candidate_hard=bool(checked['hard']),
                    certification_s=clock()-cstart)
                if checked['hard'] and clock() < deadline and checked['cost'] < report['cost']:
                    incumbent, report = deepcopy(candidate), checked
                    row['accepted'] = True
                if clock() >= deadline:
                    row['status'] = 'certification_deadline'
        row.update(cost_after=report['cost'], incumbent_after=deepcopy(incumbent),
            elapsed_s=clock()-start, step_wall_s=clock()-began)
        attempts.append(row)
    wall_s = clock()-start
    timeout = wall_s >= time_limit_s
    if timeout:
        incumbent, report = deepcopy(initial_routes), initial
    qualified = (not timeout and len(attempts) == 8 and all(
        r['decode_calls'] == 1 and r['candidate_complete'] and r['candidate_hard']
        and r['retained_edges_preserved'] for r in attempts))
    return {'routes': incumbent, 'report': report, 'wall_s': wall_s,
        'feature_s': feature_s, 'timeout': timeout, 'attempts': attempts,
        'planned_attempts': 8, 'actual_attempts': sum(r['decode_calls'] for r in attempts),
        'decode_calls': sum(r['decode_calls'] for r in attempts),
        'complete_candidates': sum(r['candidate_complete'] for r in attempts),
        'hard_candidates': sum(r['candidate_hard'] for r in attempts),
        'fixed_schedule_qualified': qualified, 'guide_role': REGISTERED['guide_role'],
        'selection_objective': 'delivery_contract_cost', 'online_gain_established': False}


def verify_source(manifest, root):
    root = Path(root).resolve()
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError('Nonempty sealed source manifest required')
    for name, expected in manifest.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Actual source mismatch: ' + name)
    # Freeze every inference/repair/evaluation module imported by this runner.
    required = ['research_20261010/delivery_destroy_eq8.py',
        'research_20261010/delivery_equal_schedule.py', 'delivery/delivery_maskco.py',
        'delivery/delivery_baselines.py', 'delivery/delivery_contract.py',
        'delivery/delivery_evaluator.py', 'coldchain/coldchain_state.py']
    for suffix in required:
        module_path = Path(__file__).resolve().parents[1] / suffix
        if not module_path.is_relative_to(root) or str(module_path.relative_to(root)).replace('\\','/') not in manifest:
            raise ValueError('Actual imported runtime source not sealed: ' + suffix)


def validate_qa(qa, identity):
    if qa.get('stage') != 'synthetic_engineering_qa' or qa.get('actual_engineering_qa_pass') is not True:
        raise ValueError('Actual synthetic QA did not pass')
    if qa.get('protocol', {}).get('version') != REGISTERED['version']:
        raise ValueError('QA belongs to another experiment')
    rows = qa.get('rows', [])
    if len(rows) != 1 or type(rows[0].get('instance_id')) is not int or rows[0]['instance_id'] != -1:
        raise ValueError('QA must use one isolated synthetic case')
    if set(rows[0].get('arms', {})) != set(REGISTERED['arms']):
        raise ValueError('QA missing an arm')
    for name, arm in rows[0]['arms'].items():
        if (arm.get('fixed_schedule_qualified') is not True or arm.get('timeout') is not False
            or arm.get('report', {}).get('hard') is not True
            or type(arm.get('actual_attempts')) is not int or arm['actual_attempts'] != 8
            or arm.get('decode_calls') != 8 or arm.get('complete_candidates') != 8
            or arm.get('hard_candidates') != 8 or len(arm.get('attempts', [])) != 8):
            raise ValueError('QA eight complete hard actual decode attempts required')
        for step, a in enumerate(arm['attempts'], 1):
            if (type(a.get('step')) is not int or a['step'] != step
                or a.get('decode_calls') != 1 or a.get('candidate_complete') is not True
                or a.get('candidate_hard') is not True or a.get('retained_edges_preserved') is not True
                or a.get('guide_outputs_used') is not (name != 'Pdiscard')
                or len(a.get('removed', [])) != 25 or len(set(a['removed'])) != 25
                or any(type(i) is not int or not 1 <= i <= 50 for i in a['removed'])
                or not a.get('logits_sha256') or a.get('candidate_report',{}).get('hard') is not True):
                raise ValueError('QA per-attempt evidence incomplete')
    for key in ('source_manifest_sha256','checkpoint_sha256','contract_sha256','model_parameters'):
        if qa.get('identity',{}).get(key) != identity.get(key):
            raise ValueError('QA actual identity differs: ' + key)


def synthetic_data():
    data = {'coords': np.stack([np.arange(51)/10., np.zeros(51)], axis=1).tolist(),
        'demands': [0]+[1]*50, 'temp_class': [0]+[i % 3 for i in range(50)],
        'reveal_time': [0.]*51, 'ready_time': [0.]*51, 'tw_start': [0.]*51,
        'tw_end': [8.]*51, 'service_time': [0.]+[.03]*50}
    return data, [list(range(1+10*k,11+10*k)) for k in range(5)]


def paired(rows, control, other, seed):
    d = np.array([(r['arms'][control]['report']['cost']-r['arms'][other]['report']['cost'])
                  /r['arms'][control]['report']['cost'] for r in rows])
    samples = np.random.default_rng(seed).integers(0,len(d),size=(2000,len(d)))
    return {'n': len(d), 'control': control, 'other': other,
        'mean_relative_gain': float(d.mean()),
        'descriptive_95_CI': np.percentile(d[samples].mean(axis=1),[2.5,97.5]).tolist(),
        'qualification_failures_retained': True, 'selection_adjusted': False}


def main(args):
    raw = Path(args.protocol).read_bytes()
    protocol = json.loads(raw)
    verify_protocol(protocol)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    try:
        manifest = json.loads(Path(args.source_manifest).read_text(encoding='utf-8'))
        verify_source(manifest, args.source_root)
        contract = DeliveryContract()
        checkpoint = Path(args.checkpoint)
        sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if sha != REGISTERED['checkpoint_sha256']:
            raise ValueError('Actual pretrained checkpoint identity mismatch')
        providers = {'P': MaskCORouteProvider(checkpoint,args.upstream),
            'R': MaskCORouteProvider(checkpoint,args.upstream,random_seed=42)}
        for provider in providers.values():
            provider.prewarm(51,contract.capacity)
        parameters = {name:parameter_digest(p) for name,p in providers.items()}
        if parameters['P']['sha256'] == parameters['R']['sha256']:
            raise ValueError('P and R actual weights identical')
        identity = {'protocol_sha256':hashlib.sha256(raw).hexdigest(),
            'source_manifest_sha256':digest(manifest), 'checkpoint_sha256':sha,
            'contract_sha256':contract.identity(), 'model_parameters':parameters,
            'upstream_root':str(Path(args.upstream).resolve()),
            'pid':os.getpid(), 'cpu_affinity':sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
            'threads':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
            'claim_scope':protocol['scope']}
        if protocol['stage'] == 'fresh_mechanism_pilot':
            if not args.synthetic_qa_artifact:
                raise ValueError('Actual synthetic QA artifact required before data generation')
            qa_raw = Path(args.synthetic_qa_artifact).read_bytes()
            if hashlib.sha256(qa_raw).hexdigest() != protocol['synthetic_qa_artifact_sha256']:
                raise ValueError('Actual synthetic QA artifact SHA mismatch')
            validate_qa(json.loads(qa_raw),identity)
        dump_new(out/'identity_start.json',identity)
        dump_new(out/'PROTOCOL.json',protocol)
        dump_new(out/'SOURCE_MANIFEST.json',manifest)
        rows = []
        ids = [-1] if protocol['stage'] == 'synthetic_engineering_qa' else protocol['instance_ids']
        with (out/'progress.jsonl').open('x',encoding='utf-8') as f:
            for iid in ids:
                data, initial = synthetic_data() if iid == -1 else generate(protocol['dataset_seed'],iid)
                checked = evaluate(data,initial,contract)
                if not checked['hard']:
                    raise ValueError('Initial instance qualification failed; no replacements')
                rng = np.random.default_rng(np.random.SeedSequence([protocol['arm_order_seed'],max(0,iid)]))
                order = list(rng.permutation(protocol['arms']))
                arms = {}
                for name in order:
                    provider = providers['R' if name == 'Rguided' else 'P']
                    arms[name] = run_arm(data,initial,contract,name,provider,
                        9001 if iid == -1 else protocol['mask_seed'],max(0,iid))
                    replay = evaluate(data,arms[name]['routes'],contract)
                    if replay['hard'] is not True or abs(replay['cost']-arms[name]['report']['cost']) > 1e-9:
                        raise RuntimeError('Submitted incumbent replay mismatch')
                row = {'instance_id':int(iid),'data':data,'initial_routes':initial,
                    'initial_report':checked, 'data_initial_sha256':digest([data,initial]),
                    'arm_order':order,'arms':arms}
                f.write(json.dumps(row,allow_nan=False)+'\n'); f.flush(); os.fsync(f.fileno())
                rows.append(row)
                print(json.dumps({'instance_id':int(iid),'completed':len(rows)}),flush=True)
        verify_source(manifest,args.source_root)
        for name,p in providers.items():
            if parameter_digest(p) != parameters[name]:
                raise RuntimeError('Actual model parameters changed')
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != sha:
            raise RuntimeError('Actual checkpoint changed')
        if Path(args.protocol).read_bytes() != raw:
            raise RuntimeError('Actual protocol changed')
        dump_new(out/'identity_end.json',identity)
        qualified = all(a['fixed_schedule_qualified'] for r in rows for a in r['arms'].values())
        result = {'protocol':protocol,'identity':identity,'stage':protocol['stage'],'rows':rows,
            'fixed_schedule_qualified':qualified,
            'actual_engineering_qa_pass':protocol['stage']=='synthetic_engineering_qa' and qualified,
            'engineering_only':protocol['stage']=='synthetic_engineering_qa',
            'online_gain_established':False,'stable_gain_established':False,
            'independent_numeric_adjudication_pending':True}
        if protocol['stage']=='fresh_mechanism_pilot':
            result['paired'] = {'Pguided_over_Pdiscard':paired(rows,'Pdiscard','Pguided',protocol['bootstrap_seed']),
                'Pguided_over_Rguided':paired(rows,'Rguided','Pguided',protocol['bootstrap_seed'])}
            # No subsequent cohort or checkpoint selection is executed by this finite pilot.
            result['qualified_numeric_signal'] = qualified and all(
                x['mean_relative_gain'] >= .01 and x['descriptive_95_CI'][0] > 0 for x in result['paired'].values())
        dump_new(out/'result.json',result)
    except Exception as exc:
        dump_new(out/'failure.json',{'error':str(exc),'exception_type':type(exc).__name__,
            'traceback':traceback.format_exc(),'old_artifacts_unchanged':True})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('protocol','out','checkpoint','upstream','source-manifest','source-root'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--synthetic-qa-artifact')
    main(parser.parse_args())
