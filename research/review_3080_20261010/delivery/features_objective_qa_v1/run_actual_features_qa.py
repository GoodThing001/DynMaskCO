"""Actual original MaskCO 3->10D mapping and paired one-step objective QA only."""
import argparse
from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'actual_checkpoint_engineering_v1'))
from features_objective import UNITS, FIELDS, digest, make_features, extend_projection, pair_label, objective_terms
from actual_full_step_qa import sha, dump, sources, strict


def run(a):
    os.environ['JAX_PLATFORMS'] = 'cpu'
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    import numpy as np
    import psutil
    reg = json.loads(a.registration.read_text(encoding='utf-8'))
    required = {'version': 'D-10D-PAIRED-ENGINEERING-v1', 'stage': 'old_fixture_engineering_only',
                'fixed_steps_each_arm': 1, 'optimizer': 'sgd_without_momentum',
                'learning_rate': .00001, 'preference_coefficient': .2,
                'research_model_eligible': False, 'dynamic_online_qualified': False}
    if not all(strict(reg.get(k), v) for k, v in required.items()):
        raise ValueError('Strict registered settings mismatch')
    for name, expected in reg['programs'].items():
        if sha(HERE / name) != expected:
            raise ValueError('Engineering code changed: ' + name)
    olddir = HERE.parent / 'actual_checkpoint_engineering_v1'
    for name, expected in reg['old_dependency_programs'].items():
        if sha(olddir / name) != expected:
            raise ValueError('Old engineering dependency changed')
    start_sources = sources(a.root, a.upstream)
    if start_sources != reg['sources']:
        raise ValueError('Actual dependency manifest changed')
    split = a.root / reg['teacher_split']
    for name, expected in reg['teacher_files'].items():
        if sha(split / name) != expected:
            raise ValueError('Old TRAIN fixture changed')
    a.out.mkdir(parents=True, exist_ok=False)
    available = psutil.virtual_memory().available
    if available < 8 * 1024 ** 3:
        dump(a.out / 'RESOURCE_FAILURE.json', {'available_ram_bytes': available, 'required_gib': 8})
        raise RuntimeError('Insufficient RAM for actual complete architecture')
    process = psutil.Process()
    if a.cpu not in process.cpu_affinity():
        raise ValueError('Requested CPU unavailable')
    process.cpu_affinity([a.cpu])
    start = time.perf_counter()
    dump(a.out / 'START_IDENTITY.json', {'registration_sha256': sha(a.registration),
         'cpu_affinity': process.cpu_affinity(), 'available_ram_bytes': available, 'sources': start_sources})
    from inspect_checkpoint import load_trusted, inspect
    saved, checkpoint_sha = load_trusted(a.checkpoint, a.upstream)
    inspection = inspect(saved, checkpoint_sha)
    if inspection['parameter_leaves'] != 234 or inspection['parameter_elements'] != 76259840:
        raise ValueError('Not actual original architecture')
    if digest(inspection['model_config']) != digest(reg['original_model_config']):
        raise ValueError('Actual original config identity mismatch')
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from helpers import sol2adj
    from modules.functional import coord_normalize
    for name in ('delivery', 'coldchain', 'research_20261010'):
        sys.path.insert(0, str(a.root / 'scripts' / name))
    from delivery_equal_schedule import generate, reconstruct
    from delivery_contract import DeliveryContract
    from delivery_evaluator import evaluate
    from delivery_maskco import partial_adjacency
    contract = DeliveryContract()
    if contract.identity() != reg['contract_hash']:
        raise ValueError('Actual delivery/physics contract changed')
    identity = json.loads((split / 'TRAIN_SPLIT_IDENTITY.json').read_text(encoding='utf-8'))
    teacher = next(json.loads(s) for s in (split / 'samples.jsonl').open(encoding='utf-8')
                   if json.loads(s)['outcome'] == 'certified')
    data, _ = generate(identity['train_seed'], teacher['instance'])
    before_report = evaluate(data, teacher['incumbent_routes_before'], contract)
    target_report = evaluate(data, teacher['candidate_routes'], contract)
    if abs(target_report['cost'] - teacher['cost_after_candidate']) > 1e-8:
        raise ValueError('Full teacher C0 cost replay changed')
    if abs(before_report['cost'] - teacher['cost_before']) > 1e-8:
        raise ValueError('Full incumbent C0 cost replay changed')
    label = pair_label(before_report, target_report)
    if label != 1:
        raise ValueError('Pinned old first engineering fixture is not the expected positive pair')
    coords = np.asarray(coord_normalize(jnp.asarray(data['coords'], jnp.float32)[None]))[0]
    features = make_features(data, contract, coords, UNITS, contract.identity())
    stripped = features.copy()
    stripped[:, :, 3:] = 0.
    # These deliberate invalid fixtures must be rejected before model use.
    rejection = {}
    changed = deepcopy(data)
    changed['reveal_time'][1] = .1
    changed['ready_time'][1] = .1
    checks = {'future_request': (changed, UNITS, contract.identity()),
              'hours_as_minutes': (data, dict(UNITS, times='min'), contract.identity()),
              'wrong_contract': (data, UNITS, '0' * 64)}
    for name, (d, units, chash) in checks.items():
        try:
            make_features(d, contract, coords, units, chash)
        except ValueError:
            rejection[name] = True
        else:
            rejection[name] = False
    if not all(rejection.values()):
        raise ValueError('Feature rejection tests failed')
    nodes = len(data['demands'])
    removed = set(teacher['mask'])
    observed = partial_adjacency(teacher['incumbent_routes_before'], removed, nodes)
    context = jnp.asarray(observed[None])
    timestep = jnp.asarray([observed[1:].sum() / (2 * (nodes - 1))], jnp.float32)
    def route_adj(routes):
        vector = [0]
        for route in routes:
            vector.extend(route + [0])
        return sol2adj(jnp.asarray([vector], jnp.int32), dtype=jnp.float32, num_nodes=nodes, is_cvrp=True)
    target, alternative = route_adj(teacher['candidate_routes']), route_adj(teacher['incumbent_routes_before'])
    cfg3 = deepcopy(saved['model_config'])
    cfg3.dtype = 'float32'
    original = cfg3.construct_model()
    graph3 = nnx.graphdef(original)
    original_params = saved['params']
    del original, saved
    gc.collect()
    cfg10 = deepcopy(cfg3)
    cfg10.encoder_input_dim = 10
    expanded_model = cfg10.construct_model()
    graph10 = nnx.graphdef(expanded_model)
    params = deepcopy(original_params)
    old_kernel = np.asarray(params['init_proj']['kernel'].value)
    params['init_proj']['kernel'].value = jnp.asarray(extend_projection(old_kernel))
    del expanded_model
    gc.collect()
    def inventory(p):
        return {str(path): {'shape': list(np.asarray(v.value).shape),
                'sha256': hashlib.sha256(np.asarray(v.value).tobytes()).hexdigest()}
                for path, v in p.flat_state()}
    original_inventory, initial_inventory = inventory(original_params), inventory(params)
    changed_paths = [p for p in original_inventory if original_inventory[p] != initial_inventory[p]]
    if changed_paths != ["('init_proj', 'kernel')"]:
        raise ValueError('Anything except zero-extended input projection changed')
    def f3(p, x):
        m = nnx.merge(graph3, p)
        return m.decode(m.encode(x), timestep, context)
    def f10(p, x):
        m = nnx.merge(graph10, p)
        return m.decode(m.encode(x), timestep, context)
    infer3, infer10 = jax.jit(f3), jax.jit(f10)
    l3 = np.asarray(infer3(original_params, jnp.asarray(features[:, :, :3])))
    l10 = np.asarray(infer10(params, jnp.asarray(features)))
    off = ~np.eye(nodes, dtype=bool)
    mapping_error = float(np.max(np.abs(l3[0][off] - l10[0][off])))
    initialization_attribute_error = float(np.max(np.abs(l10[0][off] - np.asarray(infer10(params, jnp.asarray(stripped)))[0][off])))
    if mapping_error > 1e-4 or initialization_attribute_error > 1e-6:
        raise ValueError('Actual original logits equivalence failed')
    dump(a.out / 'INPUT_AND_TEACHER_QUALIFICATION.json', {'features': features.tolist(), 'fields': FIELDS,
        'units': UNITS, 'contract': contract.payload(), 'contract_hash': contract.identity(), 'teacher': teacher,
        'teacher_full_C0_report': target_report, 'incumbent_full_C0_report': before_report,
        'pair_sign': label, 'feature_rejections': rejection, 'mapping_max_logit_error': mapping_error,
        'initial_attribute_removal_max_logit_error': initialization_attribute_error,
        'original_inventory': original_inventory, 'mapped_inventory': initial_inventory,
        'mapped_parameter_elements': 76259840 + 7 * 512,
        'physical_calibration_claim': False, 'dynamic_online_qualified': False})
    del original_params, infer3
    gc.collect()
    def loss(p, coefficient):
        return objective_terms(f10(p, jnp.asarray(features)), target, alternative, label, coefficient)
    grad_fn = jax.jit(jax.value_and_grad(loss, has_aux=True))
    arm_results = {}
    for name, coefficient in (('reconstruction_control', 0.), ('paired_objective', .2)):
        (total, components), grads = grad_fn(params, jnp.asarray(coefficient, jnp.float32))
        # Every arm updates complete model from the exact same mapped original.
        updated = jax.tree.map(lambda p, g: p - .00001 * g, params, grads)
        stats = {}
        for group in ('full', 'encoder', 'decoder'):
            entries = [(path, np.asarray(v.value)) for path, v in grads.flat_state()
                       if group == 'full' or path[0] == group]
            stats[group] = {'leaves': len(entries), 'finite': sum(bool(np.isfinite(v).all()) for _, v in entries),
                            'nonzero': sum(bool(np.any(v != 0)) for _, v in entries)}
            if stats[group]['finite'] != len(entries) or stats[group]['nonzero'] == 0:
                raise ValueError('Actual full gradient failed: ' + group)
        extra_gradient = float(np.linalg.norm(np.asarray(grads['init_proj']['kernel'].value)[3:]))
        logits = np.asarray(infer10(updated, jnp.asarray(features)))[0]
        stripped_logits = np.asarray(infer10(updated, jnp.asarray(stripped)))[0]
        attribute_effect = float(np.max(np.abs(logits[off] - stripped_logits[off])))
        candidate, remaining, insertions, status = reconstruct(data, teacher['incumbent_routes_before'],
            contract, removed, observed, logits, 1., float('inf'), time.perf_counter)
        projected = evaluate(data, candidate, contract)
        preserved = bool(np.all(partial_adjacency(candidate, set(), nodes)[observed > 0] > 0))
        checkpoint = a.out / (name + '_ENGINEERING_ONLY.pkl')
        with checkpoint.open('xb') as stream:
            pickle.dump({'params': updated, 'model_config': cfg10, 'research_model_eligible': False}, stream)
        with checkpoint.open('rb') as stream:
            reload_params = pickle.load(stream)['params']
        reload_equal = bool(np.array_equal(logits, np.asarray(infer10(reload_params, jnp.asarray(features)))[0]))
        updated_inventory = inventory(updated)
        changes = {group: sum(updated_inventory[p] != initial_inventory[p] for p in initial_inventory
                             if group == 'full' or p.startswith("('" + group + "',"))
                   for group in ('full', 'encoder', 'decoder')}
        arm_results[name] = {'objective_coefficient': coefficient, 'fixed_steps': 1,
            'same_initial_state_digest': digest(initial_inventory), 'loss_total': float(total),
            'loss_CE': float(components[0]), 'loss_preference': float(components[1]),
            'score_delta': float(components[2]), 'gradient_stats': stats,
            'extra_feature_projection_gradient_norm': extra_gradient, 'parameter_change_counts': changes,
            'attribute_removal_max_logit_effect_after_step': attribute_effect,
            'routes': candidate, 'projection': projected, 'remaining': remaining, 'status': status,
            'retained_edges_preserved': preserved, 'save_sha256': sha(checkpoint),
            'reload_logits_exact': reload_equal, 'parameter_inventory': updated_inventory}
        dump(a.out / (name + '_QA.json'), arm_results[name])
        del grads, updated, reload_params
        gc.collect()
    end_sources = sources(a.root, a.upstream)
    qa = {'version': required['version'], 'registration_sha256': sha(a.registration),
        'original_checkpoint_sha256': checkpoint_sha, 'fields': FIELDS, 'arms': arm_results,
        'mapping_max_logit_error': mapping_error,
        'source_start': start_sources, 'source_end': end_sources,
        'source_unchanged': start_sources == end_sources,
        'checkpoint_unchanged': sha(a.checkpoint) == checkpoint_sha,
        'wall_s': time.perf_counter() - start, 'research_model_eligible': False,
        'dynamic_online_qualified': False, 'stable_gain_established': False,
        'physical_calibration_claim': False}
    qa['engineering_pass'] = (start_sources == end_sources and qa['checkpoint_unchanged']
        and all(x['projection']['hard'] is True and x['projection']['complete'] is True
                and x['retained_edges_preserved'] and not x['remaining'] and x['reload_logits_exact']
                and x['extra_feature_projection_gradient_norm'] > 0
                and all(x['parameter_change_counts'][g] > 0 for g in ('encoder', 'decoder'))
                for x in arm_results.values())
        and digest(arm_results['paired_objective']['parameter_inventory'])
            != digest(arm_results['reconstruction_control']['parameter_inventory']))
    dump(a.out / 'FEATURES_OBJECTIVE_QA.json', qa)
    print(json.dumps({'engineering_pass': qa['engineering_pass'], 'wall_s': qa['wall_s']}), flush=True)
    if not qa['engineering_pass']:
        raise ValueError('Paired feature engineering not qualified')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    for key in ('root', 'upstream', 'checkpoint', 'registration', 'out'):
        ap.add_argument('--' + key, type=Path, required=True)
    ap.add_argument('--cpu', type=int, required=True)
    args = ap.parse_args()
    try:
        run(args)
    except Exception as error:
        if args.out.is_dir() and not (args.out / 'ENGINEERING_FAILURE.json').exists():
            dump(args.out / 'ENGINEERING_FAILURE.json', {'exception': repr(error),
                 'traceback': traceback.format_exc(), 'engineering_pass': False,
                 'research_model_eligible': False})
        raise
