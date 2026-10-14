"""One actual 76M-parameter masked-route SGD step, engineering only.

Reuses a pinned OLD TRAIN fixture solely to verify the complete path. This
artifact is permanently ineligible as a research model or held-out estimate.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import time
import traceback


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def strict(value, expected):
    return type(value) is type(expected) and value == expected


def sources(root, upstream):
    paths = []
    for group in ('models', 'modules', 'helpers'):
        paths.extend(sorted((upstream / group).rglob('*.py')))
    for group in ('delivery', 'coldchain'):
        paths.extend(sorted((root / 'scripts' / group).rglob('*.py')))
    paths += [root / 'scripts/research_20261010/delivery_equal_schedule.py']
    return {'upstream/' + str(p.relative_to(upstream)).replace('\\', '/') if p.is_relative_to(upstream)
            else 'extension/' + str(p.relative_to(root)).replace('\\', '/'): sha(p) for p in paths}


def run(args):
    os.environ['JAX_PLATFORMS'] = 'cpu'
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        os.environ[name] = '1'
    import numpy as np
    import psutil
    available = psutil.virtual_memory().available
    reg = json.loads(args.registration.read_text(encoding='utf-8'))
    required = {'version': 'D-ACTUAL-FULLSTEP-ENGINEERING-v1', 'stage': 'actual_checkpoint_engineering_only',
        'fixed_steps': 1, 'optimizer': 'sgd_without_momentum', 'learning_rate': .00001,
        'research_model_eligible': False, 'research_gain_claim': False, 'teacher_first_certified_ordinal': 0}
    if not all(strict(reg.get(k), v) for k, v in required.items()):
        raise ValueError('Registered engineering settings changed or wrong type')
    if type(reg.get('minimum_available_ram_gib')) is not int or reg['minimum_available_ram_gib'] < 8:
        raise ValueError('Explicit safe RAM floor required')
    args.out.mkdir(parents=True, exist_ok=False)
    dump(args.out / 'RESOURCE_PREFLIGHT.json', {'available_ram_bytes': available,
         'minimum_available_ram_gib': reg['minimum_available_ram_gib'],
         'registration_sha256': sha(args.registration)})
    if available < reg['minimum_available_ram_gib'] * 1024 ** 3:
        raise RuntimeError('Resource gate: insufficient free RAM for actual full architecture')
    process = psutil.Process()
    permitted = process.cpu_affinity()
    cpu = args.cpu if args.cpu is not None else permitted[-1]
    if cpu not in permitted:
        raise ValueError('Requested isolated CPU unavailable')
    process.cpu_affinity([cpu])
    start = time.perf_counter()
    dump(args.out / 'RESOURCE_START.json', {'available_ram_bytes': available, 'cpu_affinity': process.cpu_affinity(),
         'jax_backend_requested': 'cpu', 'registration_sha256': sha(args.registration)})
    source_start = sources(args.root, args.upstream)
    for name, expected in reg['engineering_scripts'].items():
        if sha(Path(__file__).resolve().parent / name) != expected:
            raise ValueError('Actual engineering program source changed: ' + name)
    if source_start != reg['sources']:
        raise ValueError('Frozen actual source manifest mismatch')
    split = args.root / reg['teacher_split']
    for name, expected in reg['teacher_files'].items():
        if sha(split / name) != expected:
            raise ValueError('Original TRAIN fixture changed: ' + name)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from inspect_checkpoint import load_trusted, inspect
    saved, checkpoint_sha = load_trusted(args.checkpoint, args.upstream)
    original_inspection = inspect(saved, checkpoint_sha)
    if original_inspection['parameter_elements'] != 76259840 or original_inspection['parameter_leaves'] != 234:
        raise ValueError('Not the actual registered original full architecture')
    cfg = deepcopy(saved['model_config'])
    expected_cfg = reg['model_config']
    if digest(vars(cfg)) != digest(expected_cfg):
        raise ValueError('Model configuration identity changed')
    cfg.dtype = 'float32'  # Same arithmetic choice as the existing inference adapter.
    dump(args.out / 'ACTUAL_MODEL_INSPECTION.json', original_inspection)
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from helpers import sol2adj
    from modules.functional import coord_normalize
    for folder in ('delivery', 'coldchain', 'research_20261010'):
        sys.path.insert(0, str(args.root / 'scripts' / folder))
    from delivery_equal_schedule import generate, reconstruct
    from delivery_maskco import partial_adjacency
    from delivery_contract import DeliveryContract
    from delivery_evaluator import evaluate
    identity = json.loads((split / 'TRAIN_SPLIT_IDENTITY.json').read_text(encoding='utf-8'))
    teacher = next(json.loads(line) for line in (split / 'samples.jsonl').open(encoding='utf-8')
                   if json.loads(line)['outcome'] == 'certified')
    # Regenerate the OLD fixture, never a new candidate research data split.
    data, _ = generate(identity['train_seed'], teacher['instance'])
    contract = DeliveryContract()
    target_routes = teacher['candidate_routes']
    target_report = evaluate(data, target_routes, contract)
    if target_report['hard'] is not True or target_report['complete'] is not True:
        raise ValueError('Old teacher full delivery projection not hard feasible')
    if abs(target_report['cost'] - teacher['cost_after_candidate']) > 1e-8:
        raise ValueError('Independent old teacher C0 cost replay mismatch')
    nodes = len(data['demands'])
    removed = set(teacher['mask'])
    observed = partial_adjacency(teacher['incumbent_routes_before'], removed, nodes)
    target_all_edges = partial_adjacency(target_routes, set(), nodes)
    if not np.all(target_all_edges[observed > 0] > 0):
        raise ValueError('Teacher does not preserve the actual supplied retained context')
    vector = [0]
    for route in target_routes:
        vector.extend(route + [0])
    target = sol2adj(jnp.asarray([vector], jnp.int32), dtype=jnp.float32, num_nodes=nodes, is_cvrp=True)
    raw = jnp.concatenate((coord_normalize(jnp.asarray(data['coords'], jnp.float32)[None]),
          jnp.asarray(data['demands'], jnp.float32)[None, :, None] / contract.capacity), axis=-1)
    timestep = jnp.asarray([observed[1:].sum() / (2 * (nodes - 1))], jnp.float32)
    context = jnp.asarray(observed[None], jnp.float32)
    dump(args.out / 'ACTUAL_TEACHER_FIXTURE.json', {'data': data, 'teacher': teacher,
        'teacher_report_independent_replay': target_report, 'retained_adjacency': observed.tolist(),
        'raw_features': np.asarray(raw).tolist(), 'raw_feature_sha256': hashlib.sha256(np.asarray(raw).tobytes()).hexdigest(),
        'target_adjacency': np.asarray(target).tolist(), 'all_nodes_revealed': True,
        'old_fixture_is_not_heldout_data': True, 'extended_coldchain_features_qualified': False})
    constructed = cfg.construct_model()
    graph = nnx.graphdef(constructed)
    # Record same architecture random-initialization identity before loading P.
    random_params = nnx.state(constructed, nnx.Param)
    def inventory(params):
        group_hash = {g: hashlib.sha256() for g in ('full', 'encoder', 'decoder')}
        leaves = []
        for path, value in params.flat_state():
            a = np.asarray(value.value)
            raw_digest = hashlib.sha256(a.tobytes()).hexdigest()
            leaves.append({'path': list(path), 'shape': list(a.shape), 'sha256': raw_digest})
            for group in group_hash:
                if group == 'full' or path[0] == group:
                    group_hash[group].update(repr(path).encode() + str(a.shape).encode() + str(a.dtype).encode() + a.tobytes())
        return {'group_sha256': {g: h.hexdigest() for g, h in group_hash.items()}, 'leaves': leaves}
    random_identity = inventory(random_params)
    params = saved['params']
    before = inventory(params)
    if before['group_sha256']['full'] == random_identity['group_sha256']['full']:
        raise ValueError('Checkpoint is indistinguishable from configuration initialization')
    del constructed, random_params, saved
    import gc
    gc.collect()
    def forward(p, adjacency):
        model = nnx.merge(graph, p)
        return model.decode(model.encode(raw), timestep, adjacency)
    infer = jax.jit(forward)
    old_logits = np.asarray(infer(params, context))[0]
    empty_logits = np.asarray(infer(params, jnp.zeros_like(context)))[0]
    off = ~np.eye(nodes, dtype=bool)
    mask_effect = float(np.max(np.abs(old_logits[off] - empty_logits[off])))
    def loss(p):
        logits = forward(p, context)
        logp = jax.nn.log_softmax(logits, axis=-1)
        return -(logp[:, 1:] * target[:, 1:]).mean() * (nodes / 2)
    value_and_grad = jax.jit(jax.value_and_grad(loss))
    gradient_start = time.perf_counter()
    value, grads = value_and_grad(params)
    loss_before = float(value)
    gradient_time = time.perf_counter() - gradient_start
    grad_stats = {g: {'l2_squared': 0., 'leaves_nonzero': 0, 'leaves_finite': 0, 'leaves': 0}
                  for g in ('full', 'encoder', 'decoder')}
    for path, value in grads.flat_state():
        grad = np.asarray(value.value)
        for group in grad_stats:
            if group == 'full' or path[0] == group:
                grad_stats[group]['l2_squared'] += float(np.square(grad.astype(np.float64)).sum())
                grad_stats[group]['leaves_nonzero'] += int(np.any(grad != 0))
                grad_stats[group]['leaves_finite'] += int(np.all(np.isfinite(grad)))
                grad_stats[group]['leaves'] += 1
    dump(args.out / 'ACTUAL_GRADIENT_QUALIFICATION.json', {'loss_before': loss_before,
         'groups': grad_stats, 'actual_backward_including_compile_s': gradient_time,
         'actual_mask_context_max_logit_effect': mask_effect})
    if any(s['l2_squared'] <= 0 or s['leaves_finite'] != s['leaves'] for s in grad_stats.values()):
        raise ValueError('Complete encoder/decoder gradient qualification failed')
    new_params = jax.tree.map(lambda p, g: p - reg['learning_rate'] * g, params, grads)
    del grads
    gc.collect()
    after = inventory(new_params)
    loss_after = float(jax.jit(loss)(new_params))
    updated_logits = np.asarray(infer(new_params, context))[0]
    finite_logits = bool(np.all(np.isfinite(updated_logits[off])))
    # Actual full model -> original C0 certified reconstruction projection.
    projection_start = time.perf_counter()
    candidate, remaining, insertions, status = reconstruct(data, teacher['incumbent_routes_before'],
        contract, removed, observed, updated_logits, 1., float('inf'), time.perf_counter)
    candidate_report = evaluate(data, candidate, contract)
    preserved = bool(np.all(partial_adjacency(candidate, set(), nodes)[observed > 0] > 0))
    projection_ok = not remaining and status == 'complete' and candidate_report['hard'] is True and preserved
    dump(args.out / 'ACTUAL_PROJECTED_CANDIDATE.json', {'routes': candidate, 'remaining': remaining,
        'insertions': insertions, 'status': status, 'report': candidate_report,
        'retained_edges_preserved': preserved, 'wall_s': time.perf_counter() - projection_start,
        'total10s_budget_qualified': False, 'no_gain_adjudication': True})
    checkpoint_out = args.out / 'ENGINEERING_ONLY_FULL_FINAL.pkl'
    with checkpoint_out.open('xb') as stream:
        pickle.dump({'params': new_params, 'model_config': cfg, 'engineering_fixed_step': 1,
                     'research_model_eligible': False}, stream)
    out_sha = sha(checkpoint_out)
    with checkpoint_out.open('rb') as stream:
        reloaded = pickle.load(stream)
    reloaded_logits = np.asarray(infer(reloaded['params'], context))[0]
    reload_equal = bool(np.array_equal(updated_logits, reloaded_logits))
    source_end = sources(args.root, args.upstream)
    qa = {'stage': required['stage'], 'registration_sha256': sha(args.registration),
        'original_checkpoint_sha256': checkpoint_sha, 'original_checkpoint_step_metadata': original_inspection['checkpoint_step'],
        'original_parameters_distinct_from_configuration_init': True,
        'model_config_loaded': original_inspection['model_config'], 'runtime_dtype': 'float32',
        'actual_parameter_elements': original_inspection['parameter_elements'], 'actual_parameter_leaves': 234,
        'trained_fixed_steps': 1, 'training_scope': 'all_234_leaves_not_head_only',
        'random_configuration_initialization_identity': random_identity,
        'random_control_note': 'same_architecture_rng888_configuration_identity_only_not_random42_panel',
        'reconstruction_control_note': 'this_is_reconstruction_only_no_preference_term; preference_vs_CE_pair_not_run',
        'before': before, 'after': after, 'gradient_groups': grad_stats,
        'loss_before': loss_before, 'loss_after': loss_after,
        'loss_is_engineering_surrogate_not_gain': True,
        'teacher_complete_independent_replay': True, 'projection_complete_hard_preserved': projection_ok,
        'actual_mask_context_effect': mask_effect, 'finite_actual_logits': finite_logits,
        'saved_sha256': out_sha, 'saved_reload_logits_equal': reload_equal,
        'source_start': source_start, 'source_end': source_end,
        'source_unchanged': source_start == source_end,
        'checkpoint_input_unchanged': sha(args.checkpoint) == checkpoint_sha,
        'total_wall_s_including_compile_and_training': time.perf_counter() - start,
        'engineering_pass': projection_ok and finite_logits and reload_equal and mask_effect > 0
            and all(before['group_sha256'][g] != after['group_sha256'][g] for g in ('encoder', 'decoder'))
            and source_start == source_end and sha(args.checkpoint) == checkpoint_sha,
        'research_model_eligible': False, 'full_scientific_training_qualified': False,
        'stable_online_gain_established': False, 'extended10d_features_qualified': False,
        'total10s_full_path_qualified': False,
        'physical_calibration_claim': False}
    dump(args.out / 'ACTUAL_FULLSTEP_QA.json', qa)
    print(json.dumps({k: qa[k] for k in ('engineering_pass', 'actual_parameter_elements', 'trained_fixed_steps', 'saved_reload_logits_equal')}), flush=True)
    if not qa['engineering_pass']:
        raise RuntimeError('Actual full architecture engineering qualification failed; preserve all records')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--upstream', type=Path, required=True)
    ap.add_argument('--checkpoint', type=Path, required=True)
    ap.add_argument('--registration', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--cpu', type=int)
    args = ap.parse_args()
    try:
        run(args)
    except Exception as error:
        if args.out.is_dir() and not (args.out / 'ENGINEERING_FAILURE.json').exists():
            dump(args.out / 'ENGINEERING_FAILURE.json', {'exception': repr(error), 'traceback': traceback.format_exc(),
                 'engineering_pass': False, 'research_model_eligible': False})
        raise
