"""D-INFERIDENT-v1: read-only inference identity and positive controls for the delivery adapter.

Why this is a prerequisite, not a result. The delivery line's one untested path is training a
delivery-specific reconstruction component. Before any training can be interpreted, the frozen
zero-shot path it is to be compared against must be shown to be a well-defined, reproducible
function whose mask channel actually reaches the output. Three failure modes would invalidate
every number downstream:

  * non-determinism  - the same input yields different logits across calls or processes;
  * inert mask       - the decoder output does not depend on the supplied adjacency, so any
                       "masked reconstruction" comparison would be measuring nothing;
  * live weights     - inference mutates the parameters, so the "frozen" reference drifts.

Therefore this module runs MANDATORY positive controls that must FAIL to be safe: if changing
the mask does not change the logits, or if a random same-architecture model produces the same
logits as the real checkpoint, the harness reports that as a defect rather than passing.

Scope: engineering qualification of inference only. No route is improved, no arm is compared,
no online gain, no MaskCO attribution, no impossibility claim. No research data is generated or
consumed; the instance is a fixed synthetic shape.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


def bootstrap(scripts):
    scripts = Path(scripts).resolve()
    for sub in ('delivery', 'research_20261010'):
        sys.path.insert(0, str(scripts / sub))
    return scripts


def arr_hash(a):
    a = np.ascontiguousarray(np.asarray(a))
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(str(a.dtype).encode())
    h.update(a.tobytes())
    return h.hexdigest()


def tree_hash(obj):
    import jax
    h = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(obj):
        h.update(arr_hash(leaf).encode())
    return h.hexdigest()


def main(args):
    started = time.perf_counter()
    bootstrap(args.scripts)
    from delivery_maskco import MaskCORouteProvider, partial_adjacency
    from delivery_equal_schedule import generate
    import jax
    from flax import nnx

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    checks = {}

    def record(name, ok, **extra):
        checks[name] = dict(passed=bool(ok), **extra)
        return bool(ok)

    instance, initial_routes = generate(args.seed, args.instance)
    n = len(instance['demands'])
    capacity = 50.0

    provider = MaskCORouteProvider(checkpoint=args.checkpoint, upstream_root=args.upstream_root)
    provider.prewarm(n, capacity)

    # --- read-only guarantee: parameters must not move ------------------------------
    params_before = tree_hash(nnx.state(provider.model, nnx.Param))
    feats = provider.features(instance, capacity)
    if hasattr(feats, 'block_until_ready'):
        feats.block_until_ready()

    # --- determinism -----------------------------------------------------------------
    feats2 = provider.features(instance, capacity)
    if hasattr(feats2, 'block_until_ready'):
        feats2.block_until_ready()
    record('features_deterministic_within_process',
           arr_hash(feats) == arr_hash(feats2), sha=arr_hash(feats))

    chain = np.zeros((n, n), np.float32)
    for route in initial_routes:
        seq = [0] + list(route) + [0]
        for a, b in zip(seq, seq[1:]):
            if a != b:
                chain[a, b] = chain[b, a] = 1.0

    logits_a = provider.logits(feats, chain)
    logits_a2 = provider.logits(feats, chain)
    record('logits_deterministic_within_process',
           arr_hash(logits_a) == arr_hash(logits_a2), sha=arr_hash(logits_a))

    # --- POSITIVE CONTROL: the mask channel must reach the output --------------------
    removed = set(int(x) for x in np.linspace(1, n - 1, max(1, n // 3)).astype(int))
    partial = partial_adjacency(initial_routes, removed, n)
    logits_b = provider.logits(feats, partial)
    delta_mask = float(np.max(np.abs(logits_a - logits_b)))
    record('POSITIVE_control_mask_changes_logits',
           arr_hash(logits_a) != arr_hash(logits_b) and delta_mask > 0.0,
           max_abs_delta=delta_mask, sha_partial=arr_hash(logits_b))

    empty = np.zeros((n, n), np.float32)
    logits_empty = provider.logits(feats, empty)
    record('POSITIVE_control_empty_vs_full_chain_differs',
           arr_hash(logits_empty) != arr_hash(logits_a),
           max_abs_delta=float(np.max(np.abs(logits_a - logits_empty))))

    # --- POSITIVE CONTROL: the checkpoint must matter --------------------------------
    rnd = MaskCORouteProvider(checkpoint=args.checkpoint, upstream_root=args.upstream_root,
                              random_seed=args.random_seed)
    feats_r = rnd.features(instance, capacity)
    if hasattr(feats_r, 'block_until_ready'):
        feats_r.block_until_ready()
    logits_r = rnd.logits(feats_r, chain)
    record('POSITIVE_control_checkpoint_changes_logits',
           arr_hash(logits_r) != arr_hash(logits_a),
           max_abs_delta=float(np.max(np.abs(logits_a - logits_r))),
           random_same_architecture=True)

    # --- retained-edge contract ------------------------------------------------------
    bad = 0
    for a in range(n):
        for b in range(n):
            if partial[a, b] > 0 and (a in removed or b in removed):
                bad += 1
    expected = sum(1 for route in initial_routes
                   for a, b in zip([0] + list(route), list(route) + [0])
                   if a != b and a not in removed and b not in removed)
    record('partial_adjacency_never_touches_removed_nodes',
           bad == 0, violating_entries=bad)
    record('partial_adjacency_edge_count_matches_rule',
           int(partial.sum()) == 2 * expected, stored=float(partial.sum()),
           expected_undirected=expected)
    restored = partial_adjacency([r for r in initial_routes], set(), n)
    record('unmasked_edges_are_exactly_the_original_chain',
           bool(np.array_equal(restored > 0, chain > 0)))

    # --- diagnostic, NOT an assertion: node-reindex covariance ------------------------
    perm = np.arange(n)
    rng = np.random.default_rng(args.seed)
    perm[1:] = rng.permutation(np.arange(1, n))
    inv = np.argsort(perm)
    coords_p = np.asarray(instance['coords'], np.float32)[perm]
    demands_p = np.asarray(instance['demands'], np.float32)[perm]
    inst_p = dict(instance)
    inst_p['coords'], inst_p['demands'] = coords_p, demands_p
    feats_p = provider.features(inst_p, capacity)
    if hasattr(feats_p, 'block_until_ready'):
        feats_p.block_until_ready()
    logits_p = provider.logits(feats_p, chain[np.ix_(perm, perm)])
    relabel_err = float(np.max(np.abs(logits_p - logits_a[np.ix_(perm, perm)])))
    checks['DIAGNOSTIC_node_reindex_equivariance_max_abs_error'] = dict(
        passed=None, max_abs_error=relabel_err,
        note='reported, not asserted: the adapter normalises coordinates internally, so exact '
             'equivariance is not expected and is not required by any registered claim')

    # --- read-only guarantee, after all inference ------------------------------------
    params_after = tree_hash(nnx.state(provider.model, nnx.Param))
    record('inference_does_not_mutate_parameters',
           params_before == params_after, params_before=params_before,
           params_after=params_after)

    failed = sorted(k for k, v in checks.items() if v['passed'] is False)
    result = {
        'kind': 'delivery_readonly_inference_identity_qualification',
        'protocol_version': 'D-INFERIDENT-v1',
        'checkpoint': str(args.checkpoint),
        'checkpoint_sha256': hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        'upstream_root': str(args.upstream_root),
        'synthetic_instance': {'seed': args.seed, 'instance': args.instance, 'nodes': n,
                               'capacity': capacity},
        'both_encoder_and_decoder_used': True,
        'n_checks': len(checks), 'n_failed': len(failed), 'failed': failed,
        'all_asserted_checks_pass': not failed,
        'checks': checks,
        'logits_sha256_full_chain': arr_hash(logits_a),
        'features_sha256': arr_hash(feats),
        'adapter_identity': dict(provider.identity),
        'jax_backend': jax.default_backend(),
        'what_this_qualifies': [
            'the frozen zero-shot delivery adapter is a deterministic function within a process',
            'the supplied adjacency demonstrably reaches the decoder output (positive control), '
            'so a masked-reconstruction comparison is measuring a live channel',
            'the real checkpoint demonstrably differs from a random same-architecture model',
            'inference leaves the parameters bit-identical (read-only)'],
        'what_this_does_not_qualify': [
            'cross-process determinism is NOT established by this run alone; the logits digest '
            'must be compared against a second independent process',
            'no route is improved, no arm is compared, no online gain and no MaskCO attribution',
            'physics/objective layers are untouched and uncalibrated here',
            'this says nothing about whether a TRAINED component could help'],
        'online_gain_established': False, 'maskco_attributable': False,
        'impossibility_proven': False,
        'wall_s': time.perf_counter() - started,
    }
    (out / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False, default=str),
                                     encoding='utf-8')
    print(json.dumps({k: result[k] for k in
                      ('n_checks', 'n_failed', 'failed', 'all_asserted_checks_pass',
                       'logits_sha256_full_chain', 'jax_backend', 'wall_s')}, indent=2,
                     default=str))
    return 0 if not failed else 3


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scripts', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--upstream-root', required=True)
    ap.add_argument('--seed', type=int, default=2026120151)
    ap.add_argument('--instance', type=int, default=0)
    ap.add_argument('--random-seed', type=int, default=42)
    sys.exit(main(ap.parse_args()))
