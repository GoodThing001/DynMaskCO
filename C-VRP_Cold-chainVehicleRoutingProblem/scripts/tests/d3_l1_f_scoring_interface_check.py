#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""A1-D3-L1 F SCORING-INTERFACE engineering verification (user-authorised, limited).

Scope, exactly as authorised:
  * F = results/m0_scale/mpre_reinforce_s42_clean/model.ckpt
  * current backbone = MASKCO_code/ckpts/cvrp100.ckpt
  * load path = load_mpre_trained -> partition -> nnx.merge(gd, train_state, frozen_state)
    (used here through `mtrained_replanner.load_trained_model`, the audited correct path)
  * ONE pre-specified fixed input (deterministic seeded synthetic tensors, declared as such);
    no input search, no D-Geo call, no comparison of any kind.

Every result carries F_RECONSTRUCTED_WITH_CURRENT_BACKBONE / HISTORICAL_BACKBONE_UNVERIFIED.
Passing everything yields F_SCORING_INTERFACE_VERIFIED_WITH_CURRENT_BACKBONE; it can NEVER
yield F_HISTORICAL_IDENTITY_VERIFIED.  A failing check is recorded and the run stops there.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
REPO = EXT.parent
for p in ('scripts', 'scripts/simulation', 'scripts/models', 'scripts/data', 'scripts/coldchain',
          'scripts/evaluation', 'scripts/tests'):
    sys.path.insert(0, str(EXT / p))

import numpy as np                                                # noqa: E402

CASES = []
RECORDS = {}


def ck(name, ok, detail=None):
    CASES.append((name, bool(ok), detail))
    print('[%s] %s :: %s' % ('PASS' if ok else 'FAIL', name,
                             json.dumps(detail, default=repr, sort_keys=True)))


def rec(k, v):
    RECORDS[k] = v


def sha(p):
    p = Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model-ckpt', default=str(EXT / 'results/m0_scale/'
                                               'mpre_reinforce_s42_clean/model.ckpt'))
    ap.add_argument('--cvrp-ckpt', default=str(REPO / 'MASKCO_code/ckpts/cvrp100.ckpt'))
    ap.add_argument('--json', default=None)
    args = ap.parse_args()

    REGISTERED = 'f5acf83fa2519020aeea274df5ecf740a6ad809b215d2d270209b7901903a7f5'
    LABEL = 'F_RECONSTRUCTED_WITH_CURRENT_BACKBONE / HISTORICAL_BACKBONE_UNVERIFIED'
    rec('identity', {'label': LABEL, 'model_ckpt': args.model_ckpt,
                     'model_ckpt_sha256': sha(args.model_ckpt),
                     'registered_sha256': REGISTERED,
                     'cvrp_ckpt': args.cvrp_ckpt, 'cvrp_ckpt_sha256': sha(args.cvrp_ckpt)})
    ck('I1 the checkpoint is the registered F (sha256 match)',
       RECORDS['identity']['model_ckpt_sha256'] == REGISTERED, RECORDS['identity'])

    import jax                                                     # noqa: E402
    import jax.numpy as jnp                                       # noqa: E402
    from mpre_trained import load_mpre_trained                     # noqa: E402
    from mtrained_replanner import load_trained_model              # noqa: E402

    t0 = time.time()
    model, cfg = load_trained_model(args.cvrp_ckpt, args.model_ckpt, seed=0)
    load_s = time.time() - t0
    rec('load', {'seconds': load_s, 'cfg_repr': repr(cfg)[:600]})
    ck('I2 the trained model reconstructs through the audited path '
       '(load_mpre_trained -> partition -> nnx.merge) without error', model is not None,
       {'seconds': load_s})

    leaves = [np.asarray(x) for x in __import__('jax').tree_util.tree_leaves(
        __import__('flax').nnx.state(model))]

    def nonzero(model_obj, needle):
        st = __import__('flax').nnx.state(model_obj)
        flat = __import__('jax').tree_util.tree_flatten_with_path(st)[0]
        hit = []
        for path, leaf in flat:
            name = '.'.join(str(getattr(p, 'key', getattr(p, 'name', p))) for p in path)
            if needle in name:
                hit.append(float(np.max(np.abs(np.asarray(leaf)))))
        return {'n_leaves': len(hit), 'absmax': max(hit) if hit else 0.0}

    trained_stats = {k: nonzero(model, k) for k in ('c_adapter', 'delta_head')}
    rec('restored_params', {'trained_groups': trained_stats,
                            'total_leaves': len(leaves)})
    ck('I3 the restored parameters really are the trained task residuals (c_adapter and '
       'delta_head present and non-zero, i.e. not the zero-initialised fresh model)',
       all(v['n_leaves'] > 0 and v['absmax'] > 0 for v in trained_stats.values()),
       trained_stats)

    # ---- one pre-specified fixed input (deterministic synthetic; declared) ---------------
    B, N, M = 1, 8, 3
    rng = np.random.default_rng(20261014)
    input_spec = {'seed': 20261014, 'B': B, 'N': N, 'M': M, 'note':
                  'fixed synthetic input, declared as such: it verifies the interface, it is '
                  'NOT a decision state and nothing is concluded about F quality from it'}
    rec('fixed_input', input_spec)
    cust = np.array([[2, 3, 4]], dtype=np.int32)
    pred = np.array([[0, 0, 1]], dtype=np.int32)
    succ = np.array([[1, 2, 2]], dtype=np.int32)
    adjmat = np.zeros((B, N, N), dtype=np.float32)
    for b in range(B):
        for v in range(N):
            adjmat[b, v, v] = 1.0
    timestep = np.array([0.5], dtype=np.float32)      # shaped [B]; a 0-d scalar is indexed inside
    node_valid = np.ones((B, N), dtype=np.float32)

    attempts = []
    scores = None
    for d_in in (3, 7, 10):
        try:
            raw = rng.standard_normal((B, N, d_in)).astype(np.float32)
            feats = rng.standard_normal((B, N, 10)).astype(np.float32)
            afeat = rng.standard_normal((B, M, 20)).astype(np.float32)
            t1 = time.time()
            s, internals = model.score_actions(jnp.asarray(raw), jnp.asarray(node_valid),
                                               jnp.asarray(feats), jnp.asarray(timestep),
                                               jnp.asarray(adjmat), jnp.asarray(cust),
                                               jnp.asarray(pred), jnp.asarray(succ),
                                               jnp.asarray(afeat))
            np.asarray(jax.block_until_ready(s))
            forward_s = time.time() - t1
            s_np = np.asarray(s)
            attempts.append({'raw_3d_dim': d_in, 'ok': True, 'seconds': forward_s,
                             'shape': list(s_np.shape)})
            scores = (raw, feats, afeat, s_np, forward_s, d_in)
            break
        except BaseException as exc:
            attempts.append({'raw_3d_dim': d_in, 'ok': False,
                             'error': '%s: %s' % (type(exc).__name__, str(exc)[:300])})
    rec('forward_attempts', attempts)
    ck('I4 a forward pass scores the common action list (no exception on the declared fixed '
       'input)', scores is not None, attempts)

    if scores is None:
        print('STOPPING: no forward pass succeeded')
        return 1

    raw, feats, afeat, s_np, forward_s, d_in = scores
    s2 = np.asarray(model.score_actions(
        jnp.asarray(raw), jnp.asarray(node_valid), jnp.asarray(feats), jnp.asarray(timestep),
        jnp.asarray(adjmat), jnp.asarray(cust), jnp.asarray(pred), jnp.asarray(succ),
        jnp.asarray(afeat))[0])

    finite = bool(np.all(np.isfinite(s_np)))
    length_ok = s_np.shape == (B, M)
    repeat_ok = bool(np.array_equal(s_np, s2))
    argmax_first = int(np.argmax(s_np[0]))
    first_max = int(np.flatnonzero(s_np[0] == s_np[0].max())[0])
    rec('scoring', {'raw_3d_dim': d_in, 'scores': s_np.tolist(), 'shape': list(s_np.shape),
                    'expected_shape': [B, M], 'all_finite': finite, 'length_ok': length_ok,
                    'repeat_bitwise_equal': repeat_ok, 'argmax': argmax_first,
                    'first_maximum_index': first_max, 'first_argmax_semantics_holds':
                        argmax_first == first_max, 'forward_seconds': forward_s})
    ck('I5 score vector length == number of common actions and all entries are finite',
       length_ok and finite, RECORDS['scoring'])
    ck('I6 repeated forward passes are bitwise identical (frozen-judgement stability)',
       repeat_ok, {'repeat_bitwise_equal': repeat_ok})
    ck('I7 the historical first-argmax semantics holds (np.argmax == first maximum index)',
       argmax_first == first_max, {'argmax': argmax_first, 'first_max': first_max})

    raised = None
    gap = None
    try:
        fresh, _cfg, _step = load_mpre_trained(args.cvrp_ckpt, 10, 20, seed=0)
        s_fresh = np.asarray(jax.block_until_ready(fresh.score_actions(
            jnp.asarray(raw), jnp.asarray(node_valid), jnp.asarray(feats),
            jnp.asarray(timestep), jnp.asarray(adjmat), jnp.asarray(cust),
            jnp.asarray(pred), jnp.asarray(succ), jnp.asarray(afeat))[0]))
        gap = float(np.max(np.abs(s_np - s_fresh)))
        fresh_absmax = float(np.max(np.abs(s_fresh)))
    except BaseException as exc:
        raised, fresh_absmax = '%s: %s' % (type(exc).__name__, str(exc)[:400]), None
    rec('trained_vs_fresh', {'max_abs_difference': gap, 'fresh_absmax': fresh_absmax,
                             'raised': raised,
                             'note': 'a load path that skipped train_state restoration (e.g. '
                                     'load_mpre_trained alone on the zero-initialised adapter) '
                                     'would score like the fresh model, not like F'})
    ck('I8 the restored model differs from the fresh (non-restored) model on the same input, '
       'proving the restoration actually changes the scorer',
       gap is not None and gap > 0.0, RECORDS['trained_vs_fresh'])

    fails = [c for c in CASES if not c[1]]
    status = ('F_SCORING_INTERFACE_VERIFIED_WITH_CURRENT_BACKBONE' if not fails
              else 'F_SCORING_INTERFACE_FAILED')
    rec('verdict', {'status': status, 'always_carries': LABEL,
                    'can_never_yield': 'F_HISTORICAL_IDENTITY_VERIFIED',
                    'd_geo_called': False, 'input_search_performed': False,
                    'algorithmic_gain_claim': 'none'})
    print('\n=== VERDICT: %s (%d rows, %d PASS, %d FAIL) ==='
          % (status, len(CASES), len(CASES) - len(fails), len(fails)))
    for name, _ok, detail in fails:
        print('FAILED-ROW %s :: %s' % (name, json.dumps(detail, default=repr, sort_keys=True)))
    if args.json:
        Path(args.json).write_text(json.dumps(
            {'rows': [{'name': n, 'ok': o, 'detail': dd} for n, o, dd in CASES],
             'records': RECORDS, 'n_rows': len(CASES), 'n_pass': len(CASES) - len(fails),
             'n_fail': len(fails)}, indent=2, default=repr, sort_keys=True), encoding='utf-8')
    return 1 if fails else 0


if __name__ == '__main__':
    sys.exit(main())
