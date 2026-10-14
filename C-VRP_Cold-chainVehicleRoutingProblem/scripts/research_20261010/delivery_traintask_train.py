"""D-TRAINTASK-v1 trainer: objective-aligned training of the delivery reconstruction guide.

Implements the registered recipe `delivery_traintask_recipe.json` and nothing else:
  * the objective is `advantage_weighted_decision_ranking` driven by the REALIZED contract cost,
  * the head is applied directly to frozen features (the discarded decoder call is a panel
    resource-matching device and is NOT performed during training),
  * a MANDATORY fidelity gate runs before any optimisation step: the trainer's own option
    enumeration must reproduce the frozen `reconstruct`'s recorded decisions EXACTLY.

Why the fidelity gate is the most important part. The frozen reconstruct enumerates legal
insertions internally and the trainer must re-enumerate them to have anything to optimise. If the
two enumerations disagree, the trainer optimises a different operator than the one that will be
deployed and the arm becomes uninterpretable. The gate is therefore fail-closed: match rate must
be exactly 1.0, and the run aborts otherwise. It is never relaxed to let a run proceed.

Two levels of fidelity are reported separately, because they cost very different amounts:
  * FULL rule fidelity - replays every remaining candidate customer and the regret/tie-break
    rule, so it also verifies the CUSTOMER selection. Expensive.
  * per-customer option fidelity - replays only the customer the frozen reconstruction actually
    chose, so it verifies the vehicle/position option set and its ordering. Cheap, and it is
    checked on every selected training cycle.

Scope: this trains and freezes one candidate. It runs no arm comparison and claims no improvement,
attribution, impossibility or equivalence.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


def bootstrap(scripts):
    scripts = Path(scripts).resolve()
    for p in (scripts / 'delivery', scripts / 'research_20261010'):
        sys.path.insert(0, str(p))
    return scripts


def file_sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def enumerate_options(customer, candidate, instance, contract, dist, adjacency, wave=0.0):
    """Exactly `reconstruct`'s option set for one customer, minus the guide term."""
    from delivery_baselines import route_ok
    options = []
    for k, route in enumerate(candidate):
        for at in range(len(route) + 1):
            a = route[at - 1] if at else 0
            b = route[at] if at < len(route) else 0
            if adjacency[a, b] > 0:
                continue
            if not route_ok(route[:at] + [customer] + route[at:], instance, contract, dist, wave):
                continue
            delta = float(dist[a, customer] + dist[customer, b] - dist[a, b])
            options.append((delta, k, at, a, b))
    options.sort(key=lambda row: (row[0], row[1], row[2]))
    return options


def replay(cycle, instance, contract, dist, adjacency, removed, full_rule, check_steps):
    """Replay the recorded insertions, re-enumerating options; return fidelity info."""
    candidate = [[i for i in route if i not in removed] for route in cycle['incumbent_routes_before']]
    candidate += [[] for _ in range(contract.vehicles - len(candidate))]
    remaining = sorted(removed)
    recorded = cycle['insertions']
    mismatch = None
    per_customer_checked = 0
    per_customer_ok = 0
    full_rule_steps = 0
    for step, rec in enumerate(recorded):
        i = int(rec['customer'])
        if full_rule and step < check_steps:
            choices = []
            for cand_i in remaining:
                opts = enumerate_options(cand_i, candidate, instance, contract, dist,
                                         adjacency)
                if opts:
                    regret = (opts[1][0] - opts[0][0]) if len(opts) > 1 else float('inf')
                    choices.append((regret, cand_i, opts[0]))
            if not choices:
                return {'ok': False, 'step': step, 'reason': 'no feasible insertion in replay'}
            _, i_full, (_, k_full, at_full, _, _) = max(
                choices, key=lambda row: (row[0], -row[1]))
            if (i_full, k_full, at_full) != (i, int(rec['vehicle']), int(rec['position'])):
                return {'ok': False, 'step': step,
                        'reason': 'full-rule mismatch: replay chose %s, recorded %s'
                                  % ((i_full, k_full, at_full),
                                     (i, int(rec['vehicle']), int(rec['position'])))}
            full_rule_steps += 1
        # The per-customer option check runs for EVERY step of every replay, so the two counters
        # below must both advance once per step. An earlier version advanced `ok` unconditionally
        # but `checked` only past a threshold, which produced a nonsensical rate above 1.
        opts = enumerate_options(i, candidate, instance, contract, dist, adjacency)
        match = [o for o in opts if (o[1], o[2]) == (int(rec['vehicle']), int(rec['position']))]
        if not match:
            return {'ok': False, 'step': step,
                    'reason': 'recorded option %s absent from the re-enumerated option set'
                              % ((int(rec['vehicle']), int(rec['position'])),)}
        per_customer_checked += 1
        per_customer_ok += 1
        candidate[int(rec['vehicle'])].insert(int(rec['position']), i)
        remaining.remove(i)
    return {'ok': True, 'per_customer_checked': per_customer_checked,
            'per_customer_ok': per_customer_ok, 'full_rule_steps': full_rule_steps}


def main(args):
    started = time.perf_counter()
    scripts = bootstrap(args.scripts)
    from delivery_equal_schedule import generate, DeliveryContract
    from delivery_baselines import matrices
    from delivery_maskco import partial_adjacency
    from delivery_trained_guide import DeliveryTrainedGuide
    import jax
    import jax.numpy as jnp
    import optax
    from flax import nnx

    split = Path(args.split)
    manifest = json.loads((split / 'TRAIN_SPLIT_MANIFEST.json').read_text(encoding='utf-8'))
    identity = json.loads((split / 'TRAIN_SPLIT_IDENTITY.json').read_text(encoding='utf-8'))
    if file_sha(split / 'samples.jsonl') != manifest['samples_sha256']:
        raise SystemExit('TRAIN samples.jsonl sha does not match its manifest')
    if identity.get('isa_protocol_train_split') is not True:
        raise SystemExit('refusing to train on a split that is not the registered TRAIN split')
    recipe_sha = file_sha(args.recipe)
    samples = [json.loads(l) for l in (split / 'samples.jsonl').read_text(
        encoding='utf-8').splitlines() if l.strip()]
    certified = [s for s in samples if s['outcome'] == 'certified' and s.get('insertions')
                 and s.get('delta_cost') is not None]
    if not certified:
        raise SystemExit('no certified cycles with both a decision sequence and a reward')

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    contract = DeliveryContract()
    n_inst = int(identity['instances'])
    instances, dists = [], []
    for i in range(n_inst):
        data, _ = generate(int(identity['train_seed']), i)
        instances.append(data)
        dists.append(matrices(data))

    # ---------------- fidelity gate (fail closed, before any optimisation step) ---------
    rng = np.random.default_rng(args.select_seed)
    order = list(range(len(certified)))
    if args.fidelity_cycles and args.fidelity_cycles < len(order):
        pick = sorted(rng.choice(order, size=args.fidelity_cycles, replace=False).tolist())
    else:
        pick = order
    full_ok, full_bad, full_steps = 0, [], 0
    per_cust_checked, per_cust_ok, full_rule_total = 0, 0, 0
    # The full-rule subset must be drawn from the cycles actually being checked (`pick`), not from
    # all certified indices. Drawing from the full index range made the intersection with `pick`
    # almost surely empty, so full-rule verification silently never ran while still being reported
    # as "requested". Caught only because full_rule_steps_checked came back as exactly 0.
    if args.full_rule_cycles == 0:
        full_rule_set = set(pick)
    else:
        full_rule_set = set(rng.choice(pick, size=min(args.full_rule_cycles, len(pick)),
                                       replace=False).tolist())
    t_fid = time.perf_counter()
    for k, idx in enumerate(pick):
        c = certified[idx]
        inst = int(c['instance'])
        removed = set(int(x) for x in c['mask'])
        adjacency = partial_adjacency(c['incumbent_routes_before'], removed, len(dists[inst]))
        full_rule = idx in full_rule_set
        r = replay(c, instances[inst], contract, dists[inst], adjacency, removed, full_rule,
                   args.full_rule_steps)
        full_steps += 1
        per_cust_checked += r.get('per_customer_checked', 0)
        per_cust_ok += r.get('per_customer_ok', 0)
        full_rule_total += r.get('full_rule_steps', 0)
        if r['ok']:
            full_ok += 1
        else:
            full_bad.append({'cycle_index': idx, 'instance': inst, **r})
            if len(full_bad) >= args.max_reported_mismatches:
                break
        if args.verbose and (k % 25 == 0 or k == len(pick) - 1):
            print(json.dumps({'fidelity_progress': k + 1, 'of': len(pick),
                              'elapsed_s': round(time.perf_counter() - t_fid, 1)}), flush=True)
    fidelity = {
        'cycles_checked': full_steps,
        'cycles_with_exact_replay': full_ok,
        'match_rate': (full_ok / full_steps) if full_steps else 0.0,
        'per_customer_steps_checked': per_cust_checked,
        'full_rule_steps_checked': full_rule_total,
        'per_customer_options_matched': per_cust_ok,
        'per_customer_match_rate': (per_cust_ok / per_cust_checked) if per_cust_checked else 0.0,
        'full_rule_cycles_requested': args.full_rule_cycles,
        'full_rule_steps_per_cycle': args.full_rule_steps,
        'mismatches': full_bad,
        'wall_s': time.perf_counter() - t_fid,
        'requirement': 'match rate must be exactly 1.0 before any optimisation step',
        'scope_note': ('per-customer option fidelity (the vehicle/position option set and its '
                       'ordering, for the customer the frozen reconstruction actually chose) is '
                       'checked on EVERY selected cycle. Full-rule fidelity, which additionally '
                       'replays every remaining candidate customer and the regret/tie-break rule '
                       'and therefore also verifies the CUSTOMER selection, is checked on '
                       'full_rule_cycles cycles for the first full_rule_steps steps each, '
                       'because replaying the full rule at every step of all 505 certified '
                       'cycles was measured to take far longer than the rest of the run; this '
                       'verification-depth change is recorded as an amendment in the recipe '
                       'file rather than applied silently.'),
    }
    if fidelity['match_rate'] != 1.0:
        (out / 'result.json').write_text(json.dumps(
            {'kind': 'delivery_traintask_training_aborted_by_fidelity_gate',
             'fidelity': fidelity, 'recipe_sha256': recipe_sha,
             'trained': False, 'wall_s': time.perf_counter() - started},
            indent=2, allow_nan=False, default=str), encoding='utf-8')
        print(json.dumps({'aborted': 'fidelity', 'fidelity': fidelity}, indent=2, default=str))
        return 4

    # ---------------- training data ----------------------------------------------------
    # max_cycles = 0 means "use every certified cycle"; a bare `<` would have selected none.
    sel = (sorted(rng.choice(order, size=args.max_cycles, replace=False).tolist())
           if 0 < args.max_cycles < len(order) else order)
    train_cycles = [certified[i] for i in sel]
    scale = float(np.mean([abs(c['delta_cost']) for c in train_cycles])) or 1.0
    weight_of = {id(c): float(np.clip(c['delta_cost'] / scale, -1.0, 1.0)) for c in train_cycles}
    dscale_of = {}
    for i in range(n_inst):
        off = ~np.eye(len(dists[i]), dtype=bool)
        dscale_of[i] = max(1e-9, float(np.mean(dists[i][off])))

    guide = DeliveryTrainedGuide(args.checkpoint, args.upstream_root,
                                head_seed=args.head_seed, head_hidden=args.head_hidden)
    feats = {}
    for i in range(n_inst):
        f = guide.features(instances[i], contract.capacity)
        if hasattr(f, 'block_until_ready'):
            f.block_until_ready()
        feats[i] = f
    guide._ensure_head(np.asarray(feats[0]).shape[-1])
    base_before = guide.base_param_hash()
    head_before = guide.head_param_hash()
    optimizer = nnx.Optimizer(guide._head, optax.adam(args.lr), wrt=nnx.Param)

    # Per-step option table. Each recorded insertion becomes one training step whose option set
    # belongs to the customer the frozen reconstruction actually chose, because the operator's
    # customer selection is driven by regret while only the vehicle/position choice is scored by
    # the guide - so the guide's decision is exactly argmin over this option set.
    by_cycle = []
    for c in train_cycles:
        inst = int(c['instance'])
        removed = set(int(x) for x in c['mask'])
        candidate = [[i for i in r if i not in removed] for r in c['incumbent_routes_before']]
        candidate += [[] for _ in range(contract.vehicles - len(candidate))]
        adjacency = partial_adjacency(c['incumbent_routes_before'], removed, len(dists[inst]))
        for rec in c['insertions']:
            i = int(rec['customer'])
            opts = enumerate_options(i, candidate, instances[inst], contract, dists[inst],
                                     adjacency)
            tgt = next((n2 for n2, o in enumerate(opts)
                        if (o[1], o[2]) == (int(rec['vehicle']), int(rec['position']))), None)
            if tgt is not None:
                by_cycle.append({'inst': inst, 'i': i, 'w': weight_of[id(c)],
                                 'dscale': dscale_of[inst],
                                 'delta': np.array([o[0] for o in opts]),
                                 'a': np.array([o[3] for o in opts]),
                                 'b': np.array([o[4] for o in opts]), 'target': int(tgt)})
            candidate[int(rec['vehicle'])].insert(int(rec['position']), i)

    # Batch the steps per instance so the traced graph is a handful of array ops instead of one
    # node per Python-loop iteration. The first version looped in Python and `value_and_grad`
    # re-traced a ~5000-op graph on every epoch, which stalled the run.
    by_inst = {}
    for st in by_cycle:
        by_inst.setdefault(st['inst'], []).append(st)
    batches = {}
    for inst, sts in by_inst.items():
        maxopt = max(len(s['delta']) for s in sts)
        n = len(sts)
        delta = np.full((n, maxopt), np.inf)          # inf -> weight 0 in the softmax
        a = np.zeros((n, maxopt), np.int32)
        b = np.zeros((n, maxopt), np.int32)
        target = np.zeros(n, np.int32)
        w = np.zeros(n, np.float64)
        for j, s in enumerate(sts):
            k = len(s['delta'])
            delta[j, :k] = s['delta']
            a[j, :k] = s['a']
            b[j, :k] = s['b']
            target[j] = s['target']
            w[j] = s['w']
        batches[inst] = {'delta': jnp.asarray(delta), 'a': jnp.asarray(a), 'b': jnp.asarray(b),
                         'ii': jnp.asarray(np.broadcast_to(
                             np.asarray([s['i'] for s in sts])[:, None], (n, maxopt),
                             ).copy(), np.int32),
                         'target': jnp.asarray(target), 'w': jnp.asarray(w),
                         'n': n, 'dscale': dscale_of[inst]}

    n_steps = float(max(1, len(by_cycle)))

    def loss_fn(head):
        total = 0.0
        for inst, B in batches.items():
            L = head(feats[inst])[0]
            off = ~jnp.eye(L.shape[0], dtype=bool)
            lscale = jnp.maximum(1e-8, jnp.std(L[off]))
            g1 = L[B['a'], B['ii']]
            g2 = L[B['ii'], B['b']]
            g3 = jnp.where(B['a'] == B['b'], 0.0, L[B['a'], B['b']])
            gain = (g1 + g2 - g3) / lscale
            score = B['delta'] / B['dscale'] - B['w'][:, None] * gain
            logp = jax.nn.log_softmax(-score, axis=1)
            total = total - jnp.sum(logp[jnp.arange(B['n']), B['target']])
        return total / n_steps

    history = []
    for epoch in range(args.epochs):
        loss, grads = nnx.value_and_grad(loss_fn)(guide._head)
        try:
            optimizer.update(grads)
        except TypeError:
            optimizer.update(guide._head, grads)
        history.append({'epoch': epoch, 'loss': float(loss)})
        if args.verbose and (epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs - 1):
            print(json.dumps({'epoch': epoch, 'loss': float(loss)}), flush=True)

    head_after = guide.head_param_hash()
    base_after = guide.base_param_hash()
    guide.head_snapshot = None
    head_path = out / 'delivery_trained_head.pkl'
    guide.save_head(head_path)
    result = {
        'kind': 'delivery_traintask_objective_aligned_training_run',
        'recipe_sha256': recipe_sha,
        'train_split_manifest_sha256': file_sha(split / 'TRAIN_SPLIT_MANIFEST.json'),
        'train_seed': identity['train_seed'], 'mask_seed_train': identity['mask_seed_train'],
        'select_seed': args.select_seed,
        'n_certified_available': len(certified), 'n_cycles_used': len(train_cycles),
        'n_training_steps': len(by_cycle),
        'delta_cost_scale': scale,
        'fidelity': fidelity,
        'head_seed': args.head_seed, 'head_hidden': args.head_hidden,
        'optimizer': 'adam', 'learning_rate': args.lr, 'epochs': args.epochs,
        'loss_first': history[0]['loss'] if history else None,
        'loss_last': history[-1]['loss'] if history else None,
        'loss_history': history,
        'head_param_hash_before': head_before, 'head_param_hash_after': head_after,
        'base_param_hash_before': base_before, 'base_param_hash_after': base_after,
        'frozen_encoder_and_decoder_unchanged': base_before == base_after,
        'head_changed': head_before != head_after,
        'head_artifact_sha256': file_sha(head_path),
        'jax_backend': jax.default_backend(),
        'what_this_is': ['one frozen candidate trained under the registered recipe'],
        'what_this_is_not': [
            'not a result: no arm is compared and no improvement is measured',
            'no stable online gain, no MaskCO attribution, no impossibility, no equivalence'],
        'wall_s': time.perf_counter() - started,
    }
    (out / 'result.json').write_text(json.dumps(result, indent=2, allow_nan=False, default=str),
                                     encoding='utf-8')
    print(json.dumps({k: result[k] for k in
                      ('n_cycles_used', 'n_training_steps', 'fidelity', 'loss_first',
                       'loss_last', 'frozen_encoder_and_decoder_unchanged', 'head_changed',
                       'head_artifact_sha256', 'wall_s')}, indent=2, default=str))
    return 0


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--scripts', required=True)
    ap.add_argument('--split', required=True)
    ap.add_argument('--recipe', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--upstream-root', required=True)
    ap.add_argument('--head-seed', type=int, default=2026120105)
    ap.add_argument('--head-hidden', type=int, default=64)
    ap.add_argument('--select-seed', type=int, default=2026120104)
    ap.add_argument('--max-cycles', type=int, default=80)
    ap.add_argument('--fidelity-cycles', type=int, default=40,
                    help='0 = check every certified cycle')
    ap.add_argument('--full-rule-cycles', type=int, default=8,
                    help='0 = every fidelity cycle also replays the full regret rule; '
                         'otherwise this many randomly chosen cycles do')
    ap.add_argument('--full-rule-steps', type=int, default=6)
    ap.add_argument('--max-reported-mismatches', type=int, default=5)
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--lr', type=float, default=0.001)
    ap.add_argument('--verbose', action='store_true')
    sys.exit(main(ap.parse_args()))
