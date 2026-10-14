"""Delivery n13: action-matching head -- held-out generalisation and reproducible update.

Fixed pre-registered split of the 29 sealed supervised states into 20 train / 9
held out.  Three pre-declared arms are trained head-only over the sealed latents
and all are reported; no arm is selected by outcome.

Primary arm  : plain SGD at the matched initial step scale (reproducible by
               construction because the step is proportional to the gradient).
Controls     : plain SGD at a conservative rate; the round-5 Adam optimiser
               whose cross-library trajectory divergence is already recorded.

No backbone forward, no C0, no solver.  Engineering measurement only.
"""
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / 'tools'))
os.environ.update(PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                  MKL_NUM_THREADS='1')

import numpy as np
import psutil

import engineering_stage_closure_v1 as seal
import head_action_matching as ham

REG = json.loads((HERE / 'REGISTRATION.json').read_text(encoding='utf-8'))
TOL = REG['tolerances']
ARM = ham.ARMS[0]
EPOCHS = REG['update']['epochs']
RAM_GATE_GIB = REG['resource']['min_available_ram_gib_without_backbone']
BACKBONE_THRESHOLD_GIB = REG['resource']['full_backbone_threshold_gib']


def save(name, value):
    seal.save(HERE / name, value)


def digest_leaves(leaves):
    return {k: ham.sha256_array(np.asarray(v, np.float32)) for k, v in leaves.items()}


def main():
    seal.initialize()
    import jax
    import jax.numpy as jnp
    import optax

    for name, want in REG['runtime'].items():
        mod = __import__(name)
        if getattr(mod, '__version__', None) != want:
            raise ValueError('Registered %s version mismatch' % name)

    devices = [str(d) for d in jax.devices()]
    avail = float(psutil.virtual_memory().available)
    if avail < RAM_GATE_GIB * 2 ** 30:
        raise ValueError('No-backbone resource gate failed before stage')
    if avail >= BACKBONE_THRESHOLD_GIB * 2 ** 30:
        raise ValueError('Full-backbone threshold reached; this stage is registered without backbone')

    import encodings.cp437  # noqa: F401  (warm the lazy codec before the closure)
    w = {'k1': jnp.ones((3075, 64), jnp.float32), 'b1': jnp.zeros((64,), jnp.float32),
         'k2': jnp.ones((64, 1), jnp.float32), 'b2': jnp.zeros((1,), jnp.float32)}

    def warm_ce(leaves):
        x = jnp.ones((4, 3075), jnp.float32)
        s = (jnp.tanh(x @ leaves['k1'] + leaves['b1']) @ leaves['k2'] + leaves['b2'])[:, 0]
        return -jax.nn.log_softmax(s)[0]

    np.asarray(jax.grad(warm_ce)(w))
    ham.construct_head_leaves(seed=ham.HEAD_SEED)
    ham.head_scores_jax(np.zeros((2, 3075), np.float32), {k: np.asarray(v) for k, v in w.items()})
    ham.load_record(0, ARM)

    start_closure = seal.closure(tuple(Path(p) for p in REG['sources']))
    save('SOURCE_START.json', start_closure)

    checks = []

    def ck(name, value):
        checks.append(dict(name=name, pass_=bool(value)))

    # ---- sealed identity -------------------------------------------------------
    manifest = ham.sealed_manifest_hashes()
    input_hashes = {}
    for n in range(ham.RECORDS):
        name = '%s_%02d.npz' % (ARM, n)
        got = ham.sha256_file(ham.record_path(n, ARM))
        input_hashes[name] = got
        ck('sealed_input_sha256:' + name, got == manifest.get(name))
    ck('sealed_action_options_sha256', ham.sha256_file(ham.ACTION_OPTIONS) == REG['sealed']['action_options_sha256'])
    ck('sealed_head_source_sha256', ham.sha256_file(ham.SEALED_HEAD_SOURCE) == REG['sealed']['head_source_sha256'])
    ck('round5_grad_artifact_sha256',
       ham.sha256_file(REG['round5_artifacts']['step0_grad']) == REG['round5_artifacts']['step0_grad_sha256'])

    records = [ham.load_record(n, ARM) for n in range(ham.RECORDS)]
    options = ham.action_options()
    base_leaves = {k: np.asarray(records[0]['leaves'][k], np.float32) for k in ham.LEAF_KEYS}
    ck('all_records_share_one_head_identity',
       all(all(np.array_equal(r['leaves'][k], base_leaves[k]) for k in ham.LEAF_KEYS) for r in records))
    constructed = ham.construct_head_leaves(seed=ham.HEAD_SEED)
    for k in ham.LEAF_KEYS:
        ck('sealed_leaf_is_construct_head_seed20261012:' + k, np.array_equal(constructed[k], base_leaves[k]))

    # ---- reconstruction and original forward gate ------------------------------
    concats = [ham.concat_numpy(r) for r in records]
    max_forward_diff = 0.0
    forward_exact = 0
    for rec in records:
        jout = ham.head_scores_jax(concats[rec['index']], base_leaves)
        max_forward_diff = max(max_forward_diff,
                               float(np.max(np.abs(np.asarray(jout, np.float64)
                                                   - rec['sealed_scores'].astype(np.float64)))))
        forward_exact += int(np.array_equal(np.asarray(jout), rec['sealed_scores']))
    ck('original_forward_within_registered_bound', max_forward_diff <= TOL['forward_vs_sealed_scores_max_abs'])
    inter = np.load(ham.SEALED_INTERMEDIATES)
    ck('record0_in_jax_concat_is_bit_exact_to_sealed_intermediate',
       np.array_equal(np.asarray(ham.concat_jax(records[0])), inter['eager_concat']))

    # ---- groups and the fixed pre-registered split -----------------------------
    groups = []
    for n in range(ham.RECORDS):
        g = ham.group_of(n, records[n], options[n])
        if g['target_local'] is None:
            continue
        g['rows'] = concats[n][g['eligible']]
        groups.append(g)
    supervised = [g['index'] for g in groups]
    held_positions = REG['split']['held_out_positions']
    held = [supervised[p] for p in held_positions]
    train = [i for i in supervised if i not in held]
    ck('split_sizes_match_registration',
       len(train) == REG['split']['expected_train'] and len(held) == REG['split']['expected_held_out'])
    ck('split_partition_is_exact', sorted(train + held) == sorted(supervised) and not set(train) & set(held))
    by_index = {g['index']: g for g in groups}

    def scores_of(index, leaves):
        return ham.head_scores_jax(by_index[index]['rows'], leaves)

    def metrics(indices, leaves):
        ce, hits, full_hits = [], 0, 0
        for i in indices:
            g = by_index[i]
            s = np.asarray(scores_of(i, leaves), np.float64)
            ce.append(ham.ce_loss64(s, g['target_local']))
            hits += int(int(np.argmax(s)) == g['target_local'])
            full = np.asarray(ham.head_scores_jax(concats[i], leaves), np.float64)
            full_hits += int(int(np.argmax(full)) == g['target_row'])
        return dict(mean_ce=float(np.mean(ce)), argmax_correct=hits, n=len(indices),
                    full_candidate_argmax_correct=full_hits)

    before = dict(train=metrics(train, base_leaves), held_out=metrics(held, base_leaves))

    rows_by_index = {i: (jnp.asarray(by_index[i]['rows']), by_index[i]['target_local']) for i in train}

    def group_ce(leaves, index):
        x, t = rows_by_index[index]
        s = (jnp.tanh(x @ leaves['k1'] + leaves['b1']) @ leaves['k2'] + leaves['b2'])[:, 0]
        return -jax.nn.log_softmax(s)[t]

    def make_optimizer(spec):
        if spec['kind'] == 'sgd':
            return optax.sgd(learning_rate=spec['learning_rate'])
        return optax.adam(learning_rate=spec['learning_rate'], b1=spec['b1'], b2=spec['b2'], eps=spec['eps'])

    arms = {}
    grad_fn = jax.grad(group_ce)
    for spec in REG['arms']:
        leaves = {k: jnp.asarray(base_leaves[k]) for k in ham.LEAF_KEYS}
        opt = make_optimizer(spec)
        st = opt.init(leaves)
        traj, step_log = [], []
        step0_grad = None
        for epoch in range(EPOCHS):
            for index in train:
                grads = grad_fn(leaves, index)
                if step0_grad is None:
                    step0_grad = {k: np.asarray(grads[k], np.float32) for k in ham.LEAF_KEYS}
                up, st = opt.update(grads, st)
                leaves = optax.apply_updates(leaves, up)
            traj.append(metrics(train, leaves)['mean_ce'])
            step_log.append(dict(epoch=epoch + 1, train_mean_ce=traj[-1],
                                 train_argmax=metrics(train, leaves)['argmax_correct'],
                                 param_digest=digest_leaves(leaves)))
        final = {k: np.asarray(leaves[k], np.float32) for k in ham.LEAF_KEYS}
        name = spec['name']
        np.savez(HERE / ('HEAD_AFTER_%s.npz' % name), **{k: final[k] for k in ham.LEAF_KEYS})
        np.savez(HERE / ('HEAD_STEP0_GRAD_%s.npz' % name), **{k: step0_grad[k] for k in ham.LEAF_KEYS})
        reloaded = np.load(HERE / ('HEAD_AFTER_%s.npz' % name))
        reload_exact = {k: bool(np.array_equal(np.asarray(reloaded[k], np.float32), final[k]))
                        for k in ham.LEAF_KEYS}
        for k in ham.LEAF_KEYS:
            ck('reload_bit_exact:%s:%s' % (name, k), reload_exact[k])
        rl = {k: np.asarray(reloaded[k], np.float32) for k in ham.LEAF_KEYS}
        diffs = [float(np.max(np.abs(np.asarray(scores_of(i, final), np.float64)
                                      - np.asarray(scores_of(i, rl), np.float64)))) for i in supervised]
        ck('reload_forward_bit_exact:%s' % name, max(diffs) == 0.0)
        arms[name] = dict(
            spec=spec, steps=EPOCHS * len(train), train_order=train,
            train_ce_trajectory=traj, step_log=step_log,
            before=before, after=dict(train=metrics(train, final), held_out=metrics(held, final)),
            reload_bit_exact=reload_exact, reload_forward_max_abs=max(diffs),
            changed=all(not np.array_equal(final[k], base_leaves[k]) for k in ham.LEAF_KEYS),
            saved_sha256={('%s.npz' % f): ham.sha256_file(HERE / ('%s.npz' % f))
                          for f in ('HEAD_AFTER_' + name, 'HEAD_STEP0_GRAD_' + name)},
        )

    # ---- mechanism measurements on the training step-0 gradients ---------------
    g0 = arms[REG['primary_arm']]['saved_sha256']
    gfile = np.load(HERE / ('HEAD_STEP0_GRAD_%s.npz' % REG['primary_arm']))
    allg = np.concatenate([np.asarray(gfile[k], np.float64).ravel() for k in ham.LEAF_KEYS])
    adam_lr = next(s['learning_rate'] for s in REG['arms'] if s['kind'] == 'adam')
    adam_eps = next(s['eps'] for s in REG['arms'] if s['kind'] == 'adam')
    adam_step = adam_lr * np.abs(allg) / (np.abs(allg) + adam_eps)
    mechanism = dict(
        train_step0_zero_gradient_fraction=float(np.mean(allg == 0.0)),
        train_step0_mean_abs_gradient=float(np.mean(np.abs(allg))),
        adam_t1_mean_abs_step=float(np.mean(adam_step)),
        adam_t1_saturated_fraction=float(np.mean(adam_step > 0.5 * adam_lr)),
        matched_lr_from_round5_full_grad=REG['update']['matched_lr_derivation']['matched_lr'],
        note=('Adam normalises the update: for |g| >> eps its step is +/-lr regardless of gradient magnitude, so '
              'coordinates at the float32 rounding floor can move by a full step whose sign is not reproducible '
              'across arithmetics. Plain SGD keeps the step proportional to |g| and is therefore reproducible.'),
    )

    end_closure = seal.closure(tuple(Path(p) for p in REG['sources']))
    save('SOURCE_END.json', end_closure)
    ck('actual_loaded_files_and_maps_start_end', seal.same(start_closure, end_closure))
    after_hashes = {name: ham.sha256_file(ham.record_path(int(name[-6:-4]), ARM)) for name in input_hashes}
    ck('sealed_inputs_unchanged_after_stage', after_hashes == input_hashes)
    ck('cpu_devices_only', jax.default_backend() == 'cpu'
       and all(str(getattr(dv, 'platform', '')) == 'cpu' for dv in jax.devices()))
    backbone = [p for p in list(end_closure['files']) + list(end_closure['modules'].values())
                if 'MASKCO_code' in p or p.lower().endswith('.ckpt')]
    ck('no_backbone_or_checkpoint_in_loaded_closure', not backbone)
    ck('head_leaf_identity_bit_exact',
       all(np.array_equal(constructed[k], base_leaves[k]) for k in ham.LEAF_KEYS))

    result = dict(
        checks_count=len(checks), failed=sum(not c['pass_'] for c in checks),
        failure_names=[c['name'] for c in checks if not c['pass_']], checks=checks,
        stage='delivery_n13_action_matching_generalization',
        scope='head-only CE over sealed latents; fixed 20/9 train/held-out split; three pre-declared arms; no backbone/C0/solver',
        sealed_arm=ARM, records=ham.RECORDS,
        head_parameter_elements=int(sum(int(np.size(base_leaves[k])) for k in ham.LEAF_KEYS)),
        input_hashes=input_hashes,
        forward_vs_sealed_scores=dict(max_abs_diff=max_forward_diff, bit_exact_records=forward_exact,
                                      tolerance=TOL['forward_vs_sealed_scores_max_abs']),
        split=dict(train_records=train, held_out_records=held,
                   held_out_positions=held_positions, total=len(supervised)),
        arms=arms,
        primary_arm=REG['primary_arm'],
        mechanism=mechanism,
        resource=dict(available_ram_gib_before_stage=avail / 2 ** 30,
                      full_backbone_threshold_gib=BACKBONE_THRESHOLD_GIB, backbone_loaded=False,
                      devices=devices, cpu_affinity=psutil.Process().cpu_affinity()),
        memory_limitation=('Local available RAM %.3f GiB was below the registered %.1f GiB full-backbone threshold, '
                           'so this stage trained only the small scoring head over already sealed node '
                           'representations and never loaded the backbone.' % (avail / 2 ** 30, BACKBONE_THRESHOLD_GIB)),
        source_actual_files=len(end_closure['files']), source_actual_maps=len(end_closure['modules']),
        formal_trained_control_qualified=False, formal_controls_qualified=False,
        full_request_budget_qualified=False, full_scientific_loaded_closure_qualified=False,
        scientific_data_generated=False, new_science_family=False, scientific_family3_started=False,
        optimizer_updates_of_model=0, learned_output_qualified=False, stable_online_gain=False,
    )
    save('RESULT.json', result)
    save('RUN_EXIT.json', dict(exit_code=int(result['failed'] > 0)))
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ('checks', 'input_hashes')}))
    return int(result['failed'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
