"""Delivery n12: action-matching scoring-head update / order / save / reload.

Head-only per-customer feasible-action cross-entropy on the sealed saved node
latents of the `actual_action_head_execution_root_v1` pretrained_initial arm.
No backbone forward, no C0, no solver, no teacher cost input.  Engineering
component only; it is not a formal trained control.
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
LR = REG['update']['learning_rate']
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

    if np.__version__ != '1.26.4' or sys.version_info[:3] != (3, 12, 7):
        raise ValueError('Registered runtime mismatch')
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

    # Warm up every lazily loaded native path and submodule before the closure.
    w = {'k1': jnp.ones((3075, 64), jnp.float32), 'b1': jnp.zeros((64,), jnp.float32),
         'k2': jnp.ones((64, 1), jnp.float32), 'b2': jnp.zeros((1,), jnp.float32)}

    def warm_ce(leaves):
        x = jnp.ones((4, 3075), jnp.float32)
        s = (jnp.tanh(x @ leaves['k1'] + leaves['b1']) @ leaves['k2'] + leaves['b2'])[:, 0]
        return -jax.nn.log_softmax(s)[0]

    np.asarray(jax.grad(warm_ce)(w))
    ham.construct_head_leaves(seed=ham.HEAD_SEED)
    ham.head_scores_jax(np.zeros((2, 3075), np.float32), {kk: np.asarray(v) for kk, v in w.items()})

    start_closure = seal.closure(tuple(Path(p) for p in REG['sources']))
    save('SOURCE_START.json', start_closure)

    checks = []

    def ck(name, value):
        checks.append(dict(name=name, pass_=bool(value)))

    # ---- sealed input identity -------------------------------------------------
    manifest = ham.sealed_manifest_hashes()
    input_hashes = {}
    for n in range(ham.RECORDS):
        name = '%s_%02d.npz' % (ARM, n)
        got = ham.sha256_file(ham.record_path(n, ARM))
        input_hashes[name] = got
        ck('sealed_input_sha256:' + name, got == manifest.get(name))
    ck('sealed_action_options_sha256', ham.sha256_file(ham.ACTION_OPTIONS) == REG['sealed']['action_options_sha256'])
    ck('sealed_head_source_sha256', ham.sha256_file(ham.SEALED_HEAD_SOURCE) == REG['sealed']['head_source_sha256'])
    ck('sealed_intermediates_sha256', ham.sha256_file(ham.SEALED_INTERMEDIATES) == REG['sealed']['intermediates_sha256'])

    records = [ham.load_record(n, ARM) for n in range(ham.RECORDS)]
    options = ham.action_options()
    ck('action_options_record_count', len(options) == ham.RECORDS)

    # ---- head leaf identity ----------------------------------------------------
    base_leaves = records[0]['leaves']
    same_across = all(all(np.array_equal(r['leaves'][kk], base_leaves[kk]) for kk in ham.LEAF_KEYS)
                      for r in records)
    ck('all_records_share_one_head_identity', same_across)
    constructed = ham.construct_head_leaves(seed=ham.HEAD_SEED)
    leaf_exact = {kk: bool(np.array_equal(constructed[kk], base_leaves[kk])) for kk in ham.LEAF_KEYS}
    for kk in ham.LEAF_KEYS:
        ck('sealed_leaf_is_construct_head_seed20261012:' + kk, leaf_exact[kk])
    leaf_elements = int(sum(int(np.size(base_leaves[kk])) for kk in ham.LEAF_KEYS))
    leaf_shapes = {kk: list(np.shape(base_leaves[kk])) for kk in ham.LEAF_KEYS}
    ck('head_leaf_count_four', len(ham.LEAF_KEYS) == 4)
    ck('head_parameter_elements', leaf_elements == REG['sealed']['head_parameter_elements'])

    # ---- concat reconstruction and original forward gate -----------------------
    concats = []
    forward_rows = []
    max_forward_diff = 0.0
    forward_exact = 0
    for rec in records:
        cat = ham.concat_numpy(rec)
        concats.append(cat)
        jout = ham.head_scores_jax(cat, base_leaves)
        got = np.asarray(jout, np.float64)
        want = rec['sealed_scores'].astype(np.float64)
        d = float(np.max(np.abs(got - want)))
        max_forward_diff = max(max_forward_diff, d)
        forward_exact += int(np.array_equal(np.asarray(jout), rec['sealed_scores']))
        forward_rows.append(dict(index=rec['index'], max_abs_diff=d,
                                 concat_sha256=ham.sha256_array(cat)))
    ck('original_forward_within_registered_bound', max_forward_diff <= TOL['forward_vs_sealed_scores_max_abs'])

    inter = np.load(ham.SEALED_INTERMEDIATES)
    cat0_jax = np.asarray(ham.concat_jax(records[0]))
    ck('record0_in_jax_concat_is_bit_exact_to_sealed_intermediate',
       np.array_equal(cat0_jax, inter['eager_concat']))
    sealed_concat_scores = np.asarray(ham.head_scores_jax(inter['eager_concat'], base_leaves), np.float64)
    sealed_concat_diff = float(np.max(np.abs(sealed_concat_scores - records[0]['sealed_scores'].astype(np.float64))))

    # ---- supervised per-customer feasible-action groups ------------------------
    groups = []
    for n in range(ham.RECORDS):
        g = ham.group_of(n, records[n], options[n])
        g['eligible_concat'] = concats[n][g['eligible']]
        g['candidate_concat'] = concats[n][g['candidates']]
        groups.append(g)
    supervised = [g for g in groups if g['target_local'] is not None]
    skipped = [dict(index=g['index'], key=g['key'], customer=g['customer'],
                    eligible=int(len(g['eligible'])), reason=g['reason'])
               for g in groups if g['target_local'] is None]
    ck('one_group_per_record', len(groups) == ham.RECORDS)
    ck('supervised_group_count', len(supervised) == REG['groups']['expected_supervised'])
    ck('eligible_group_sizes_in_registered_range',
       all(REG['groups']['min_eligible'] <= len(g['eligible']) <= REG['groups']['max_eligible'] for g in groups))

    initial_ce = [ham.ce_loss64(records[g['index']]['sealed_scores'][g['eligible']], g['target_local'])
                  for g in supervised]
    initial_argmax = [int(np.argmax(records[g['index']]['sealed_scores'][g['eligible']])) == g['target_local']
                      for g in supervised]

    # ---- head-only update ------------------------------------------------------
    order = [g['index'] for g in supervised]
    rows_by_index = {g['index']: (jnp.asarray(g['eligible_concat']), g['target_local']) for g in supervised}

    def group_ce(leaves, index):
        x, t = rows_by_index[index]
        s = (jnp.tanh(x @ leaves['k1'] + leaves['b1']) @ leaves['k2'] + leaves['b2'])[:, 0]
        return -jax.nn.log_softmax(s)[t]

    def scores_for(concat_rows, cur):
        x = jnp.asarray(concat_rows)
        return (jnp.tanh(x @ cur['k1'] + cur['b1']) @ cur['k2'] + cur['b2'])[:, 0]

    def mean_group_ce(cur):
        return float(np.mean([np.asarray(-jax.nn.log_softmax(scores_for(g['eligible_concat'], cur))[g['target_local']],
                                         np.float64) for g in supervised]))

    def argmax_correct(cur):
        return int(sum(int(np.argmax(np.asarray(scores_for(g['eligible_concat'], cur), np.float64))) == g['target_local']
                       for g in supervised))

    leaves = {kk: jnp.asarray(base_leaves[kk]) for kk in ham.LEAF_KEYS}
    optimizer = optax.adam(learning_rate=LR, b1=REG['update']['b1'], b2=REG['update']['b2'],
                           eps=REG['update']['eps'])
    opt_state = optimizer.init(leaves)
    grad_fn = jax.grad(group_ce)

    step_log = []
    epoch_log = [dict(epoch=0, mean_ce=mean_group_ce(leaves), argmax_correct=argmax_correct(leaves),
                      param_digest=digest_leaves(leaves))]
    step0_grad = None
    head_step1 = None
    last_grad = None
    started = time.perf_counter()
    for epoch in range(EPOCHS):
        for index in order:
            grads = grad_fn(leaves, index)
            if step0_grad is None:
                step0_grad = {kk: np.asarray(grads[kk], np.float32) for kk in ham.LEAF_KEYS}
            last_grad = {kk: np.asarray(grads[kk], np.float32) for kk in ham.LEAF_KEYS}
            updates, opt_state = optimizer.update(grads, opt_state)
            leaves = optax.apply_updates(leaves, updates)
            if head_step1 is None:
                head_step1 = {kk: np.asarray(leaves[kk], np.float32) for kk in ham.LEAF_KEYS}
            g = next(x for x in supervised if x['index'] == index)
            loss = float(np.asarray(-jax.nn.log_softmax(scores_for(g['eligible_concat'], leaves))[g['target_local']],
                                    np.float64))
            gnorm = float(np.sqrt(sum(float(np.sum(np.asarray(grads[kk], np.float64) ** 2)) for kk in ham.LEAF_KEYS)))
            step_log.append(dict(epoch=epoch, index=index, loss=loss, loss_hex=float(loss).hex(),
                                 grad_norm_hex=float(gnorm).hex(), param_digest=digest_leaves(leaves)))
        epoch_log.append(dict(epoch=epoch + 1, mean_ce=mean_group_ce(leaves),
                              argmax_correct=argmax_correct(leaves), param_digest=digest_leaves(leaves)))
    finished = time.perf_counter()

    final_leaves = {kk: np.asarray(leaves[kk], np.float32) for kk in ham.LEAF_KEYS}
    final_ce = [ham.ce_loss64(np.asarray(scores_for(g['eligible_concat'], leaves), np.float64), g['target_local'])
                for g in supervised]
    final_argmax = [int(np.argmax(np.asarray(scores_for(g['eligible_concat'], leaves), np.float64))) == g['target_local']
                    for g in supervised]
    changed = 0
    for g in supervised:
        before = records[g['index']]['sealed_scores'][g['candidates']].astype(np.float64)
        after = np.asarray(scores_for(g['candidate_concat'], leaves), np.float64)
        changed += int(int(np.argmax(after)) != int(np.argmax(before)))

    # ---- save and reload -------------------------------------------------------
    np.savez(HERE / 'HEAD_STEP0_GRAD.npz', **{kk: step0_grad[kk] for kk in ham.LEAF_KEYS})
    np.savez(HERE / 'HEAD_LAST_GRAD.npz', **{kk: last_grad[kk] for kk in ham.LEAF_KEYS})
    np.savez(HERE / 'HEAD_STEP1.npz', **{kk: head_step1[kk] for kk in ham.LEAF_KEYS})
    np.savez(HERE / 'HEAD_AFTER.npz', **{kk: final_leaves[kk] for kk in ham.LEAF_KEYS})

    reloaded = np.load(HERE / 'HEAD_AFTER.npz')
    reload_exact = {kk: bool(np.array_equal(np.asarray(reloaded[kk], np.float32), final_leaves[kk]))
                    for kk in ham.LEAF_KEYS}
    for kk in ham.LEAF_KEYS:
        ck('reload_leaf_bit_exact:' + kk, reload_exact[kk])
    reloaded_leaves = {kk: np.asarray(reloaded[kk], np.float32) for kk in ham.LEAF_KEYS}
    reload_forward_max = 0.0
    for rec in records:
        a = np.asarray(ham.head_scores_jax(concats[rec['index']], final_leaves))
        b = np.asarray(ham.head_scores_jax(concats[rec['index']], reloaded_leaves))
        if not np.array_equal(a, b):
            reload_forward_max = max(reload_forward_max,
                                     float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))))
    ck('reloaded_leaves_reproduce_post_update_forward_bit_exact', reload_forward_max == 0.0)

    step1 = np.load(HERE / 'HEAD_STEP1.npz')
    ck('step1_saved_and_reloadable',
       all(np.array_equal(np.asarray(step1[kk], np.float32), head_step1[kk]) for kk in ham.LEAF_KEYS))

    round_trip_max = 0.0
    for rec in records:
        b = np.asarray(ham.head_scores_jax(concats[rec['index']], base_leaves), np.float64)
        round_trip_max = max(round_trip_max, float(np.max(np.abs(b - rec['sealed_scores'].astype(np.float64)))))
    ck('sealed_leaf_round_trip_within_bound', round_trip_max <= TOL['forward_vs_sealed_scores_max_abs'])

    # ---- post-run input immutability and no-backbone guards ---------------------
    after_hashes = {name: ham.sha256_file(ham.record_path(int(name[-6:-4]), ARM)) for name in input_hashes}
    ck('sealed_inputs_unchanged_after_stage', after_hashes == input_hashes)
    ck('cpu_devices_only', all('CpuDevice' in d for d in devices))

    end_closure = seal.closure(tuple(Path(p) for p in REG['sources']))
    ck('actual_loaded_files_and_maps_start_end', seal.same(start_closure, end_closure))
    backbone = [p for p in list(end_closure['files']) + list(end_closure['modules'].values())
                if 'MASKCO_code' in p or p.lower().endswith('.ckpt')]
    ck('no_backbone_or_checkpoint_in_loaded_closure', not backbone)

    result = dict(
        checks_count=len(checks), failed=sum(not c['pass_'] for c in checks),
        failure_names=[c['name'] for c in checks if not c['pass_']], checks=checks,
        stage='delivery_n12_action_matching_head_update',
        scope='head-only four-leaf per-customer feasible-action CE; no backbone forward, no C0, no solver',
        sealed_arm=ARM, records=ham.RECORDS,
        head_parameter_elements=leaf_elements, head_leaf_shapes=leaf_shapes,
        head_leaf_identity_bit_exact=leaf_exact, head_identical_across_records=same_across,
        input_hashes=input_hashes,
        forward_vs_sealed_scores=dict(max_abs_diff=max_forward_diff, bit_exact_records=forward_exact,
                                      tolerance=TOL['forward_vs_sealed_scores_max_abs'],
                                      per_record=forward_rows),
        record0_in_jax_concat_bit_exact_to_sealed_intermediate=True,
        record0_sealed_concat_forward_max_abs_diff=sealed_concat_diff,
        groups=dict(total=len(groups), supervised=len(supervised), skipped=skipped,
                    eligible_sizes=[int(len(g['eligible'])) for g in groups],
                    candidate_sizes=[int(len(g['candidates'])) for g in groups]),
        update=dict(optimizer='optax.adam', learning_rate=LR, b1=REG['update']['b1'], b2=REG['update']['b2'],
                    eps=REG['update']['eps'], epochs=EPOCHS, steps=len(step_log),
                    order=order, fixed_order=True),
        loss=dict(initial_ce=initial_ce, final_ce=final_ce,
                  initial_mean_ce=float(np.mean(initial_ce)), final_mean_ce=float(np.mean(final_ce)),
                  initial_argmax_correct=int(sum(initial_argmax)), final_argmax_correct=int(sum(final_argmax)),
                  epoch_log=epoch_log, step_log=step_log),
        mechanism=dict(candidate_argmax_changed_groups=changed,
                       note='head-only memorisation of 29 sealed groups; not a trained control and not a gain'),
        reload=dict(reload_leaf_bit_exact=reload_exact,
                    reloaded_forward_bit_exact_max_abs_diff=reload_forward_max,
                    step1_reload_ok=True, sealed_leaf_round_trip_max_abs_diff=round_trip_max),
        saved_files=['HEAD_STEP0_GRAD.npz', 'HEAD_LAST_GRAD.npz', 'HEAD_STEP1.npz', 'HEAD_AFTER.npz'],
        saved_sha256={name: ham.sha256_file(HERE / name) for name in
                      ('HEAD_STEP0_GRAD.npz', 'HEAD_LAST_GRAD.npz', 'HEAD_STEP1.npz', 'HEAD_AFTER.npz')},
        resource=dict(available_ram_gib_before_stage=avail / 2 ** 30,
                      full_backbone_threshold_gib=BACKBONE_THRESHOLD_GIB, backbone_loaded=False,
                      devices=devices, cpu_affinity=psutil.Process().cpu_affinity()),
        memory_limitation=('Local available RAM %.3f GiB was below the registered %.1f GiB full-backbone threshold, '
                           'so this stage updated only the small scoring head over already sealed node '
                           'representations and never loaded the backbone.' % (avail / 2 ** 30, BACKBONE_THRESHOLD_GIB)),
        update_seconds=finished - started,
        source_actual_files=len(end_closure['files']), source_actual_maps=len(end_closure['modules']),
        formal_trained_control_qualified=False, formal_controls_qualified=False,
        full_request_budget_qualified=False, full_scientific_loaded_closure_qualified=False,
        scientific_data_generated=False, new_science_family=False, scientific_family3_started=False,
        optimizer_updates_of_model=0, learned_output_qualified=False, stable_online_gain=False,
    )
    save('RESULT.json', result)
    save('RUN_EXIT.json', dict(exit_code=int(result['failed'] > 0)))
    print(json.dumps({kk: v for kk, v in result.items()
                      if kk not in ('checks', 'forward_vs_sealed_scores', 'loss', 'input_hashes')}))
    return int(result['failed'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
