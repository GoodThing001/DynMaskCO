"""Independent audit of the delivery n12 scoring-head update / save / reload.

The audit re-reads the sealed original artifacts, uses the sealed original head
module as the forward reference, and re-derives the per-customer feasible-action
CE update arithmetic (softmax CE, analytic four-leaf gradient, Adam recursion)
with its own float64 implementation.  It does not import run.py.
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
RESULT = json.loads((HERE / 'RESULT.json').read_text(encoding='utf-8'))
RUN_EXIT = json.loads((HERE / 'RUN_EXIT.json').read_text(encoding='utf-8'))
EPOCHS = REG['update']['epochs']
LR = REG['update']['learning_rate']
B1, B2, EPS = REG['update']['b1'], REG['update']['b2'], REG['update']['eps']
LEAF_KEYS = ham.LEAF_KEYS


def save(name, value):
    seal.save(HERE / name, value)


def own_concat(rec):
    """Audit-owned reconstruction of the recorded head input."""
    h = np.array(rec['latent'][0], np.float32)
    a = np.array(rec['actions'], np.int32)
    parts = [h[a[:, 0]], h[a[:, 3]], h[a[:, 4]]]
    for key in ('route_weights', 'prefix_weights', 'suffix_weights'):
        parts.append(np.asarray(rec[key], np.float32).dot(h))
    parts.append(np.asarray(rec['slot_position_length'], np.float32))
    return np.concatenate(parts, axis=1)


def own_scores64(cat, leaves):
    z1 = np.asarray(cat, np.float64) @ np.asarray(leaves['k1'], np.float64) + np.asarray(leaves['b1'], np.float64)
    a = np.tanh(z1)
    return (a @ np.asarray(leaves['k2'], np.float64))[:, 0] + np.asarray(leaves['b2'], np.float64)


def own_ce64(s, t):
    s = np.asarray(s, np.float64)
    m = s.max()
    return float(m + np.log(np.exp(s - m).sum()) - s[t])


def own_grad64(cat, leaves, t):
    x = np.asarray(cat, np.float64)
    k1 = np.asarray(leaves['k1'], np.float64)
    b1 = np.asarray(leaves['b1'], np.float64)
    k2 = np.asarray(leaves['k2'], np.float64)
    a = np.tanh(x @ k1 + b1)
    s = (a @ k2)[:, 0]
    m = s.max()
    p = np.exp(s - m)
    p = p / p.sum()
    ds = p.copy()
    ds[t] -= 1.0
    ds = ds[:, None]
    dz1 = (ds @ k2.T) * (1.0 - a * a)
    return {'k1': x.T @ dz1, 'b1': dz1.sum(axis=0), 'k2': a.T @ ds, 'b2': ds.sum(axis=0)}


def own_adam64(leaves, grads, m_state, v_state, step):
    out, m_new, v_new = {}, {}, {}
    for k in LEAF_KEYS:
        g = np.asarray(grads[k], np.float64)
        m = B1 * np.asarray(m_state[k], np.float64) + (1.0 - B1) * g
        v = B2 * np.asarray(v_state[k], np.float64) + (1.0 - B2) * g * g
        mh = m / (1.0 - B1 ** step)
        vh = v / (1.0 - B2 ** step)
        out[k] = np.asarray(leaves[k], np.float64) - LR * mh / (np.sqrt(vh) + EPS)
        m_new[k], v_new[k] = m, v
    return out, m_new, v_new


def main():
    seal.initialize()
    import jax
    import jax.numpy as jnp

    # Warm up every lazily loaded native path, submodule and stdlib codec before
    # the closure (see ../action_matching_head_update_root_v1/CLOSURE_DIAGNOSIS.json).
    import encodings.cp437  # noqa: F401
    w = {'k1': jnp.ones((3075, 64), jnp.float32), 'b1': jnp.zeros((64,), jnp.float32),
         'k2': jnp.ones((64, 1), jnp.float32), 'b2': jnp.zeros((1,), jnp.float32)}

    def warm_ce(leaves):
        x = jnp.ones((4, 3075), jnp.float32)
        s = (jnp.tanh(x @ leaves['k1'] + leaves['b1']) @ leaves['k2'] + leaves['b2'])[:, 0]
        return -jax.nn.log_softmax(s)[0]

    np.asarray(jax.grad(warm_ce)(w))
    ham.construct_head_leaves(seed=ham.HEAD_SEED)
    ham.head_scores_jax(np.zeros((2, 3075), np.float32), {kk: np.asarray(v) for kk, v in w.items()})
    ham.load_record(0, ARM)

    start_closure = seal.closure(tuple(Path(p) for p in REG['sources']))
    save('AUDIT_SOURCE_START.json', start_closure)
    started = time.perf_counter()
    checks = []

    def ck(name, value):
        checks.append(dict(name=name, pass_=bool(value)))

    def close(a, b, tol):
        return float(np.max(np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64)))) <= tol

    # 1. registered source identity
    for path, want in REG['sources'].items():
        ck('registered_source:' + path, ham.sha256_file(path) == want)

    # 2. sealed input identity recomputed
    manifest = ham.sealed_manifest_hashes()
    input_hashes = {}
    for n in range(ham.RECORDS):
        name = '%s_%02d.npz' % (ARM, n)
        got = ham.sha256_file(ham.record_path(n, ARM))
        input_hashes[name] = got
        ck('sealed_input_sha256:' + name, got == manifest.get(name))
    ck('audit_matches_recorded_input_hashes', input_hashes == RESULT['input_hashes'])
    ck('sealed_action_options_sha256', ham.sha256_file(ham.ACTION_OPTIONS) == REG['sealed']['action_options_sha256'])
    ck('sealed_head_source_sha256', ham.sha256_file(ham.SEALED_HEAD_SOURCE) == REG['sealed']['head_source_sha256'])
    ck('sealed_intermediates_sha256', ham.sha256_file(ham.SEALED_INTERMEDIATES) == REG['sealed']['intermediates_sha256'])

    # 3. sealed head leaf identity re-derived in this process
    records = [ham.load_record(n, ARM) for n in range(ham.RECORDS)]
    options = ham.action_options()
    sealed_leaves = {k: np.asarray(records[0]['leaves'][k], np.float32) for k in LEAF_KEYS}
    ck('all_records_share_one_head_identity',
       all(all(np.array_equal(r['leaves'][k], sealed_leaves[k]) for k in LEAF_KEYS) for r in records))
    ref = ham.construct_head_leaves(seed=ham.HEAD_SEED)
    for k in LEAF_KEYS:
        ck('sealed_leaf_is_construct_head_seed20261012:' + k, np.array_equal(ref[k], sealed_leaves[k]))

    # 4. audit-owned concat reconstruction vs recorded digests and sealed intermediate
    concats = [own_concat(r) for r in records]
    ck('audit_concat_sha256_matches_recorded',
       all(ham.sha256_array(concats[i]) == RESULT['forward_vs_sealed_scores']['per_record'][i]['concat_sha256']
           for i in range(ham.RECORDS)))
    inter = np.load(ham.SEALED_INTERMEDIATES)
    cat0_jax = np.asarray(ham.concat_jax(records[0]))
    ck('record0_in_jax_concat_bit_exact_to_sealed_intermediate',
       np.array_equal(cat0_jax, inter['eager_concat']))
    ck('audit_concat_numpy_vs_sealed_intermediate_denormal_only',
       float(np.max(np.abs(concats[0].astype(np.float64) - inter['eager_concat'].astype(np.float64))))
       <= TOL['audit_concat_max_abs'])

    # 5. original forward reference: audit float64 vs sealed scores
    audit_forward_diff = 0.0
    for i, rec in enumerate(records):
        d = float(np.max(np.abs(own_scores64(concats[i], sealed_leaves) - rec['sealed_scores'].astype(np.float64))))
        audit_forward_diff = max(audit_forward_diff, d)
    ck('audit_float64_forward_vs_sealed_scores_within_bound',
       audit_forward_diff <= TOL['forward_vs_sealed_scores_max_abs'])
    ck('recorded_forward_diff_matches_audit_bound',
       RESULT['forward_vs_sealed_scores']['max_abs_diff'] <= TOL['forward_vs_sealed_scores_max_abs'])

    # 6. groups, targets and initial CE
    groups = []
    for n in range(ham.RECORDS):
        g = ham.group_of(n, records[n], options[n])
        groups.append(g)
    supervised = [g for g in groups if g['target_local'] is not None]
    ck('audit_supervised_group_count', len(supervised) == RESULT['groups']['supervised'])
    ck('audit_skipped_records_match', [g['index'] for g in groups if g['target_local'] is None]
       == [s['index'] for s in RESULT['groups']['skipped']])
    audit_initial = [own_ce64(records[g['index']]['sealed_scores'][g['eligible']], g['target_local'])
                     for g in supervised]
    ck('audit_initial_ce_bit_exact_to_recorded',
       np.array_equal(np.asarray(audit_initial, np.float64), np.asarray(RESULT['loss']['initial_ce'], np.float64)))

    # 7. independent full re-derivation of the head-only update (float64)
    order = [g['index'] for g in supervised]
    ck('audit_order_matches_recorded', order == RESULT['update']['order'])
    cat_by_index = {g['index']: concats[g['index']][g['eligible']] for g in supervised}
    tgt_by_index = {g['index']: g['target_local'] for g in supervised}
    cur = {k: np.asarray(sealed_leaves[k], np.float64) for k in LEAF_KEYS}
    m_state = {k: np.zeros_like(cur[k]) for k in LEAF_KEYS}
    v_state = {k: np.zeros_like(cur[k]) for k in LEAF_KEYS}
    traj = []
    step = 0
    for epoch in range(EPOCHS):
        for index in order:
            step += 1
            g = own_grad64(cat_by_index[index], cur, tgt_by_index[index])
            cur, m_state, v_state = own_adam64(cur, g, m_state, v_state, step)
            if step == 1:
                first_grad64 = g
        traj.append(float(np.mean([own_ce64(own_scores64(cat_by_index[g['index']], cur), g['target_local'])
                                   for g in supervised])))
    ck('audit_step_count_matches_recorded', step == RESULT['update']['steps'])

    recorded_traj = [e['mean_ce'] for e in RESULT['loss']['epoch_log'][1:]]
    traj_diff = float(np.max(np.abs(np.asarray(traj) - np.asarray(recorded_traj))))
    ck('audit_epoch_ce_trajectory_within_bound', traj_diff <= TOL['audit_trajectory_loss_max_abs'])

    # 8. recorded step-0 gradient versus audit analytic gradient
    g0 = np.load(HERE / 'HEAD_STEP0_GRAD.npz')
    grad_diff = max(float(np.max(np.abs(np.asarray(g0[k], np.float64) - first_grad64[k]))) for k in LEAF_KEYS)
    ck('audit_step0_gradient_within_bound', grad_diff <= TOL['audit_gradient_max_abs'])

    # 9. Adam step-1 arithmetic independently reproduced
    s1 = np.load(HERE / 'HEAD_STEP1.npz')
    pred1 = {k: np.asarray(sealed_leaves[k], np.float64)
                - LR * np.asarray(g0[k], np.float64) / (np.abs(np.asarray(g0[k], np.float64)) + EPS) for k in LEAF_KEYS}
    step1_diff = max(float(np.max(np.abs(pred1[k] - np.asarray(s1[k], np.float64)))) for k in LEAF_KEYS)
    ck('adam_step1_arithmetic_within_bound', step1_diff <= TOL['adam_step1_max_abs'])

    # 10. final parameters and final loss
    aft = np.load(HERE / 'HEAD_AFTER.npz')
    aft_leaves = {k: np.asarray(aft[k], np.float32) for k in LEAF_KEYS}
    final_param_diff = max(float(np.max(np.abs(aft_leaves[k].astype(np.float64) - cur[k]))) for k in LEAF_KEYS)
    ck('audit_final_parameters_within_bound', final_param_diff <= TOL['audit_final_param_max_abs'])
    ck('head_actually_changed_by_update',
       all(not np.array_equal(aft_leaves[k], sealed_leaves[k]) for k in LEAF_KEYS))
    audit_final_ce = [own_ce64(own_scores64(cat_by_index[g['index']], aft_leaves), g['target_local'])
                      for g in supervised]
    final_ce_diff = float(np.max(np.abs(np.asarray(audit_final_ce) - np.asarray(RESULT['loss']['final_ce']))))
    ck('audit_final_ce_within_bound', final_ce_diff <= TOL['audit_reference_loss_max_abs'])
    audit_final_argmax = int(sum(int(np.argmax(own_scores64(cat_by_index[g['index']], aft_leaves))) == g['target_local']
                                 for g in supervised))
    ck('audit_final_argmax_correct_matches',
       audit_final_argmax == RESULT['loss']['final_argmax_correct'])

    # 11. saved artifacts and reload
    for name, want in RESULT['saved_sha256'].items():
        ck('saved_artifact_sha256:' + name, ham.sha256_file(HERE / name) == want)
    for k in LEAF_KEYS:
        ck('reload_leaf_bit_exact:' + k, bool(RESULT['reload']['reload_leaf_bit_exact'][k]))
        ck('audit_reload_matches_final_leaf:' + k, np.array_equal(aft_leaves[k], np.asarray(aft[k], np.float32)))
    reload_forward_max = 0.0
    for rec in records:
        x = jnp.asarray(concats[rec['index']])
        fresh = np.load(HERE / 'HEAD_AFTER.npz')
        a = np.asarray((jnp.tanh(x @ jnp.asarray(aft_leaves['k1']) + jnp.asarray(aft_leaves['b1']))
                        @ jnp.asarray(aft_leaves['k2']) + jnp.asarray(aft_leaves['b2']))[:, 0])
        b = np.asarray((jnp.tanh(x @ jnp.asarray(np.asarray(fresh['k1'], np.float32))
                                 + jnp.asarray(np.asarray(fresh['b1'], np.float32)))
                        @ jnp.asarray(np.asarray(fresh['k2'], np.float32))
                        + jnp.asarray(np.asarray(fresh['b2'], np.float32)))[:, 0])
        if not np.array_equal(a, b):
            reload_forward_max = max(reload_forward_max, float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64)))))
    ck('reloaded_leaves_reproduce_forward_bit_exact', reload_forward_max == 0.0)
    round_trip = 0.0
    for rec in records:
        x = jnp.asarray(concats[rec['index']])
        out = np.asarray((jnp.tanh(x @ jnp.asarray(sealed_leaves['k1']) + jnp.asarray(sealed_leaves['b1']))
                          @ jnp.asarray(sealed_leaves['k2']) + jnp.asarray(sealed_leaves['b2']))[:, 0], np.float64)
        round_trip = max(round_trip, float(np.max(np.abs(out - rec['sealed_scores'].astype(np.float64)))))
    ck('sealed_leaf_round_trip_within_registered_bound',
       round_trip <= TOL['forward_vs_sealed_scores_max_abs'])

    # 12. sealed inputs untouched and no backbone anywhere in this audit
    after_hashes = {name: ham.sha256_file(ham.record_path(int(name[-6:-4]), ARM)) for name in input_hashes}
    ck('sealed_inputs_unchanged_after_audit', after_hashes == input_hashes)
    end_closure = seal.closure(tuple(Path(p) for p in REG['sources']))
    save('AUDIT_SOURCE_END.json', end_closure)
    ck('audit_loaded_files_and_maps_start_end', seal.same(start_closure, end_closure))
    v1 = ROOT / 'research/review_3080_20261010/delivery/action_matching_head_update_root_v1'
    v1_result = json.loads((v1 / 'RESULT.json').read_text(encoding='utf-8'))
    v1_exit = json.loads((v1 / 'RUN_EXIT.json').read_text(encoding='utf-8'))
    v1_diag = json.loads((v1 / 'CLOSURE_DIAGNOSIS.json').read_text(encoding='utf-8'))
    ck('v1_failures_preserved_not_converted', v1_result['failed'] == 2 and v1_exit['exit_code'] == 1)
    ck('v1_has_no_source_end_closure', not (v1 / 'SOURCE_END.json').exists())
    ck('v1_closure_delta_was_codec_only',
       v1_diag['changed_files'] == [] and v1_diag['removed_files'] == []
       and v1_diag['added_modules'] == ['encodings.cp437'])
    backbone = [p for p in list(end_closure['files']) + list(end_closure['modules'].values())
                if 'MASKCO_code' in p or p.lower().endswith('.ckpt')]
    ck('audit_no_backbone_or_checkpoint_loaded', not backbone)
    ck('cpu_devices_only', jax.default_backend() == 'cpu'
       and all(str(getattr(dv, 'platform', '')) == 'cpu' for dv in jax.devices()))

    # 13. recorded ledger consistency and memory limitation
    ck('exit_codes_consistent_with_failed_count',
       RUN_EXIT['exit_code'] == int(RESULT['failed'] > 0)
       and len(RESULT['failure_names']) == RESULT['failed']
       and len(RESULT['checks']) == RESULT['checks_count'])
    ck('recorded_failed_count_zero', RESULT['failed'] == 0)
    ck('memory_limitation_recorded', bool(RESULT.get('memory_limitation'))
       and RESULT['resource']['available_ram_gib_before_stage'] < REG['resource']['full_backbone_threshold_gib'])
    ck('backbone_recorded_not_loaded', RESULT['resource']['backbone_loaded'] is False)
    for flag in ('formal_trained_control_qualified', 'formal_controls_qualified', 'full_request_budget_qualified',
                 'full_scientific_loaded_closure_qualified', 'scientific_data_generated', 'new_science_family',
                 'scientific_family3_started', 'learned_output_qualified', 'stable_online_gain'):
        ck('recorded_flag_false:' + flag, RESULT[flag] is False)
    ck('no_model_optimizer_update_recorded', RESULT['optimizer_updates_of_model'] == 0)

    finished = time.perf_counter()
    audit = dict(checks_count=len(checks), failed=sum(not c['pass_'] for c in checks),
                 failure_names=[c['name'] for c in checks if not c['pass_']], checks=checks,
                 audit_forward_float64_vs_sealed_max_abs=audit_forward_diff,
                 audit_concat_numpy_vs_sealed_intermediate_max_abs=
                 float(np.max(np.abs(concats[0].astype(np.float64) - inter['eager_concat'].astype(np.float64)))),
                 audit_step0_gradient_max_abs_diff=grad_diff,
                 audit_adam_step1_max_abs_diff=step1_diff,
                 audit_epoch_ce_trajectory_max_abs_diff=traj_diff,
                 audit_final_parameter_max_abs_diff=final_param_diff,
                 audit_final_ce_max_abs_diff=final_ce_diff,
                 audit_final_argmax_correct=audit_final_argmax,
                 audit_reload_forward_max_abs_diff=reload_forward_max,
                 sealed_leaf_round_trip_max_abs_diff=round_trip,
                 recorded_update_steps=RESULT['update']['steps'], audit_update_steps=step,
                 recorded_mean_ce=dict(initial=RESULT['loss']['initial_mean_ce'],
                                       final=RESULT['loss']['final_mean_ce']),
                 audit_mean_ce_final=float(np.mean(audit_final_ce)),
                 actual_source_files=len(start_closure['files']), actual_source_maps=len(start_closure['modules']),
                 audit_seconds=finished - started,
                 formal_trained_control_qualified=False, formal_controls_qualified=False,
                 full_request_budget_qualified=False, full_scientific_loaded_closure_qualified=False,
                 scientific_data_generated=False, new_science_family=False, scientific_family3_started=False,
                 stable_online_gain=False)
    save('INDEPENDENT_AUDIT.json', audit)
    save('AUDIT_RUN_EXIT.json', dict(exit_code=int(audit['failed'] > 0)))
    print(json.dumps({k: v for k, v in audit.items() if k != 'checks'}))
    return int(audit['failed'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
