"""Independent audit of delivery n13: held-out generalisation and reproducible update.

Own float64 re-derivation of the split, the head arithmetic, the per-customer
feasible-action CE, the SGD/Adam recursions and the held-out metrics.  Does not
import run.py.
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
KEYS = ham.LEAF_KEYS


def save(name, value):
    seal.save(HERE / name, value)


def own_concat(rec):
    h = np.array(rec['latent'][0], np.float32)
    a = np.array(rec['actions'], np.int32)
    parts = [h[a[:, 0]], h[a[:, 3]], h[a[:, 4]]]
    for key in ('route_weights', 'prefix_weights', 'suffix_weights'):
        parts.append(np.asarray(rec[key], np.float32).dot(h))
    parts.append(np.asarray(rec['slot_position_length'], np.float32))
    return np.concatenate(parts, axis=1)


def scores64(cat, leaves):
    a = np.tanh(np.asarray(cat, np.float64) @ np.asarray(leaves['k1'], np.float64)
                + np.asarray(leaves['b1'], np.float64))
    return (a @ np.asarray(leaves['k2'], np.float64))[:, 0] + np.asarray(leaves['b2'], np.float64)


def ce64(s, t):
    s = np.asarray(s, np.float64)
    m = s.max()
    return float(m + np.log(np.exp(s - m).sum()) - s[t])


def grad64(cat, leaves, t):
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


def main():
    seal.initialize()
    import jax
    import jax.numpy as jnp

    import encodings.cp437  # noqa: F401
    w = {'k1': jnp.ones((3075, 64), jnp.float32), 'b1': jnp.zeros((64,), jnp.float32),
         'k2': jnp.ones((64, 1), jnp.float32), 'b2': jnp.zeros((1,), jnp.float32)}

    def warm_ce(leaves):
        x = jnp.ones((4, 3075), jnp.float32)
        s = (jnp.tanh(x @ leaves['k1'] + leaves['b1']) @ leaves['k2'] + leaves['b2'])[:, 0]
        return -jax.nn.log_softmax(s)[0]

    np.asarray(jax.grad(warm_ce)(w))
    ham.construct_head_leaves(seed=ham.HEAD_SEED)
    ham.load_record(0, ARM)

    start_closure = seal.closure(tuple(Path(p) for p in REG['sources']))
    save('AUDIT_SOURCE_START.json', start_closure)
    started = time.perf_counter()
    checks = []

    def ck(name, value):
        checks.append(dict(name=name, pass_=bool(value)))

    # 1. source and sealed identity
    for path, want in REG['sources'].items():
        ck('registered_source:' + path, ham.sha256_file(path) == want)
    manifest = ham.sealed_manifest_hashes()
    input_hashes = {}
    for n in range(ham.RECORDS):
        name = '%s_%02d.npz' % (ARM, n)
        got = ham.sha256_file(ham.record_path(n, ARM))
        input_hashes[name] = got
        ck('sealed_input_sha256:' + name, got == manifest.get(name))
    ck('audit_matches_recorded_input_hashes', input_hashes == RESULT['input_hashes'])
    ck('round5_grad_artifact_sha256',
       ham.sha256_file(REG['round5_artifacts']['step0_grad']) == REG['round5_artifacts']['step0_grad_sha256'])
    records = [ham.load_record(n, ARM) for n in range(ham.RECORDS)]
    options = ham.action_options()
    sealed_leaves = {k: np.asarray(records[0]['leaves'][k], np.float32) for k in KEYS}
    ref = ham.construct_head_leaves(seed=ham.HEAD_SEED)
    for k in KEYS:
        ck('sealed_leaf_is_construct_head_seed20261012:' + k, np.array_equal(ref[k], sealed_leaves[k]))

    # 2. audit-owned reconstruction and forward reference
    concats = [own_concat(r) for r in records]
    fwd_diff = max(float(np.max(np.abs(scores64(concats[i], sealed_leaves)
                                        - records[i]['sealed_scores'].astype(np.float64))))
                   for i in range(ham.RECORDS))
    ck('audit_float64_forward_vs_sealed_scores_within_bound',
       fwd_diff <= TOL['forward_vs_sealed_scores_max_abs'])

    # 3. audit-owned split and metrics
    groups = []
    for n in range(ham.RECORDS):
        g = ham.group_of(n, records[n], options[n])
        if g['target_local'] is None:
            continue
        g['rows'] = concats[n][g['eligible']]
        groups.append(g)
    supervised = [g['index'] for g in groups]
    by_index = {g['index']: g for g in groups}
    held = [supervised[p] for p in REG['split']['held_out_positions']]
    train = [i for i in supervised if i not in held]
    ck('audit_split_matches_recorded',
       train == RESULT['split']['train_records'] and held == RESULT['split']['held_out_records'])

    def metrics64(indices, leaves):
        ce, hits, full_hits = [], 0, 0
        for i in indices:
            g = by_index[i]
            s = scores64(g['rows'], leaves)
            ce.append(ce64(s, g['target_local']))
            hits += int(int(np.argmax(s)) == g['target_local'])
            full_hits += int(int(np.argmax(scores64(concats[i], leaves))) == g['target_row'])
        return dict(mean_ce=float(np.mean(ce)), argmax_correct=hits, n=len(indices),
                    full_candidate_argmax_correct=full_hits)

    before = dict(train=metrics64(train, sealed_leaves), held_out=metrics64(held, sealed_leaves))
    ck('audit_before_metrics_ce_within_bound',
       max(abs(before['train']['mean_ce'] - RESULT['arms'][REG['primary_arm']]['before']['train']['mean_ce']),
           abs(before['held_out']['mean_ce'] - RESULT['arms'][REG['primary_arm']]['before']['held_out']['mean_ce']))
       <= TOL['audit_metric_max_abs'])
    for part in ('train', 'held_out'):
        ck('audit_before_argmax_matches:' + part,
           before[part]['argmax_correct'] == RESULT['arms'][REG['primary_arm']]['before'][part]['argmax_correct'])

    # 4. audit-owned trajectory re-derivation per arm
    arm_report = {}
    for spec in REG['arms']:
        name = spec['name']
        arm = RESULT['arms'][name]
        cur = {k: np.asarray(sealed_leaves[k], np.float64) for k in KEYS}
        m = {k: np.zeros_like(cur[k]) for k in KEYS}
        v = {k: np.zeros_like(cur[k]) for k in KEYS}
        traj = []
        g_first = None
        step = 0
        for _ in range(EPOCHS):
            for index in train:
                step += 1
                g = grad64(by_index[index]['rows'], cur, by_index[index]['target_local'])
                if g_first is None:
                    g_first = g
                if spec['kind'] == 'sgd':
                    cur = {k: cur[k] - spec['learning_rate'] * g[k] for k in KEYS}
                else:
                    b1, b2, eps = spec['b1'], spec['b2'], spec['eps']
                    for k in KEYS:
                        m[k] = b1 * m[k] + (1.0 - b1) * g[k]
                        v[k] = b2 * v[k] + (1.0 - b2) * g[k] * g[k]
                        cur[k] = cur[k] - spec['learning_rate'] * (m[k] / (1.0 - b1 ** step)) / (
                            np.sqrt(v[k] / (1.0 - b2 ** step)) + eps)
            traj.append(float(np.mean([ce64(scores64(by_index[i]['rows'], cur), by_index[i]['target_local'])
                                       for i in train])))
        aft = {k: np.asarray(np.load(HERE / ('HEAD_AFTER_%s.npz' % name))[k], np.float32) for k in KEYS}
        param_diff = max(float(np.max(np.abs(aft[k].astype(np.float64) - cur[k]))) for k in KEYS)
        traj_diff = float(np.max(np.abs(np.asarray(traj) - np.asarray(arm['train_ce_trajectory']))))
        after = dict(train=metrics64(train, aft), held_out=metrics64(held, aft))
        ce_diff = max(abs(after[part]['mean_ce'] - arm['after'][part]['mean_ce'])
                      for part in ('train', 'held_out'))
        arm_report[name] = dict(
            kind=spec['kind'], learning_rate=spec['learning_rate'], steps=step,
            param_max_abs_diff=param_diff, trajectory_max_abs_diff=traj_diff,
            after_ce_max_abs_diff=ce_diff,
            after_argmax_matches=all(after[part]['argmax_correct'] == arm['after'][part]['argmax_correct']
                                     for part in ('train', 'held_out')),
            audit_held_out_after=after['held_out'], recorded_held_out_after=arm['after']['held_out'],
            audit_train_after=after['train'],
        )
        if spec['kind'] == 'sgd':
            ck('sgd_trajectory_reproduced:' + name,
               traj_diff <= TOL['audit_sgd_trajectory_max_abs']
               and param_diff <= TOL['audit_sgd_final_param_max_abs'])
        else:
            arm_report[name]['note'] = ('Adam control: cross-library divergence is reported, not asserted; '
                                        'round 5 already recorded it before this registration.')
        ck('audit_after_metrics_reproduced:' + name, ce_diff <= TOL['audit_metric_max_abs'])

    ck('primary_arm_is_sgd', next(s['kind'] for s in REG['arms'] if s['name'] == REG['primary_arm']) == 'sgd')

    # 5. mechanism: float32 versus float64 gradient and the resulting step
    first = train[0]
    x = jnp.asarray(by_index[first]['rows'])
    t = by_index[first]['target_local']

    def ce_one(leaves):
        s = (jnp.tanh(x @ leaves['k1'] + leaves['b1']) @ leaves['k2'] + leaves['b2'])[:, 0]
        return -jax.nn.log_softmax(s)[t]

    gf32 = jax.grad(ce_one)({k: jnp.asarray(sealed_leaves[k]) for k in KEYS})
    gf32 = {k: np.asarray(gf32[k], np.float64) for k in KEYS}
    gf64 = grad64(by_index[first]['rows'], sealed_leaves, t)
    gdiff = max(float(np.max(np.abs(gf32[k] - gf64[k]))) for k in KEYS)
    a32 = np.concatenate([gf32[k].ravel() for k in KEYS])
    a64 = np.concatenate([gf64[k].ravel() for k in KEYS])
    nz = (a32 != 0.0) | (a64 != 0.0)
    sign_agree = float(np.mean(np.sign(a32[nz]) == np.sign(a64[nz]))) if nz.any() else None
    adam_lr = next(s['learning_rate'] for s in REG['arms'] if s['kind'] == 'adam')
    adam_eps = next(s['eps'] for s in REG['arms'] if s['kind'] == 'adam')
    step32 = adam_lr * np.abs(a32) / (np.abs(a32) + adam_eps)
    step64 = adam_lr * np.abs(a64) / (np.abs(a64) + adam_eps)
    adam_step_diff = float(np.max(np.abs(step32 - step64)))
    sgd_lr = next(s['learning_rate'] for s in REG['arms'] if s['name'] == REG['primary_arm'])
    sgd_step_diff = float(np.max(np.abs(sgd_lr * a32 - sgd_lr * a64)))
    mechanism = dict(
        gradient_float32_vs_float64_max_abs=gdiff,
        nonzero_union_coordinates=int(nz.sum()),
        gradient_sign_agreement_on_nonzero_union=sign_agree,
        adam_t1_step_max_abs_diff_f32_vs_f64=adam_step_diff,
        sgd_step_max_abs_diff_f32_vs_f64=sgd_step_diff,
        adam_saturated_fraction_on_first_group=float(np.mean(np.abs(step32) > 0.5 * adam_lr)),
        recorded_train_step0_zero_gradient_fraction=RESULT['mechanism']['train_step0_zero_gradient_fraction'],
        recorded_adam_t1_saturated_fraction=RESULT['mechanism']['adam_t1_saturated_fraction'],
        conclusion=('Adam makes the update depend on the gradient sign and saturates it to +/-lr, so float32 '
                    'rounding-floor coordinates receive full-size steps whose sign is not reproducible across '
                    'arithmetics (%.6g Adam step difference) whereas the SGD step stays proportional to the '
                    'gradient (%.6g difference), which is why only the SGD trajectory reproduces.'
                    % (adam_step_diff, sgd_step_diff)),
    )

    # 6. guards and ledger
    after_hashes = {name: ham.sha256_file(ham.record_path(int(name[-6:-4]), ARM)) for name in input_hashes}
    ck('sealed_inputs_unchanged_after_audit', after_hashes == input_hashes)
    end_closure = seal.closure(tuple(Path(p) for p in REG['sources']))
    save('AUDIT_SOURCE_END.json', end_closure)
    ck('audit_loaded_files_and_maps_start_end', seal.same(start_closure, end_closure))
    backbone = [p for p in list(end_closure['files']) + list(end_closure['modules'].values())
                if 'MASKCO_code' in p or p.lower().endswith('.ckpt')]
    ck('audit_no_backbone_or_checkpoint_loaded', not backbone)
    ck('cpu_devices_only', jax.default_backend() == 'cpu'
       and all(str(getattr(dv, 'platform', '')) == 'cpu' for dv in jax.devices()))
    ck('exit_codes_consistent',
       RUN_EXIT['exit_code'] == int(RESULT['failed'] > 0)
       and len(RESULT['failure_names']) == RESULT['failed']
       and len(RESULT['checks']) == RESULT['checks_count'])
    ck('memory_limitation_recorded', bool(RESULT.get('memory_limitation'))
       and RESULT['resource']['available_ram_gib_before_stage'] < REG['resource']['full_backbone_threshold_gib'])
    ck('backbone_recorded_not_loaded', RESULT['resource']['backbone_loaded'] is False)
    for flag in ('formal_trained_control_qualified', 'formal_controls_qualified', 'full_request_budget_qualified',
                 'full_scientific_loaded_closure_qualified', 'scientific_data_generated', 'new_science_family',
                 'scientific_family3_started', 'learned_output_qualified', 'stable_online_gain'):
        ck('recorded_flag_false:' + flag, RESULT[flag] is False)
    ck('held_out_records_disjoint_from_train', not set(train) & set(held))
    ck('all_three_arms_saved', all((HERE / ('HEAD_AFTER_%s.npz' % s['name'])).exists() for s in REG['arms']))
    ck('no_arm_selected_by_outcome',
       RESULT['primary_arm'] == REG['primary_arm']
       and set(RESULT['arms']) == {s['name'] for s in REG['arms']})

    finished = time.perf_counter()
    audit = dict(checks_count=len(checks), failed=sum(not c['pass_'] for c in checks),
                 failure_names=[c['name'] for c in checks if not c['pass_']], checks=checks,
                 audit_forward_float64_vs_sealed_max_abs=fwd_diff,
                 audit_before=before, audit_arms=arm_report, mechanism=mechanism,
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
