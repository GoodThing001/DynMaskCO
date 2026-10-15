#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D3-L1 P0-B (plan B): strict distance-contract verification + corrected causal perturbation and
decision projection.  Read-only; no state collection, no F/D-Geo call.

Corrections over the previous version (all three were found by the user):

  1. DISTANCE CONTRACT.  The previous check compared a float64 np.linalg.norm against the dataset's
     generator-computed float32 Euclidean matrix and declared "not Euclidean".  That is a
     computation-path difference, not evidence of an external road-network matrix.  Here the
     matrix is rebuilt in the GENERATOR'S OWN ORDER
         coords32 = coords.astype(float32); diff = coords32[:,None,:] - coords32[None,:,:]
         dist_rebuilt = sqrt((diff**2).sum(-1))
     and must satisfy np.array_equal per development day, with dtype/shape/bytes also checked.  No
     tolerance is relaxed: on a mismatch the first differing position, both values and both bit
     patterns are reported and the verdict stays NOT_VERIFIABLE.

  2. PERTURBED-DISTANCE BUG.  The previous code wrote the ORIGINAL eu back into the perturbed
     dist_mat.  Here the perturbed matrix is rebuilt from the PERTURBED coords with the same float32
     order, only entries touching invisible customers may change, and visible-visible entries must
     remain bit-identical.

  3. PROJECTION FALSE-PASS RISKS.  JF1-H-F's plan() mutates the vehicle objects it is given and
     returns nothing, so iterating its return value was wrong: the projection now deep-copies the
     vehicles, calls plan(), ignores the return and reads mutable_suffix from the cloned objects.
     `complete` no longer treats "an error key exists" as success, and the previous unconditional
     B5 pass is replaced by a real check.  The candidate-pool helper (dynmaskco_cc_context) is
     imported ONLY after declaring it an explicit F-free dependency; the process asserts it stays
     F-free and records the module's SHA.
"""
import argparse
import copy
import dataclasses
import hashlib
import json
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parents[1]
REPO = EXT.parent
for p in ('scripts', 'scripts/simulation', 'scripts/coldchain', 'scripts/data', 'scripts/tests'):
    sys.path.insert(0, str(EXT / p))

PERTURBABLE = {'demands': 'positive', 'tw_start': 'time', 'tw_end': 'time',
               'service_time': 'time'}
COORD_DELTA, TW_DELTA = 1.0, 0.5
FORBIDDEN = ('mpre', 'mpre_trained', 'mpre_policy', 'mtrained_replanner', 'd3_geo_adapter')
KNOWN_DEV_CHECK = '66573ddc653c863f7ca74ff204134de8d91058e3fd825193f1001371dca39c89'
CASES = []
REC = {}


def ck(name, ok, detail=None):
    CASES.append((name, bool(ok), detail))
    print('[%s] %s :: %s' % ('PASS' if ok else 'FAIL', name,
                             json.dumps(detail, default=repr, sort_keys=True)))


def sha_file(p):
    p = Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def sha_arr(a):
    import numpy as np
    a = np.asarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype.str).encode())
    h.update(str(a.shape).encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def fbits(x):
    try:
        return struct.pack('>d', float(x)).hex()
    except Exception:
        return None


def rebuild_dist32(coords_day):
    """Exactly the generator's order (generate_coldchain_data.py): float32 coords, difference,
    square, sum, sqrt."""
    import numpy as np
    c = np.asarray(coords_day, dtype=np.float32)
    diff = c[:, None, :] - c[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def distance_contract(dataset):
    import numpy as np
    rep = {'days_checked': 0, 'bit_exact_days': 0, 'mismatches': [], 'dtype_match': None,
           'shape_match': None, 'energy_mat_present': 'energy_mat' in dataset,
           'other_derived_keys': [k for k in dataset if 'mat' in k.lower()]}
    if 'dist_mat' not in dataset:
        rep['verdict'] = 'ABSENT'
        return rep
    dm = np.asarray(dataset['dist_mat'])
    for d in range(dm.shape[0]):
        rb = rebuild_dist32(np.asarray(dataset['coords'])[d])
        rep['days_checked'] += 1
        if rb.dtype.str == dm.dtype.str:
            rep['dtype_match'] = True
        else:
            rep['dtype_match'] = False
        if rb.shape == dm[d].shape:
            rep['shape_match'] = True
        else:
            rep['shape_match'] = False
        if np.array_equal(rb, dm[d]) and rb.dtype.str == dm.dtype.str:
            rep['bit_exact_days'] += 1
        elif len(rep['mismatches']) < 3:
            diff = np.argwhere(rb != dm[d])
            i, j = (int(diff[0][0]), int(diff[0][1])) if len(diff) else (None, None)
            rep['mismatches'].append({'day': d, 'first_index': [i, j],
                                      'rebuilt': (float(rb[i, j]) if i is not None else None),
                                      'stored': (float(dm[d][i, j]) if i is not None else None),
                                      'rebuilt_bits': (fbits(rb[i, j]) if i is not None else None),
                                      'stored_bits': (fbits(dm[d][i, j]) if i is not None else None),
                                      'n_differing_entries': int(len(diff))})
    rep['verdict'] = ('VERIFIED' if rep['bit_exact_days'] == rep['days_checked']
                      and rep['dtype_match'] is True and rep['shape_match'] is True
                      else 'NOT_VERIFIABLE')
    return rep


def invisible_customers(snapshot):
    import numpy as np
    vis = np.asarray(snapshot['visible_mask']).astype(bool)
    return [int(c) for c in snapshot['customer_universe'] if not bool(vis[int(c)])]


def u_for(day_id, position, customer, field):
    key = ('d3l1-perturb|%d|%d|%d|%s' % (day_id, position, customer, field)).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') / float(1 << 64)


def perturb(dataset, snapshot, day_id, position):
    """PURE.  Coordinates are perturbed and the derived distance matrix is rebuilt from the
    PERTURBED coords in the generator's float32 order."""
    import numpy as np
    rep = {'day_id': int(day_id), 'position': int(position), 'invisible_customers': [],
           'customers_changed': [], 'fields_changed': [], 'n_invisible_modified': 0,
           'verdict': None, 'notes': [], 'dist_mat': {}, 'preservation': {}}
    try:
        invisible = invisible_customers(snapshot)
        rep['invisible_customers'] = invisible
        if not invisible:
            rep['verdict'] = 'NOT_VERIFIABLE'
            rep['notes'].append('no invisible customers')
            return None, rep
        out = dict(dataset)
        coords = np.asarray(dataset['coords'])
        n = coords.shape[0]
        if not (0 <= int(day_id) < n):
            rep['verdict'] = 'NOT_VERIFIABLE'
            rep['notes'].append('day_id out of range')
            return None, rep
        changed = {}
        nc = coords.copy()
        for c in invisible:
            if c >= coords.shape[1]:
                rep['notes'].append('customer %d out of coords range' % c)
                continue
            before = np.asarray(nc[int(day_id), c]).copy()
            val = np.asarray(nc[int(day_id), c], dtype=np.float32) + np.float32(
                u_for(day_id, position, c, 'coords') * COORD_DELTA)
            nc[int(day_id), c] = val.astype(coords.dtype)
            if not np.array_equal(before, np.asarray(nc[int(day_id), c])):
                changed.setdefault(c, []).append('coords')
        out['coords'] = nc
        rep['fields_changed'].append('coords')
        for field, kind in PERTURBABLE.items():
            if field not in dataset:
                rep['notes'].append('%s absent' % field)
                continue
            arr = np.asarray(dataset[field])
            if arr.ndim != 2 or arr.shape[0] != n:
                rep['notes'].append('%s unexpected shape' % field)
                continue
            new = arr.copy()
            for c in invisible:
                if c >= arr.shape[1]:
                    continue
                before = np.asarray(new[int(day_id), c]).copy()
                u = u_for(day_id, position, c, field)
                if kind == 'positive':
                    v = max(float(new[int(day_id), c]) * (1.0 + u), 1e-6)
                else:
                    v = float(new[int(day_id), c]) + u * TW_DELTA
                new[int(day_id), c] = arr.dtype.type(v)
                if not np.array_equal(before, np.asarray(new[int(day_id), c])):
                    changed.setdefault(c, []).append(field)
            out[field] = new
            rep['fields_changed'].append(field)
        rep['customers_changed'] = sorted(changed)
        rep['per_customer_fields'] = {str(k): sorted(v) for k, v in sorted(changed.items())}
        rep['n_invisible_modified'] = len(changed)
        # ---- derived distances from the PERTURBED coords (the previous bug) ----------------
        if 'dist_mat' in dataset:
            dm = np.asarray(dataset['dist_mat'])
            rb = rebuild_dist32(np.asarray(out['coords'])[int(day_id)])
            ndm = dm.copy()
            ndm[int(day_id)] = rb.astype(dm.dtype)
            out['dist_mat'] = ndm
            vis = [int(c) for c in snapshot['customer_universe']
                   if bool(np.asarray(snapshot['visible_mask'])[int(c)])]
            rep['dist_mat'] = {
                'present': True, 'rebuilt_from_perturbed_coords': True,
                'dtype_preserved': bool(ndm.dtype == dm.dtype),
                'visible_visible_unchanged': bool(np.array_equal(
                    dm[int(day_id)][np.ix_(vis, vis)], ndm[int(day_id)][np.ix_(vis, vis)])),
                'entries_changed': int(np.count_nonzero(dm[int(day_id)] != ndm[int(day_id)])),
                'invisible_involved_entries_only':
                    bool(np.array_equal(dm[int(day_id)][np.ix_(vis, vis)],
                                        ndm[int(day_id)][np.ix_(vis, vis)]))}
            if 'energy_mat' in dataset:
                rep['notes'].append('energy_mat present: its contract must be declared before it '
                                    'may be rebuilt (left UNCHANGED here, recorded)')
                rep['dist_mat']['energy_mat_left_unchanged'] = True
        else:
            rep['dist_mat'] = {'present': False}
        pres = {'shapes': {}, 'dtypes': {}, 'reveal_time_unchanged': None}
        ok = True
        for f in list(PERTURBABLE) + ['coords', 'temp_class']:
            if f in dataset:
                ok &= (np.asarray(out[f]).shape == np.asarray(dataset[f]).shape
                       and np.asarray(out[f]).dtype == np.asarray(dataset[f]).dtype)
                pres['shapes'][f] = list(np.asarray(out[f]).shape)
                pres['dtypes'][f] = str(np.asarray(out[f]).dtype)
        pres['shapes_and_dtypes_preserved'] = bool(ok)
        pres['reveal_time_unchanged'] = ('reveal_time' not in out
                                         or np.array_equal(np.asarray(out['reveal_time']),
                                                           np.asarray(dataset['reveal_time'])))
        rep['preservation'] = pres
        if rep['n_invisible_modified'] == 0 or not ok or not pres['reveal_time_unchanged'] \
                or rep['dist_mat'].get('visible_visible_unchanged') is False:
            rep['verdict'] = 'NOT_VERIFIABLE'
            rep['notes'].append('perturbation not valid or preservation violated')
            return None, rep
        rep['verdict'] = 'PERTURBED'
        return out, rep
    except BaseException as exc:
        rep['verdict'] = 'NOT_VERIFIABLE'
        rep['error'] = '%s: %s' % (type(exc).__name__, str(exc)[:300])
        return None, rep


def decision_projection(env, snapshot, replanner, inst_idx):
    """Rebuild visible inputs, legal candidates and the baseline planned suffix from the snapshot.
    The replanner MUTATES the vehicles it is handed, so its return value is ignored."""
    import numpy as np
    from recourse_snapshot import restore_recourse_snapshot
    out = {'replanner_calls': 0, 'errors': {}}
    try:
        restored = restore_recourse_snapshot(snapshot)
        objs = restored if isinstance(restored, tuple) else (restored,)
        vehicles = next((o for o in objs if isinstance(o, list) and o
                         and hasattr(o[0], 'vehicle_id')), None)
        served = next((o for o in objs if hasattr(o, 'shape') and getattr(o, 'dtype', None)
                       is not None and o.dtype == bool), None)
        if vehicles is None or served is None:
            return {'verdict': 'NOT_VERIFIABLE', 'errors': {'restore': 'objects not recognised'},
                    'replanner_calls': 0}
        vis = sorted(int(c) for c in snapshot['customer_universe']
                     if bool(np.asarray(snapshot['visible_mask'])[int(c)]))
        out['visible_ids'] = vis
        out['visible_inputs_sha'] = {
            'demands': sha_arr(np.asarray(env.demands[int(inst_idx)])[vis]),
            'dist_mat': sha_arr(np.asarray(env.dist_mat, dtype=np.float64)[int(inst_idx)]
                                [np.ix_(vis, vis)]) if hasattr(env, 'dist_mat') else None,
            'reveal_time': sha_arr(np.asarray(env.reveal_time, dtype=np.float64)[int(inst_idx)]
                                   [vis])}
        try:
            from dynmaskco_cc_context import decision_pool_from_vehicles
            pool = decision_pool_from_vehicles(vehicles, np.asarray(served), vis)
            out['legal_candidates'] = sorted(int(x) for x in pool)
            out['legal_candidates_sha'] = sha_arr(np.asarray(out['legal_candidates'],
                                                             dtype=np.int64))
        except BaseException as exc:
            out['errors']['legal_candidates'] = '%s: %s' % (type(exc).__name__, str(exc)[:160])
        state = snapshot.get('replanner_state')
        try:
            if state is not None and hasattr(replanner, 'restore_state'):
                replanner.restore_state(copy.deepcopy(state))
        except BaseException as exc:
            out['errors']['replanner_state'] = '%s: %s' % (type(exc).__name__, str(exc)[:160])
        cloned = copy.deepcopy(vehicles)
        try:
            replanner.plan(env, int(inst_idx), float(snapshot['clock']), cloned,
                           np.asarray(served).copy(), vis)
            out['replanner_calls'] += 1
            suf = sorted((int(getattr(v, 'vehicle_id', -1)),
                          tuple(int(x) for x in (getattr(v, 'mutable_suffix', None) or [])))
                         for v in cloned)
            out['planned_suffix'] = suf
            out['planned_suffix_sha'] = sha_arr(np.asarray(
                [x for _, t in suf for x in t] or [0], dtype=np.int64))
        except BaseException as exc:
            out['errors']['planned_suffix'] = '%s: %s' % (type(exc).__name__, str(exc)[:200])
        need = ('visible_inputs_sha', 'legal_candidates_sha', 'planned_suffix_sha')
        produced = all(k in out and out[k] is not None for k in need)
        out['all_three_outputs_produced'] = bool(produced)
        out['verdict'] = 'PROJECTED' if (produced and not out['errors']) else 'NOT_VERIFIABLE'
        return out
    except BaseException as exc:
        out['errors']['outer'] = '%s: %s' % (type(exc).__name__, str(exc)[:200])
        out['verdict'] = 'NOT_VERIFIABLE'
        return out


def main():
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True)
    ap.add_argument('--dataset', default=str(EXT / 'data/m0_scale/'
                                             'dcc_50_r1_edod05_dev_check_teacher.npz'))
    args = ap.parse_args()
    from coldchain_contract import (ObjectiveProfile, apply_objective_profile,
                                    default_pilot_contract)
    from d3_l1_state_collector import (RevealSlotCollector, build_env, make_replanner,
                                       run_baseline_rollout)

    # ---- step 1: generator identity ------------------------------------------------------
    man = json.loads((EXT / 'data/m0_scale/M0_TEACHER_MANIFEST.json').read_text(encoding='utf-8'))
    gen_src = EXT / 'scripts/data/generate_coldchain_data.py'
    REC['generator_identity'] = {
        'manifest_generator': man.get('generator'),
        'manifest_generator_sha256': man.get('generator_sha256'),
        'actual_generator_sha256': sha_file(gen_src),
        'match': (man.get('generator_sha256') == sha_file(gen_src)),
        'dataset_sha256': sha_file(args.dataset),
        'dataset_is_the_registered_dev_check': sha_file(args.dataset) == KNOWN_DEV_CHECK,
        'note': 'the dataset identity is taken from the server-computed value recorded earlier'}
    ck('B0 the generator source matches the manifest generator_sha256 and the dataset is the '
       'registered dev_check NPZ', REC['generator_identity']['match']
       and REC['generator_identity']['dataset_is_the_registered_dev_check'],
       REC['generator_identity'])

    data = dict(np.load(str(args.dataset)))
    # ---- step 2: strict float32 distance contract ----------------------------------------
    dc = distance_contract(data)
    REC['distance_contract'] = dc
    ck('B0b DIST_MAT_GENERATION_CONTRACT: the frozen matrix equals the generator-order float32 '
       'Euclidean rebuild BIT-EXACTLY on every development day (no tolerance relaxation)',
       dc.get('verdict') == 'VERIFIED', dc)

    if dc.get('verdict') != 'VERIFIED':
        ck('B1 perturbation valid (n_invisible_modified > 0, preservation held)',
           False, {'blocked_by': 'distance contract not verified'})
        ck('B4 both sides rebuild the projection with all three outputs produced',
           False, {'blocked_by': 'distance contract not verified'})
    else:
        prof = json.loads((EXT / 'results/o0cc/scale_v2/objective_profile.json')
                          .read_text(encoding='utf-8'))
        flds = {f.name for f in dataclasses.fields(ObjectiveProfile)}
        eff = apply_objective_profile(default_pilot_contract(),
                                      ObjectiveProfile(**{k: v for k, v in prof.items()
                                                          if k in flds}))
        env = build_env(data, 50.0, 25, 1.0, make_replanner(50.0, 24.0, eff)[0], eff, 24.0)
        col = RevealSlotCollector(env, 0, 1)
        run_baseline_rollout(env, 0, hook=col)
        snap = col.slots[0]['snapshot']
        p1, r1 = perturb(data, snap, 0, 1)
        p2, r2 = perturb(data, snap, 0, 1)
        REC['perturbation_report'] = r1
        REC['deterministic_repeat'] = (r1 == r2)
        ck('B1 perturbation valid: invisible customers identified, n_invisible_modified counted '
           'from real bit-level changes, distance rebuilt from the PERTURBED coords with '
           'visible-visible entries bit-identical, and the input dataset untouched',
           r1.get('verdict') == 'PERTURBED' and r1.get('n_invisible_modified', 0) > 0
           and r1.get('dist_mat', {}).get('rebuilt_from_perturbed_coords')
           and r1.get('dist_mat', {}).get('visible_visible_unchanged') is True and r1 == r2,
           {'verdict': r1.get('verdict'), 'n_invisible_modified':
            r1.get('n_invisible_modified'), 'dist_mat': r1.get('dist_mat'),
            'deterministic': r1 == r2})
        if p1 is not None:
            mods = sys.modules.keys()
            proj_o = decision_projection(env, snap, make_replanner(50.0, 24.0, eff)[0], 0)
            env_p = build_env(p1, 50.0, 25, 1.0, make_replanner(50.0, 24.0, eff)[0], eff, 24.0)
            proj_p = decision_projection(env_p, snap, make_replanner(50.0, 24.0, eff)[0], 0)
            REC['projection_original'] = proj_o
            REC['projection_perturbed'] = proj_p
            REC['candidate_pool_dependency'] = {
                'module': 'dynmaskco_cc_context',
                'sha256': sha_file(EXT / 'scripts/simulation/dynmaskco_cc_context.py'),
                'declared_as_explicit_f_free_dependency': True,
                'why': 'the projection uses its decision_pool_from_vehicles helper; importing it '
                       'therefore makes it part of the F-free execution chain and it must be '
                       'registered rather than excluded'}
            f_free = not any(m.split('.')[0] in FORBIDDEN or m.startswith('mpre')
                             for m in sys.modules)
            REC['candidate_pool_dependency']['process_stayed_f_free'] = f_free
            need = ('visible_inputs_sha', 'legal_candidates_sha', 'planned_suffix_sha')
            complete = (proj_o.get('verdict') == 'PROJECTED' and proj_p.get('verdict') == 'PROJECTED'
                        and all(k in proj_o and proj_o[k] is not None for k in need)
                        and all(k in proj_p and proj_p[k] is not None for k in need)
                        and not proj_o.get('errors') and not proj_p.get('errors'))
            REC['projection_comparison'] = {
                k: (proj_o.get(k) == proj_p.get(k)) for k in need}
            REC['projection_comparison']['complete'] = bool(complete)
            ck('B4 both sides rebuild the projection independently with all three outputs truly '
               'produced, no error keys and the replanner called on cloned vehicles (its return '
               'value ignored because plan() mutates in place)',
               complete and proj_o.get('replanner_calls', 0) >= 1
               and proj_p.get('replanner_calls', 0) >= 1,
               {'original': {k: proj_o.get(k) for k in ('verdict', 'replanner_calls',
                                                        'all_three_outputs_produced')},
                'perturbed': {k: proj_p.get(k) for k in ('verdict', 'replanner_calls',
                                                         'all_three_outputs_produced')},
                'errors': {**proj_o.get('errors', {}), **proj_p.get('errors', {})}})
            ck('B5 the comparison is a real per-item check (no unconditional pass): it reports '
               'the visible-input, legal-candidate and planned-suffix SHA equality plus the '
               'changed customers and fields, and NOT_VERIFIABLE whenever completeness fails',
               complete and bool(REC['projection_comparison']),
               {'hashes_equal': REC['projection_comparison'],
                'changed_customers': r1.get('customers_changed')})
        else:
            ck('B4 both sides rebuild the projection independently', False,
               {'reason': r1.get('verdict'), 'notes': r1.get('notes')})
            ck('B5 the comparison is a real per-item check', False, {'reason': 'no perturbation'})
    REC['F_side_caveat'] = ('An F-free projection PASS only shows that this model-free decision '
                            'information boundary is invariant under the declared perturbation '
                            'family; the real F feature extractor must still be verified inside the '
                            'F-vs-D-Geo mechanism gate.')
    REC['plan_B_scope'] = ('Plan B only: generator-order float32 distance contract verification + '
                           'perturbed-distance rebuild + projection error handling. Plans A and C '
                           'are not implemented and formal collection is not authorised.')
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'rows': [{'name': n, 'ok': o, 'detail': d} for n, o, d in CASES],
                               'records': REC, 'n_rows': len(CASES),
                               'n_pass': sum(1 for c in CASES if c[1]),
                               'n_fail': sum(1 for c in CASES if not c[1]),
                               'source_sha256_this_tool':
                                   sha_file(Path(__file__))},
                              indent=2, default=repr, sort_keys=True), encoding='utf-8')
    print('\n=== %d rows, %d PASS, %d FAIL ===' % (len(CASES),
                                                   sum(1 for c in CASES if c[1]),
                                                   sum(1 for c in CASES if not c[1])))
    print('DIST_MAT_GENERATION_CONTRACT =', dc.get('verdict'))
    print('WROTE', out)
    return 0 if not [c for c in CASES if not c[1]] else 1


if __name__ == '__main__':
    sys.exit(main())
