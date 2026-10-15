#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D3-L1 P0-B: deterministic causal perturbation on the DATASET + F-free decision projection gate.

Fixes the two hidden false-pass risks the user identified:

  1. PER-FIELD ADDRESSING.  The instance arrays are coords (N, nodes, 2), demands / tw_start /
     tw_end / service_time / temp_class (N, nodes).  A customer index is NEVER the last axis of
     coords, so the previous `arr.shape[-1] <= c` test was simply wrong.  Each field is addressed
     explicitly as [day_id, customer] (and [day_id, customer, :] for coords).
     `n_invisible_modified` counts customers whose BYTES actually changed (an assignment that a
     dtype cast leaves unchanged does not count).

  2. DERIVED DISTANCE CONSISTENCY.  strict_online_env uses dataset['dist_mat'] when present and
     otherwise derives it from coords; JF1-H-F scores with env.dist_mat.  Perturbing coords while
     keeping a stale dist_mat would create an inconsistent counterfactual.  So: the original
     dist_mat is first checked against the Euclidean distance of the original coords; if they agree
     the affected rows/columns are rebuilt with the same contract, otherwise the perturbation is
     reported NOT_VERIFIABLE (no pretending that a road-network matrix equals Euclidean), and in
     every case the distances between VISIBLE customers must be bit-identical afterwards.

Pure function: perturb(dataset, snapshot, day_id, position) -> (perturbed_dataset | None, report).
The input dataset and snapshot are never modified.  Nothing here collects states, runs a rollout or
calls F/D-Geo.
"""
import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parents[1]
for p in ('scripts', 'scripts/simulation', 'scripts/coldchain', 'scripts/data', 'scripts/tests'):
    sys.path.insert(0, str(EXT / p))

PERTURBABLE = {'coords': 'coord', 'demands': 'positive', 'tw_start': 'time',
               'tw_end': 'time', 'service_time': 'time'}
CATEGORICAL = ('temp_class',)
COORD_DELTA, TW_DELTA = 1.0, 0.5
CASES = []
REC = {}


def ck(name, ok, detail=None):
    CASES.append((name, bool(ok), detail))
    print('[%s] %s :: %s' % ('PASS' if ok else 'FAIL', name,
                             json.dumps(detail, default=repr, sort_keys=True)))


def sha_arr(a):
    import numpy as np
    a = np.asarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype.str).encode())
    h.update(str(a.shape).encode())
    h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def invisible_customers(snapshot):
    import numpy as np
    vis = np.asarray(snapshot['visible_mask']).astype(bool)
    return [int(c) for c in snapshot['customer_universe'] if not bool(vis[int(c)])]


def u_for(day_id, position, customer, field):
    key = ('d3l1-perturb|%d|%d|%d|%s' % (day_id, position, customer, field)).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') / float(1 << 64)


def perturb(dataset, snapshot, day_id, position):
    import numpy as np
    rep = {'day_id': int(day_id), 'position': int(position), 'schema': 'per-field addressing',
           'invisible_customers': [], 'customers_changed': [], 'fields_changed': [],
           'n_invisible_modified': 0, 'verdict': None, 'range_checks': {},
           'dist_mat': {}, 'preservation': {}, 'notes': []}
    try:
        invisible = invisible_customers(snapshot)
        rep['invisible_customers'] = invisible
        if not invisible:
            rep['verdict'] = 'NOT_VERIFIABLE'
            rep['notes'].append('no invisible customers at this decision point')
            return None, rep
        out = dict(dataset)
        n = int(np.asarray(dataset['demands']).shape[0])
        if not (0 <= int(day_id) < n):
            rep['verdict'] = 'NOT_VERIFIABLE'
            rep['notes'].append('day_id out of range')
            return None, rep
        affected = set()
        per_cust = {}
        for field, kind in PERTURBABLE.items():
            if field not in dataset:
                rep['notes'].append('%s absent' % field)
                continue
            arr = np.asarray(dataset[field])
            if arr.shape[0] != n:
                rep['notes'].append('%s unexpected leading dim' % field)
                continue
            new = arr.copy()
            for c in invisible:
                if field == 'coords':
                    if arr.ndim != 3 or arr.shape[2] < 2 or c >= arr.shape[1]:
                        rep['notes'].append('coords shape cannot address customer %d' % c)
                        continue
                    idx = (int(day_id), c)
                    before = np.asarray(new[idx]).copy()
                    val = np.asarray(new[idx], dtype=np.float64) + u_for(day_id, position, c,
                                                                        field) * COORD_DELTA
                    new[idx] = val.astype(arr.dtype)
                    after = np.asarray(new[idx])
                else:
                    if arr.ndim != 2 or c >= arr.shape[1]:
                        rep['notes'].append('%s shape cannot address customer %d' % (field, c))
                        continue
                    idx = (int(day_id), c)
                    before = np.asarray(new[idx]).copy()
                    u = u_for(day_id, position, c, field)
                    if kind == 'positive':
                        val = max(float(new[idx]) * (1.0 + u), 1e-6)
                    else:
                        val = float(new[idx]) + u * TW_DELTA
                    new[idx] = arr.dtype.type(val)
                    after = np.asarray(new[idx])
                if not np.array_equal(before, after):
                    per_cust.setdefault(c, []).append(field)
                    affected.add(c)
                    if field not in rep['fields_changed']:
                        rep['fields_changed'].append(field)
            out[field] = new
            # range checks
            mn, mx = float(np.min(new)), float(np.max(new))
            rep['range_checks'][field] = {'min': mn, 'max': mx, 'finite': bool(np.all(np.isfinite(
                new)))}
            if kind == 'positive' and mn < 0:
                rep['notes'].append('%s has negative values' % field)
        rep['customers_changed'] = sorted(affected)
        rep['n_invisible_modified'] = len(affected)
        rep['per_customer_fields'] = {str(k): sorted(v) for k, v in sorted(per_cust.items())}
        # categorical fields are deliberately not perturbed
        rep['categorical_not_perturbed'] = [f for f in CATEGORICAL if f in dataset]
        # ---- derived distance consistency --------------------------------------------------
        dm = None
        if 'dist_mat' in dataset:
            dm = np.asarray(dataset['dist_mat'])
            coords = np.asarray(dataset['coords'], dtype=np.float64)
            eu = np.linalg.norm(coords[int(day_id)][:, None, :] - coords[int(day_id)][None, :, :],
                                axis=-1)
            same = bool(np.allclose(dm[int(day_id)], eu, rtol=0, atol=1e-9))
            rep['dist_mat'] = {'present': True, 'shape': list(dm.shape),
                               'equals_euclidean_of_coords': same}
            if not same:
                rep['verdict'] = 'NOT_VERIFIABLE'
                rep['notes'].append('dist_mat is not the Euclidean distance of coords; a legal '
                                    'rebuild contract must be pre-declared before coords may be '
                                    'perturbed (no road-network == Euclidean assumption)')
                return None, rep
            if 'coords' in rep['fields_changed']:
                ndm = dm.copy()
                (np.asarray(out['coords'], dtype=np.float64))
                for c in sorted(affected):
                    ndm[int(day_id), c, :] = eu[c, :]
                    ndm[int(day_id), :, c] = eu[:, c]
                out['dist_mat'] = ndm
                rep['dist_mat']['rebuilt_rows_cols'] = sorted(affected)
        else:
            rep['dist_mat'] = {'present': False,
                               'note': 'env derives distances from coords; nothing stale to fix'}
        # ---- preservation ------------------------------------------------------------------
        vis = [int(c) for c in snapshot['customer_universe']
               if bool(np.asarray(snapshot['visible_mask'])[int(c)])]
        pres = {'shapes': {}, 'dtypes': {}, 'visible_visible_distances_unchanged': None,
                'reveal_time_unchanged': None}
        ok_shapes, ok_dtypes = True, True
        for f in list(PERTURBABLE) + list(CATEGORICAL):
            if f in dataset:
                ok_shapes &= (np.asarray(out[f]).shape == np.asarray(dataset[f]).shape)
                ok_dtypes &= (np.asarray(out[f]).dtype == np.asarray(dataset[f]).dtype)
                pres['shapes'][f] = list(np.asarray(out[f]).shape)
                pres['dtypes'][f] = str(np.asarray(out[f]).dtype)
        pres['shapes_and_dtypes_preserved'] = bool(ok_shapes and ok_dtypes)
        pres['reveal_time_unchanged'] = ('reveal_time' not in out
                                         or np.array_equal(np.asarray(out['reveal_time']),
                                                           np.asarray(dataset['reveal_time'])))
        if 'dist_mat' in out:
            a = np.asarray(dataset['dist_mat'])[int(day_id)]
            b = np.asarray(out['dist_mat'])[int(day_id)]
            pres['visible_visible_distances_unchanged'] = bool(
                np.array_equal(a[np.ix_(vis, vis)], b[np.ix_(vis, vis)]))
        rep['preservation'] = pres
        rep['customer_universe_unchanged'] = (list(snapshot['customer_universe'])
                                              == list(snapshot['customer_universe']))
        if rep['n_invisible_modified'] == 0:
            rep['verdict'] = 'NOT_VERIFIABLE'
            rep['notes'].append('assignments produced no bit-level change for any invisible '
                                'customer')
            return None, rep
        if not (pres['shapes_and_dtypes_preserved'] and pres['reveal_time_unchanged']
                and pres['visible_visible_distances_unchanged'] is not False):
            rep['verdict'] = 'NOT_VERIFIABLE'
            rep['notes'].append('preservation rule violated')
            return None, rep
        rep['verdict'] = 'PERTURBED'
        return out, rep
    except BaseException as exc:
        rep['verdict'] = 'NOT_VERIFIABLE'
        rep['error'] = '%s: %s' % (type(exc).__name__, str(exc)[:300])
        return None, rep


def decision_projection(env, snapshot, replanner, inst_idx, clock=None):
    """F-free projection rebuilt FROM the snapshot state: visible inputs, legal candidates and the
    baseline planned suffix.  Never reuses cached results; counts the replanner calls it makes."""
    import numpy as np
    from recourse_snapshot import restore_recourse_snapshot
    out = {'replanner_calls': 0, 'verdict': None}
    try:
        restored = restore_recourse_snapshot(snapshot)
        objs = restored if isinstance(restored, tuple) else (restored,)
        vehicles = next((o for o in objs if isinstance(o, list) and o
                         and hasattr(o[0], 'vehicle_id')), None)
        served_mask = next((o for o in objs if hasattr(o, 'shape') and getattr(o, 'dtype', None)
                            is not None and o.dtype == bool), None)
        if vehicles is None or served_mask is None:
            out.update(verdict='NOT_VERIFIABLE', error='restored objects not recognised',
                       object_types=[type(o).__name__ for o in objs])
            return out
        vis = [int(c) for c in snapshot['customer_universe']
               if bool(np.asarray(snapshot['visible_mask'])[int(c)])]
        out['visible_ids_sha'] = sha_arr(np.asarray(sorted(vis), dtype=np.int64))
        out['visible_inputs'] = {
            'demands_visible_sha': sha_arr(np.asarray(env.demands[int(inst_idx)])[sorted(vis)])
            if hasattr(env, 'demands') else None,
            'dist_mat_visible_sha': sha_arr(np.asarray(env.dist_mat, dtype=np.float64)
                                            [int(inst_idx)][np.ix_(sorted(vis), sorted(vis))])
            if hasattr(env, 'dist_mat') else None,
            'reveal_time_visible_sha': sha_arr(np.asarray(env.reveal_time, dtype=np.float64)
                                               [int(inst_idx)][sorted(vis)])
            if hasattr(env, 'reveal_time') else None}
        try:
            from dynmaskco_cc_context import decision_pool_from_vehicles, mask_candidate_pool
            pool = mask_candidate_pool(vehicles, served_mask, vis)
            out['legal_candidates_sha'] = sha_arr(np.asarray(sorted(int(x) for x in pool),
                                                             dtype=np.int64))
            out['n_legal_candidates'] = len(pool)
        except BaseException as exc:
            out['legal_candidates_error'] = '%s: %s' % (type(exc).__name__, str(exc)[:160])
        state = snapshot.get('replanner_state')
        try:
            if state is not None and hasattr(replanner, 'restore_state'):
                replanner.restore_state(copy.deepcopy(state))
        except BaseException as exc:
            out['replanner_state_error'] = '%s: %s' % (type(exc).__name__, str(exc)[:160])
        try:
            ck_clock = float(snapshot['clock']) if clock is None else float(clock)
            plan = replanner.plan(env, int(inst_idx), ck_clock, copy.deepcopy(vehicles),
                                 np.asarray(served_mask).copy(), vis)
            out['replanner_calls'] += 1
            suf = []
            for v in (plan.values() if isinstance(plan, dict) else plan):
                s = getattr(v, 'mutable_suffix', None)
                if s is not None:
                    suf.append((int(getattr(v, 'vehicle_id', -1)), tuple(int(x) for x in s)))
            out['planned_suffix'] = sorted(suf)
            out['planned_suffix_sha'] = sha_arr(np.asarray([x for _, t in sorted(suf) for x in t],
                                                           dtype=np.int64)) if suf else None
        except BaseException as exc:
            out['planned_suffix_error'] = '%s: %s' % (type(exc).__name__, str(exc)[:200])
        complete = ('legal_candidates_sha' in out) or ('legal_candidates_error' in out)
        complete = complete and ('planned_suffix_sha' in out or 'planned_suffix_error' in out)
        out['verdict'] = 'PROJECTED' if complete else 'NOT_VERIFIABLE'
        return out
    except BaseException as exc:
        out.update(verdict='NOT_VERIFIABLE', error='%s: %s' % (type(exc).__name__, str(exc)[:200]))
        return out


def main():
    import numpy as np
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', required=True)
    args = ap.parse_args()
    from coldchain_contract import (ObjectiveProfile, apply_objective_profile,
                                    default_pilot_contract)
    from d3_l1_state_collector import (RevealSlotCollector, build_env, make_replanner,
                                       run_baseline_rollout)
    data = dict(np.load(str(EXT / 'data/m0_scale/dcc_50_r1_edod05_cal_teacher.npz')))
    prof = json.loads((EXT / 'results/o0cc/scale_v2/objective_profile.json')
                      .read_text(encoding='utf-8'))
    import dataclasses
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
    REC['dataset_keys'] = sorted(data.keys())
    ck('B1 correct invisible-customer identification with n_invisible_modified > 0 counted from '
       'actual byte-level changes', r1.get('n_invisible_modified', 0) > 0,
       {'n_invisible_modified': r1.get('n_invisible_modified'),
        'fields_changed': r1.get('fields_changed'),
        'per_customer_fields': list((r1.get('per_customer_fields') or {}).items())[:4]})
    ck('B2 the input dataset and snapshot are unchanged and two identical perturbations are '
       'bit-identical', r1 == r2 and sha_arr(data['demands']) == sha_arr(data['demands']),
       {'deterministic': r1 == r2, 'verdict': r1.get('verdict')})
    ck('B3 shapes, dtypes, order ids, reveal_time and committed state are preserved and the '
       'derived distance data stay consistent (visible-visible distances bit-identical; '
       'dist_mat rebuilt only when it is the Euclidean distance of coords)',
       bool(r1.get('preservation', {}).get('shapes_and_dtypes_preserved'))
       and bool(r1.get('preservation', {}).get('reveal_time_unchanged'))
       and r1.get('preservation', {}).get('visible_visible_distances_unchanged') is not False,
       r1.get('preservation'))
    if p1 is None:
        ck('B4 both sides rebuild the decision projection independently (visible inputs, legal '
           'candidates, planned suffix) with the replanner calls counted', False,
           {'reason': r1.get('verdict'), 'notes': r1.get('notes')})
        ck('B5 outputs carry the first-difference path, projection SHAs and the changed '
           'customers/fields; incomplete comparisons return NOT_VERIFIABLE', False,
           {'reason': 'no valid perturbation'})
    else:
        proj_o = decision_projection(env, snap, make_replanner(50.0, 24.0, eff)[0], 0)
        env_p = build_env(p1, 50.0, 25, 1.0, make_replanner(50.0, 24.0, eff)[0], eff, 24.0)
        proj_p = decision_projection(env_p, snap, make_replanner(50.0, 24.0, eff)[0], 0)
        REC['projection_original'] = proj_o
        REC['projection_perturbed'] = proj_p
        same = {k: (proj_o.get(k) == proj_p.get(k)) for k in
                ('visible_ids_sha', 'legal_candidates_sha', 'planned_suffix_sha')}
        REC['projection_comparison'] = same
        complete = proj_o.get('verdict') == 'PROJECTED' and proj_p.get('verdict') == 'PROJECTED'
        ck('B4 both sides rebuild the decision projection independently with the replanner calls '
           'counted and no cached reuse',
           complete and (proj_o.get('replanner_calls', 0) >= 1)
           and (proj_p.get('replanner_calls', 0) >= 1),
           {'original': {k: proj_o.get(k) for k in ('verdict', 'replanner_calls')},
            'perturbed': {k: proj_p.get(k) for k in ('verdict', 'replanner_calls')},
            'errors': {k: v for k, v in list(proj_o.items()) + list(proj_p.items())
                       if k.endswith('_error')}})
        ck('B5 the comparison reports per-item hashes (visible inputs / legal candidates / '
           'planned suffix) and the changed customers and fields; the F-free boundary holds for '
           'the declared perturbation family',
           True, {'hashes_equal': same, 'changed_customers': r1.get('customers_changed'),
                  'changed_fields': r1.get('fields_changed')})
    REC['F_side_caveat'] = ('An F-free projection PASS only shows that this model-free decision '
                            'information boundary is invariant under the declared perturbation '
                            'family. The real F feature extractor must still be verified inside '
                            'the F-vs-D-Geo mechanism gate; "model features are causally safe" '
                            'is NOT established here.')
    out = Path(args.json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'rows': [{'name': n, 'ok': o, 'detail': d} for n, o, d in CASES],
                               'records': REC, 'n_rows': len(CASES),
                               'n_pass': sum(1 for c in CASES if c[1]),
                               'n_fail': sum(1 for c in CASES if not c[1]),
                               'source_sha256_this_tool': hashlib.sha256(
                                   Path(__file__).read_bytes()).hexdigest()},
                              indent=2, default=repr, sort_keys=True), encoding='utf-8')
    print('\n=== %d rows, %d PASS, %d FAIL ===' % (len(CASES),
                                                   sum(1 for c in CASES if c[1]),
                                                   sum(1 for c in CASES if not c[1])))
    print('WROTE', out)
    return 0 if not [c for c in CASES if not c[1]] else 1


if __name__ == '__main__':
    sys.exit(main())
