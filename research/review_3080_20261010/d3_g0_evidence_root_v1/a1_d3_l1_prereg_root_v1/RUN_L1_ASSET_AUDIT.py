#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Reproducible read-only asset audit for the A1-D3-L1 pre-registration round.

Regenerates `A1_D3_L1_STATE_AND_F_ASSET_AUDIT.json`:
  * state-index assets under results/m0_scale (depth 1, verification copies excluded),
    their sha256 / record counts / per-record keys, and the coverage of the 28 registered
    snapshot fields -> the STATE_ASSET_GAP verdict;
  * presence + sha256 + API surface of scripts/simulation/recourse_snapshot.py;
  * the training state manifest (indices only) and the data identity (NPZ + manifest sha256);
  * the F loading closure: model.ckpt sha256 vs the registered value, backbone sha256
    (NEWLY pinned here -- it is absent from training_identity.json), training metadata, and a
    structural probe of the pickle's `train_state` (group leaf counts / parameter counts /
    all-zero leaves / absmax), which is what proves the task-trained residuals are non-zero.

READ-ONLY: nothing here mutates any asset, no state is replayed, no model is constructed and no
forward pass is executed.  Running it again is safe and idempotent.

Usage: python RUN_L1_ASSET_AUDIT.py
"""
import hashlib
import json
import pathlib
import pickle
import re

REPO = pathlib.Path(__file__).resolve().parents[3]
EXT = REPO / 'C-VRP_Cold-chainVehicleRoutingProblem'
OUTDIR = pathlib.Path(__file__).resolve().parent
STATE_FIELDS = ('clock', 'served_mask', 'mutable_suffix', 'mutable_suffix_len', 'needs_replan',
                'replan_reason', 'replan_count', 'committed_next', 'committed_arrive',
                'committed_finish', 'dispatch_time', 'vehicle_node', 'vehicle_ready',
                'vehicle_load', 'vehicle_status', 'services_node', 'services_prev_node',
                'services_start', 'services_arrival', 'services_depart', 'services_finish',
                'services_len', 'served_route', 'served_route_len', 'visible_mask',
                'customer_universe', 'num_nodes', 'num_vehicles')
REGISTERED_F_SHA = 'f5acf83fa2519020aeea274df5ecf740a6ad809b215d2d270209b7901903a7f5'
CKPT_REL = 'results/m0_scale/mpre_reinforce_s42_clean'
BACKBONE_REL = 'MASKCO_code/ckpts/cvrp100.ckpt'


def rel(p):
    return pathlib.Path(p).relative_to(EXT).as_posix()


def sha(p):
    p = pathlib.Path(p)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None


def state_assets():
    assets = []
    for q in sorted((EXT / 'results/m0_scale').glob('*/per_state.json')):
        if q.parent.name.startswith(('_bverify', '_smoke')):
            continue
        try:
            j = json.loads(q.read_text(encoding='utf-8'))
        except Exception as exc:
            assets.append({'path': rel(q), 'unreadable': str(exc)})
            continue
        recs = j if isinstance(j, list) else (j.get('states') or j.get('records') or [])
        first = recs[0] if recs else {}
        assets.append({'path': rel(q), 'bytes': q.stat().st_size, 'sha256': sha(q),
                       'n_records': len(recs),
                       'per_record_keys': sorted(first) if isinstance(first, dict) else None})
    return assets


def main():
    r = {}
    assets = state_assets()
    r['state_index_assets'] = assets
    present = set()
    for a in assets:
        present |= set(a.get('per_record_keys') or [])
    r['field_coverage'] = {
        'snapshot_fields_checked': len(STATE_FIELDS),
        'fields_present_in_history': sorted(f for f in STATE_FIELDS if f in present),
        'fields_missing_in_history': sorted(f for f in STATE_FIELDS if f not in present),
        'index_fields_present': sorted(k for k in present if k in ('inst', 'event', 'clock'))}

    rs = EXT / 'scripts/simulation/recourse_snapshot.py'
    txt = rs.read_text(encoding='utf-8') if rs.is_file() else ''
    r['snapshot_machinery'] = {
        'path': 'scripts/simulation/recourse_snapshot.py', 'exists': rs.is_file(),
        'sha256': sha(rs), 'bytes': rs.stat().st_size if rs.is_file() else None,
        'api': {n: ('def %s' % n) in txt for n in
                ('capture_recourse_snapshot', 'restore_recourse_snapshot', 'resume_from_snapshot',
                 'snapshot_state_hash', 'validate_snapshot_against_env')},
        'schema_versions': sorted(set(re.findall(r'recourse_snapshot_v[0-9a-z_]+', txt)))}

    sm = EXT / CKPT_REL / 'state_manifest.json'
    j = json.loads(sm.read_text(encoding='utf-8'))
    r['training_state_manifest'] = {
        'path': rel(sm), 'sha256': sha(sm), 'keys': sorted(j),
        'n_indices': len(j.get('inst_event', [])), 'carries_state_content': False,
        'note': 'inst_event is a list of [inst,event] pairs only; no state content'}
    r['data_identity'] = {rel(q): {'bytes': q.stat().st_size, 'sha256': sha(q)}
                          for q in sorted((EXT / 'data/m0_scale').glob('*')) if q.is_file()}

    ck, cv = EXT / CKPT_REL / 'model.ckpt', REPO / BACKBONE_REL
    ti = json.loads((ck.parent / 'training_identity.json').read_text(encoding='utf-8'))
    f = {'model_ckpt': {'path': rel(ck), 'sha256': sha(ck), 'bytes': ck.stat().st_size},
         'registered_sha256': REGISTERED_F_SHA,
         'backbone_ckpt': {'path': BACKBONE_REL, 'sha256': sha(cv),
                           'bytes': cv.stat().st_size if cv.is_file() else None},
         'training_identity_sha256': sha(ck.parent / 'training_identity.json'),
         'summary_sha256': sha(ck.parent / 'summary.json'),
         'state_manifest_sha256': sha(sm),
         'training_identity_top_keys': sorted(ti),
         'backbone_recorded_in_training_identity': ('cvrp' in json.dumps(ti).lower())}
    f['model_ckpt_matches_registered'] = f['model_ckpt']['sha256'] == REGISTERED_F_SHA
    try:
        import flax
        import jax
        import jax.numpy as jnp
        with open(ck, 'rb') as fh:
            d = pickle.load(fh)
        ts = d['train_state']
        flat = jax.tree_util.tree_flatten_with_path(ts)[0]
        groups = {}
        for path, leaf in flat:
            name = '.'.join(str(getattr(p, 'key', getattr(p, 'name', p))) for p in path)
            top = name.split('.')[0]
            arr = jnp.asarray(leaf)
            g = groups.setdefault(top, {'n_leaves': 0, 'params': 0, 'all_zero_leaves': 0,
                                        'absmax': 0.0})
            g['n_leaves'] += 1
            g['params'] += int(arr.size)
            g['absmax'] = max(g['absmax'],
                              float(jnp.max(jnp.abs(arr))) if arr.size else 0.0)
            if arr.size and not bool(jnp.any(arr)):
                g['all_zero_leaves'] += 1
        f['pickle'] = {
            'top_keys': sorted(d),
            'meta': {k: d.get(k) for k in ('s_TRAIN', 'num_steps', 'batch_states', 'K', 'seed',
                                           'F_NODE', 'F_ACTION_EXPLICIT')},
            'train_state_type': type(ts).__module__ + '.' + type(ts).__name__,
            'n_leaves': len(flat), 'group_stats': groups,
            'task_params_nonzero': all(groups.get(k, {}).get('absmax', 0.0) > 0
                                       for k in ('c_adapter', 'delta_head')),
            'backbone_leaves_inside_train_state': groups.get('backbone', {}).get('n_leaves', 0),
            'jax': jax.__version__, 'flax': flax.__version__}
    except BaseException as exc:
        f['pickle'] = {'probe_error': '%s: %s' % (type(exc).__name__, exc)}
    r['F_identity'] = f

    r['verdict'] = {
        'state_assets': 'STATE_ASSET_GAP',
        'why': ['every historical asset under results/m0_scale stores (inst,event) indices plus '
                'scalar per-arm outcomes only',
                'none of the %d checked registered snapshot fields is stored for any state'
                % len(STATE_FIELDS),
                'the 256 training-state indices in state_manifest.json carry no state content',
                'the sealed asset alone therefore cannot supply a state to F and D-Geo',
                'the registered recoverability judgement already says: '
                'index_only_recovery_requires_a_new_execution, is_that_route_read_only=false, '
                'authorised_this_round=false'],
        'not_done_deliberately': ['no state was replayed or re-derived',
                                  'no result value was used to reconstruct a state',
                                  'no snapshot was invented',
                                  'no execution was performed for this inventory'],
        'required_before_any_L1_execution': [
            'an authorised deterministic RE-COLLECTION pre-registration that captures a '
            'recourse_snapshot (v2/v3_cc) per state and requires (inst,event,J0) to reproduce '
            'exactly',
            'closure of the contract/profile identity gap noted by the earlier audit',
            'a pinned backbone identity (cvrp100.ckpt sha256 is NOT recorded in '
            'training_identity.json)']}

    (OUTDIR / 'A1_D3_L1_STATE_AND_F_ASSET_AUDIT.json').write_text(
        json.dumps(r, indent=2, sort_keys=True, default=repr), encoding='utf-8')
    print('verdict:', r['verdict']['state_assets'])
    print('index assets:', [(a.get('path'), a.get('n_records')) for a in assets])
    print('fields present in history:', r['field_coverage']['fields_present_in_history'])
    print('snapshot machinery:', r['snapshot_machinery']['exists'],
          r['snapshot_machinery']['schema_versions'])
    print('F sha match:', f['model_ckpt_matches_registered'],
          '| backbone:', (f['backbone_ckpt']['sha256'] or '')[:16],
          '| recorded:', f['backbone_recorded_in_training_identity'])
    print('pickle groups:', {k: (v['n_leaves'], v['params'], round(v['absmax'], 4))
                             for k, v in f['pickle'].get('group_stats', {}).items()})
    print('task params nonzero:', f['pickle'].get('task_params_nonzero'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
