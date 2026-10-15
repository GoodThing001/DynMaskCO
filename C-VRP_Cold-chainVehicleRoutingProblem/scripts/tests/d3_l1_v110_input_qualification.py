"""One bounded engineering pass over existing states. No rollout, F score or install."""
import copy
import dataclasses
import hashlib
import json
import pickle
import sys
import time
import traceback
from pathlib import Path

EXT = Path(__file__).resolve().parents[2]
for part in ('scripts/simulation', 'scripts/coldchain', 'scripts/data', 'scripts/models', 'scripts/tests'):
    sys.path.insert(0, str(EXT / part))

T_IMPORT = time.monotonic()
import numpy as np
from action_contract import build_vehicle_plans, certify_route, enumerate_actions_from_plans
from coldchain_contract import ObjectiveProfile, apply_objective_profile, default_pilot_contract
from d3_geo_adapter import build_pool_B
from d3_l1_causal_perturbation import perturb
from d3_l1_state_collector import build_env, make_replanner
from d3_l1_v18_readonly_audit import typed_bytes
from dynmaskco_cc_context import validate_full_partition
from dynmaskco_cc_graph import build_plan_adjacency, build_visible_index
from recourse_snapshot import restore_recourse_snapshot, snapshot_state_hash
from repair_state import extract_repair_state_v1
import coldchain_visible_features  # include the real lazy feature dependency in the start closure
IMPORT_S = time.monotonic() - T_IMPORT

ROOT = EXT.parent / 'research/review_3080_20261010/d3_g0_evidence_root_v1/a1_d3_l1_prereg_root_v1'
E19 = ROOT / 'v19_evidence'
OUT = ROOT / 'v110_evidence'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def digest(x):
    return hashlib.sha256(typed_bytes(x)).hexdigest()


def write_new(p, content):
    with open(p, 'x', encoding='utf-8') as fh:
        json.dump(content, fh, indent=2, sort_keys=True)
        fh.flush()
        import os
        os.fsync(fh.fileno())


def origins():
    out = {}
    for name, mod in list(sys.modules.items()):
        p = getattr(mod, '__file__', None)
        if not p:
            continue
        p = Path(p).resolve()
        if p.is_file() and p.suffix == '.py' and p.is_relative_to(EXT):
            out[name] = {'path': p.relative_to(EXT).as_posix(), 'sha256': sha(p),
                         'file': str(p), 'spec_origin': getattr(getattr(mod, '__spec__', None), 'origin', None)}
    return out


def materialize(data, snap, contract):
    planner, _ = make_replanner(50., 24., contract)
    env = build_env(data, 50., 25, 1., planner, contract, 24.)
    env.event_id = int(snap['event_id'])
    vs, _, served = restore_recourse_snapshot(snap)
    clock, day = float(snap['clock']), int(snap['instance_id'])
    before = [(v.vehicle_id, v.ready_time) for v in vs]
    env.prepare_decision_point(clock, vs)
    ready_prepared = all(v.ready_time >= clock for v in vs if v.status in ('ready', 'idle'))
    visible = [int(c) for c in snap['customer_universe'] if snap['visible_mask'][int(c)]]
    mutable = {v.vehicle_id for v in vs if v.status in ('idle', 'ready') and v.needs_replan}
    reserved = set(sorted((v.vehicle_id for v in vs if v.status == 'idle'), reverse=True)[:1])
    planner.restore_state(copy.deepcopy(snap['replanner_state']))
    planner.plan(env, day, clock, vs, served, visible, replan_ids=mutable)
    plans = build_vehicle_plans(env, day, vs)
    protected = {v.vehicle_id for v in vs if v.vehicle_id in reserved and v.status == 'idle'
                 and v.current_node == 0 and not plans[v.vehicle_id].suffix}
    mutable -= protected
    pool, pool_diag = build_pool_B(vs, served, visible, plans, mutable, protected=protected)
    mask = pool[:2]
    partial = copy.deepcopy(plans)
    for p in partial.values():
        p.suffix = tuple(c for c in p.suffix if c not in mask)
    legal = []
    for c in mask:
        candidates, _ = enumerate_actions_from_plans(env, day, partial, c, allowed_vehicle_ids=mutable)
        legal.extend((c, cand.action, cand) for cand in candidates if cand.feasible)
    idx, nodes, nv = build_visible_index(visible)
    inputs = extract_repair_state_v1(env, day, clock, vs, served, visible, partial, mask, legal,
                                     50., 24., allowed_vehicle_ids=mutable,
                                     node_to_local=idx, local_to_node=nodes, Nv=nv)
    a0 = np.triu(build_plan_adjacency(plans, idx, nv), k=1) > .5
    ap = np.triu(inputs['adjmat'][0], k=1) > .5
    inputs['timestep'] = float((a0 & ap).sum() / a0.sum()) if a0.sum() else 0.
    committed = {int(v.committed_next) for v in vs
                 if v.status == 'committed' and v.committed_next not in (None, 0)}
    universe = set(visible) - {int(c) for c in visible if served[int(c)]} - committed
    partition_ok, partition = validate_full_partition(committed, plans, planner.deferred_customers,
        universe, served_mask=served, future_fn=lambda c: c not in visible)
    routing = {vid: certify_route(env, day, p.anchor_node, p.anchor_time, p.anchor_load, p.suffix)
               for vid, p in plans.items()}
    tensor_keys = ('raw_3d', 'node_feats', 'node_valid', 'adjmat', 'cust', 'pred', 'succ',
                   'action_valid', 'action_feats', 'cargo', 'cargo_valid')
    finite = all(np.isfinite(inputs[k]).all() for k in tensor_keys)
    no_future = set(nodes) <= set(visible) | {0}
    index_ok = all(inputs[k].size == 0 or (inputs[k].min() >= 0 and inputs[k].max() < nv)
                   for k in ('cust', 'pred', 'succ'))
    targets_ok = all(vid in mutable for vid in inputs['vehicle_target_vid'])
    return {'P0': plans, 'partial': partial, 'vehicles': vs, 'served': served,
        'deferred': sorted(planner.deferred_customers), 'visible_ids': visible,
        'mutable_ids': sorted(mutable), 'protected_ids': sorted(protected), 'pool': pool,
        'mask': mask, 'legal': legal, 'inputs': inputs, 'pool_diag': pool_diag,
        'routing': routing, 'partition': partition, 'ready_before': before,
        'ready_after': [(v.vehicle_id, v.ready_time) for v in vs],
        'checks': {'prepared_anchor_clock': ready_prepared, 'partition': partition_ok,
                   'routing_only': all(r['feasible'] for r in routing.values()),
                   'finite_inputs': finite, 'only_visible_nodes': no_future,
                   'endpoint_indices': index_ok, 'targets_in_mutable': targets_ok,
                   'mask_size_two': len(mask) == 2, 'legal_nonempty': bool(legal),
                   'action_shape': inputs['action_feats'].shape == (1, len(legal), 20)},
        'cargo_overflow_lots': sum(max(0, len(getattr(v.coldchain_state, 'cargo_manifest', ()) or ()) - 8)
                                   for v in vs)}


def main():
    OUT.mkdir(exist_ok=False)
    (OUT / 'derived').mkdir()
    started = time.monotonic()
    start_sources = origins()
    frozen = json.loads((E19 / 'RUNTIME_IDENTITY_V19.json').read_text())
    manifest = json.loads((E19 / 'STATE_MANIFEST_64_V19.json').read_text())
    assert len(manifest) == 64
    assert [(r['day_id'], r['position']) for r in manifest] == [(d, p) for d in range(16) for p in range(1, 5)]
    data_path = EXT / 'data/m0_scale/dcc_50_r1_edod05_dev_check_teacher.npz'
    assert sha(data_path) == frozen['dataset_sha256']
    assert all(sha(EXT / r['path']) == r['sha256'] for r in frozen['modules'].values())
    data = dict(np.load(data_path))
    profile_path = EXT / 'results/o0cc/scale_v2/objective_profile.json'
    assert sha(profile_path) == frozen['profile_file_sha256']
    profile = json.loads(profile_path.read_text())
    fields = {f.name for f in dataclasses.fields(ObjectiveProfile)}
    contract = apply_objective_profile(default_pilot_contract(), ObjectiveProfile(**{k: v for k, v in profile.items() if k in fields}))
    assert contract.contract_hash == frozen['effective_contract_hash']
    write_new(OUT / 'START.json', {'sources': start_sources, 'initial_import_wall_s': IMPORT_S,
        'dataset_sha256': sha(data_path), 'profile_sha256': sha(profile_path),
        'effective_contract_hash': frozen['effective_contract_hash'],
        'state_manifest_sha256': sha(E19 / 'STATE_MANIFEST_64_V19.json'),
        'prereg_sha256': sha(ROOT / 'A1_D3_L1_INPUT_QUALIFICATION_PREREGISTRATION_V1_10.md'),
        'classification': 'engineering_input_only_no_F_forward_no_science_comparison'})
    rows, baseline_calls = [], 0
    for entry in manifest:
        t0 = time.monotonic()
        row = {'day_id': entry['day_id'], 'position': entry['position'], 'state_hash': entry['state_hash']}
        payload = {'entry': entry}
        try:
            if t0 - started > 180.:
                row.update(status='NOT_ATTEMPTED_BATCH_DEADLINE')
            else:
                path = E19 / 'formal_collection' / entry['snapshot_path']
                assert sha(path) == entry['snapshot_sha256']
                snap = pickle.loads(path.read_bytes())
                assert snapshot_state_hash(snap) == entry['state_hash']
                changed, perturb_report = perturb(data, snap, entry['day_id'], entry['position'])
                assert perturb_report['verdict'] == 'PERTURBED'
                a = materialize(data, snap, contract)
                baseline_calls += 1
                b = materialize(changed, snap, contract)
                baseline_calls += 1
                payload.update(original=a, hidden_perturbed=b, perturbation=perturb_report)
                exact = {k: typed_bytes(a[k]) == typed_bytes(b[k]) for k in
                         ('P0', 'deferred', 'pool', 'mask', 'legal', 'inputs', 'mutable_ids', 'protected_ids')}
                original_ev = pickle.loads((E19 / 'formal_collection' / entry['validation_path']).read_bytes())
                stored = dict(original_ev['original_projection']['planned_suffix'])
                prepared = {v.vehicle_id: tuple(v.mutable_suffix) for v in a['vehicles']}
                checks = dict(a['checks'], **{'hidden_' + k: v for k, v in b['checks'].items()},
                              **{'exact_' + k: v for k, v in exact.items()})
                negative = copy.deepcopy(a['inputs']);negative['raw_3d'][0, 0, 0] += np.float32(.125)
                checks['visible_input_change_detected'] = digest(a['inputs']) != digest(negative)
                checks['dtype_change_detected'] = typed_bytes(np.array([0.], np.float32)) != typed_bytes(np.array([0.], np.float64))
                checks['signed_zero_change_detected'] = typed_bytes(np.array([0.], np.float64)) != typed_bytes(np.array([-0.], np.float64))
                checks['engineering_slot_cap'] = time.monotonic() - t0 <= 60.
                row.update(checks=checks, status='INPUT_CHECKS_PASS' if all(checks.values()) else 'INPUT_CHECKS_FAILED',
                    input_sha256=digest(a['inputs']), P0_sha256=digest(a['P0']), legal_sha256=digest(a['legal']),
                    n_legal=len(a['legal']), n_B=len(a['pool']), Nv=a['inputs']['Nv'], mask=a['mask'],
                    stored_projection_suffix_equal={str(k): tuple(stored[k]) == v for k, v in prepared.items()},
                    cargo_overflow_lots=a['cargo_overflow_lots'])
        except Exception as exc:
            row.update(status='FAILED_EXCEPTION', exception_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        row['wall_s'] = time.monotonic() - t0
        row['C0_full_certification'] = 'NOT_RUN_MISSING_REGISTERED_BUDGET_AND_STATE_CONTRACT_BOUND_CERTIFIER'
        raw = OUT / ('derived/day%02d_pos%d.pkl' % (entry['day_id'], entry['position']))
        # Keep both successful and failed contexts; no object here is an executable publication.
        with open(raw, 'xb') as fh:
            pickle.dump(payload, fh, protocol=4)
        row['derived_path'] = str(raw.relative_to(OUT));row['derived_sha256'] = sha(raw)
        write_new(OUT / ('derived/day%02d_pos%d.json' % (entry['day_id'], entry['position'])), row)
        rows.append(row)
    end_sources = origins()
    source_stable = all(sha(EXT / r['path']) == r['sha256'] for r in start_sources.values())
    forbidden = sorted(n for n in sys.modules if n.startswith(('mpre', 'mtrained')))
    result = {'rows': rows, 'denominator': 64, 'states_checked': len(rows),
        'input_checks_pass': sum(r['status'] == 'INPUT_CHECKS_PASS' for r in rows),
        'input_checks_failed_or_missing': sum(r['status'] != 'INPUT_CHECKS_PASS' for r in rows),
        'sources_at_start': start_sources, 'sources_at_end': end_sources, 'source_stable': source_stable,
        'F_module_imports': forbidden, 'baseline_calls': baseline_calls, 'F_forward_calls': 0,
        'F_D_Geo_comparisons': 0, 'install_calls': 0, 'rollouts': 0,
        'wall_s': time.monotonic() - started, 'C0_full_certification': 'BLOCKED_MISSING_REGISTERED_BUDGET_AND_BOUND_CERTIFIER',
        'science_ready': False, 'stable_online_gain': False, 'algorithmic_gain_claim': 'none'}
    write_new(OUT / 'INPUT_QUALIFICATION_V110.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('rows', 'sources_at_start', 'sources_at_end')}))
    return int(result['input_checks_failed_or_missing'] != 0 or not source_stable or bool(forbidden))


if __name__ == '__main__':
    sys.exit(main())
