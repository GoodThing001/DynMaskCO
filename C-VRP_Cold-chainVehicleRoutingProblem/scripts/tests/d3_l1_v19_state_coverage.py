"""Read stored state/projection bytes only: manifest and B-domain coverage, no new plan calls."""
import copy
import hashlib
import json
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

EXT = Path(__file__).resolve().parents[2]
for part in ('scripts/simulation', 'scripts/coldchain', 'scripts/tests'):
    sys.path.insert(0, str(EXT / part))
from recourse_snapshot import restore_recourse_snapshot
from d3_geo_adapter import build_pool_B

ROOT = EXT.parent / 'research/review_3080_20261010/d3_g0_evidence_root_v1/a1_d3_l1_prereg_root_v1'
E = ROOT / 'v19_evidence'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    audit = json.loads((E / 'FORMAL_64_INDEPENDENT_AUDIT.json').read_text())
    assert audit['n_fail'] == 0 and audit['qualified_slots'] == 64
    collection = E / 'formal_collection'
    rows = [json.loads(s) for s in (collection / 'STATE_LEDGER.jsonl').read_text().splitlines()]
    manifest, coverage = [], []
    for r in rows:
        if r['event'] != 'PUBLISHED':
            continue
        snap = pickle.loads((collection / r['snapshot_path']).read_bytes())
        ev = pickle.loads((collection / r['validation_path']).read_bytes())
        vs, ts, sm = restore_recourse_snapshot(snap)
        visible = [int(c) for c in snap['customer_universe'] if snap['visible_mask'][int(c)]]
        mutable = {v.vehicle_id for v in vs if v.status in ('idle', 'ready') and v.needs_replan}
        reserved = set(sorted((v.vehicle_id for v in vs if v.status == 'idle'), reverse=True)[:1])
        raw_plans = {v.vehicle_id: SimpleNamespace(suffix=tuple(v.mutable_suffix)) for v in vs}
        raw_pool, _ = build_pool_B(vs, sm, visible, raw_plans, mutable)
        projected_suffixes = dict(ev['original_projection']['planned_suffix'])
        projected_vs = copy.deepcopy(vs)
        for v in projected_vs:
            v.mutable_suffix = list(projected_suffixes[v.vehicle_id])
        plans = {v.vehicle_id: SimpleNamespace(suffix=tuple(v.mutable_suffix)) for v in projected_vs}
        protected = {v.vehicle_id for v in projected_vs if v.vehicle_id in reserved
                     and v.status == 'idle' and v.current_node == 0 and not plans[v.vehicle_id].suffix}
        pool, diag = build_pool_B(projected_vs, sm, visible, plans, mutable - protected, protected=protected)
        key = {'day_id': r['day_id'], 'position': r['position']}
        manifest.append(dict(key, event_id=snap['event_id'], clock=snap['clock'],
            reveal_idx=snap['reveal_idx'], state_hash=r['state_hash'], snapshot_path=r['snapshot_path'],
            snapshot_sha256=r['snapshot_sha256'], validation_path=r['validation_path'],
            validation_sha256=r['validation_sha256'], qualified=True,
            eligible_use='L1_ENGINEERING_MECHANISM_STATE_SOURCE_ONLY',
            L2_L3_science_qualified=False))
        coverage.append(dict(key, pre_plan_B_pool=raw_pool, stored_projection_B_pool=pool,
            mutable_ids=sorted(mutable - protected), protected_ids=sorted(protected),
            projection_pool_diagnostic=diag,
            warning='Post-baseline projection is stored evidence, not a newly certified common incumbent.'))
    assert len(manifest) == 64
    assert not any(n.startswith(('mpre', 'mtrained')) or n == 'repair_state' for n in sys.modules)
    report = {'rows': coverage, 'denominator': 64, 'formal_states_qualified': len(manifest),
        'pre_plan_nonempty': sum(bool(r['pre_plan_B_pool']) for r in coverage),
        'stored_projection_nonempty': sum(bool(r['stored_projection_B_pool']) for r in coverage),
        'stored_projection_size_ge2': sum(len(r['stored_projection_B_pool']) >= 2 for r in coverage),
        'new_rollouts': 0, 'new_baseline_plan_calls': 0, 'F_scoring_calls': 0,
        'F_D_Geo_comparisons': 0, 'input_independent_audit_sha256': sha(E / 'FORMAL_64_INDEPENDENT_AUDIT.json'),
        'source_sha256': sha(__file__), 'pool_B_source_sha256': sha(EXT / 'scripts/simulation/d3_geo_adapter.py'),
        'next_gate': 'Materialize and certify the shared POST-BASELINE incumbent with frozen identity; never use the empty pre-plan suffix as P0.',
        'algorithmic_gain_claim': 'none', 'stable_online_gain': False}
    for name, content in (('STATE_MANIFEST_64_V19.json', manifest), ('STATE_COVERAGE_V19.json', report)):
        with open(E / name, 'x', encoding='utf-8') as fh:
            json.dump(content, fh, indent=2, sort_keys=True)
    print(json.dumps({k: report[k] for k in ('formal_states_qualified', 'pre_plan_nonempty',
                                           'stored_projection_nonempty', 'stored_projection_size_ge2')}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
