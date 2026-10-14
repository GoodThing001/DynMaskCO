"""Read-only provenance/numeric checks of the actual continuation artifacts.

No model import, training, search or numerical experiment. Saved JSON rows and
archives are inspected; raw simulator execution is deliberately not certified.
"""
import argparse
import json
import math
from pathlib import Path
from independent_support_guard import GuardError, bind_fields, load, require, sha


def walk_inventory(root, alias, entries):
    for p in sorted(root.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts or p.suffix == '.pyc': continue
        rec = {'root': alias, 'path': p.relative_to(root).as_posix(),
               'sha256': sha(p), 'bytes': p.stat().st_size}
        if p.suffix == '.json':
            try:
                value = load(p)
                rec.update(json_parsed=True, root_type=type(value).__name__)
                if type(value) is dict:
                    rec['top_keys'] = sorted(value)
            except (ValueError, OSError) as exc:
                rec.update(json_parsed=False, error=str(exc))
        entries.append(rec)


def close(a, b, label):
    require(type(a) in (int, float) and type(b) in (int, float)
            and math.isfinite(a) and math.isfinite(b) and abs(a - b) < 1e-10,
            label + ' numerical mismatch')


def audit(repo):
    repo = Path(repo).resolve(); ext = repo / 'C-VRP_Cold-chainVehicleRoutingProblem'
    current = ext / 'results/research_20261010_continuation_v2'
    inventory = []
    roots = ((current, 'continuation_results'),
             (repo / 'research/pickup_20round_audit', 'pickup_review_artifacts'),
             (ext / 'scripts/research_20261010', 'research_sources'),
             (ext / 'scripts/research_claim_gate', 'actual_claim_gate'))
    for root, alias in roots:
        require(root.is_dir(), 'input root missing: ' + str(root))
        walk_inventory(root, alias, inventory)
    manifest = load(ext / 'scripts/research_claim_gate/NEGATIVE_ARCHIVE_CURRENT_INVENTORY.json')
    original = {p.relative_to(ext).as_posix(): sha(p) for p in (ext / 'negative_result_paper').rglob('*') if p.is_file()}
    require(len(original) == 183 and original == manifest['archive_inventory'], '183-file seal mismatch')
    formal = []
    for folder in ('09_unified_v1', '10_confirm_c1'):
        for p in sorted((ext / 'negative_result_paper/results' / folder).glob('*.json')):
            if not p.name.endswith(('.analysis.json', '.summary.json')): continue
            mirror = ext / 'negative_result_paper/release/results' / folder / p.name
            require(mirror.is_file() and sha(p) == sha(mirror), 'formal mirror identity mismatch')
            formal.append({'path': p.relative_to(ext).as_posix(), 'sha256': sha(p)})
    require(len(formal) == 14, 'formal JSON count mismatch')

    registration = load(current / 'pickup/P_LIVE_DEV_PANEL_V2_RUN_REGISTRATION.json')
    pins = []
    for name, ref in registration['pins'].items():
        if 'sha256' not in ref: continue
        candidates = [ext / ref['path'], repo / ref['path']]
        found = [p for p in candidates if p.is_file()]
        require(len(found) == 1 and sha(found[0]) == ref['sha256'], 'run pin mismatch: ' + name)
        pins.append({'name': name, 'path': ref['path'], 'sha256': ref['sha256'], 'matched': True})
    require(len(pins) == 7, 'registration pins not seven')

    # Independent arithmetic of saved development rows only, not full route replay.
    dev = load(current / 'delivery/dev_gate_v1/result.json')
    rows = dev['per_instance']; require(len(rows) == 32, 'delivery saved row count')
    require({row['instance'] for row in rows} == set(range(32)), 'delivery saved row identities')
    for arm, expected in dev['mean_final_cost'].items():
        close(sum(row[arm] for row in rows) / 32, expected, arm + ' mean')
    for label, contrast in dev['paired_contrasts'].items():
        arm = label.split('_minus_', 1)[1]
        gains = [(row[arm] - row['trained']) / row[arm] for row in rows]
        close(sum(gains) / 32, contrast['mean_relative_cost_reduction'], label)
        require(sum(g > 1e-12 for g in gains) == contrast['n_improved'], 'improved count mismatch')
        require(sum(g < -1e-12 for g in gains) == contrast['n_worse'], 'worse count mismatch')
    require(dev['stable_online_gain_established'] is False and dev['maskco_attributable'] is False,
            'development incorrectly promoted')

    remote_path = current / 'delivery/dev_gate_3080_v2/result.json'
    remote = load(remote_path)
    local_without_wall = {k: v for k, v in dev.items() if k != 'wall_s'}
    remote_without_wall = {k: v for k, v in remote.items() if k != 'wall_s'}
    require(local_without_wall == remote_without_wall, 'cross-host DEV rows or non-wall fields differ')
    cross_host = {'local_path': 'delivery/dev_gate_v1/result.json',
                  'local_sha256': sha(current / 'delivery/dev_gate_v1/result.json'),
                  'replacement_remote_path': 'delivery/dev_gate_3080_v2/result.json',
                  'replacement_remote_sha256': sha(remote_path),
                  'n_rows': len(remote['per_instance']),
                  'all_non_wall_fields_exact_equal': True,
                  'local_wall_s': dev['wall_s'], 'remote_wall_s': remote['wall_s'],
                  'new_fresh_confirmation_established': False,
                  'historical_failed_attempt_placeholder_preserved':
                      (current / 'delivery/dev_gate_3080_v1/result.json').stat().st_size == 0}
    invalid = [r for r in inventory if r.get('json_parsed') is False]
    for record in invalid:
        if record['path'] == 'delivery/dev_gate_3080_v1/result.json' and record['bytes'] == 0:
            record['classification'] = 'historical_failed_attempt_invalid_placeholder'
            record['superseding_valid_artifact'] = 'delivery/dev_gate_3080_v2/result.json'
            record['does_not_invalidate_replacement_cross_host_DEV'] = True

    authority = load(ext / 'configs/i3_20261008_v1/protocol.json')
    # This is a current-value/pin check; it does not prove registration timing.
    budget = registration['operating_point']['budget_B_kwh']
    close(authority['teacher_collection']['budget'], budget, 'registered budget')

    lineage = {'parent_PROGRAM_sha256': sha(ext / 'results/research_20261010/PROGRAM.json'),
               'continuation_PROGRAM_sha256': sha(current / 'PROGRAM.json'),
               'continuation_parent_declared_sha256': load(current / 'PROGRAM.json')['parent_program_sha256']}
    require(lineage['parent_PROGRAM_sha256'] == lineage['continuation_parent_declared_sha256'], 'PROGRAM parent mismatch')
    return {'schema': 'independent-continuation-provenance-review-v2',
            'lineage': lineage, 'original_files_unchanged': 183, 'formal_json_mirrors': formal,
            'registration_pins': pins, 'budget_current_value': budget,
            'delivery_dev_saved_row_arithmetic_pass': True,
            'cross_host_same_DEV_check': cross_host,
            'delivery_dev_mean_relative_cost_reductions': {k: v['mean_relative_cost_reduction']
                                                         for k, v in dev['paired_contrasts'].items()},
            'new_input_inventory': inventory,
            'n_new_inventory_files': len(inventory),
            'json_parse_failures': invalid,
            'stable_online_gain_established': False, 'MaskCO_attribution_established': False,
            'full_historical_reproduction_established': False, 'equivalence_established': False,
            'limits': ['Saved row arithmetic and field pins, not independent raw route/simulator replay.',
                       'A stored registered_at or preregistered_primary flag is not immutable pre-data proof.',
                       'Source inventory is a present-day snapshot, not an executed-source start/end identity.',
                       'Inventory coverage does not establish semantic correctness of every new artifact.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--repo', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); out = Path(args.output).resolve()
    require(not out.is_relative_to(Path(args.repo).resolve()), 'output must be outside input repository')
    result = audit(args.repo)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ('original_files_unchanged', 'n_new_inventory_files', 'json_parse_failures',
                                           'delivery_dev_saved_row_arithmetic_pass')}, ensure_ascii=False))
