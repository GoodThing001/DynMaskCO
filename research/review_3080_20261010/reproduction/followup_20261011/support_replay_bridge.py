"""Hash-bound source/runtime coverage plus actually executed engineering replay.

Only the two installed engineering validators below may execute. Bundle-supplied
code is never imported. No result can certify original-training reproduction,
independent semantic implementation, physical calibration, or scientific gain.
"""
import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path


class ClosureError(ValueError): pass


def require(ok, why):
    if not ok: raise ClosureError(why)


def load(path):
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, 'duplicate JSON key: ' + key); out[key] = value
        return out
    def constant(value): raise ClosureError('nonfinite JSON: ' + value)
    value = json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=pairs, parse_constant=constant)
    def inspect(v):
        if type(v) is float: require(math.isfinite(v), 'nonfinite numeric value')
        if isinstance(v, dict):
            for x in v.values(): inspect(x)
        elif isinstance(v, list):
            for x in v: inspect(x)
    inspect(value); return value


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()


def inside(root, relative):
    require(type(relative) is str and bool(relative), 'missing relative artifact path')
    rel = Path(relative)
    require(not rel.is_absolute() and '..' not in rel.parts, 'artifact path outside project')
    p = (root / rel).resolve(); require(p.is_relative_to(root), 'artifact path outside project'); return p


def artifact(root, entry):
    require(type(entry) is dict, 'artifact reference object required')
    digest = entry.get('sha256')
    require(type(digest) is str and len(digest) == 64 and all(c in '0123456789abcdef' for c in digest), 'invalid SHA256')
    p = inside(root, entry.get('path')); require(p.is_file(), 'missing artifact: ' + str(p))
    require(sha(p) == digest, 'artifact bytes changed: ' + str(p)); return p


VALIDATORS = {
    'delivery_event_contract_only': 'scripts/correction_engineering/delivery_trace_contract.py',
    'pickup_observer_evaluator_only': 'scripts/correction_engineering/pickup_trace_runner.py',
}
IDENTITY_ROLES = ('runner_source', 'training_source', 'weights', 'dataset', 'contract', 'environment')


def load_validator(root, registry):
    name = registry['validator']; relative = VALIDATORS.get(name)
    require(relative is not None, 'unregistered replay adapter')
    pins = registry.get('validator_sources'); require(type(pins) is list and pins, 'missing validator source closure')
    require(any(x.get('path') == relative for x in pins), 'main replay source not pinned')
    if name == 'pickup_observer_evaluator_only':
        require(any(x.get('path') == 'scripts/correction_engineering/audit_io.py' for x in pins), 'journal dependency not pinned')
    for entry in pins: artifact(root, entry)
    path = root / relative
    module_name = '_support_registered_' + name
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    # All callable code comes from the externally supplied trusted registry.
    sys.path.insert(0, str(path.parent)); sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module, pins


def close_tree(actual, expected, where='replay'):
    if type(expected) is bool:
        require(actual is expected, where + ': literal boolean mismatch')
    elif type(expected) in (int, float):
        require(type(actual) in (int, float) and math.isfinite(actual) and math.isfinite(expected), where + ': finite number required')
        require(abs(actual - expected) <= 1e-9, where + ': numeric mismatch')
    elif type(expected) is dict:
        require(type(actual) is dict and actual.keys() == expected.keys(), where + ': field coverage mismatch')
        for key in expected: close_tree(actual[key], expected[key], where + '.' + key)
    elif type(expected) is list:
        require(type(actual) is list and len(actual) == len(expected), where + ': list coverage mismatch')
        for i, x in enumerate(expected): close_tree(actual[i], x, where + '[' + str(i) + ']')
    else: require(type(actual) is type(expected) and actual == expected, where + ': value mismatch')


def audit(project, bundle_path, registry_path):
    root = Path(project).resolve(); bundle = load(bundle_path); registry = load(registry_path)
    require(bundle.get('schema') == 'support_closure_bundle_v1', 'bundle schema')
    require(registry.get('schema') == 'support_closure_registry_v1', 'external registry schema')
    require(bundle.get('registry_sha256') == sha(registry_path), 'external registry identity mismatch')
    require(type(registry.get('data_seed')) is int and registry['data_seed'] not in (20260927, 20260928), 'protected or missing data seed')
    require(bundle['protocol'] == registry['protocol'], 'protocol not externally bound')
    artifact(root, registry['protocol'])
    expected = registry.get('identities'); supplied = bundle.get('identities')
    require(type(expected) is dict and type(supplied) is dict, 'identity manifest object required')
    missing = [role for role in IDENTITY_ROLES if role not in expected or role not in supplied]
    require(not missing, 'missing identity roles: ' + ','.join(missing))
    for role in expected:
        require(supplied.get(role) == expected[role], 'identity reference differs: ' + role)
        artifact(root, expected[role])
    identity_hashes = {k: v['sha256'] for k, v in expected.items()}
    for label in ('source_start', 'source_end'):
        ledger = load(artifact(root, bundle[label]))
        require(ledger.get('schema') == 'runtime_identity_ledger_v1', 'runtime identity ledger schema')
        require(ledger.get('identity_hashes') == identity_hashes, label + ' runtime identities mismatch')
        require(ledger.get('protocol_sha256') == bundle['protocol']['sha256'], label + ' protocol mismatch')
    expected_keys = registry.get('unit_arm_keys')
    require(type(expected_keys) is list and bool(expected_keys), 'expected unit/arm coverage missing')
    def key(entry):
        require(type(entry.get('day_id')) is int and type(entry.get('arm')) is str and bool(entry['arm']), 'invalid day/arm key')
        return (entry['day_id'], entry['arm'])
    keys = [key(x) for x in expected_keys]; require(len(set(keys)) == len(keys), 'duplicate registered unit/arm key')
    traces = bundle.get('traces'); require(type(traces) is list, 'raw trace manifest missing')
    actual_keys = [key(x) for x in traces]
    require(len(set(actual_keys)) == len(actual_keys) and set(actual_keys) == set(keys), 'raw day/arm coverage differs')
    raw_hashes = [x['raw']['sha256'] for x in traces]
    require(len(set(raw_hashes)) == len(raw_hashes), 'same raw file reused for independent day/arm keys')
    module, pins = load_validator(root, registry)
    replayed = []
    for entry in traces:
        raw = artifact(root, entry['raw'])
        if registry['validator'] == 'delivery_event_contract_only':
            document = load(raw); require(document.get('schema') == 'support_delivery_trace_v1', 'delivery raw schema')
            require(key(document) == key(entry), 'embedded raw day/arm identity differs')
            frozen_contract = load(artifact(root, expected['contract']))
            frozen_data = load(artifact(root, expected['dataset']))
            require(document['contract'] == frozen_contract.get('delivery_event_contract'), 'raw contract differs from registered actual contract')
            day_data = frozen_data.get('days', {}).get(str(entry['day_id']))
            require(type(day_data) is dict and document['orders'] == day_data.get('orders') and document['coordinates'] == day_data.get('coordinates'), 'raw day data differs from registered actual dataset')
            require(all(str(int(k)) == k for k in document['orders']) and all(str(int(k)) == k for k in document['coordinates']), 'noncanonical or colliding order coordinate keys')
            orders = {int(k): v for k, v in document['orders'].items()}
            coords = {int(k): v for k, v in document['coordinates'].items()}
            result = module.validate_trace(orders, document['events'], coordinates=coords, **document['contract'])
        else:
            result = module.replay(raw)
            require(result.get('formal_eligible') is False, 'engineering journal promoted')
            require(result.get('days_replayed') == [entry['day_id']], 'one journal per registered day/arm required')
        close_tree(result, entry['expected_replay'])
        replayed.append({'day_id': entry['day_id'], 'arm': entry['arm'], 'raw_sha256': sha(raw), 'recomputed_result': result})
    # Catch sources or inputs changed while the validator executed.
    loaded_sources = {}
    registered_sources = {entry['path']: entry['sha256'] for entry in pins}
    for mod in list(sys.modules.values()):
        filename = getattr(mod, '__file__', None)
        if filename and str(filename).endswith('.py'):
            path = Path(filename).resolve()
            if path.is_relative_to(root):
                rel = path.relative_to(root).as_posix()
                require(rel in registered_sources, 'executed project source absent from validator closure: ' + rel)
                require(sha(path) == registered_sources[rel], 'loaded source bytes differ: ' + rel)
                loaded_sources[rel] = registered_sources[rel]
    for entry in pins: artifact(root, entry)
    for entry in expected.values(): artifact(root, entry)
    for entry in traces: artifact(root, entry['raw'])
    return {'schema': 'support_closure_replay_v1', 'registered_artifact_byte_identity_pass': True,
            'runtime_ledger_consistency_pass': True, 'unit_arm_raw_coverage_pass': True,
            'actual_registered_validator_replay_pass': True, 'n_unit_arm_traces_replayed': len(replayed),
            'scope': registry['validator'], 'replays': replayed, 'loaded_project_source_sha256': loaded_sources,
            'registry_sha256': sha(registry_path), 'protocol_sha256': bundle['protocol']['sha256'],
            'independent_semantic_implementation_established': False,
            'original_training_execution_reproduced': False, 'full_from_zero_reproduction': False,
            'physical_calibration_established': False, 'formal_online_qualification_established': False,
            'stable_maskco_core_online_gain': False,
            'limitations': ['Trusted registry supplied externally is a source anchor, not pre-data time proof.',
                           'Stored start/end ledgers prove recorded consistency, not execution authenticity.',
                           'Weights bytes do not certify actual activation; training source bytes do not prove training executed.',
                           'Delivery validator checks event/ownership/distance; quality and energy are not certified.',
                           'Pickup observer reuses its registered evaluator; independent physical evaluator is still required.',
                           'No statistical confirmation or MaskCO attribution implemented.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('project', 'bundle', 'trusted-registry', 'output'): parser.add_argument('--' + field, required=True)
    args = parser.parse_args(); out = Path(args.output).resolve()
    require(not out.is_relative_to(Path(args.project).resolve()), 'output must be outside authoritative inputs')
    try:
        result = audit(args.project, args.bundle, args.trusted_registry); code = 0
    except (ClosureError, ValueError, KeyError, TypeError, OSError, ImportError, RuntimeError) as exc:
        result = {'schema': 'support_closure_replay_v1', 'engineering_replay_failed_closed': True,
                  'error': type(exc).__name__ + ': ' + str(exc), 'full_from_zero_reproduction': False,
                  'formal_online_qualification_established': False, 'stable_maskco_core_online_gain': False}; code = 2
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as stream: json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'code': code, 'report': str(out), 'stable_maskco_core_online_gain': False})); return code


if __name__ == '__main__': raise SystemExit(main())
