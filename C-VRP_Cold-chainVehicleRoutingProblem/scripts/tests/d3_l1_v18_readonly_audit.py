"""Independent process, read-only artifact verification; never executes an episode."""
import argparse
import dataclasses
import hashlib
import json
import pickle
import struct
import sys
from pathlib import Path

EXT = Path(__file__).resolve().parents[2]
for part in ('scripts/simulation', 'scripts/coldchain', 'scripts/tests'):
    sys.path.insert(0, str(EXT / part))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def typed_bytes(x):
    """Independent exact encoding: all types and float payload bits are mandatory."""
    import numpy as np
    tag = (type(x).__module__ + '.' + type(x).__qualname__).encode()
    def frame(b):
        return len(b).to_bytes(8, 'big') + b
    if isinstance(x, (np.ndarray, np.generic)):
        payload = frame(x.dtype.str.encode()) + frame(repr(x.shape).encode()) + frame(x.tobytes())
    elif type(x) is float:
        payload = struct.pack('>d', x)
    elif x is None or type(x) in (int, bool, str):
        payload = repr(x).encode()
    elif dataclasses.is_dataclass(x):
        payload = b''.join(frame(f.name.encode()) + frame(typed_bytes(getattr(x, f.name)))
                           for f in dataclasses.fields(x))
    elif type(x) is dict:
        payload = b''.join(frame(typed_bytes(k)) + frame(typed_bytes(x[k]))
                           for k in sorted(x, key=repr))
    elif type(x) in (tuple, list):
        payload = b''.join(frame(typed_bytes(v)) for v in x)
    else:
        raise TypeError('unverifiable exact object: %s' % tag)
    return frame(tag) + frame(payload)


def audit(root, runtime_identity):
    from recourse_snapshot import snapshot_state_hash, restore_recourse_snapshot
    root = Path(root)
    frozen = json.loads(Path(runtime_identity).read_text(encoding='utf-8'))
    checks = []
    def ck(name, ok, detail=None):
        checks.append({'name': name, 'ok': bool(ok), 'detail': detail})
    rows = [json.loads(line) for line in (root / 'STATE_LEDGER.jsonl').read_text().splitlines()]
    publications = [r for r in rows if r['event'] == 'PUBLISHED']
    for mod, rec in frozen['modules'].items():
        ck('current_source:' + mod, sha(EXT / rec['path']) == rec['sha256'])
    for filename in ('SUPERVISOR_IDENTITY_START.json', 'SUPERVISOR_IDENTITY_END.json',
                     'WORKER_IDENTITY_START_DAY00.json', 'WORKER_IDENTITY_END_DAY00.json'):
        path = root / filename
        if not path.exists():
            ck(filename, False, 'missing')
            continue
        state = json.loads(path.read_text())
        ck(filename, state['ok'] and state['runtime_identity_sha256'] == sha(runtime_identity)
           and all(state['identity']['frozen_checks'].values()))
        ck(filename + ':sources', {k: v['sha256'] for k, v in state['identity']['modules'].items()}
           == {k: v['sha256'] for k, v in frozen['modules'].items()})
    manifest = [json.loads(line) for line in (root / 'STATE_MANIFEST.jsonl').read_text().splitlines()]
    published_paths = set()
    for pub in publications:
        pos = pub['position']
        prefix = 'day%d/slot%d:' % (pub['day_id'], pos)
        path = root / pub['snapshot_path']
        published_paths.add(path.resolve())
        matching = [r for r in manifest if (r['day_id'], r['position']) == (pub['day_id'], pos)]
        ck(prefix + 'one_manifest', len(matching) == 1)
        ck(prefix + 'snapshot_sha_bytes', sha(path) == pub['snapshot_sha256']
           and path.stat().st_size == pub['bytes'])
        snap = pickle.loads(path.read_bytes())
        ck(prefix + 'hash', snapshot_state_hash(snap) == snap['state_hash'] == pub['state_hash'])
        ck(prefix + 'contract', snap['coldchain_contract_hash'] == frozen['effective_contract_hash'])
        ck(prefix + 'scalar_version', snap['runtime_scalar_encoding'] == 'lossless_scalar_v18')
        restore_recourse_snapshot(snap)  # integrity and reconstructability only; no run
        evpath = root / pub['validation_path']
        ck(prefix + 'validation_sha', sha(evpath) == pub['validation_sha256'])
        ev = pickle.loads(evpath.read_bytes())
        ck(prefix + 'full_trace_exact', typed_bytes(ev['original_traces']) == typed_bytes(ev['resumed_traces']))
        ck(prefix + 'served_exact', typed_bytes(ev['original_served']) == typed_bytes(ev['resumed_served']))
        ck(prefix + 'snapshot_serialization_exact', typed_bytes(snap) == typed_bytes(ev['reloaded_snapshot']))
        ck(prefix + 'recapture_exact', typed_bytes(snap) == typed_bytes(ev['recaptured_snapshot']))
        p, q = ev['original_projection'], ev['perturbed_projection']
        ck(prefix + 'two_calls', p['replanner_calls'] == q['replanner_calls'] == 1)
        for key in ('visible_inputs_sha', 'legal_candidates_sha', 'planned_suffix_sha',
                    'planned_suffix_sha_boundary'):
            ck(prefix + key, p[key] == q[key] and p[key] is not None)
        ck(prefix + 'no_projection_errors', not p['errors'] and not q['errors'])
        started = [r for r in rows if r['event'] == 'ATTEMPT_STARTED'
                   and (r['day_id'], r['position']) == (pub['day_id'], pos)]
        ck(prefix + 'one_capture_wal', len(started) == 1 and started[0]['mono'] < pub['mono'])
        ck(prefix + 'span', 0 <= pub['slot_span_wall_s'] <= 60.)
        completion = [r for r in rows if r['event'] == 'SLOT_FINISHED'
                      and (r['day_id'], r['position']) == (pub['day_id'], pos)]
        ck(prefix + 'durable_completion', len(completion) == 1
           and completion[0]['mono'] >= pub['mono'] and completion[0]['slot_span_wall_s'] <= 60.)
    result = json.loads((root / 'SUPERVISOR_RESULT.json').read_text())
    ck('denominator64', result['slots_in_denominator'] == 64)
    ck('publication_count', result['slots_published'] == len(publications))
    ck('no_duplicate_publication', len({(r['day_id'], r['position']) for r in publications}) == len(publications))
    orphans = [str(p.relative_to(root)) for p in (root / 'snapshots').glob('*.pkl')
               if p.resolve() not in published_paths]
    return {'purpose': 'independent_readonly_no_rollout', 'checks': checks,
            'n_pass': sum(r['ok'] for r in checks), 'n_fail': sum(not r['ok'] for r in checks),
            'publications_verified': len(publications), 'orphan_objects_never_accepted': orphans,
            'staged_objects_never_accepted': [str(p.relative_to(root)) for p in (root / 'staged').glob('*.pkl')],
            'model_identity': 'F_FREE_NO_MODEL', 'runtime_identity_sha256': sha(runtime_identity)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True)
    ap.add_argument('--runtime-identity', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    try:
        result = audit(args.root, args.runtime_identity)
    except Exception as exc:
        result = {'n_fail': 1, 'error': '%s: %s' % (type(exc).__name__, exc)}
    result.update(auditor_sha256=sha(__file__), argv=sys.argv)
    with open(args.out, 'x', encoding='utf-8') as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    print(json.dumps({k: v for k, v in result.items() if k != 'checks'}))
    return int(result['n_fail'] != 0)


if __name__ == '__main__':
    sys.exit(main())
