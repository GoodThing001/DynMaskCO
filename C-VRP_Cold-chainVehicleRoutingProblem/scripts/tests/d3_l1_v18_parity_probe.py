"""Development day 0 only. Preserve actual typed objects and first-difference evidence."""
import argparse
import copy
import dataclasses
import hashlib
import json
import pickle
import re
import struct
import sys
from pathlib import Path

EXT = Path(__file__).resolve().parents[2]
for part in ('scripts', 'scripts/simulation', 'scripts/coldchain', 'scripts/tests'):
    sys.path.insert(0, str(EXT / part))


def leaf(value):
    import numpy as np
    row = {'type': type(value).__module__ + '.' + type(value).__qualname__,
           'value_repr': repr(value)}
    if isinstance(value, (np.ndarray, np.generic)):
        row.update(dtype=value.dtype.str, shape=list(value.shape), raw_hex=value.tobytes().hex())
    if isinstance(value, (float, np.floating)):
        row['ieee754_binary64_big_endian'] = struct.pack('>d', value).hex()
    return row


def at_path(obj, path):
    for attr, index in re.findall(r'\.([^\.\[\]]+)|\[(\d+)\]', path[4:]):
        obj = obj[int(index)] if index else (obj[attr] if isinstance(obj, dict)
                                             else getattr(obj, attr))
    return obj


def comparison(a, b):
    from d3_l1_bitwise_parity import bitwise_equal
    verdict, diffs = bitwise_equal(a, b)
    rec = {'verdict': verdict, 'diffs': diffs}
    if diffs:
        path = diffs[0]['path']
        try:
            rec['first'] = {'path': path, 'original': leaf(at_path(a, path)),
                            'restored': leaf(at_path(b, path))}
        except (AttributeError, KeyError, TypeError, IndexError) as exc:
            rec['first_unresolved'] = repr(exc)
    return rec


def main():
    import numpy as np
    from coldchain_contract import ObjectiveProfile, apply_objective_profile, default_pilot_contract
    from d3_l1_state_collector import RevealSlotCollector, build_env, make_replanner
    from recourse_snapshot import (capture_recourse_snapshot, restore_recourse_snapshot,
                                   snapshot_state_hash)
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--exact', action='store_true')
    args = ap.parse_args()
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=False)
    dataset = EXT / 'data/m0_scale/dcc_50_r1_edod05_dev_check_teacher.npz'
    profile = EXT / 'results/o0cc/scale_v2/objective_profile.json'
    data = dict(np.load(dataset))
    prof = json.loads(profile.read_text(encoding='utf-8'))
    fields = {f.name for f in dataclasses.fields(ObjectiveProfile)}
    eff = apply_objective_profile(default_pilot_contract(),
                                  ObjectiveProfile(**{k: v for k, v in prof.items() if k in fields}))

    def env():
        return build_env(data, 50., 25, 1., make_replanner(50., 24., eff)[0], eff, 24.)

    e = env()
    col = RevealSlotCollector(e, 0, 4, capture=False)
    captured = []

    def hook(e, inst, clock, event, reveal, vehicles, traces, served, universe):
        col(e, inst, clock, event, reveal, vehicles, traces, served, universe)
        if len(col.slots) > len(captured):
            runtime = copy.deepcopy((vehicles, traces, served))
            kw = {'preserve_scalar_types': True} if args.exact else {}
            snap = capture_recourse_snapshot(e, inst, clock, event, reveal, vehicles, traces,
                                             served, all_customers=universe, **kw)
            captured.append((snap, runtime, comparison(runtime, (vehicles, traces, served))))

    e.snapshot_hook = hook
    original, original_served = e.run(0)
    rows, objects = [], []
    for n, (snap, runtime, unchanged) in enumerate(captured, 1):
        restored = restore_recourse_snapshot(snap)
        er = env()
        recap = []

        def recapture(e, inst, clock, event, reveal, vehicles, traces, served, universe):
            if not recap:
                kw = {'preserve_scalar_types': True} if args.exact else {}
                recap.append(capture_recourse_snapshot(e, inst, clock, event, reveal, vehicles,
                                                       traces, served, all_customers=universe, **kw))

        er.snapshot_hook = recapture
        resumed, served = er.run_resumed(snap)
        rows.append({'slot': n, 'clock': snap['clock'], 'event_id': snap['event_id'],
                     'state_hash': snapshot_state_hash(snap),
                     'capture_did_not_mutate_runtime': unchanged,
                     'capture_restore': comparison(runtime, restored),
                     'recapture_hash_equal': snapshot_state_hash(recap[0]) == snapshot_state_hash(snap),
                     'full_resume_traces': comparison(original, resumed),
                     'full_resume_served': comparison(original_served, served)})
        objects.append({'snapshot': snap, 'runtime_at_capture': runtime,
                        'restored_runtime': restored, 'resumed_traces': resumed})
    blob = pickle.dumps({'original_traces': original, 'original_served': original_served,
                         'slots': objects}, protocol=4)
    (out / 'TYPED_OBJECTS.pkl').write_bytes(blob)
    result = {'use_exact_extension': args.exact, 'purpose': 'engineering_dev_day0_only',
              'formal_slots_collected': 0, 'argv': sys.argv, 'rows': rows,
              'typed_objects_sha256': hashlib.sha256(blob).hexdigest(),
              'dataset_sha256': hashlib.sha256(dataset.read_bytes()).hexdigest(),
              'effective_contract': eff.contract_hash,
              'sources': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                          [Path(__file__), EXT / 'scripts/simulation/recourse_snapshot.py',
                           EXT / 'scripts/simulation/strict_online_env.py']}}
    (out / 'PARITY.json').write_text(json.dumps(result, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
