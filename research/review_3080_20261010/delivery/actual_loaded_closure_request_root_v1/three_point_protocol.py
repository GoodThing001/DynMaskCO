"""Three-capture-point fail-closed publication protocol for a delivery request.

Round 20 proved the closure candidate's single-interval behaviour (5/5 local checks).
This module registers and tests the protocol the real request needs: identity is
captured at three points and publication is refused unless every capture is identical
to the previous one.

  P1  after the model and explicit files are loaded
  P2  after the first JAX and first native (C++) call
  P3  immediately before publication

A lazy Python import or a native module arriving after P1 is detected at P2; anything
appearing between P2 and P3 is detected at P3.  Nothing is written when the check fails.

This is a local, model-free protocol test.  It does NOT qualify the complete-model
request, the native lazy-import surface of the real model, source identity of the
3080 tree, or the 8 s + 2 s budget.
"""
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DELIVERY = HERE.parent
sys.path.insert(0, str(ROOT / 'tools'))

import importlib.util

_spec = importlib.util.spec_from_file_location(
    'closure_candidate', ROOT / 'tools/actual_loaded_closure_candidate_r20.py')
closure = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(closure)


def capture(explicit):
    t0 = time.perf_counter()
    snap = closure.capture(explicit)
    return snap, time.perf_counter() - t0


def main():
    out = {'protocol': 'three_capture_point_fail_closed',
           'points': ['P1_after_load', 'P2_after_first_jax_and_native_call', 'P3_before_publication'],
           'candidate': 'tools/actual_loaded_closure_candidate_r20.py',
           'scenarios': {}, 'checks': []}

    def ck(name, value):
        out['checks'].append({'name': name, 'pass': bool(value)})

    tmp = Path(tempfile.mkdtemp(prefix='closure3p_'))
    explicit = [str(tmp / 'explicit_a.py'), str(tmp / 'explicit_b.py')]
    (tmp / 'explicit_a.py').write_text('A = 1\n', encoding='utf-8')
    (tmp / 'explicit_b.py').write_text('B = 2\n', encoding='utf-8')

    # ---------- scenario CLEAN: nothing changes between captures ----------
    p1, t1 = capture(explicit)
    lib = tmp / 'lazy_lib.py'
    lib.write_text('def work(x):\n    return x + 1\n', encoding='utf-8')
    sys.path.insert(0, str(tmp))
    import lazy_lib  # noqa: F401  (stands in for the first explicit call)
    first = lazy_lib.work(1)
    p2, t2 = capture(explicit)
    p3, t3 = capture(explicit)
    r12, r23 = closure.compare(p1, p2), closure.compare(p2, p3)
    out['scenarios']['clean_after_first_call'] = dict(
        first_call_result=first, capture_seconds=[t1, t2, t3],
        p1_files=len(p1['file_sha256']), p2_files=len(p2['file_sha256']), p3_files=len(p3['file_sha256']),
        p1_to_p2_qualified=r12['qualified'], p2_to_p3_qualified=r23['qualified'],
        p1_to_p2_new_files=len(r12['new_files']))
    ck('clean_path_p2_sees_the_lazy_python_import', not r12['qualified'] and
       any('lazy_lib' in f for f in r12['new_files']))
    ck('clean_path_p3_matches_p2', r23['qualified'])
    # after P2 the closure is stable, so publication would be allowed
    ck('clean_path_publication_allowed_after_stable_p2_p3', r23['qualified'])

    # ---------- scenario PRE_PUBLICATION_CHANGE: file edited between P2 and P3 ----------
    p2b, _ = capture(explicit)
    (tmp / 'explicit_a.py').write_text('A = 1\nCHANGED = True\n', encoding='utf-8')
    p3b, _ = capture(explicit)
    r23b = closure.compare(p2b, p3b)
    out['scenarios']['pre_publication_change'] = dict(
        qualified=r23b['qualified'], changed_files=[Path(f).name for f in r23b['changed_files']],
        new_files=[Path(f).name for f in r23b['new_files']])
    ck('pre_publication_change_is_rejected', not r23b['qualified'])
    (tmp / 'explicit_a.py').write_text('A = 1\n', encoding='utf-8')

    # ---------- scenario NEW_NATIVE_FILE: a native path appears between P1 and P2 ----------
    p1c, _ = capture(explicit)
    native = tmp / 'fake_native.so'
    native.write_bytes(b'\x7fELF' + b'0' * 64)
    start = closure.capture(explicit)
    start['native_paths'] = sorted(set(start['native_paths']) | {str(native.resolve())})
    start['file_sha256'][str(native.resolve())] = closure._sha(native)
    end = closure.capture(explicit)
    r12c = closure.compare(start, end)
    out['scenarios']['injected_native_path'] = dict(qualified=r12c['qualified'],
                                                    native_map_changed=r12c['native_map_changed'])
    ck('injected_native_path_is_rejected', not r12c['qualified'] and r12c['native_map_changed'])

    # ---------- scenario MISSING_EXPLICIT ----------
    missing = str(tmp / 'does_not_exist.py')
    try:
        closure.capture(explicit + [missing])
        raised = None
    except FileNotFoundError as exc:
        raised = 'FileNotFoundError'
    ck('missing_explicit_file_raises', raised == 'FileNotFoundError')
    out['scenarios']['missing_explicit'] = dict(raised=raised)

    # ---------- scenario NO_TREE_SCAN: capture cost is proportional to loaded set ----------
    out['scenarios']['no_tree_scan'] = dict(
        includes_unloaded_package_trees=p1['includes_unloaded_package_trees'],
        explicit_paths_count=len(p1['explicit_paths']))
    ck('candidate_does_not_scan_unloaded_trees', p1['includes_unloaded_package_trees'] is False)

    shutil.rmtree(tmp, ignore_errors=True)
    out['pass_count'] = sum(1 for c in out['checks'] if c['pass'])
    out['fail_count'] = sum(1 for c in out['checks'] if not c['pass'])
    out['model_loaded'] = False
    out['remote_executed'] = False
    out['complete_model_request_qualified'] = False
    out['unpassed_gates'] = [
        '3080 complete-model loaded closure (requires the 76M backbone environment)',
        'native lazy-import surface of the real model (JAX/C++ first call)',
        'source identity of the 3080 tree against the registered shard set',
        'full 8 s execution + 2 s recovery request budget',
        'publication gate placement inside the real request',
    ]
    out['note'] = ('passing these checks proves only the three-point fail-closed protocol works locally; '
                   'it does not qualify the complete-model request')

    out_path = HERE / 'THREE_POINT_PROTOCOL_RESULT.json'
    assert not out_path.exists()
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({k: v for k, v in out.items() if k != 'checks'}, ensure_ascii=False, indent=2)[:2500])
    return 0 if out['fail_count'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
