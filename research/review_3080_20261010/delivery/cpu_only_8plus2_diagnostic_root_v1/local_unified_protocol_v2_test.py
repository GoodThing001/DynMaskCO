"""Local test of the unified protocol r2, including the case that crashed r1 on the server.

r1 failed on the 3080 host with `TypeError: expected str, bytes or os.PathLike object,
not _OpNamespace` inside the candidate's module loop.  That input is reproduced here, so
the fix is verified locally before it is trusted on the server.
"""
import json
import shutil
import sys
import tempfile
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import unified_closure_protocol_r2 as U  # noqa: E402


def main():
    out = dict(protocol='unified_r2', candidate=str(U.CANDIDATE),
               points=list(U.POINTS), warmup=list(U.WARMUP),
               gated=['%s->%s' % (a.split('_')[0], b.split('_')[0]) for a, b in U.GATED_ADJACENCIES],
               informational=['%s->%s' % (a.split('_')[0], b.split('_')[0]) for a, b in U.INFORMATIONAL_ADJACENCIES],
               scenarios={}, checks=[])

    def ck(name, value):
        out['checks'].append(dict(name=name, pass_=bool(value)))

    tmp = Path(tempfile.mkdtemp(prefix='unified_closure_r2_'))
    explicit = [str(tmp / 'explicit_a.py'), str(tmp / 'explicit_b.py')]
    (tmp / 'explicit_a.py').write_text('A = 1\n', encoding='utf-8')
    (tmp / 'explicit_b.py').write_text('B = 2\n', encoding='utf-8')
    (tmp / 'lazy_lib.py').write_text('def work(x):\n    return x + 1\n', encoding='utf-8')
    sys.path.insert(0, str(tmp))

    # ---- scenario A: warm-up BEFORE P1, nothing changes afterwards ---------
    p0 = U.snapshot(explicit)
    import lazy_lib
    first = lazy_lib.work(1)
    p1 = U.snapshot(explicit)
    p2 = U.snapshot(explicit)
    p3 = U.snapshot(explicit)
    r = U.evaluate(p0, p1, p2, p3)
    out['scenarios']['warmup_before_p1_clean'] = dict(
        first_call=first, warmup_delta_qualified=r['informational']['P0->P1']['qualified'],
        warmup_new_files=len(r['informational']['P0->P1']['new_files']),
        p1_p2_qualified=r['gated']['P1->P2']['qualified'],
        p2_p3_qualified=r['gated']['P2->P3']['qualified'], publishable=r['publishable'])
    ck('warmup_delta_is_recorded_but_not_gated',
       r['informational']['P0->P1']['qualified'] is False)
    ck('clean_path_is_publishable', r['publishable'] is True)

    # ---- scenario B: file module imported inside the timed region ----------
    (tmp / 'lazy_lib2.py').write_text('VALUE = 42\n', encoding='utf-8')
    p1b = U.snapshot(explicit)
    import lazy_lib2  # noqa: F401
    p2b = U.snapshot(explicit)
    p3b = U.snapshot(explicit)
    rb = U.evaluate(p1b, p1b, p2b, p3b)
    out['scenarios']['file_module_inside_timed_region'] = dict(
        p1_p2_qualified=rb['gated']['P1->P2']['qualified'], publishable=rb['publishable'],
        refusal_reasons=rb['refusal_reasons'],
        new_files=[Path(f).name for f in rb['gated']['P1->P2']['new_files']])
    ck('file_module_inside_timed_region_rejects_publication', rb['publishable'] is False)
    ck('rejection_names_the_p1_to_p2_adjacency', rb['refusal_reasons'] == ['P1->P2'])

    # ---- scenario C: change between P2 and P3 -------------------------------
    p1c = U.snapshot(explicit)
    p2c = U.snapshot(explicit)
    (tmp / 'explicit_a.py').write_text('A = 1\nCHANGED = True\n', encoding='utf-8')
    p3c = U.snapshot(explicit)
    rc = U.evaluate(p2c, p1c, p2c, p3c)
    out['scenarios']['change_between_p2_and_p3'] = dict(
        publishable=rc['publishable'], refusal_reasons=rc['refusal_reasons'])
    ck('change_before_publication_rejects', rc['publishable'] is False
       and 'P2->P3' in rc['refusal_reasons'])
    (tmp / 'explicit_a.py').write_text('A = 1\n', encoding='utf-8')

    # ---- scenario D: the exact r1 server crash input ------------------------
    fabricated = types.ModuleType('fabricated_attrs_module')
    fabricated.__getattr__ = lambda name: object()      # PEP 562, prints as _OpNamespace-alike
    sys.modules['fabricated_attrs_module'] = fabricated
    fabricated_hasattr_file = hasattr(fabricated, '__file__')
    raised = None
    try:
        pd = U.snapshot(explicit)
    except BaseException as exc:                        # r20 raised exactly here
        raised = repr(exc)
        pd = None
    out['scenarios']['r1_server_crash_input'] = dict(
        hasattr_file_is_true=bool(fabricated_hasattr_file), raised=raised,
        recorded_as_module_without_file=bool(pd and 'fabricated_attrs_module' in pd['modules_without_file']))
    ck('non_path_module_attribute_does_not_crash_capture', raised is None)
    ck('non_path_module_is_still_seen_by_name',
       bool(pd and 'fabricated_attrs_module' in pd['module_files'])
       and pd['module_files']['fabricated_attrs_module'] is None)

    # ---- scenario E: a file-less module arriving inside the timed region ----
    p1e = U.snapshot(explicit)
    late = types.ModuleType('late_fileless_module')
    late.__getattr__ = lambda name: object()
    sys.modules['late_fileless_module'] = late
    p2e = U.snapshot(explicit)
    re_ = U.evaluate(p1e, p1e, p2e, p2e)
    out['scenarios']['fileless_module_inside_timed_region'] = dict(
        p1_p2_qualified=re_['gated']['P1->P2']['qualified'], publishable=re_['publishable'],
        module_added=re_['gated']['P1->P2']['module_added'],
        new_files=len(re_['gated']['P1->P2']['new_files']))
    ck('fileless_module_inside_timed_region_rejects', re_['publishable'] is False
       and 'late_fileless_module' in re_['gated']['P1->P2']['module_added'])

    # ---- scenario F: the r20 FAIL is documented, not rewritten -------------
    fail = HERE / 'run_r1' / 'DIAG_FAILURE.traceback.txt'
    out['scenarios']['r1_failure_origin'] = dict(
        path='run_r1/DIAG_FAILURE.traceback.txt', preserved=fail.is_file(),
        type_error_recorded=bool(fail.is_file() and '_OpNamespace' in fail.read_text(encoding='utf-8')))
    ck('r1_failure_original_is_preserved_and_not_rewritten',
       fail.is_file() and '_OpNamespace' in fail.read_text(encoding='utf-8'))

    sys.modules.pop('fabricated_attrs_module', None)
    sys.modules.pop('late_fileless_module', None)
    shutil.rmtree(tmp, ignore_errors=True)
    out['pass_count'] = sum(1 for c in out['checks'] if c['pass_'])
    out['fail_count'] = sum(1 for c in out['checks'] if not c['pass_'])
    out['publication_performed'] = False
    out['model_loaded'] = False
    out_path = HERE / 'LOCAL_UNIFIED_PROTOCOL_v2_RESULT.json'
    assert not out_path.exists()
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n',
                        encoding='utf-8', newline='\n')
    print(json.dumps({k: v for k, v in out.items() if k != 'checks'}, ensure_ascii=False, indent=2)[:2600])
    return 0 if out['fail_count'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
