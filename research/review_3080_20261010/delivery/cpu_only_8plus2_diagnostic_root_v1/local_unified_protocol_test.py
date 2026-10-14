"""Local test of the unified protocol, including the case the earlier test got wrong."""
import json
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[3] / 'tools'))

import unified_closure_protocol as U  # noqa: E402


def main():
    out = dict(protocol='unified', points=list(U.POINTS), warmup=list(U.WARMUP),
               gated=['%s->%s' % (a.split('_')[0], b.split('_')[0]) for a, b in U.GATED_ADJACENCIES],
               informational=['%s->%s' % (a.split('_')[0], b.split('_')[0]) for a, b in U.INFORMATIONAL_ADJACENCIES],
               scenarios={}, checks=[])

    def ck(name, value):
        out['checks'].append(dict(name=name, pass_=bool(value)))

    tmp = Path(tempfile.mkdtemp(prefix='unified_closure_'))
    explicit = [str(tmp / 'explicit_a.py'), str(tmp / 'explicit_b.py')]
    (tmp / 'explicit_a.py').write_text('A = 1\n', encoding='utf-8')
    (tmp / 'explicit_b.py').write_text('B = 2\n', encoding='utf-8')
    (tmp / 'lazy_lib.py').write_text('def work(x):\n    return x + 1\n', encoding='utf-8')
    sys.path.insert(0, str(tmp))

    # ---- scenario A: warm-up BEFORE P1, nothing changes afterwards ----
    p0 = U.snapshot(explicit)
    import lazy_lib  # declared warm-up: this is the lazy import the warm-up exists to provoke
    first = lazy_lib.work(1)
    p1 = U.snapshot(explicit)
    p2 = U.snapshot(explicit)          # timed execution segment completes with no new import
    p3 = U.snapshot(explicit)
    r = U.evaluate(p0, p1, p2, p3)
    out['scenarios']['warmup_before_p1_clean'] = dict(
        first_call=first, warmup_delta_qualified=r['informational']['P0->P1']['qualified'],
        warmup_new_files=len(r['informational']['P0->P1']['new_files']),
        p1_p2_qualified=r['gated']['P1->P2']['qualified'],
        p2_p3_qualified=r['gated']['P2->P3']['qualified'], publishable=r['publishable'])
    ck('warmup_delta_is_recorded_but_not_gated',
       r['informational']['P0->P1']['qualified'] is False
       and any('lazy_lib' in f for f in r['informational']['P0->P1']['new_files']))
    ck('clean_path_is_publishable', r['publishable'] is True)

    # ---- scenario B: lazy import INSIDE the timed region (P1 -> P2) ----
    for name in list(sys.modules):
        if name == 'lazy_lib2':
            del sys.modules[name]
    (tmp / 'lazy_lib2.py').write_text('VALUE = 42\n', encoding='utf-8')
    p0b = U.snapshot(explicit)
    p1b = U.snapshot(explicit)
    import lazy_lib2  # noqa: F401  arrives during the timed execution, after P1
    p2b = U.snapshot(explicit)
    p3b = U.snapshot(explicit)
    rb = U.evaluate(p0b, p1b, p2b, p3b)
    out['scenarios']['lazy_import_inside_timed_region'] = dict(
        p1_p2_qualified=rb['gated']['P1->P2']['qualified'], p2_p3_qualified=rb['gated']['P2->P3']['qualified'],
        publishable=rb['publishable'], refusal_reasons=rb['refusal_reasons'],
        new_files=[Path(f).name for f in rb['gated']['P1->P2']['new_files']])
    ck('lazy_import_inside_timed_region_rejects_publication', rb['publishable'] is False)
    ck('rejection_names_the_p1_to_p2_adjacency', rb['refusal_reasons'] == ['P1->P2'])

    # ---- scenario C: change between P2 and P3 ----
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

    # ---- scenario D: the earlier test's judgement is now rejected ----
    out['scenarios']['earlier_test_case_reversed'] = dict(
        earlier_test_judged_publishable_on_p2_p3_only=True,
        unified_verdict='rejected because P1->P2 differed')
    ck('contradiction_with_the_earlier_test_is_resolved', rb['publishable'] is False)

    shutil.rmtree(tmp, ignore_errors=True)
    out['pass_count'] = sum(1 for c in out['checks'] if c['pass_'])
    out['fail_count'] = sum(1 for c in out['checks'] if not c['pass_'])
    out['publication_performed'] = False
    out['model_loaded'] = False
    out_path = HERE / 'LOCAL_UNIFIED_PROTOCOL_RESULT.json'
    assert not out_path.exists()
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({k: v for k, v in out.items() if k != 'checks'}, ensure_ascii=False, indent=2)[:2200])
    return 0 if out['fail_count'] == 0 else 1


if __name__ == '__main__':
    raise SystemExit(main())
