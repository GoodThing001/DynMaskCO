"""P2-C-CAMR-STATIC-01  read-only L1-B adjudication (errata).

Purpose: recompute, from the two frozen raw Pass A / Pass B artefacts only, the L1-B membership
relation that `REPORT.md` got wrong, and prove the error deterministically.

The defect (frozen in commit 94a517b, NOT modified here): `run_pass.py:155-156` tested

    sd['plan_hash'] in d_hashes        # sd = distance-greedy, d_hashes = distance-SGBS set

i.e. "is distance-greedy covered by distance-SGBS", which is true almost by construction because
the SGBS skeleton records the branch-free greedy completion first. The registered question was

    sm['plan_hash'] in d_hashes        # sm = MaskCO-greedy

This script recomputes the second quantity from the artefacts, per unit and per run.

It does NOT load the model, does NOT re-run any search, and does NOT touch T1/T2/registration:
it reads four JSON files and one source file. Its receipt has its own source identity and does
NOT inherit the earlier 12/12 verification, whose F2 check recomputed only the L1-A aggregates
and copied the L1-B field from the (wrong) summary.

Usage: python readonly_l1b_adjudication.py [--out readonly_l1b_adjudication.json]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent
SELF = Path(__file__).resolve()

INPUTS = ['pass_a_run1.json', 'pass_b_run1.json', 'pass_a_run2.json', 'pass_b_run2.json',
          'run_pass.py', 'verification.json', 'REPORT.md']

OFFENDING_SOURCE_LINES = [155, 156]


def sha_file(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def load(name):
    return json.loads((OUT / name).read_text(encoding='utf-8'))


def ukey(r):
    u = r['unit_id']
    return (u['arm'], int(u['day_id']), int(u['ordinal']), tuple(int(x) for x in u['mask']))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, default=OUT / 'readonly_l1b_adjudication.json')
    a = ap.parse_args()
    if a.out.exists():
        raise SystemExit('refuse overwrite: %s already exists' % a.out)

    pa = {r: load('pass_a_run%d.json' % r) for r in (1, 2)}
    pb = {r: load('pass_b_run%d.json' % r) for r in (1, 2)}
    ver = load('verification.json')

    rows = {}
    for run in (1, 2):
        A = {ukey(r): r for r in pa[run]['rows']}
        B = {ukey(r): r for r in pb[run]['rows']}
        if set(A) != set(B):
            raise SystemExit('unit key mismatch between pass A and pass B (run %d)' % run)
        rows[run] = []
        for k in A:
            mh = A[k]['sides']['model']['plan_hash']
            dh = A[k]['sides']['distance']['plan_hash']
            d_set = set(B[k]['sides']['distance']['valid_plan_hashes'])
            m_set = set(B[k]['sides']['model']['valid_plan_hashes'])
            rows[run].append({
                'unit_id': A[k]['unit_id'],
                'model_greedy_hash': mh,
                'distance_greedy_hash': dh,
                'model_greedy_in_distance_set': (mh in d_set) if mh else None,
                'distance_greedy_in_distance_set': (dh in d_set) if dh else None,
                'model_greedy_in_own_set': (mh in m_set) if mh else None,
                'recorded_field': B[k].get('model_greedy_in_distance_set'),
                'distance_set_size': len(d_set),
                'model_set_size': len(m_set),
                'model_set_plans_covered': len(m_set & d_set),
            })
        rows[run].sort(key=lambda r: (r['unit_id']['arm'], r['unit_id']['day_id'],
                                      r['unit_id']['ordinal'], r['unit_id']['mask']))

    def agg(run, field):
        vals = [r[field] for r in rows[run]]
        return {'true': sum(1 for v in vals if v is True),
                'false': sum(1 for v in vals if v is False),
                'none': sum(1 for v in vals if v is None)}

    corrected = {run: agg(run, 'model_greedy_in_distance_set') for run in (1, 2)}
    distance_self = {run: agg(run, 'distance_greedy_in_distance_set') for run in (1, 2)}
    own = {run: agg(run, 'model_greedy_in_own_set') for run in (1, 2)}

    records_wrong_object = all(
        r['recorded_field'] == r['distance_greedy_in_distance_set'] for r in rows[1] + rows[2])
    records_not_model_object = all(
        r['recorded_field'] != r['model_greedy_in_distance_set'] for r in rows[1] + rows[2])
    runs_agree = ([(r['unit_id'], r['model_greedy_hash'], r['model_greedy_in_distance_set'])
                   for r in rows[1]]
                  == [(r['unit_id'], r['model_greedy_hash'], r['model_greedy_in_distance_set'])
                      for r in rows[2]])

    src_lines = (OUT / 'run_pass.py').read_text(encoding='utf-8').split('\n')
    offending = [{'line': n, 'text': src_lines[n - 1].strip()} for n in OFFENDING_SOURCE_LINES]

    checks = {
        'C1_nineteen_units_in_both_runs': len(rows[1]) == len(rows[2]) == 19,
        'C2_recorded_field_is_the_distance_quantity': records_wrong_object,
        'C3_recorded_field_is_not_the_model_quantity': records_not_model_object,
        'C4_corrected_model_greedy_is_never_covered':
            corrected[1]['false'] == 19 and corrected[2]['false'] == 19,
        'C5_distance_greedy_is_always_covered':
            distance_self[1]['true'] == 19 and distance_self[2]['true'] == 19,
        'C6_model_greedy_is_in_its_own_set':
            own[1]['true'] == 19 and own[2]['true'] == 19,
        'C7_two_runs_agree_on_the_corrected_quantity': runs_agree,
        'C8_no_model_loaded_no_search_rerun': True,
    }

    payload = {
        'schema': 'p2_c_camr_static_01_readonly_l1b_adjudication_v1',
        'run_id': pa[1]['run_id'],
        'kind': 'errata_readonly_adjudication',
        'frozen_commit_under_review': '94a517bca711f039bb262ac4059706119b270a70',
        'status_label': 'L1B_REPORTED_VERDICT_INVALID_PENDING_INDEPENDENT_READONLY_ADJUDICATION',
        'defect': {
            'file': 'run_pass.py',
            'lines': offending,
            'wrong_object': "sd['plan_hash']  (distance-greedy)",
            'required_object': "sm['plan_hash']  (MaskCO-greedy)",
            'effect': ('the L1-B coverage test compared distance-greedy against the '
                       'distance-SGBS set, which is true almost by construction because the SGBS '
                       'skeleton records the branch-free greedy completion first, so the field '
                       'was near-tautologically 19/19'),
        },
        'corrected_relation': {
            'definition': ('model_greedy_in_distance_set = PassA[unit].sides.model.plan_hash in '
                           'set(PassB[unit].sides.distance.valid_plan_hashes)'),
            'run1': corrected[1], 'run2': corrected[2],
            'covered': corrected[1]['true'], 'uncovered': corrected[1]['false'],
        },
        'controls': {
            'distance_greedy_in_distance_set': distance_self[1],
            'model_greedy_in_own_set': own[1],
            'model_set_plans_total': sum(r['model_set_size'] for r in rows[1]),
            'model_set_plans_covered_by_distance': sum(r['model_set_plans_covered'] for r in rows[1]),
        },
        'original_acceptance_does_not_cover_this': {
            'verification_sha256': sha_file(OUT / 'verification.json'),
            'reported_l1b_covered': ver['recomputed']['l1b_model_greedy_covered'],
            'reported_l1b_uncovered': ver['recomputed']['l1b_model_greedy_uncovered'],
            'why': ('verify_camr_static.py F2 recomputed only the L1-A aggregates and copied the '
                    'L1-B field from the summary; it never executed the membership test, so its '
                    '12/12 cannot certify the L1-B relation in either direction'),
        },
        'scope': ('read-only: no model load, no search re-execution, no change to registration '
                  'T1, run identity T2, preflight, Pass A/B artefacts or REPORT.md'),
        'inputs_sha256': {name: sha_file(OUT / name) for name in INPUTS},
        'source_sha256': sha_file(SELF),
        'checks': checks,
        'all_ok': all(checks.values()),
        'rows': rows[1],
    }
    (a.out).open('x', encoding='utf-8', newline='\n').write(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + '\n')

    for k, v in checks.items():
        print('%-5s %s' % ('PASS' if v else 'FAIL', k))
    print('corrected L1-B: covered %d / uncovered %d (run1)  covered %d / uncovered %d (run2)'
          % (corrected[1]['true'], corrected[1]['false'],
             corrected[2]['true'], corrected[2]['false']))
    print('recorded field is the DISTANCE quantity in all 19 units:', records_wrong_object)
    sys.exit(0 if payload['all_ok'] else 7)


if __name__ == '__main__':
    main()
