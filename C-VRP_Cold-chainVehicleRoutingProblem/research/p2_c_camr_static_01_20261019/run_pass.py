"""P2-C-CAMR-STATIC-01  Pass A and Pass B, one complete run.

Refuses to start unless the registered chain is intact: registration -> run_identity ->
preflight, all bound by sha256, with the degeneration gate passed and no bias in either
direction (a failed gate is a hard stop, not something to work around).

Pass A (primary, L1-A): one greedy completion per side over the 19 registered units.
  side 1 = distance-greedy   (non-learned: rank by incremental distance)
  side 2 = MaskCO-greedy     (frozen backbone insertion scores)
Pass B (secondary, L1-B): the layer-by-layer beam completion sets of both sides.

Failure accounting is frozen: a unit that dies keeps its slot and is recorded as
PAIR_INVALID_<reason>; pair-level status is never "identical" when either side has no valid plan.

Usage: python run_pass.py --run 1        # -> pass_a_run1.json, pass_b_run1.json
       python run_pass.py --run 2
"""
import argparse
import sys
import time
import traceback
from pathlib import Path

OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))

import camr_shared as cs                                            # noqa: E402

BEAM, PER_PARENT, MAX_COMP = 2, 2, 8


def load_chain():
    reg = cs.load_json(OUT / 'registration.json')
    ident = cs.load_json(OUT / 'run_identity.json')
    pre = cs.load_json(OUT / 'preflight.json')
    reg_sha = cs.sha_file(OUT / 'registration.json')
    ident_sha = cs.sha_file(OUT / 'run_identity.json')
    if ident['T1']['registration_sha256'] != reg_sha:
        raise SystemExit('STOP: run_identity does not bind the current registration')
    if pre['run_identity_sha256'] != ident_sha:
        raise SystemExit('STOP: preflight does not bind the current run_identity')
    if pre['registration_sha256'] != reg_sha:
        raise SystemExit('STOP: preflight does not bind the current registration')
    if not pre['all_ok']:
        raise SystemExit('STOP: degeneration gate failed; Pass A must not run')
    for rel, want in ident['source_files'].items():
        if cs.sha_file(cs.WORKSPACE / rel) != want:
            raise SystemExit('STOP: source identity drift since T2: %s' % rel)
    return reg, ident, pre, ident_sha


def side_record(g, unit, plans, view, cvid, anchor, cn, pending):
    P = g['plan']
    if P is None:
        return {'status': g['reason'], 'plan_hash': None, 'n_completions': 0,
                'scoring_calls': g['scoring_calls']}
    ok, why = cs.check_plan(view, P, cvid, anchor, cn, pending)
    return {'status': 'OK' if ok else why, 'plan_hash': cs.signature(P), 'n_completions': 1,
            'suffix_first_after_anchor': cs.suffix_first(P, cvid),
            'anchor': list(cs.anchor_of(P, cvid)),
            'cvid_suffix': [int(x) for x in P[cvid].suffix],
            'scoring_calls': g['scoring_calls']}


def set_record(c, unit, view, cvid, anchor, cn, pending):
    hashes, n_ok, reasons = [], 0, {}
    for P in c['completions']:
        ok, why = cs.check_plan(view, P, cvid, anchor, cn, pending)
        reasons[why] = reasons.get(why, 0) + 1
        if ok:
            n_ok += 1
            hashes.append(cs.signature(P))
    return {'n_unique': c['stats']['n_unique'], 'n_total': c['stats']['n_total'],
            'n_fail': c['stats']['n_fail'], 'n_valid': n_ok, 'scoring_calls': c['stats']['scoring_calls'],
            'first_reason': c['first_reason'], 'valid_plan_hashes': hashes,
            'check_reasons': reasons, 'params': c['params']}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=int, choices=(1, 2), required=True)
    a = ap.parse_args()
    pa = OUT / ('pass_a_run%d.json' % a.run)
    pb = OUT / ('pass_b_run%d.json' % a.run)
    for p in (pa, pb):
        if p.exists():
            raise SystemExit('refuse overwrite: %s already exists' % p)

    reg, ident, pre, ident_sha = load_chain()
    res = reg['resources']
    start = time.time()

    from p2_pos01_mask_value import FrozenBackbone
    bb = FrozenBackbone(str(cs.WORKSPACE / ident['model']['checkpoint']))
    if dict((k, list(v) if isinstance(v, tuple) else v) for k, v in bb.manifest.items()) \
            != ident['model']['manifest']:
        raise SystemExit('STOP: model manifest differs from the T2 record')

    rows_a, rows_b = [], []
    cache = {}
    for unit in reg['units']['list']:
        uid = unit['unit_id']
        key = (uid['arm'], uid['day_id'], uid['ordinal'])
        if key not in cache:
            cache[key] = cs.unit_from_row({'arm': uid['arm'], 'day': uid['day_id'],
                                           'ordinal': uid['ordinal']})
        u = cache[key]
        mask = tuple(int(x) for x in uid['mask'])
        if (u['public_identity'] != uid['public_identity']
                or u['committed_next'] != unit['committed_next']
                or [int(x) for x in u['queue']] != [int(x) for x in unit['queue']]
                or list(mask) not in [list(m) for m in u['masks']]):
            raise SystemExit('STOP: unit reconstruction mismatch %s' % (uid,))

        view, plans, env, allowed = u['view'], u['plans'], u['env'], u['allowed']
        cvid, anchor, cn, pending = (u['cvid'], u['anchor'], u['committed_next'], u['pending'])
        t0 = time.time()
        notes = {}
        try:
            # ---- Pass A --------------------------------------------------
            gd = cs.greedy(cs.proposer(view, env, allowed, score='distance'), plans, mask, allowed)
            gm = cs.greedy(cs.proposer(view, env, allowed, score='model', backbone=bb, base=plans),
                           plans, mask, allowed)
            A = {'unit_id': uid, 'committed_vid': cvid, 'committed_next': cn,
                 'anchor': list(anchor), 'queue': [int(x) for x in u['queue']],
                 'n_allowed': len(allowed), 'n_pending': len(pending),
                 'sides': {'distance': side_record(gd, unit, plans, view, cvid, anchor, cn, pending),
                           'model': side_record(gm, unit, plans, view, cvid, anchor, cn, pending)}}
            sd, sm = A['sides']['distance'], A['sides']['model']
            if sd['status'] != 'OK' and sm['status'] != 'OK':
                pair = 'PAIR_INVALID_NO_OUTPUT'
            elif sd['status'] != 'OK':
                pair = 'PAIR_INVALID_DISTANCE_' + sd['status']
            elif sm['status'] != 'OK':
                pair = 'PAIR_INVALID_MODEL_' + sm['status']
            else:
                pair = 'OK'
            A['pair_status'] = pair
            A['same_plan'] = (sd['plan_hash'] == sm['plan_hash']) if pair == 'OK' else None
            A['same_first_after_anchor'] = ((sd['suffix_first_after_anchor']
                                             == sm['suffix_first_after_anchor'])
                                            if pair == 'OK' else None)

            # ---- Pass B --------------------------------------------------
            cd = cs.completions(cs.proposer(view, env, allowed, score='distance'), plans, mask,
                                allowed, BEAM, PER_PARENT, MAX_COMP)
            cm = cs.completions(cs.proposer(view, env, allowed, score='model', backbone=bb,
                                            base=plans), plans, mask, allowed, BEAM, PER_PARENT,
                                MAX_COMP)
            B = {'unit_id': uid,
                 'sides': {'distance': set_record(cd, unit, view, cvid, anchor, cn, pending),
                           'model': set_record(cm, unit, view, cvid, anchor, cn, pending)}}
            d_hashes = set(B['sides']['distance']['valid_plan_hashes'])
            m_hashes = set(B['sides']['model']['valid_plan_hashes'])
            B['model_greedy_in_distance_set'] = (
                sd['plan_hash'] in d_hashes if sd['plan_hash'] else None)
            B['model_set_subset_of_distance'] = (m_hashes <= d_hashes) if m_hashes else None
            B['set_intersection'] = len(m_hashes & d_hashes)
            B['set_union'] = len(m_hashes | d_hashes)
            B['uncovered_model_plans'] = sorted(m_hashes - d_hashes)
        except FloatingPointError as exc:                               # NONFINITE_SCORE
            notes['exception'] = 'NONFINITE_SCORE: %s' % exc
            A = {'unit_id': uid, 'pair_status': 'PAIR_INVALID_NONFINITE',
                 'sides': {'distance': {'status': 'NOT_RUN_NONFINITE'},
                           'model': {'status': 'NONFINITE_SCORE'}}, 'same_plan': None,
                 'same_first_after_anchor': None}
            B = {'unit_id': uid, 'sides': {}, 'model_greedy_in_distance_set': None}
        except Exception as exc:                                        # noqa: BLE001
            notes['exception'] = '%s: %s' % (type(exc).__name__, exc)
            notes['traceback'] = traceback.format_exc().splitlines()[-6:]
            A = {'unit_id': uid, 'pair_status': 'PAIR_INVALID_EXCEPTION',
                 'sides': {}, 'same_plan': None, 'same_first_after_anchor': None}
            B = {'unit_id': uid, 'sides': {}, 'model_greedy_in_distance_set': None}

        wall = time.time() - t0
        if wall > res['unit_wall_s']:
            A['pair_status'] = 'PAIR_INVALID_TIMEOUT'
            A['same_plan'] = None
            A['same_first_after_anchor'] = None
            notes['timeout'] = {'wall_s': round(wall, 3), 'cap_s': res['unit_wall_s']}
        A['wall_s'] = round(wall, 3)
        B['wall_s'] = round(wall, 3)
        A.update(notes)
        rows_a.append(A)
        rows_b.append(B)

    elapsed = time.time() - start
    valid = [r for r in rows_a if r['pair_status'] == 'OK']
    invalid = [r for r in rows_a if r['pair_status'] != 'OK']
    summary_a = {        'schema': 'p2_c_camr_static_01_pass_a_v1',
        'run': a.run, 'run_id': reg['run_id'],
        'run_identity_sha256': ident_sha,
        'preflight_sha256': cs.sha_file(OUT / 'preflight.json'),
        'registration_sha256': ident['T1']['registration_sha256'],
        'n_units_registered': reg['units']['n'], 'n_units_run': len(rows_a),
        'n_evaluable': len(valid), 'n_invalid': len(invalid),
        'invalid_units': [{'unit_id': r['unit_id'], 'pair_status': r['pair_status']}
                          for r in invalid],
        'n_different_plan': sum(1 for r in valid if not r['same_plan']),
        'n_same_first_after_anchor_but_different_plan':
            sum(1 for r in valid if r['same_plan'] is False and r['same_first_after_anchor']),
        'wall_s_total': round(elapsed, 3),
        'package_wall_cap_s': res['package_wall_s'],
        'rows': rows_a,
    }
    payload_b = {
        'schema': 'p2_c_camr_static_01_pass_b_v1',
        'run': a.run, 'run_id': reg['run_id'],
        'run_identity_sha256': ident_sha,
        'n_units_registered': reg['units']['n'], 'n_units_run': len(rows_b),
        'n_model_greedy_covered': sum(1 for r in rows_b
                                      if r.get('model_greedy_in_distance_set') is True),
        'n_model_greedy_uncovered': sum(1 for r in rows_b
                                        if r.get('model_greedy_in_distance_set') is False),
        'n_model_greedy_unknown': sum(1 for r in rows_b
                                      if r.get('model_greedy_in_distance_set') is None),
        'routes': ('a: pass_a_run%d.json' % a.run, 'b: pass_b_run%d.json' % a.run),
        'rows': rows_b,
    }
    cs.write_json(pa, summary_a)
    cs.write_json(pb, payload_b)
    print('run', a.run, 'units', len(rows_a), 'evaluable', len(valid), 'invalid', len(invalid),
          'different', summary_a['n_different_plan'], 'wall', round(elapsed, 1))
    for r in rows_a:
        if r['pair_status'] == 'OK':
            print('  %-7s d%s ord%-3s %-10s same=%s first_diff=%s'
                  % (r['unit_id']['arm'], r['unit_id']['day_id'], r['unit_id']['ordinal'],
                     str(r['unit_id']['mask']), r['same_plan'],
                     r['same_first_after_anchor'] is False))
        else:
            print('  %-7s d%s ord%-3s %-10s %s'
                  % (r['unit_id']['arm'], r['unit_id']['day_id'], r['unit_id']['ordinal'],
                     str(r['unit_id']['mask']), r['pair_status']))


if __name__ == '__main__':
    main()
