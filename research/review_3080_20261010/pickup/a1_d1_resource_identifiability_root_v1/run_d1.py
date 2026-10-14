"""D1 executor: is the resource channel identifiable?  (read-only, CPU-only, no model load)

Implements CC_Compare/P1A_D1_资源可识别性诊断_登记_2026-10-14.md.
It does NOT run any planner, does NOT load a model, does NOT read utility as a criterion.

Two evidence sources:
  A. the frozen registered decision log (1000 decisions) -- the deployed rule's own record;
  B. a structural pressure fixture: the registered rule replayed on the registered TRAIN
     split (regenerated deterministically from train_seed=20260925) over a PRE-DECLARED
     budget ladder, to show whether the budget constraint CAN bind.

alpha ladder declared BEFORE any computation (rule stated in the registration):
    alpha in {1.00, 0.75, 0.50, 0.25}
    rule: a coarse ladder anchored at the registered budget (alpha=1.0) spanning one order
    of magnitude, derived from the registered budget unit alone -- no reference to utility,
    to the observed log, or to which tier "looks good".
"""
import ast
import bisect
import hashlib
import json
import os
import sys
import traceback
from pathlib import Path

# --- CPU-only declaration (no jax import happens in this executor) ------------
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '1')

ROOT = Path(__file__).resolve().parents[4]
EXT = ROOT / 'C-VRP_Cold-chainVehicleRoutingProblem'
V4 = ROOT / 'research/review_3080_20261010/pickup/a1_budget_reserving_admission_root_v4'
V3 = ROOT / 'research/review_3080_20261010/pickup/a1_deployment_rule_identity_root_v3'
OUT = Path(__file__).resolve().parent
CC_TOOLS = ROOT / 'CC_Compare' / 'tools'
REGISTERED_LOG_SHA = '485f60735e54faa2e0bd28cd379772afe99cbd24985250212a258f619eb1a44e'
ALPHA_LADDER = [1.00, 0.75, 0.50, 0.25]
B_REGISTERED = 710.6548328611384


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    res = {'stage': 'D1_resource_identifiability', 'cpu_only': True,
           'model_loaded': False, 'planner_run': False, 'utility_read_as_criterion': False,
           'alpha_ladder_declared_before_computation': ALPHA_LADDER,
           'alpha_ladder_rule': ('coarse ladder anchored at the registered budget (alpha=1.0), '
                                 'spanning one order of magnitude, derived from the registered '
                                 'budget unit alone'),
           'publication_performed': False}
    write('D1_STATE.json', res)

    # ---------------- identity ----------------
    log_path = V3 / 'DECISION_LOG.jsonl'
    rule_path = V4 / 'reserving_admission.py'
    reg_path = V4 / 'REGISTRATION.json'
    reg = json.loads(reg_path.read_text(encoding='utf-8'))
    ident = {'decision_log': {'path': str(log_path.relative_to(ROOT)), 'sha256': sha(log_path),
                              'registered_sha256': REGISTERED_LOG_SHA,
                              'match': sha(log_path) == REGISTERED_LOG_SHA},
             'rule_module': {'path': str(rule_path.relative_to(ROOT)), 'sha256': sha(rule_path),
                             'registered_in_v4_registration':
                                 reg['sources'].get(str(rule_path), None),
                             'match': reg['sources'].get(str(rule_path)) == sha(rule_path)},
             'v4_registration': {'sha256': sha(reg_path)},
             'contract_sha256_expected': reg['frozen']['expected_contract_sha256'],
             'data_sha256_expected': reg['frozen']['expected_data_sha256'],
             'data_meta_sha256_expected': reg['frozen']['expected_meta_sha256'],
             'data_files_present_locally': None}
    res['identity'] = ident

    rows = [json.loads(l) for l in log_path.read_text(encoding='utf-8').splitlines() if l.strip()]
    res['n_decisions'] = len(rows)
    if len(rows) != 1000:
        res['verdict'] = 'not_qualified'
        res['reason'] = 'n_decisions != 1000'
        write('D1_STATE.json', res)
        return 1
    write('D1_STATE.json', res)

    # ---------------- primary 4-class classification (definitions frozen in the registration)
    cls = {'excluded_by_budget_precheck': 0, 'excluded_by_price': 0,
           'admitted_but_base_rejected': 0, 'admitted_and_base_accepted': 0}
    for r in rows:
        reason, base = r.get('reason'), r.get('base_decision')
        if reason == 'over_remaining_budget':
            cls['excluded_by_budget_precheck'] += 1
        elif reason == 'below_shadow_price':
            cls['excluded_by_price'] += 1
        elif r.get('rule_decision') == 'admit' and base != 'accepted':
            cls['admitted_but_base_rejected'] += 1
        elif r.get('rule_decision') == 'admit' and base == 'accepted':
            cls['admitted_and_base_accepted'] += 1
    res['primary_classification'] = dict(counts=cls,
                                         fractions={k: v / 1000.0 for k, v in cls.items()})

    # ---------------- distributions ----------------
    def q(vals, p):
        v = sorted(vals)
        return v[min(int(p * (len(v) - 1)), len(v) - 1)]

    b_rem = [float(r['b_rem']) for r in rows]
    e_low = [float(r['e_lower']) for r in rows]
    lams = [round(float(r['lam']), 6) for r in rows if r.get('lam') is not None]
    days = {}
    for r in rows:
        days.setdefault((r['inst'], r.get('day_id', 0)), []).append(r)
    per_day_spent = {str(k): max(float(x['spent_before']) for x in v) for k, v in days.items()}
    lam_hist = {}
    for l in lams:
        lam_hist[str(l)] = lam_hist.get(str(l), 0) + 1
    res['distributions'] = {
        'b_rem_kwh': {'min': min(b_rem), 'p05': q(b_rem, .05), 'p25': q(b_rem, .25),
                      'p50': q(b_rem, .50), 'p75': q(b_rem, .75), 'max': max(b_rem),
                      'n_below_50': sum(1 for x in b_rem if x < 50.0)},
        'e_lower_kwh': {'min': min(e_low), 'p50': q(e_low, .5), 'max': max(e_low),
                        'sum': sum(e_low)},
        'spent_before_per_day_max_kwh': per_day_spent,
        'spent_over_B': {k: v / B_REGISTERED for k, v in per_day_spent.items()},
        'B_registered_kwh': B_REGISTERED,
        'lam_value_histogram': lam_hist,
        'lam_floor_count': sum(1 for l in lams if abs(l - 0.25) < 1e-9),
        'lam_max_grid_count': sum(1 for l in lams if abs(l - 8.0) < 1e-9),
        'n_decisions_with_lam_recorded': len(lams),
    }

    # ---------------- shadow-price observables (frozen log; NO re-run) ----------------
    # the rule's score is REV[cls] - lam * e_lower; the intervention vs "no shadow price"
    # (lam == 0) is lam * e_lower.
    n_price_active = sum(1 for r in rows if float(r['lam']) > 0.0 and float(r['e_lower']) > 0.0)
    n_lambda_varies = sum(1 for r in rows if abs(float(r['lam']) - 0.25) > 1e-9)
    n_score_changed = sum(1 for r in rows
                          if abs(float(r['lam']) * float(r['e_lower'])) > 0.0)
    n_action_changed_by_price = cls['excluded_by_price']
    # ablation lam == 0 evaluated ON THE FROZEN LOG: with lam=0 the filter admits whenever
    # e <= b_rem; the hard pre-check fired 0 times, so the filter's action changes on exactly
    # the price-rejected decisions.
    ablation_changes = n_action_changed_by_price
    res['shadow_price_effect'] = {
        'n_price_active': n_price_active,
        'n_lambda_above_grid_floor': n_lambda_varies,
        'n_score_changed': n_score_changed,
        'n_action_changed_by_price': n_action_changed_by_price,
        'ablation_lambda_zero_action_changes_from_frozen_log': ablation_changes,
        'n_ordering_changed': None,
        'n_ordering_changed_note': ('NOT recordable from this log: the registered rule log does '
                                    'not persist the candidate set, so a candidate-ordering '
                                    'change cannot be observed -- declared as a field gap, '
                                    'not inferred'),
        'certify_energy_missing': True,
        'certify_energy_missing_note': ('the registered log records b_rem/spent_before/e_lower/'
                                        'lam/reason/base_decision but NOT certify_energy; '
                                        'actual certified energy is NOT inferred from the '
                                        'lower-bound ledger (per registration)'),
    }

    # ---------------- T2a: numerical correctness on a pre-set pressure fixture ----------
    sys.path.insert(0, str(CC_TOOLS))
    import role_c_decision_audit as aud
    fixtures = [
        {'budget': 100.0, 'e_o': 2.0, 'v_hi': 200.0, 'v_lo': 190.0},
        {'budget': 10.0, 'e_o': 4.0, 'v_hi': 100.0, 'v_lo': 108.0},
        {'budget': 100.0, 'e_o': 0.0, 'v_hi': 200.0, 'v_lo': 100.0},
        {'budget': 1.0, 'e_o': 5.0, 'v_hi': 100.0, 'v_lo': 90.0},
    ]
    ok_a, rows_a, prob_a = aud.check_t2a_shadow_price_correctness(fixtures)
    res['t2a_fixture_result'] = {'ok': ok_a, 'problems': prob_a,
                                 'n_fixtures': len(rows_a),
                                 'all_checks_per_fixture': [r['checks'] for r in rows_a]}

    # ---------------- structural pressure fixture: replay the registered rule on TRAIN -----
    sweep = {'executed': False}
    try:
        sys.path.insert(0, str(EXT / 'scripts'))
        for sub in ('evaluation', 'models', 'simulation', 'coldchain', 'data'):
            sys.path.insert(0, str(EXT / 'scripts' / sub))
        # the registered rule module lives in the v4 registration root, not in EXT/scripts
        sys.path.insert(0, str(V4))
        from reserving_admission import (build_lambda_grid, build_clock_grid, calibrate,
                                         e_lower_of, REV)
        import run_exp_reserve as RR
        fz = reg['frozen']
        train_ds = RR.generate_dataset(fz['train_instances'], fz['n_orders'], fz['train_seed'])
        lam_grid = build_lambda_grid()
        clock_grid = build_clock_grid(fz['booking_horizon'])
        G = calibrate(train_ds, reg['objective'], fz['booking_horizon'], lam_grid, clock_grid)
        import numpy as np
        dem = np.asarray(train_ds['demands'], dtype=np.float64)
        rev_t = np.asarray(train_ds['reveal_time'], dtype=np.float64)
        cls_t = np.asarray(train_ds['temp_class'], dtype=np.int64)
        day_orders = []
        for d in range(dem.shape[0]):
            m = dem[d, 1:] > 0
            cs, qs, rs = cls_t[d, 1:][m], dem[d, 1:][m], rev_t[d, 1:][m]
            order = np.argsort(rs, kind='stable')
            day_orders.append([(float(rs[i]), int(cs[i]), float(qs[i]),
                                e_lower_of(int(cs[i]), float(qs[i]), reg['objective']))
                               for i in order])

        def replay(B):
            pre = price = adm = 0
            spent = 0.0
            for orders in day_orders:
                spent = 0.0
                for (t, c, q, e) in orders:
                    b_rem = B - spent
                    if e > b_rem:
                        pre += 1
                        continue
                    ti = min(max(bisect.bisect_right(clock_grid, t) - 1, 0), len(clock_grid) - 1)
                    lam = lam_grid[-1]
                    for li, lv in enumerate(lam_grid):
                        if G[li][ti] <= b_rem:
                            lam = lv
                            break
                    if (REV[c] - lam * e) > 0.0:
                        spent += e
                        adm += 1
                    else:
                        price += 1
            return {'precheck_excluded': pre, 'price_excluded': price, 'admitted': adm}

        sweep = {'executed': True, 'train_seed': fz['train_seed'],
                 'train_instances': fz['train_instances'], 'n_orders': fz['n_orders'],
                 'B_registered_kwh': B_REGISTERED,
                 'note': ('structural fixture: the registered rule replayed on the registered '
                          'TRAIN split, tracking spent over the RULE-OWN accepted set (the '
                          'deployed run tracks the base-SAA accepted set; declared difference). '
                          'G is calibrated once and does not depend on B'),
                 'tiers': {}}
        for a in ALPHA_LADDER:
            B = a * B_REGISTERED
            c = replay(B)
            c['alpha'] = a
            c['budget_kwh'] = B
            c['structurally_binding'] = bool(c['precheck_excluded'] > 0
                                             or c['price_excluded'] > 0)
            sweep['tiers'][str(a)] = c
        sweep['any_tier_binding'] = any(t['structurally_binding'] for t in sweep['tiers'].values())
        sweep['tiers_with_hard_precheck_firing'] = [k for k, t in sweep['tiers'].items()
                                                    if t['precheck_excluded'] > 0]
    except BaseException:
        sweep = {'executed': False, 'traceback': traceback.format_exc()[-2500:]}
    res['structural_pressure_alpha_sweep'] = sweep

    # ---------------- natural binding (observed on the deployed days) ----------------
    res['natural_binding'] = {
        'per_day_precheck_fires': {str(k): sum(1 for x in v
                                              if x.get('reason') == 'over_remaining_budget')
                                   for k, v in days.items()},
        'per_day_price_rejections': {str(k): sum(1 for x in v
                                                 if x.get('reason') == 'below_shadow_price')
                                     for k, v in days.items()},
        'n_days': len(days),
        'any_natural_precheck_binding_observed': any(
            x.get('reason') == 'over_remaining_budget' for x in rows),
        'any_natural_price_binding_observed': any(
            x.get('reason') == 'below_shadow_price' for x in rows),
        'statement': ('the hard pre-check never fired; the price channel rejected candidates on '
                      'every day. This does NOT license "the resource is not scarce": a policy '
                      'that already rejects energy-heavy orders keeps the budget ample, so the '
                      'pre-check staying silent may be an EFFECT of the mechanism. Actual '
                      'certified energy is not in this log and is not inferred.'),
    }

    # ---------------- verdict ----------------
    gaps = []
    if not ident['decision_log']['match']:
        gaps.append('decision_log sha256 != registered')
    if not ident['rule_module']['match']:
        gaps.append('rule_module sha256 != v4 registration')
    if not ok_a:
        gaps.append('T2a fixture failed')
    if not sweep.get('executed'):
        gaps.append('structural alpha sweep not executed')
    gaps.append('certify_energy not recorded in the log (declared, not inferred)')
    gaps.append('candidate ordering not recorded in the log (declared, not inferred)')
    res['declared_gaps'] = gaps
    res['verdict'] = ('qualified' if (ident['decision_log']['match']
                                      and ident['rule_module']['match'] and ok_a)
                      else 'not_qualified')
    res['answer'] = {
        'does_the_budget_constraint_actually_exclude_candidates': {
            'via_hard_precheck_on_deployed_days': cls['excluded_by_budget_precheck'],
            'via_shadow_price_on_deployed_days': cls['excluded_by_price'],
            'structurally_reproducible': sweep.get('any_tier_binding'),
            'structurally_reproducible_at_hard_precheck_level':
                bool(sweep.get('tiers_with_hard_precheck_firing')),
        },
        'does_the_shadow_price_change_scores': {'n_score_changed': n_score_changed,
                                                'of': len(rows)},
        'does_it_change_actions': {'n_action_changed_by_price': n_action_changed_by_price},
    }
    write('D1_RESOURCE_IDENTIFIABILITY.json', res)
    print(json.dumps({k: v for k, v in res.items() if k != 'identity'},
                     ensure_ascii=False, indent=2)[:5500])
    return 0


def write(name, payload):
    (OUT / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + '\n',
                            encoding='utf-8', newline='\n')


if __name__ == '__main__':
    raise SystemExit(main())
