"""阶段 C 生产函数定向测试：select_improving / outcome_protocol_error / service_ok。

直接覆盖生产函数（非测试自写逻辑），验证：
  1. select_improving 顺序无关（两个改善候选成本差 < τ，倒序后选择相同）；
  2. select_improving τ 阈值（< τ 不离开 KEEP，≥ τ 取最小 cost）；
  3. outcome_protocol_error 对 NaN/Inf/缺字段报错；
  4. service_ok 对 distance/coldchain 及缺字段语义正确。

用法：python scripts/tests/test_oracle_selection.py
产物：results/o0cc/oracle_selection_tests.json
"""
import sys, os, json

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CVRPTW = os.path.dirname(_BASE)
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'evaluation'))

from hard_gate import select_improving, improvement_tau, outcome_protocol_error, service_ok

RESULTS = []


def record(name, ok, detail=''):
    RESULTS.append({'test': name, 'status': 'PASS' if ok else 'FAIL', 'detail': str(detail)})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def test_select_improving_order_invariant():
    # 两个改善候选成本差 1e-12 < τ，且都明显优于 keep；倒序应选 (cost, action_id) 最小者。
    keep = 10.0
    tau = improvement_tau(keep)  # ~1e-8
    pairs = [(8.0, 'a'), (8.0 + 1e-12, 'b')]
    fwd = select_improving(pairs, keep, tau)
    rev = select_improving(list(reversed(pairs)), keep, tau)
    ok = (fwd == rev == (8.0, 'a'))
    record('select_improving_order_invariant', ok, f"fwd={fwd} rev={rev}")
    return ok


def test_select_improving_tau_threshold():
    keep = 10.0
    tau = 1e-6
    # 低于 τ 的「改善」不离开 KEEP
    below = select_improving([(10.0 - 0.5e-6, 'x')], keep, tau)
    # 高于 τ 取最小 cost
    above = select_improving([(9.0, 'x'), (8.5, 'y')], keep, tau)
    ok = (below is None and above == (8.5, 'y'))
    record('select_improving_tau_threshold', ok, f"below={below} above={above}")
    return ok


def test_outcome_protocol_error():
    base = {'complete': True, 'n_unserved': 0, 'n_duplicate': 0, 'tw_feasible': True,
            'capacity_feasible': True, 'depot_return_feasible': True, 'distance_cost': 1.0}
    good = dict(base, distance_km=1.0, quality_loss=0.0, energy_kwh=0.0, coldchain_cost=1.0,
                temperature_hard_feasible=True, all_orders_picked=True,
                all_cargo_delivered_to_depot=True, terminal_manifests_empty=True,
                trace_accounting_consistent=True, distance_accounting_consistent=True)
    ok_good = (outcome_protocol_error(good, 'coldchain') is None)
    ok_nan = ('non-finite' in (outcome_protocol_error(
        dict(good, distance_cost=float('nan')), 'coldchain') or ''))
    ok_inf = ('non-finite' in (outcome_protocol_error(
        dict(good, coldchain_cost=float('-inf')), 'coldchain') or ''))
    ok_missing = ('missing' in (outcome_protocol_error(
        {k: good[k] for k in good if k != 'n_unserved'}, 'coldchain') or ''))
    # accounting=false → 协议错误（不是 service fail）
    ok_acct = ('accounting' in (outcome_protocol_error(
        dict(good, trace_accounting_consistent=False), 'coldchain') or ''))
    # 小数计数 → 协议错误（不截断）
    ok_frac = ('non-integer' in (outcome_protocol_error(
        dict(good, n_unserved=0.5), 'coldchain') or ''))
    # ownership 破坏 → 协议错误（轨迹 outcome 正常但候选 repair 审计异常）
    ok_own = ('ownership' in (outcome_protocol_error(
        dict(good, ownership_violations=1), 'coldchain') or ''))
    # 小数 repair 字段 → 协议错误（不 int() 截断）
    ok_repair_frac = ('invalid repair' in (outcome_protocol_error(
        dict(good, ownership_violations=0, terminal_unresolved=0.5), 'coldchain') or ''))
    # strict_repair：缺 repair 字段 → 协议错误（正式 JF1-H-F 路径）
    ok_strict_missing = ('missing required repair' in (outcome_protocol_error(
        good, 'coldchain', strict_repair=True) or ''))
    # distance objective 不需 coldchain 字段
    ok_dist = (outcome_protocol_error(base, 'distance') is None)
    ok = (ok_good and ok_nan and ok_inf and ok_missing and ok_acct and ok_frac and ok_own
          and ok_repair_frac and ok_strict_missing and ok_dist)
    record('outcome_protocol_error', ok,
           f"good={ok_good} nan={ok_nan} inf={ok_inf} missing={ok_missing} "
           f"acct={ok_acct} frac={ok_frac} own={ok_own} repair_frac={ok_repair_frac} "
           f"strict_missing={ok_strict_missing} dist={ok_dist}")
    return ok


def test_service_ok_objective():
    base = {'complete': True, 'n_unserved': 0, 'n_duplicate': 0, 'tw_feasible': True,
            'capacity_feasible': True, 'depot_return_feasible': True}
    ok_dist = service_ok(base, 'distance')
    ok_cc_missing = (not service_ok(base, 'coldchain'))
    cc = dict(base, temperature_hard_feasible=True, all_orders_picked=True,
              all_cargo_delivered_to_depot=True, terminal_manifests_empty=True,
              trace_accounting_consistent=True, distance_accounting_consistent=True)
    ok_cc_full = service_ok(cc, 'coldchain')
    # terminal unresolved > 0 → service fail（候选拒绝）
    ok_term = (not service_ok(dict(cc, terminal_unresolved=1), 'coldchain'))
    ok = ok_dist and ok_cc_missing and ok_cc_full and ok_term
    record('service_ok_objective', ok,
           f"dist={ok_dist} cc_missing={ok_cc_missing} cc_full={ok_cc_full} term_reject={ok_term}")
    return ok


def main():
    ok1 = test_select_improving_order_invariant()
    ok2 = test_select_improving_tau_threshold()
    ok3 = test_outcome_protocol_error()
    ok4 = test_service_ok_objective()
    out_dir = os.path.join(_CVRPTW, 'results', 'o0cc')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'oracle_selection_tests.json'), 'w') as f:
        json.dump({'all_pass': ok1 and ok2 and ok3 and ok4, 'results': RESULTS}, f, indent=2)
    print(f"\n  ALL: {'PASS' if (ok1 and ok2 and ok3 and ok4) else 'FAIL'}")
    return 0 if (ok1 and ok2 and ok3 and ok4) else 1


if __name__ == '__main__':
    sys.exit(main())
