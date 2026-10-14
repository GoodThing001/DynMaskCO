"""核对：新基线（a1_c1fresh_reveal_40 的口径）算出的 B / cooling_share
是否与现有 a1_c1rr2_reveal_40 完全一致。

若一致 → 追加臂（CO-enriched / MAPT）可以**现在就**用 a1_c1rr2_reveal_40 作
--baseline-from 跑（B/cooling_share 相同 ⇒ 决策完全相同），等 F1 新基线落盘后
再用它重配对（共享源码哈希与新基线相同，重配对合法），从而省掉 ~8h 串行等待。
若不一致 → 必须等新基线。

用法（在 C-VRP_Cold-chainVehicleRoutingProblem 下）：
    python ../CC_Compare/tools/check_budget_equiv.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.abspath(os.path.join(HERE, '..', '..',
                                   'C-VRP_Cold-chainVehicleRoutingProblem'))
for p in (os.path.join(EXT, 'scripts'),
          os.path.join(EXT, 'scripts', 'coldchain'),
          os.path.join(EXT, 'scripts', 'simulation'),
          os.path.join(EXT, 'scripts', 'evaluation')):
    if p not in sys.path:
        sys.path.insert(0, p)

RECORDED = json.load(open(os.path.join(
    EXT, 'results', 'a1_c1rr2_reveal_40', 'gate.json'), encoding='utf-8'))['budget']


def main():
    from run_exp_reserve import generate_dataset
    from run_exp_encoder_v3 import compute_budget_and_dwell
    from scenario_saa import (make_c0_contract_v2, build_history_times_classes)
    contract = make_c0_contract_v2()
    train_ds = generate_dataset(200, 200, 20260925)
    hist_tc = build_history_times_classes(train_ds)
    B, cooling_share, e_mean = compute_budget_and_dwell(
        train_ds, contract, 15, 20260925, 0.60, hist_tc, k=10)
    print('now  B =', repr(B), 'cooling_share =', repr(cooling_share),
          'train_energy_mean =', repr(e_mean))
    print('ref  B =', repr(RECORDED['B']),
          'cooling_share =', repr(RECORDED['cooling_share']))
    print('B  equal:', abs(B - RECORDED['B']) < 1e-9)
    print('cs equal:', abs(cooling_share - RECORDED['cooling_share']) < 1e-12)


if __name__ == '__main__':
    main()
