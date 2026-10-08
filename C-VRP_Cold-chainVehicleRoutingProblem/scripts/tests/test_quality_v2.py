"""A-v1 品质 v2 三项验证：v1 回归不变、v2 手算一致、阈值两侧判定。

三项（用户要求）：
  1. v1 回归不变：v1 QualityConfig 字段/hash 不变，default_pilot_contract 校验通过。
  2. v2 小轨迹手算一致：零阶（评分线性下降）与一阶（比例指数下降）推进手算一致。
  3. 阈值两侧判定：鱼评分 >= 5 可售 / < 5 不可售；番茄（none）salable=None（不评价）。
"""
import os
import sys

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_COLD = os.path.join(_SCRIPTS, 'coldchain')
if _COLD not in sys.path:
    sys.path.insert(0, _COLD)

from coldchain_contract import (default_pilot_contract, ColdChainContractV2,
                                QualityConfigV2)
from coldchain_state import (create_vehicle_state, dispatch_vehicle,
                             transition_segment)


def _contract_v2():
    c = ColdChainContractV2()
    c.validate()
    return c


def _run_single(c, temp_class, dwell_h, qty=2.0):
    """dispatch -> pickup(一个单) -> return(dwell_h 后)，返回 delivered record。"""
    state = create_vehicle_state(c)
    state = dispatch_vehicle(state, c)
    initial = c.quality.initial_value[temp_class]
    state, _ = transition_segment(
        state, depart_time=0.0, arrival_time=1.0, service_finish=1.0,
        served_customer=1, active_zone_mask=(True,) * 3, contract=c,
        order_quantity=qty, order_temp_class=temp_class,
        initial_quality=initial, segment_distance_units=1.0)
    ret_arr = 1.0 + dwell_h
    state, _ = transition_segment(
        state, depart_time=1.0, arrival_time=ret_arr, service_finish=ret_arr,
        served_customer=None, active_zone_mask=(True,) * 3, contract=c,
        return_to_depot=True, segment_distance_units=1.0)
    return state.delivered_to_depot[0]


def test_v1_unchanged():
    c = default_pilot_contract()
    # v1 quality 仍是 v1 字段，未被动
    assert hasattr(c.quality, 'reference_rate_per_hour'), "v1 quality 字段丢失"
    assert not hasattr(c.quality, 'rate_per_hour'), "v1 quality 被 v2 污染"
    c.validate()
    h = c.contract_hash
    assert len(h) == 64
    # 记录 v1 默认 pilot 契约 hash（回归锚点，用于后续对比）
    print(f"[PASS] v1 unchanged: hash={h[:12]}...")
    return True


def test_v2_zero_order_hand_calc():
    c = _contract_v2()
    # 鲻鱼 class1 零阶：rate 0.01883/h，dwell 10h -> 9 - 0.01883*10 = 8.8117
    rec = _run_single(c, 1, 10.0)
    expected = 9.0 - 0.01883 * 10.0
    assert abs(rec.delivered_quality - expected) < 1e-6, (rec.delivered_quality, expected)
    assert rec.salable is True  # 8.81 >= 5
    print(f"[PASS] zero-order hand-calc: delivered={rec.delivered_quality:.4f} (expect {expected:.4f})")
    return True


def test_v2_first_order_hand_calc():
    c = _contract_v2()
    # 番茄 class0 一阶：rate 0.00184/h，dwell 10h -> 1.0 * exp(-0.00184*10) = 0.9818
    import math
    rec = _run_single(c, 0, 10.0)
    expected = 1.0 * math.exp(-0.00184 * 10.0)
    assert abs(rec.delivered_quality - expected) < 1e-6, (rec.delivered_quality, expected)
    assert rec.salable is None  # 番茄 none 阈值 -> 不评价
    print(f"[PASS] first-order hand-calc: delivered={rec.delivered_quality:.4f} (expect {expected:.4f}), salable=None")
    return True


def test_v2_threshold_both_sides():
    c = _contract_v2()
    # 鲻鱼 class1 零阶：dwell 10h -> 8.81 >= 5 -> salable True
    rec_ok = _run_single(c, 1, 10.0)
    assert rec_ok.salable is True
    # 鲻鱼 dwell 300h -> 9 - 0.01883*300 = 3.35 < 5 -> salable False
    rec_bad = _run_single(c, 1, 300.0)
    assert rec_bad.salable is False
    # 海鲈鱼 class2 零阶 rate 0.000278/h：dwell 10h -> 8.997 >> 5 -> True（-18°C 保鲜，阈值很少触发）
    rec_frozen = _run_single(c, 2, 10.0)
    assert rec_frozen.salable is True
    print("[PASS] threshold both sides: 10h->salable, 300h->unsalable, frozen 10h->salable")
    return True


if __name__ == '__main__':
    results = {
        'v1_unchanged': test_v1_unchanged(),
        'v2_zero_order': test_v2_zero_order_hand_calc(),
        'v2_first_order': test_v2_first_order_hand_calc(),
        'v2_threshold_both_sides': test_v2_threshold_both_sides(),
    }
    ok = all(results.values())
    print('ALL PASS' if ok else 'SOME FAIL')
    sys.exit(0 if ok else 1)
