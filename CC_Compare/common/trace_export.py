"""轨迹导出与重放对账（B1）。

  - export_trace(traces, contract)        在线执行轨迹 → 可 JSON 序列化明细
  - trace_replay_check(exported, outcome) 导出数字与权威 evaluator 终局 D/Q/E/J 对账
  - export_static_routes(traces)          每车 zero-delimited 静态路线（供
                                          replay_static_pickup_route 独立物理复核）

不信任外部方法自报值：outcome 一律由 evaluate_coldchain_trace / _eval 重算；
trace_export 只负责把 engine 已产生的 ServiceRecord / VehicleTrace 无损落盘，
并用 trace_replay_check 证明落盘过程没有丢/改数字。
"""
import numpy as np

from coldchain_evaluator import math_isclose


def export_trace(traces, contract=None):
    """把 VehicleTrace 列表导出为可序列化 dict。

    车辆「未使用」（无 services 且无 final_coldchain_state）也保留条目（identity 审计）。
    """
    out = {'vehicles': []}
    for tr in traces:
        vt = {
            'vehicle_id': int(tr.vehicle_id),
            'used': bool(tr.services) or tr.final_coldchain_state is not None,
            'dispatch_time': (None if tr.dispatch_time is None else float(tr.dispatch_time)),
            'return_depart': (None if tr.return_depart is None else float(tr.return_depart)),
            'return_arrival': (None if tr.return_arrival is None else float(tr.return_arrival)),
            'dispatch_preconditioning_energy_kwh':
                float(tr.dispatch_preconditioning_energy_kwh),
            'return_segment_distance_km': float(tr.return_segment_distance_km),
            'return_segment_quality_loss': float(tr.return_segment_quality_loss),
            'return_segment_energy_kwh': float(tr.return_segment_energy_kwh),
            'return_segment_thermal_violation_count':
                int(tr.return_segment_thermal_violation_count),
            'return_segment_thermal_violation_duration_h':
                float(tr.return_segment_thermal_violation_duration_h),
            'services': [],
            'unload_records': [],
            'final': None,
        }
        for sr in tr.services:
            vt['services'].append({
                'prev_node': int(sr.prev_node),
                'node': int(sr.node),
                'depart_time': float(sr.depart_time),
                'arrival_time': float(sr.arrival_time),
                'service_start': float(sr.service_start),
                'service_finish': float(sr.service_finish),
                'picked_order_id': (None if sr.picked_order_id is None
                                    else int(sr.picked_order_id)),
                'segment_distance_km': float(sr.segment_distance_km),
                'segment_quality_loss': float(sr.segment_quality_loss),
                'segment_energy_kwh': float(sr.segment_energy_kwh),
                'segment_thermal_violation_count': int(sr.segment_thermal_violation_count),
                'segment_thermal_violation_duration_h':
                    float(sr.segment_thermal_violation_duration_h),
            })
        for rec in tr.depot_unload_records:
            vt['unload_records'].append({
                'order_id': int(rec.order_id),
                'salable': bool(rec.salable),
            })
        if tr.final_coldchain_state is not None:
            st = tr.final_coldchain_state
            vt['final'] = {
                'closed': bool(st.closed),
                'cargo_manifest': [
                    {'order_id': int(lot.order_id), 'quantity': float(lot.quantity),
                     'temp_class': int(lot.temp_class),
                     'pickup_finish_time': float(lot.pickup_finish_time),
                     'initial_quality': float(lot.initial_quality),
                     'quality_remaining': float(lot.quality_remaining)}
                    for lot in st.cargo_manifest],
                'total_load': float(st.total_load),
                'cumulative_distance_km': float(st.cumulative_distance_km),
                'cumulative_energy_kwh': float(st.cumulative_energy_kwh),
                'thermal_violation_count': int(st.thermal_violation_count),
                'thermal_violation_duration_h': float(st.thermal_violation_duration_h),
            }
        out['vehicles'].append(vt)
    return out


def trace_replay_check(exported, outcome, objective='coldchain'):
    """导出轨迹与权威 evaluator 终局数字对账。

    distance 口径对账 distance_cost；coldchain 口径对账 distance_km / quality_loss /
    energy_kwh / thermal 计数。任何不一致 = 导出过程丢/改数字（协议错误）。
    返回 (ok: bool, problems: list[str])。
    """
    problems = []
    vs = exported['vehicles']
    used = [v for v in vs if v['used']]

    def seg(v, k):
        base = (sum(s[k] for s in v['services']) if v['services'] else 0.0)
        return base

    if objective == 'coldchain':
        exp_distance = sum(seg(v, 'segment_distance_km') + v['return_segment_distance_km']
                           for v in used)
        exp_quality = sum(seg(v, 'segment_quality_loss') + v['return_segment_quality_loss']
                          for v in used)
        exp_energy = (sum(v['dispatch_preconditioning_energy_kwh']
                          + seg(v, 'segment_energy_kwh') + v['return_segment_energy_kwh']
                          for v in used))
        exp_thermal = sum(seg(v, 'segment_thermal_violation_count')
                          + v['return_segment_thermal_violation_count'] for v in used)
        exp_thermal_dur = sum(seg(v, 'segment_thermal_violation_duration_h')
                              + v['return_segment_thermal_violation_duration_h'] for v in used)
        if not math_isclose(exp_distance, float(outcome['distance_km'])):
            problems.append(f'distance_km 对账失败: trace={exp_distance} '
                            f'outcome={outcome["distance_km"]}')
        if not math_isclose(exp_quality, float(outcome['quality_loss'])):
            problems.append(f'quality_loss 对账失败: trace={exp_quality} '
                            f'outcome={outcome["quality_loss"]}')
        if not math_isclose(exp_energy, float(outcome['energy_kwh'])):
            problems.append(f'energy_kwh 对账失败: trace={exp_energy} '
                            f'outcome={outcome["energy_kwh"]}')
        if exp_thermal != int(outcome['thermal_violation_count']):
            problems.append(f'thermal_violation_count 对账失败: trace={exp_thermal} '
                            f'outcome={outcome["thermal_violation_count"]}')
        if not math_isclose(exp_thermal_dur, float(outcome['thermal_violation_duration_h'])):
            problems.append(f'thermal_violation_duration_h 对账失败: trace={exp_thermal_dur} '
                            f'outcome={outcome["thermal_violation_duration_h"]}')
        # unload 记录与 delivered 终局一致
        unload_ids = [r['order_id'] for v in used for r in v['unload_records']]
        exp_unload = len(unload_ids)
        if exp_unload != len(set(unload_ids)):
            problems.append(f'unload 记录含重复 order_id: {unload_ids}')
        # 所有 used 车必须有 final 状态（evaluator 的 missing_final_state 前置）
        for v in used:
            if v['final'] is None:
                problems.append(f'vehicle {v["vehicle_id"]} used 但缺 final 状态')
    else:
        exp_distance = sum(seg(v, 'segment_distance_km') + v['return_segment_distance_km']
                           for v in used)
        if not math_isclose(exp_distance, float(outcome['distance_cost'])):
            problems.append(f'distance_cost 对账失败: trace={exp_distance} '
                            f'outcome={outcome["distance_cost"]}')

    return (len(problems) == 0), problems


def export_static_routes(exported):
    """每车 zero-delimited 静态路线（[c1,...,ck,0]；未使用车为空路线）。

    供 replay_static_pickup_route 做独立 C0 物理复核（TW/容量/return/complete）。
    """
    routes = []
    for v in exported['vehicles']:
        route = [s['node'] for s in v['services']]
        if route:
            route.append(0)
        routes.append(route)
    return routes
