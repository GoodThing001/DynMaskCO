"""B1.1 测试公共助手：合成数据集 + propose 模式 DummyAdapter（纯 NumPy）。

proposal 模式下 adapter 只接收 DecisionView（未来信息结构不可达），
返回 PlanProposal（纯数据）；所有违规 adapter 在 proposal 层触发 ContractViolation。
"""
import os
import sys

_TESTS = os.path.dirname(os.path.abspath(__file__))
_COMMON = os.path.dirname(_TESTS)
for p in (_TESTS, _COMMON):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np

import _bootstrap  # noqa: F401  (common 路径 + 项目脚本路径)
from method_adapter import (ExternalReplanner, PlanProposal, ContractViolation,
                            DecisionView)
from strict_online_env import StrictOnlineEnv


def make_synthetic_dataset(n_customers=6, reveal_spec=None, seed=0,
                           capacity_tight=False, tw_end_customers=10.0):
    """构造合成 DCC 数据集（1 实例，batch 维度）。

    - 前 4 个客户 t=0 可见，其余按 reveal_spec（缺省 = 全部 0）；
    - depot 位于 (0.5, 0.5)，客户坐标 [0,1]^2；
    - capacity_tight=True 时需求加大（迫使多车）。
    """
    rng = np.random.RandomState(seed)
    n = n_customers + 1
    coords = rng.uniform(0.1, 0.9, size=(n, 2))
    coords[0] = [0.5, 0.5]
    demands = rng.randint(8, 16, size=n).astype(np.float32)
    if capacity_tight:
        demands[1:] = rng.randint(18, 28, size=n_customers).astype(np.float32)
    demands[0] = 0.0
    tw_start = np.zeros(n, dtype=np.float32)
    tw_end = np.full(n, tw_end_customers, dtype=np.float32)
    tw_end[0] = tw_end_customers + 2.0          # depot return horizon
    service_time = np.full(n, 0.1, dtype=np.float32)
    service_time[0] = 0.0
    temp_class = np.zeros(n, dtype=np.int32)
    for i in range(1, n):
        temp_class[i] = (i % 3)
    initial_quality = np.ones(n, dtype=np.float32)
    reveal = np.zeros(n, dtype=np.float32)
    if reveal_spec:
        for c, t in reveal_spec.items():
            reveal[c] = t
    return {
        'coords': coords[None, ...], 'demands': demands[None, ...],
        'tw_start': tw_start[None, ...], 'tw_end': tw_end[None, ...],
        'service_time': service_time[None, ...], 'temp_class': temp_class[None, ...],
        'initial_quality': initial_quality[None, ...], 'reveal_time': reveal[None, ...],
    }


def make_early_reveal_dataset():
    """3a 专用：客户 1 离 depot 很远（腿长 > 1.0），客户 5/6 在 t=0.2 提前 reveal。

    保证在 reveal 决策点存在「committed（行程中）」车辆。
    """
    ds = make_synthetic_dataset(reveal_spec={5: 0.2, 6: 0.2}, seed=0)
    ds['coords'][0, 1] = [0.95, 0.95]
    ds['coords'][0, 0] = [0.05, 0.05]     # depot 挪到角落 → 腿长 ≈1.27
    for i in (2, 3, 4):
        ds['coords'][0, i] = [0.5, 0.9]
    return ds


def build_env(dataset, capacity, num_vehicles, adapter, contract=None):
    return StrictOnlineEnv(dataset, capacity, 1.0, num_vehicles,
                           replanner=adapter, coldchain_contract=contract)


def pool_ids(view: DecisionView):
    return [n for n, m in zip(view.node_ids, view.pool_mask) if m]


def prefix_signature(rec, until_clock):
    """扰动客户的 reveal 时刻之前的完整决策序列签名（B1.2 全 prefix parity）。

    覆盖：clock、事件类型（revealed ids）、plan hash 前后、每事件 execution
    层动作（type/vehicle/customer）。event_id 也纳入（事件序列一致才有意义）。
    """
    sig = []
    for ev in rec['events']:
        if ev['clock'] >= until_clock - 1e-9:
            break
        acts = sorted(
            (a['action_type'], a['vehicle_id'], a['customer_id'])
            for a in rec['actions']
            if a['event_id'] == ev['event_id'] and a['action_layer'] == 'execution')
        sig.append({
            'event_id': ev['event_id'],
            'clock': ev['clock'],
            'revealed': list(ev['revealed_customer_ids']),
            'plan_before': ev['plan_hash_before'],
            'plan_after': ev['plan_hash_after'],
            'execution_actions': acts,
        })
    return sig


class GreedyEDDAdapter(ExternalReplanner):
    """EDD 贪心外部 adapter（propose 模式）。

    逻辑与项目 GreedyReplanner 同构：reserved = committed_next ∪ 全部车辆尾部，
    unserved = 可见未服务 − reserved；对每辆 replan 车贪心建 suffix。
    完全确定（无随机数）；支持 WAIT。
    """

    method_name = 'dummy-greedy-edd'
    method_revision = '2'
    adapter_revision = '2'

    def __init__(self):
        super().__init__()
        self.checkpoint_hash = 'none'

    def propose(self, view):
        # view.pool 已排除非 replan 车辆的 committed_next 与完整 suffix
        # （B1.1 P0 ownership 语义）；此处只按 pool 贪心分配。
        unserved = list(pool_ids(view))
        allocated = set()
        suffixes = {}
        for v in view.vehicles:
            if v.vehicle_id not in view.replan_ids:
                continue
            route = self._greedy_route(view, v, [c for c in unserved
                                                 if c not in allocated])
            if route == [0] and v.anchor_node_id != 0 and view.has_future_reveal:
                route = []
            suffixes[v.vehicle_id] = tuple(route)
            for o in route:
                if o != 0:
                    allocated.add(int(o))
        with self.time_model():
            pass  # 贪心无模型耗时（耗时=构造，记为 0）
        return PlanProposal(
            suffixes=suffixes,
            model_input_customers=tuple(sorted(unserved)),
            fallback_triggered=False,
            model_runtime_s=0.0,
        )

    @staticmethod
    def _greedy_route(view, v, remaining):
        idx = {n: i for i, n in enumerate(view.node_ids)}
        route = []
        current = v.anchor_node_id
        cur_time = v.ready_time
        load = v.load
        unvisited = set(int(i) for i in remaining)
        while unvisited:
            best, best_score = None, float('inf')
            for j in unvisited:
                i_cur, i_j = idx[current], idx[j]
                d = view.dist_mat[i_cur][i_j]
                arr = cur_time + d / view.tw_speed
                arr = max(arr, view.tw_start[i_j])
                finish = arr + view.service_time[i_j]
                ret = finish + view.dist_mat[i_j][idx[0]] / view.tw_speed
                if (arr <= view.tw_end[i_j] + 1e-6
                        and ret <= view.depot_tw_end + 1e-6
                        and load + view.demands[i_j] <= view.capacity):
                    score = view.tw_end[i_j] + 0.01 * d
                    if score < best_score:
                        best_score, best = score, j
            if best is None:
                break
            route.append(best)
            unvisited.remove(best)
            i_best = idx[best]
            load += view.demands[i_best]
            cur_time = max(cur_time + view.dist_mat[idx[current]][i_best]
                           / view.tw_speed,
                           view.tw_start[i_best]) + view.service_time[i_best]
            current = best
        route.append(0)
        return route


# ---------------------------------------------------------------------------
# 违规 proposal adapter（B1.1 破坏测试）
# ---------------------------------------------------------------------------

class DuplicateCustomerAdapter(GreedyEDDAdapter):
    """proposal：同一客户出现在两辆车 suffix（跨车重复，必须被拒绝）。"""

    def propose(self, view):
        p = super().propose(view)
        replan = sorted(view.replan_ids)
        if len(replan) >= 2:
            pool = pool_ids(view)
            if pool:
                c = pool[0]
                s = dict(p.suffixes)
                s[replan[0]] = tuple(x for x in s[replan[0]] if x != 0)
                s[replan[1]] = tuple(x for x in s[replan[1]] if x != 0)
                s[replan[0]] = tuple([c] + list(s[replan[0]]))
                s[replan[1]] = tuple([c] + list(s[replan[1]]))
                return PlanProposal(s, p.model_input_customers, False,
                                    p.model_runtime_s)
        return p


class IntraRouteDuplicateAdapter(GreedyEDDAdapter):
    """proposal：同车 suffix 内重复客户（必须被拒绝）。"""

    def propose(self, view):
        p = super().propose(view)
        for vid in view.replan_ids:
            s = p.suffixes[vid]
            if len(s) >= 2 and s[0] != 0:
                s2 = (s[0], s[0]) + tuple(x for x in s[1:] if x != s[0])
                return PlanProposal({**p.suffixes, vid: s2},
                                    p.model_input_customers, False,
                                    p.model_runtime_s)
        return p


class NonReplanKeyAdapter(GreedyEDDAdapter):
    """proposal：包含非 replan 车辆键（必须被拒绝）。"""

    def propose(self, view):
        p = super().propose(view)
        non_replan = [v.vehicle_id for v in view.vehicles
                      if v.vehicle_id not in view.replan_ids]
        if non_replan:
            s = dict(p.suffixes)
            s[non_replan[0]] = (0,)
            return PlanProposal(s, p.model_input_customers, False,
                                p.model_runtime_s)
        return p


class MissingKeyAdapter(GreedyEDDAdapter):
    """proposal：缺少一个 replan 车辆键（必须被拒绝）。"""

    def propose(self, view):
        p = super().propose(view)
        if len(view.replan_ids) > 1:
            s = dict(p.suffixes)
            del s[sorted(view.replan_ids)[0]]
            return PlanProposal(s, p.model_input_customers, False,
                                p.model_runtime_s)
        return p


class FutureCustomerAdapter(GreedyEDDAdapter):
    """proposal：suffix 含未来客户（结构上不存在于 view；硬编码 id，必须被拒绝）。"""

    def __init__(self, future_customer):
        super().__init__()
        self.future_customer = int(future_customer)

    def propose(self, view):
        p = super().propose(view)
        for vid in view.replan_ids:
            return PlanProposal({**p.suffixes, vid: (self.future_customer, 0)},
                                p.model_input_customers, False,
                                p.model_runtime_s)
        return p



class InvalidCustomerAdapter(GreedyEDDAdapter):
    """proposal：suffix 含非法客户编号（必须被拒绝）。"""

    def propose(self, view):
        p = super().propose(view)
        for vid in view.replan_ids:
            return PlanProposal({**p.suffixes, vid: (99999, 0)},
                                p.model_input_customers, False,
                                p.model_runtime_s)
        return p


class NoisyGreedyAdapter(GreedyEDDAdapter):
    """依赖 np.random 的 adapter：打乱车辆贪心顺序（合法改变决策，
    验证 decision_hash 对决策内容敏感）。"""

    def propose(self, view):
        unserved = list(pool_ids(view))
        allocated = set()
        suffixes = {}
        replan_vehicles = [v for v in view.vehicles if v.vehicle_id in view.replan_ids]
        order = list(range(len(replan_vehicles)))
        np.random.shuffle(order)
        for i in order:
            v = replan_vehicles[i]
            route = self._greedy_route(view, v, [c for c in unserved
                                                 if c not in allocated])
            if route == [0] and v.anchor_node_id != 0 and view.has_future_reveal:
                route = []
            suffixes[v.vehicle_id] = tuple(route)
            for o in route:
                if o != 0:
                    allocated.add(int(o))
        return PlanProposal(
            suffixes=suffixes,
            model_input_customers=tuple(sorted(unserved)),
            fallback_triggered=False,
            model_runtime_s=0.0,
        )