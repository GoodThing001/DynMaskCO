"""RRNCO-Ordering-RH-D 公共适配器（R0.5 Stage A.1）。

组合 subproblem / preference / pickup_certificate / coordinator 成公共
`ExternalReplanner.propose(view) -> PlanProposal`：
  1. 每辆 replan 车生成**自包含** visible-only 子问题 + canonical hash；
  2. **每车只调用一次** preference provider，得到 orderings（计入 model_runtime_s）；
  3. 把 orderings 传给确定性协调器（跨车竞争 + open-new-vehicle 优先 + 认证）；
  4. 独立重放认证全部 suffix（防御）；
  5. 返回完整 PlanProposal。

关键约束：
  - provider 必须显式注入（禁止默认 EDD 冒充 RRNCO）；
  - provider 异常 → 带原因 EDD fallback；
  - solve_meta 记录的是**与决策同源**的 ordering（不是二次调用）。

不含 Torch / RL4CO / TensorDict（真实模型由 Stage B 的 rrnco_backend 注入）。
"""
import time

from method_adapter import ExternalReplanner, PlanProposal
from coordinator import coordinate, _coord_edd
from pickup_certificate import certify_suffix
from preference import validate_ordering
from subproblem import build_subproblem


class RRNCOGuidedAdapter(ExternalReplanner):
    method_name = 'rrnco-ordering-rh-d'
    method_revision = '0'
    adapter_revision = '0'

    def __init__(self, preference_provider):
        super().__init__()
        if preference_provider is None:
            raise ValueError('RRNCO preference provider 必须显式注入（禁止默认 EDD '
                             '冒充 RRNCO）')
        self.preference_provider = preference_provider

    @classmethod
    def compute_files(cls):
        return ('subproblem.py', 'preference.py', 'pickup_certificate.py',
                'coordinator.py', 'rrnco_guided_adapter.py')

    def propose(self, view):
        replan_ids = set(int(x) for x in view.replan_ids)
        replan = sorted((v for v in view.vehicles
                         if int(v.vehicle_id) in replan_ids),
                        key=lambda v: int(v.vehicle_id))
        pool = set(int(c) for c in view.pool_customer_ids)

        # ---- 1. 每车一次 provider（自包含子问题 + canonical hash）----
        sub_hashes = {}
        orderings = {}
        provider_error = None
        t0 = time.perf_counter()
        for v in replan:
            vid = int(v.vehicle_id)
            sp = build_subproblem(view, v)
            sub_hashes[vid] = sp.canonical_hash()
            try:
                ordering = tuple(int(c) for c in self.preference_provider.order(sp))
            except Exception as exc:  # noqa: BLE001
                provider_error = f'{type(exc).__name__}: {exc}'
                break
            orderings[vid] = ordering
        self.last_model_runtime_s = time.perf_counter() - t0

        # ---- 2. 协调（provider 异常 → 带原因 EDD fallback）----
        if provider_error is not None:
            result = _coord_edd(view, replan, pool,
                                reason=f'provider 异常: {provider_error}')
        else:
            result = coordinate(view, orderings)

        # ---- 3. 独立重放认证全部 suffix（防御：失败即 bug → EDD 兜底）----
        vehicle_views = {int(v.vehicle_id): v for v in view.vehicles}
        for vid, suffix in result.suffixes.items():
            ok, reason = certify_suffix(view, vehicle_views[vid], suffix)
            if not ok:
                result = _coord_edd(view, replan, pool,
                                    reason=f'certificate defense: {vid}: {reason}')
                break

        # ---- 4. PlanProposal ----
        model_input = tuple(sorted(pool))
        solve_meta = {
            'subproblem_hashes': {str(k): v for k, v in sub_hashes.items()},
            'model_orderings': {str(k): [int(c) for c in v]
                                for k, v in orderings.items()},
            'decisions': result.decisions,
            'deferred': [int(c) for c in result.deferred],
            'assigned': [int(c) for c in result.assigned],
            'fallback_reason': result.fallback_reason,
        }
        return PlanProposal(
            suffixes=result.suffixes,
            model_input_customers=model_input,
            fallback_triggered=(result.fallback_reason is not None),
            model_runtime_s=float(self.last_model_runtime_s),
            solve_meta=solve_meta,
        )
