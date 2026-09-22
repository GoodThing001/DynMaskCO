"""PyVRP-RH-D adapter（B2）：strict-online rolling horizon 距离优化基线。

定义（正式名称 PyVRP-RH-D）：
  在 strict-online rolling horizon 下以 travel distance 为求解目标的传统强基线，
  最终由统一 C0 evaluator 回放并报告 D/Q/E/J_CC。非冷链感知；
  不修改 PyVRP cost function、不注入 quality/energy 代理。

确定性协议（开发与 Gate 期）：
  固定 PyVRP seed + 固定 MaxIterations + 无 warm start；
  两次运行 decision_hash 完全一致。固定 MaxRuntime 的效率实验另立协议。

失败语义（第一阶段）：不静默 fallback。无完整解 / infeasible / 客户缺失 /
跨车重复 / 浮点 certificate 失败 → 显式抛 MappingError/SolveError（实例失败可见，
原因落盘）。原生完整服务率与 fallback 率分别报告（后续再立项
'PyVRP-RH-D + JF1-H-F fallback'）。
"""
import os
import sys

_DCC_VRP = os.path.dirname(os.path.abspath(__file__))
_COMMON = os.path.normpath(os.path.join(_DCC_VRP, '..', '..', 'common'))
for p in (_DCC_VRP, _COMMON):
    if p not in sys.path:
        sys.path.insert(0, p)

import _bootstrap  # noqa: F401  (项目脚本路径)

from method_adapter import ExternalReplanner
from problem_builder import build_model
from route_mapper import map_solution, MappingError, wait_or_close
import identity as pyvrp_identity


class SolveError(RuntimeError):
    pass


class PyVRPRHDAdapter(ExternalReplanner):
    """每事件构建可见子问题 → 求解 → 映射回 proposal。

    max_iterations / seed：开发与 Gate 期固定（确定性协议）。
    """

    method_name = 'pyvrp-rh-d'
    method_revision = '1'
    adapter_revision = '1'

    @classmethod
    def compute_files(cls):
        """计算依赖：adapter 本体 + 子问题构造 + 路线映射（均直接决定决策）。
        identity.py（环境身份检查）与 run_dcc.py（入口）归 control 集合。
        """
        return ('pyvrp_adapter.py', 'problem_builder.py', 'route_mapper.py')

    def __init__(self, max_iterations=1000, seed=0, check_env=True,
                 require_complete=False):
        super().__init__()
        if check_env:
            self.env_identity = pyvrp_identity.check_environment()
            self.checkpoint_hash = ('pyvrp=' + self.env_identity['pyvrp_version']
                                    + '_native=' + self.env_identity['_pyvrp_sha256'][:16])
        else:
            self.checkpoint_hash = 'pyvrp-env-check-disabled'
        self.max_iterations = int(max_iterations)
        self.seed = int(seed)
        self.require_complete = bool(require_complete)

    def propose(self, view):
        import warnings
        from pyvrp.stop import MaxIterations
        from method_adapter import PlanProposal

        pool = tuple(n for n, p in zip(view.node_ids, view.pool_mask) if p)

        # ---- 空 pool 快速路径：不调用 PyVRP ----
        # WAIT/CLOSE 规则见 route_mapper.wait_or_close（冻结，保守版）：
        # WAIT 仅限 depot+空载+有 future；其余 CLOSE。
        if not pool:
            suffixes = {}
            for vid in view.replan_ids:
                v = next(x for x in view.vehicles if x.vehicle_id == vid)
                suffixes[vid] = wait_or_close(view, v)
            return PlanProposal(
                suffixes=suffixes,
                model_input_customers=(),
                fallback_triggered=False,
                model_runtime_s=0.0,
                solve_meta={'n_pool': 0, 'empty_pool_fast_path': True,
                            'n_penalty_warnings': 0},
            )

        problem = build_model(view, require_complete=self.require_complete)
        with self.time_model():
            solve_params = self._solve_params(problem)
            with warnings.catch_warnings(record=True) as wlist:
                warnings.simplefilter('always')
                result = problem.model.solve(
                    stop=MaxIterations(self.max_iterations),
                    seed=self.seed,
                    display=False,
                    params=solve_params,
                )
        if not result.is_feasible():
            raise SolveError('PyVRP 返回 infeasible（存在约束违例的路线）')
        proposal = map_solution(problem, view, result.best,
                                runtime_s=self.last_model_runtime_s)
        meta = dict(problem.scale_meta)
        meta['n_penalty_warnings'] = len(wlist)
        proposal = PlanProposal(
            suffixes=proposal.suffixes,
            model_input_customers=proposal.model_input_customers,
            fallback_triggered=proposal.fallback_triggered,
            model_runtime_s=proposal.model_runtime_s,
            solve_meta=meta,
        )
        return proposal

    def _solve_params(self, problem):
        """deferral 模式：固定 feasibility penalty（min==max，界限计算）；
        严格模式：默认 penalty（required 客户，不需要 prize 对抗）。"""
        meta = problem.scale_meta
        if not meta.get('feasibility_penalty'):
            return None
        from pyvrp import SolveParams
        from pyvrp.PenaltyManager import PenaltyParams
        p = float(meta['feasibility_penalty'])
        return SolveParams(penalty=PenaltyParams(min_penalty=p, max_penalty=p))
