"""OR-Tools-RH-D adapter（OR5/OR6）。

定义：strict-online rolling horizon 下以 travel distance 为求解目标的
传统强基线（OR-Tools 求解器），最终由统一 C0 evaluator 重放 D/Q/E/J_CC。
非冷链感知；drop 只是 DEFER 不是 service success。

确定性预算（OR6）：固定 solution_limit + 固定 first-solution/局部搜索策略 +
单线程；wall-clock time_limit 仅作安全上限。OR-Tools 9.11.4210 无
random_seed 字段——确定性由上述固定参数保证，由 test_determinism 实测。

失败语义（OR5，protocol revision 1）：fallback=disabled。
  SOLVED / NO_SOLUTION / TIME_LIMIT / MAPPING_ERROR /
  FLOAT_CERTIFICATE_FAIL / INTEGER_OVERFLOW / EMPTY_POOL_FAST_PATH
NO_SOLUTION/TIME_LIMIT 输出安全 WAIT/CLOSE 以便完整落日志，但必须
fallback_triggered=True + solver_status 记录；正式 Gate 要求 fallback=0。

状态码（9.11.4210 未导出枚举，实测 + C++ 约定 + pyi 确认）：
  0 NOT_SOLVED / 1 SUCCESS / 2 PARTIAL / 3 FAIL / 4 FAIL_TIMEOUT /
  5 INVALID / 6 INFEASIBLE / 7 OPTIMAL（证明最优，属成功）
"""
import os
import sys

_DCC_VRP = os.path.dirname(os.path.abspath(__file__))
_COMMON = os.path.normpath(os.path.join(_DCC_VRP, '..', '..', 'common'))
for p in (_DCC_VRP, _COMMON):
    if p not in sys.path:
        sys.path.insert(0, p)
import _bootstrap  # noqa: F401

from method_adapter import ExternalReplanner, PlanProposal
from problem_builder import build_model, BuildError
from route_mapper import map_solution, MappingError
from problem_builder import empty_suffix_policy as wait_or_close
import identity as ortools_identity

STATUS_NAMES = {0: 'NOT_SOLVED', 1: 'SUCCESS', 2: 'PARTIAL_SUCCESS',
                3: 'FAIL', 4: 'FAIL_TIMEOUT', 5: 'INVALID', 6: 'INFEASIBLE',
                7: 'OPTIMAL'}
SUCCESS_STATUSES = (1, 2, 7)   # 7 = ROUTING_OPTIMAL（证明最优，属成功）


class SolveError(RuntimeError):
    pass


class ORToolsRHDAdapter(ExternalReplanner):
    """每事件构建可见子问题 → OR-Tools 求解 → 映射回 proposal。"""

    method_name = 'ortools-rh-d'
    method_revision = '1'
    adapter_revision = '1'

    @classmethod
    def compute_files(cls):
        return ('ortools_adapter.py', 'problem_builder.py', 'route_mapper.py')

    def __init__(self, solution_limit=30, time_limit_s=30.0, check_env=True):
        super().__init__()
        if check_env:
            self.env_identity = ortools_identity.check_environment()
            native = self.env_identity['native_extension_sha256']
            # checkpoint_hash：完整 64 位 native SHA-256（正式身份字段，P1-3
            # 不再截断）；checkpoint_short 另作日志简写，不与身份字段混用。
            self.checkpoint_hash = (
                'ortools=' + self.env_identity['ortools_version']
                + '_native=' + native)
            self.checkpoint_short = (
                'ortools=' + self.env_identity['ortools_version']
                + '_native=' + native[:16])
        else:
            self.checkpoint_hash = 'ortools-env-check-disabled'
            self.checkpoint_short = self.checkpoint_hash
        if int(solution_limit) <= 0:
            raise ValueError(f'solution_limit 必须为正: {solution_limit}')
        if float(time_limit_s) <= 0:
            raise ValueError(f'time_limit_s 必须为正: {time_limit_s}')
        self.solution_limit = int(solution_limit)
        self.time_limit_s = float(time_limit_s)

    def _search_params(self):
        import math
        from ortools.constraint_solver import routing_enums_pb2, pywrapcp
        sp = pywrapcp.DefaultRoutingSearchParameters()
        sp.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
        sp.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        sp.solution_limit = self.solution_limit
        # 亚秒安全上限（OR4.1）：秒取整会把 0.0001 变成 0——用毫秒，最小 1ms
        sp.time_limit.FromMilliseconds(
            max(1, int(math.ceil(self.time_limit_s * 1000))))
        sp.log_search = False
        return sp

    def _safe_suffixes(self, view):
        return {vid: wait_or_close(view, next(
            x for x in view.vehicles if x.vehicle_id == vid))
            for vid in view.replan_ids}

    def propose(self, view):
        pool = tuple(n for n, p in zip(view.node_ids, view.pool_mask) if p)

        # ---- EMPTY_POOL_FAST_PATH：不调用 OR-Tools ----
        if not pool:
            return PlanProposal(
                suffixes=self._safe_suffixes(view),
                model_input_customers=(),
                fallback_triggered=False,
                model_runtime_s=0.0,
                solve_meta={'n_pool': 0, 'empty_pool_fast_path': True,
                            'solver_status': 'EMPTY_POOL_FAST_PATH'},
            )

        def _failure(stage, status_name, code, reason, extra=None):
            """结构化失败记录：安全 suffix + fallback + 显式阶段/原因
            （不伪装成功；正式 Gate 看到任何 fallback 必须失败）。"""
            meta = {'solver_status': status_name,
                    'solver_status_code': code,
                    'failure_stage': stage,
                    'failure_reason': reason,
                    'fallback': True}
            if extra:
                meta.update(extra)
            return PlanProposal(
                suffixes=self._safe_suffixes(view),
                model_input_customers=tuple(pool),
                fallback_triggered=True,
                model_runtime_s=self.last_model_runtime_s,
                solve_meta=meta,
            )

        # ---- 层 1：模型构造（BuildError 之外的原生异常也结构化落盘）----
        try:
            problem = build_model(view)
        except BuildError as e:
            return _failure('BUILD', 'BUILD_ERROR', -1, str(e))
        except Exception as e:   # noqa: BLE001  记录异常类与 repr，不吞错误类型
            return _failure('BUILD', 'BUILD_EXCEPTION', -3,
                            f'{type(e).__name__}: {e!r}')

        # ---- 层 2：求解 ----
        try:
            with self.time_model():
                solution = problem.routing.SolveWithParameters(
                    self._search_params())
        except Exception as e:   # noqa: BLE001  求解器异常必须落盘而非中断
            return _failure('SOLVE', 'SOLVER_EXCEPTION', -2, repr(e))
        status_int = problem.routing.status()
        status_name = STATUS_NAMES.get(status_int, f'UNKNOWN_{status_int}')

        if solution is None or status_int not in SUCCESS_STATUSES:
            return _failure(
                'SOLVE',
                'TIME_LIMIT' if status_int == 4 else 'NO_SOLUTION',
                status_int, f'raw_status={status_name}',
                extra=dict(problem.scale_meta))

        # ---- 层 3：映射 / 浮点 certificate ----
        dropped = []
        for node_number in problem.customer_id_by_node_index:
            idx = problem.manager.NodeToIndex(node_number)
            if solution.Value(problem.routing.NextVar(idx)) == idx:
                dropped.append(problem.customer_id_by_node_index[node_number])
        dropped = dropped + list(getattr(problem, 'forced_dropped', []))

        try:
            proposal = map_solution(problem, view, solution,
                                    runtime_s=self.last_model_runtime_s,
                                    solver_status=status_name,
                                    dropped_customers=dropped)
        except MappingError as e:
            return _failure('MAP', e.code, status_int, str(e),
                            extra=dict(problem.scale_meta))
        except Exception as e:   # noqa: BLE001  记录异常类与 repr，不吞错误类型
            return _failure('MAP', 'MAPPING_EXCEPTION', status_int,
                            f'{type(e).__name__}: {e!r}',
                            extra=dict(problem.scale_meta))

        proposal = PlanProposal(
            suffixes=proposal.suffixes,
            model_input_customers=proposal.model_input_customers,
            fallback_triggered=False,
            model_runtime_s=proposal.model_runtime_s,
            solve_meta=dict(proposal.solve_meta,
                            solver_status_code=status_int,
                            solver_status_name=status_name,
                            partial_solution=(status_int == 2)),
        )
        return proposal
