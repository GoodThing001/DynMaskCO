"""JF1-H-F → common contract 接入（项目内部原生 baseline，主表）。

JF1-H-F（scripts/simulation/jf1h_repair.py）是项目自身的 hard-feasible baseline：
完整计划层 repair + 显式延期 + slack_vehicles=1 + urgent-defer。这里通过
NativeReplanner 包装接入 common runner——不走 proposal 模式（原生 plan 直接
接收 env/vehicles），但写保护、fleet plans 捕获、公共 ownership audit、
事件/动作记录、C0 重放与校验全部由公共合同层执行。

repair 层审计（repair_ownership_violations / repair_terminal_unresolved）由
项目 _eval 经 bridge 属性透传附加，repair_applicable=True。
"""
import os
import sys

_DCC_VRP = os.path.dirname(os.path.abspath(__file__))
_COMMON = os.path.normpath(os.path.join(_DCC_VRP, '..', '..', 'common'))
for p in (_DCC_VRP, _COMMON):
    if p not in sys.path:
        sys.path.insert(0, p)
import _bootstrap  # noqa: F401  (项目脚本路径)

from method_adapter import NativeReplanner


class Jf1hfAdapter(NativeReplanner):
    """JF1-H-F 的 common-contract 包装。"""

    method_name = 'jf1-h-f'
    method_revision = '1'
    adapter_revision = '1'

    @classmethod
    def compute_files(cls):
        """计算依赖：项目 JF1-H-F 实现链（jf1h_repair → joint_fleet）。"""
        return ('project/simulation/jf1h_repair.py',
                'project/simulation/joint_fleet.py')

    def __init__(self, slack_vehicles=1):
        from jf1h_repair import make_continuation
        self.continuation = make_continuation(slack_vehicles=slack_vehicles)
        self.checkpoint_hash = 'none'

    def plan(self, env, inst_idx, clock, vehicles, served_mask, visible_ids,
             replan_ids=None):
        self.continuation.plan(env, inst_idx, clock, vehicles, served_mask,
                               visible_ids, replan_ids=replan_ids)

    def export_state(self):
        return self.continuation.export_state()

    def restore_state(self, state):
        self.continuation.restore_state(state)

    def sync_deferred_from_vehicles(self, vehicles):
        self.continuation.sync_deferred_from_vehicles(vehicles)

    # ---- repair 审计属性透传（_eval 经 bridge 读取）----
    @property
    def repair_stats(self):
        return self.continuation.repair_stats

    @property
    def deferred_customers(self):
        return self.continuation.deferred_customers
