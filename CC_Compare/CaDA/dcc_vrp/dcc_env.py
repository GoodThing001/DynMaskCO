"""CaDA DCCEnv —— 在 MTVRPEnv 上叠加 non-anticipatory（可见性）约束。

不修改 CaDA 原始 `envs/env.py`。继承 `MTVRPEnv` 只做两件事:
  1. `_reset`: 保留 `visible_mask`，并把未来节点（visible_mask==0）预标记为已访问，
     使解码器永远不选它们，且 `done` 在「所有可见客户访问完」时触发（而非所有节点）。
  2. `select_start_nodes`: 只从**可见客户**出发（非预知 multi-start）。

基础 action mask（TW + 容量 + 距离限制）由父类保证可行性；本类只在上叠加可见性。
"""
import torch
from envs.env import MTVRPEnv


class DCCEnv(MTVRPEnv):
    def _reset(self, td=None, batch_size=None):
        td_reset = super()._reset(td, batch_size)
        # 保留可见性掩码（父类 _reset 只拷贝标准字段，会丢 visible_mask）
        td_reset["visible_mask"] = td["visible_mask"]
        # 未来节点预标记为已访问 —— 既不参与解码，也保证 done 收敛
        future = ~td["visible_mask"].bool()          # (B, N+1)
        td_reset["visited"] = td_reset["visited"] | future
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset

    @staticmethod
    def select_start_nodes(td):
        """只从可见客户出发（batch_size 必须为 1，DCC 实例可见性异构）。"""
        assert td.batch_size[0] == 1, "DCCEnv.select_start_nodes requires batch_size==1"
        visible = td["visible_mask"].bool()[0]       # (N+1,)
        start_nodes = torch.nonzero(visible[1:]).squeeze(-1) + 1  # 可见客户（1 起编号）
        num_starts = start_nodes.numel()
        return num_starts, start_nodes
