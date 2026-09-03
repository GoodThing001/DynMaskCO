"""RRNCO DCC 环境 —— 在 RMTVRPEnv（rcvrptw）上叠加 non-anticipatory（可见性）约束。

不修改 RRNCO 原始 `rrnco/envs/rmtvrp/`。做两件事:
  1. `DCCRMTVRPEnv`: 实现「causal encoder + full-service decoder」（等价 DynMaskCO 的
     causal decode）。encoder/decoder 看到的 `distance_matrix`/`duration_matrix`/
     `time_windows`/`service_time`/`demand_linehaul` 是 **masked 版**（未来节点被屏蔽，
     由 `dcc_data.py` 生成），而 env 的可行性判定（TW+容量+回程）与距离奖励必须用
     **真值**（`_true_*` 字段）。本类在计算 action mask / step / reward 时临时把标准字段
     换成真值，算完换回，从而 **serve 全部 50 客户**（不早停）。
  2. `VisibleStartNodes`: multi-start 解码只从**可见客户**出发（非预知）。
     配合 run_dcc.py 里 `num_starts = 可见客户数`，保证 batchify/reshape 一致。

基础 action mask（TW + 容量 + 距离限制 + 回程 depot）由父类 `get_action_mask` 保证；
本类只负责「环境计算用真值、模型读取用 masked」的字段分离。
"""
import torch

from rrnco.envs.rmtvrp import RMTVRPEnv
from rrnco.envs.rmtvrp.selectstartnodes import SelectStartNodes


class VisibleStartNodes(SelectStartNodes):
    """只从可见客户出发（batch_size 必须为 1，DCC 实例可见性异构）。"""

    def _select(self, td, num_starts):
        assert td.batch_size[0] == 1, "VisibleStartNodes requires batch_size==1"
        visible = td["visible_mask"].bool()[0]        # (N+1,)
        starts = torch.nonzero(visible[1:]).squeeze(-1) + 1
        return starts                                 # 长度 = 可见客户数

    def get_num_starts(self, td):
        return int(td["visible_mask"].bool()[0].sum().item()) - 1   # 排除 depot


class DCCRMTVRPEnv(RMTVRPEnv):
    """RMTVRPEnv + 可见性门控。normalize=False（原始欧氏距离，cost 可直接比）。

    non-anticipatory 语义：模型（encoder + decoder inductive bias）读到的未来节点特征是
    masked 的（坐标→0.5、距离/时长→中性、TW/service/demand→0），但 env 用 `_true_*` 真值
    做可行性 + 奖励。因此解码器仍服务全部 50 客户，且报告的 cost = 纯真实行驶距离。
    """

    # 这些字段在 masked（模型看）与 true（env 用）之间有一一对应的 `_true_*` 副本
    _TRUE_KEYS = [
        "distance_matrix", "duration_matrix", "time_windows",
        "service_time", "demand_linehaul",
    ]

    def __init__(self, **kwargs):
        # 固定 normalize=False（成本可比性，见 dcc_data.py 注释）
        kwargs["normalize"] = False
        kwargs.setdefault("select_start_nodes_fn", VisibleStartNodes())
        kwargs.setdefault("check_solution", False)
        super().__init__(**kwargs)

    # ---- 环境计算时把标准字段临时换成真值，算完换回（模型始终读到 masked）----
    def _swap_true_in(self, td):
        """把标准字段换成 `_true_*` 真值，返回 {key: 原值} 以便换回。"""
        saved = {}
        for k in self._TRUE_KEYS:
            tk = "_true_" + k
            if tk in td and k in td:
                saved[k] = td[k]
                td[k] = td[tk]
        return saved

    def _swap_back(self, td, saved):
        for k, v in saved.items():
            td[k] = v

    def _reset(self, td=None, batch_size=None):
        td_reset = super()._reset(td, batch_size)
        # 父类 _reset 会丢掉 visible_mask 与 `_true_*`（只拷贝标准字段），这里补回
        td_reset["visible_mask"] = td["visible_mask"]
        td_reset["_true_distance_matrix"] = td["_true_distance_matrix"]
        td_reset["_true_duration_matrix"] = td["_true_duration_matrix"]
        td_reset["_true_time_windows"] = td["_true_time_windows"]
        td_reset["_true_service_time"] = td["_true_service_time"]
        # demand_linehaul 真值需 prepend depot 0（与父类 _reset 的 prepend 对齐）
        dtrue = td["_true_demand_linehaul"]                       # (B, N) 客户 only
        td_reset["_true_demand_linehaul"] = torch.cat(
            [torch.zeros_like(dtrue[..., :1]), dtrue], dim=1
        )
        # 父类 _reset 内部用 masked 字段算过一次 action mask，这里用真值重算覆盖
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset

    def get_action_mask(self, td):
        saved = self._swap_true_in(td)
        mask = RMTVRPEnv.get_action_mask(td)
        self._swap_back(td, saved)
        return mask

    def _step(self, td):
        saved = self._swap_true_in(td)
        td = super()._step(td)
        self._swap_back(td, saved)
        return td

    def _get_reward(self, td, actions):
        saved = self._swap_true_in(td)
        reward = super()._get_reward(td, actions)
        self._swap_back(td, saved)
        return reward
