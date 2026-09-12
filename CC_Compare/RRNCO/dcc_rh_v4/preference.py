"""RRNCO 偏好定义与 mock provider（R0.5 Stage A，纯 stdlib/NumPy，不依赖 Torch）。

固定「RRNCO 偏好」的唯一含义：
  偏好 = 一次确定性解码序列中，删除 depot 与 anchor 后，客户**首次出现**的顺序。
  即一条解码序列（可能含多个 depot 分隔子路线）压缩成「客户偏好排序」。

规则（validate_ordering）：
  - 必须完整覆盖可见 pool（无缺失、无多余）；
  - 禁止外部客户（不在 pool）；
  - 禁止重复客户；
  - 禁止非法本地索引（无法映射到真实客户的索引）；
  - 不允许静默追加 EDD / 最近邻客户（缺了就是缺了，显式失败）。
"""
from dataclasses import dataclass


def extract_first_occurrence(sequence, depot=0, anchor=None):
    """从解码序列提取「客户首次出现顺序」（删除 depot 与 anchor）。

    sequence：节点 id 序列（真实 id；depot=0 是子路线分隔符）。
    返回 tuple[int, ...]：客户真实 id 的偏好排序（最偏好在前）。
    """
    seen = set()
    order = []
    for node in sequence:
        n = int(node)
        if n == depot or n == anchor:
            continue
        if n in seen:
            continue
        seen.add(n)
        order.append(n)
    return tuple(order)


def validate_ordering(ordering, pool_customer_ids):
    """校验偏好排序是否合法（完整覆盖 pool、无外部/重复）。返回 problem 列表。"""
    pool = set(int(c) for c in pool_customer_ids)
    problems = []
    seen = set()
    for c in ordering:
        c = int(c)
        if c not in pool:
            problems.append(f'外部客户 {c}（不在 pool）')
        if c in seen:
            problems.append(f'重复客户 {c}')
        seen.add(c)
    for c in sorted(pool - seen):
        problems.append(f'缺失客户 {c}（pool 未覆盖）')
    return problems


@dataclass
class PreferenceProvider:
    """偏好 provider 接口：order(subproblem) -> tuple[int, ...]（真实客户 id 排序）。"""
    def order(self, subproblem):
        raise NotImplementedError


class MockPreferenceProvider(PreferenceProvider):
    """确定性 mock（不依赖 Torch）。mode ∈ {edd, nearest, fixed, shuffle}。

    用于离线区分「模型给出的排序」与「外层协调器做出的安全分配」；真实模型
    由 Stage B 的 rrnco_backend 提供。
    """

    def __init__(self, mode='edd', seed=0):
        if mode not in ('edd', 'nearest', 'fixed', 'shuffle'):
            raise ValueError(f'未知 mode: {mode}')
        self.mode = mode
        self.seed = seed

    def order(self, subproblem):
        pool = list(subproblem.pool_customer_ids)
        idx = lambda c: subproblem.node_index(c)  # noqa: E731
        if self.mode == 'edd':
            return tuple(sorted(pool, key=lambda c: subproblem.tw_end[idx(c)]))
        if self.mode == 'nearest':
            ai = subproblem.anchor_idx
            return tuple(sorted(pool, key=lambda c: subproblem.dist_mat[ai][idx(c)]))
        if self.mode == 'fixed':
            return tuple(sorted(pool))
        if self.mode == 'shuffle':
            import random
            rng = random.Random(self.seed)
            p = list(pool)
            rng.shuffle(p)
            return tuple(p)
        raise ValueError(self.mode)
