"""A-v1 步骤 3：MaskCO 条件场景生成器。

- token 化：未来订单 = (时间桶 0.5h × 温区 3 × 空间桶 3 × 需求桶 2) 离散 token + NULL token；
- 生成模型：集合不变（无位置编码）自注意力：序列 = [可见订单 token][MASK×M_max]，在 MASK 槽做 softmax 重建；
- 三档编码器（共用生成头）：pretrained（冻结 cvrp100 512 维，服务器）/ random（随机同架构 MLP）/ explicit（线性投影）；
- 采样：单步 softmax（v1）；迭代去掩码留作 config-2；
- 采样器接口不变：sample(snapshot, rng, k) -> list[FutureScenario]（因果：只用 snapshot + 训练历史）。
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from dataclasses import replace

import numpy as np

import jax
import jax.numpy as jnp
import flax
from flax import nnx

from scenario_saa import (ScenarioOrder, VisibleSnapshot, SPOT_CENTERS,
                          FutureScenario)

TIME_BIN_H = 0.5
N_TIME_BINS = 32          # 0..16h / 0.5h
N_CLASSES = 3
N_SPOTS = 3
N_DEMAND_BINS = 2         # (1,2], (2,3]
N_FIELDS = (N_TIME_BINS, N_CLASSES, N_SPOTS, N_DEMAND_BINS)
NULL_TOKEN = 0
VOCAB_SIZE = 1 + N_TIME_BINS * N_CLASSES * N_SPOTS * N_DEMAND_BINS
MASK_TOKEN = VOCAB_SIZE    # [MASK] 槽（未知待生成）
PAD_TOKEN = VOCAB_SIZE + 1  # 批次补齐位（P0 修复：与 MASK/NULL 分离，屏蔽注意力与损失）
N_TOKENS = VOCAB_SIZE + 2   # NULL/真实 token + MASK + PAD

M_MAX = 200                # 序列最大未来槽数（训练/推理统一）


# --------------------------------------------------------------------------- #
# token 化
# --------------------------------------------------------------------------- #
def order_to_token(o: ScenarioOrder) -> int:
    t = int(min(float(o.reveal), 16.0 - 1e-9) / TIME_BIN_H)
    c = int(o.temp_class)
    b = int(np.argmin(((SPOT_CENTERS - [o.x, o.y]) ** 2).sum(1)))
    d = 0 if float(o.demand) <= 2.0 else 1
    return 1 + (t * N_CLASSES + c) * N_SPOTS * N_DEMAND_BINS + (b * N_DEMAND_BINS + d)


def token_to_bucket(token: int):
    """-> (time_bin, class, spot, demand_bin)。"""
    v = int(token) - 1
    d = v % N_DEMAND_BINS
    v //= N_DEMAND_BINS
    b = v % N_SPOTS
    v //= N_SPOTS
    c = v % N_CLASSES
    t = v // N_CLASSES
    return t, c, b, d


class HistoryPools:
    """解码用的历史经验分布：每 (时间桶, 空间桶) 的坐标库、需求库、lead（tw_end-reveal）库、时间桶内时间库。"""

    def __init__(self, history, rng=None):
        rng = np.random.default_rng(0) if rng is None else rng
        self.coords = defaultdict(list)      # (spot,) -> [(x,y)]
        self.demands = defaultdict(list)     # (class, spot, demand_bin) -> [demand]
        self.leads = defaultdict(list)       # (time_bin,) -> [lead]
        self.times = defaultdict(list)       # (time_bin,) -> [time in bin]
        for day in history:
            for o in day:
                b = int(np.argmin(((SPOT_CENTERS - [o.x, o.y]) ** 2).sum(1)))
                self.coords[b].append((o.x, o.y))
                d = 0 if float(o.demand) <= 2.0 else 1
                self.demands[(int(o.temp_class), b, d)].append(o.demand)
                t = int(min(float(o.reveal), 16.0 - 1e-9) / TIME_BIN_H)
                self.leads[t].append(float(o.tw_end - o.reveal))
                self.times[t].append(float(o.reveal))
        self.coords = {k: np.array(v) for k, v in self.coords.items()}
        self.demands = {k: np.array(v) for k, v in self.demands.items()}
        self.leads = {k: np.array(v) for k, v in self.leads.items()}
        self.times = {k: np.array(v) for k, v in self.times.items()}

    def decode_token(self, token: int, rng: np.random.Generator,
                     clock: float | None = None) -> ScenarioOrder:
        t, c, b, d = token_to_bucket(token)
        if self.coords.get(b) is not None and len(self.coords[b]):
            x, y = self.coords[b][int(rng.integers(0, len(self.coords[b])))]
        else:
            x, y = SPOT_CENTERS[b]
        lead = (float(self.leads[t][int(rng.integers(0, len(self.leads[t])))])
                if self.leads.get(t) is not None and len(self.leads[t]) else 2.5)
        lo, hi = t * TIME_BIN_H, (t + 1) * TIME_BIN_H
        # I1 边界桶修复（2026-09-30）：token 落在含 clock 的边界桶时，揭示时间按
        # 「尚未到来」条件截断采样 U(clock, hi)——保证解码订单 reveal>clock 且重编码
        # 回同一 token，数量头采样值 = 实际保留数（不再采后删单）。
        if clock is not None and lo <= float(clock) < hi:
            reveal = float(rng.uniform(float(clock) + 1e-9, hi))
        else:
            reveal = float(rng.uniform(lo, hi))
        # P0 修复（2026-09-30）：需求必须服从预测桶 d——按 (class, spot, bin) 条件池采样；
        # 池空回退桶中值（1.5∈(1,2]、2.5∈(2,3]），并断言重编码回 d。
        dp = self.demands.get((c, b, d))
        if dp is not None and len(dp):
            demand = float(dp[int(rng.integers(0, len(dp)))])
        else:
            demand = 1.5 if d == 0 else 2.5
        assert (demand > 1.0 and demand <= 2.0) if d == 0 else (demand > 2.0 and demand <= 3.0), \
            (token, d, demand)
        return ScenarioOrder(oid=-(1_000_000 + int(rng.integers(1 << 30))),
                             reveal=reveal, x=float(x), y=float(y),
                             demand=demand, temp_class=c, tw_start=0.0,
                             tw_end=reveal + lead, service_time=0.05)


def snapshot_tokens(snap: VisibleSnapshot) -> list[int]:
    return [order_to_token(o) for o in snap.orders]


def clock_support_mask(clock: float) -> np.ndarray:
    """P0（2026-09-30）：未来时间支持掩码——时间桶结束 ≤ clock 的 token 不可采样。
    边界桶（含 clock 的桶）允许采样，采后由安全过滤兜底。返回 bool[VOCAB_SIZE]。"""
    support = np.ones(VOCAB_SIZE, dtype=bool)
    for tok in range(1, VOCAB_SIZE):
        t, _c, _b, _d = token_to_bucket(tok)
        if (t + 1) * TIME_BIN_H <= clock + 1e-9:
            support[tok] = False
    return support


def clock_bin_of(clock: float) -> int:
    """I1 进阶（2026-09-30）：时钟所在 0.5h 桶编号（0..N_TIME_BINS-1），
    作为数量/token 头的显式条件输入。"""
    return int(min(max(float(clock), 0.0), 15.999) / TIME_BIN_H)


# --------------------------------------------------------------------------- #
# 生成模型（Flax NNX，集合不变：无位置编码）
# --------------------------------------------------------------------------- #
class EncoderMLP(nnx.Module):
    """订单 token -> D 维嵌入（random 臂 / pretrained 臂的字段 adapter 共用此结构）。"""

    def __init__(self, dim: int, arm: str, rngs: nnx.Rngs):
        self.arm = arm
        self.emb = nnx.Embed(N_TOKENS, dim, rngs=rngs)
        self.proj = nnx.Linear(dim, dim, rngs=rngs)

    def __call__(self, tokens):
        x = self.emb(tokens)
        return jax.nn.gelu(self.proj(x))


class PretrainedEncoder(nnx.Module):
    """预训练臂 A：冻结 MaskCO CVRP encoder（512 维，3D 输入 coord+demand）+ 字段投影 adapter。

    输入 raw = (空间桶中心 x, 空间桶中心 y, 需求桶值)——token 的确定性函数，训练/部署一致；
    时间/温区经可学习 bucket 嵌入与 CVRP 输出拼接。CVRP 权重冻结（stop_gradient）。
    """

    def __init__(self, dim: int, cvrp_model, rngs: nnx.Rngs, capacity=50.0):
        self.dim = dim
        self._cvrp = cvrp_model          # 非参数属性（冻结）
        self._capacity = float(capacity)
        self.bucket_emb = nnx.Embed(N_TOKENS, dim, rngs=rngs)
        self.merge = nnx.Linear(512 + dim, dim, rngs=rngs)

    def __call__(self, tokens):
        # token -> (t, c, b, d) 查表，构造 [B, L, 3] raw（NULL/MASK 用 (0,0,0) 占位）
        idx = jnp.arange(N_TOKENS)
        v = jnp.maximum(idx - 1, 0)
        d = v % N_DEMAND_BINS
        v = v // N_DEMAND_BINS
        b = v % N_SPOTS
        v = v // N_SPOTS
        c = v % N_CLASSES
        t = v // N_CLASSES
        valid = idx >= 1
        spots = jnp.asarray(SPOT_CENTERS)          # jax 数组：tracer 索引 numpy 会报 TracerArrayConversionError
        bx = jnp.where(valid, spots[b, 0], 0.0)
        by = jnp.where(valid, spots[b, 1], 0.0)
        bd = jnp.where(valid, jnp.array([1.5, 2.5])[d], 0.0)
        table = jnp.stack([bx, by, bd], axis=-1)          # [N_TOKENS, 3]
        raw = table[tokens]                                # [B, L, 3]
        coords = raw[..., :2]
        # P0 修复（2026-09-30）：归一统计只按「有效 token」计算（可见订单），
        # MASK/PAD/NULL 占位不参与均值/L2 尺度——padding 长度不得改变编码。
        valid = ((tokens != MASK_TOKEN) & (tokens != PAD_TOKEN)
                 & (tokens != NULL_TOKEN)).astype(jnp.float32)   # [B, L]
        # 广播陷阱修复（2026-09-30）：(B,1,2)/(B,1) 会右对齐广播成 (B,B,2) 垃圾——
        # 归一统计必须显式把 n_valid 升为 (B,1,1)。
        n_valid = jnp.maximum(valid.sum(axis=1), 1.0)[:, None, None]   # [B,1,1]
        mean = (coords * valid[..., None]).sum(axis=1, keepdims=True) / n_valid
        sq = (coords ** 2).sum(axis=-1, keepdims=True)
        l2 = jnp.sqrt((sq * valid[..., None]).sum(axis=1, keepdims=True) / n_valid)
        coords_n = (coords - mean) / jnp.maximum(l2, 1e-9)
        dem_n = (raw[..., 2:] / self._capacity)
        x3 = jnp.concatenate([coords_n, dem_n], axis=-1)
        # PAD 一致性修复（2026-09-30，路线图实测 0.0038 logits 偏移）：冻结 cvrp 内部
        # 注意力不支持 PAD 屏蔽，须经 attn_options 传入加性 bias（MISModel 同款通道，
        # 不改 MASKCO_code）——PAD 位对任意位及任意位对 PAD 位 −1e9，使有效位编码
        # 与 PAD 数量无关；无 PAD 时 bias 全 0，数值与旧路径一致。
        is_pad = (tokens == PAD_TOKEN)                                   # [B, L]
        pad_bias = jnp.where(is_pad[:, None, :] | is_pad[:, :, None],
                             -1e9, 0.0).astype(x3.dtype)                 # [B, L, L]
        h = jax.lax.stop_gradient(
            self._cvrp.encode(x3, attn_options={'bias': pad_bias}))      # [B, L, 512]（冻结）
        e = self.bucket_emb(tokens)
        return jax.nn.gelu(self.merge(jnp.concatenate([h, e], axis=-1)))


class ExplicitEncoder(nnx.Module):
    """显式特征臂：token 的显式字段（时间桶/类别/空间桶/需求桶）线性投影，不经可学习 embed。"""

    def __init__(self, dim: int, rngs: nnx.Rngs):
        self.proj = nnx.Linear(4, dim, rngs=rngs)

    def __call__(self, tokens):
        t = jnp.array([token_to_bucket(int(i)) for i in np.arange(1, N_TOKENS)],
                      dtype=jnp.float32)          # (N_TOKENS-1, 4)
        t = jnp.vstack([jnp.zeros((1, 4)), t])    # NULL 行
        feats = t[tokens]                         # (B, L, 4)
        return jax.nn.gelu(self.proj(feats))


class MaskCOScenarioModel(nnx.Module):
    """[visible tokens][MASK×M_max] -> 每槽 token logits（softmax 重建）+ 数量头 logits。

    I1 第一步（2026-09-30）：显式数量头 P(N_future | visible)——可见 token 池化后
    输出 (M_MAX+1) 计数分布；token 头输出条件于「有订单」的槽分布（NULL 由数量显式决定，
    不再靠逐槽 NULL 隐式计数）。槽对称保留（无位置编码）。
    I1 进阶（2026-09-30）：时钟条件——clock_bin（0..31，当前时钟所在 0.5h 桶）经
    clock_emb 广播加到每个位置（token 头条件）+ 池化表示（数量头条件）；
    clock_bin=None 时等价旧路径（回归兼容）。
    """

    def __init__(self, dim=128, arm='random', cvrp_model=None,
                 rngs: nnx.Rngs | None = None):
        rngs = rngs or nnx.Rngs(0)
        self.arm = arm
        if arm == 'explicit':
            self.encoder = ExplicitEncoder(dim, rngs)
        elif arm in ('pretrained', 'random_cvrp'):
            # random_cvrp = 同 CVRP encoder 架构随机权重（同冻结策略，同架构消融）
            if cvrp_model is None:
                raise ValueError("%s arm requires cvrp_model" % arm)
            self.encoder = PretrainedEncoder(dim, cvrp_model, rngs)
        else:
            self.encoder = EncoderMLP(dim, 'random', rngs)
        self.attn0 = nnx.MultiHeadAttention(
            num_heads=4, in_features=dim, qkv_features=dim, out_features=dim,
            rngs=rngs, decode=False, broadcast_dropout=False)
        self.mlp0 = nnx.Linear(dim, dim, rngs=rngs)
        self.attn1 = nnx.MultiHeadAttention(
            num_heads=4, in_features=dim, qkv_features=dim, out_features=dim,
            rngs=rngs, decode=False, broadcast_dropout=False)
        self.mlp1 = nnx.Linear(dim, dim, rngs=rngs)
        self.head = nnx.Linear(dim, VOCAB_SIZE, rngs=rngs)  # 输出 0..VOCAB-1（NULL=0 也在内）
        self.head_count = nnx.Linear(dim, M_MAX + 1, rngs=rngs)  # 数量头（0..M_MAX）
        self.clock_emb = nnx.Embed(N_TIME_BINS, dim, rngs=rngs)  # I1 进阶：时钟条件
        self.dim = dim

    def __call__(self, tokens, attn_mask=None, clock_bin=None):
        if attn_mask is None:
            # P0 修复（2026-09-30）：PAD 位隔离注意力——仅非 PAD 位之间互相 attend。
            valid = tokens != PAD_TOKEN            # [B, L]
            attn_mask = valid[:, None, None, :] & valid[:, None, :, None]  # [B,1,L,L]
        x = self.encoder(tokens)
        if clock_bin is not None:
            cb = jnp.asarray(clock_bin, dtype=jnp.int32)          # [B]
            x = x + self.clock_emb(cb)[:, None, :]                # token 头时钟条件
        x = self._block(x, self.attn0, self.mlp0, attn_mask)
        x = self._block(x, self.attn1, self.mlp1, attn_mask)
        tok_logits = self.head(x)
        # 数量头：仅用可见（非 MASK/PAD/NULL）token 的平均表示（+ 时钟条件）
        vis_mask = ((tokens != MASK_TOKEN) & (tokens != PAD_TOKEN)
                    & (tokens != NULL_TOKEN)).astype(jnp.float32)   # [B, L]
        n_vis = jnp.maximum(vis_mask.sum(axis=1, keepdims=True), 1.0)
        pooled = (x * vis_mask[..., None]).sum(axis=1) / n_vis       # [B, D]
        if clock_bin is not None:
            pooled = pooled + self.clock_emb(jnp.asarray(clock_bin, dtype=jnp.int32))
        cnt_logits = self.head_count(pooled)                        # [B, M_MAX+1]
        return tok_logits, cnt_logits

    @staticmethod
    def _block(x, attn, mlp, mask):
        h = attn(x, x, mask=mask)
        x = x + h
        x = x + jax.nn.gelu(mlp(x))
        return x


def _norm_probs(probs):
    """float64 重归一 + 末位取余（float32 softmax 行和可能略超 1.0）。"""
    p = np.asarray(probs, dtype=np.float64)
    p /= np.maximum(p.sum(-1, keepdims=True), 1e-12)
    p[..., -1] = np.clip(1.0 - p[..., :-1].sum(-1), 0.0, None)
    return p


def sample_from_logits(token_logits, count_logits, clock, rng, k, pools):
    """I1（2026-09-30）：显式数量头采样（纯 numpy，可测试）。

    - n ~ softmax(count_logits)（0..M_MAX）；
    - 每个槽从「非 NULL 条件分布」独立采样（NULL 质量剔除后重归一；槽对称）；
    - 时钟支持掩码：结束 ≤ clock 的 token 不可采样；边界桶采后安全过滤。
    返回 list[FutureScenario]（每场景恰 n 个未来订单，n 可随场景不同）。"""
    support = clock_support_mask(float(clock))
    tl = np.where(support[None, :], np.asarray(token_logits), -1e9)
    non_null = np.arange(VOCAB_SIZE) != NULL_TOKEN
    tl = np.where(non_null[None, :], tl, -1e9)          # 条件于「有订单」
    probs_tok = _norm_probs(jax.nn.softmax(tl, axis=-1))
    probs_cnt = _norm_probs(jax.nn.softmax(np.asarray(count_logits)[None, :], axis=-1))[0]
    m_max = probs_tok.shape[0]
    out = []
    for _ in range(k):
        n = int(np.argmax(rng.multinomial(1, probs_cnt)))
        n = min(n, m_max)
        idx = rng.multinomial(1, probs_tok).argmax(-1)   # 每槽独立（槽对称）
        scen = []
        for j in range(n):
            tok = int(idx[j])
            # 边界桶截断解码：reveal ∈ (clock, 桶尾] → 数量头采样值 = 实际保留数
            o = pools.decode_token(tok, rng, clock=float(clock))
            o = replace(o, oid=-(1_000_000 + j + 1))
            if o.reveal > float(clock) + 1e-6:           # 兜底安全过滤（正常不触发）
                scen.append(o)
        out.append(scen)
    return out


def keep_mask_by_prob(sample_probs, remask_frac, rng):
    """I2（2026-09-30）：按采样 token 概率保留顶部 (1−remask_frac) 槽（初版可解释规则）。
    sample_probs: (n,)；返回 keep 布尔数组。"""
    probs = np.asarray(sample_probs, dtype=np.float64)
    n = len(probs)
    n_keep = max(int(round(n * (1.0 - remask_frac))), 0)
    keep = np.zeros(n, dtype=bool)
    if n_keep > 0:
        order = np.argsort(-probs)
        keep[order[:n_keep]] = True
    return keep


def _cond_probs(tok_logits, support):
    """非 NULL 条件分布 + 时钟支持：返回 float64 概率 (m, VOCAB)。"""
    tl = np.where(support[None, :], np.asarray(tok_logits), -1e9)
    non_null = np.arange(VOCAB_SIZE) != NULL_TOKEN
    tl = np.where(non_null[None, :], tl, -1e9)
    return _norm_probs(jax.nn.softmax(tl, axis=-1))


# --------------------------------------------------------------------------- #
# 采样器（接口与步骤 2 一致）
# --------------------------------------------------------------------------- #
class MaskCOScenarioSampler:
    """sample(snapshot, rng, k) -> list[FutureScenario]。

    I1：显式数量头 + 槽条件分布；因果保证 = 只用 snapshot 可见 token + 训练历史解码池。
    """

    name = "maskco"

    def __init__(self, model: MaskCOScenarioModel, pools: HistoryPools, m_max: int = M_MAX,
                 iterative: bool = False, remask_frac: float = 0.5, rounds: int = 1):
        self.model = model
        self.pools = pools
        self.m_max = m_max
        self.iterative = bool(iterative)
        self.remask_frac = float(remask_frac)
        self.rounds = int(rounds)

    def _visible_tokens(self, snap):
        toks = [order_to_token(o) for o in snap.orders]
        # 路线图 09-30：可见订单数超过 m_max 时显式报错，禁止静默丢可见订单
        if len(toks) > self.m_max:
            raise ValueError('visible orders %d exceed m_max %d（禁静默截断：请调大 --m-max）'
                             % (len(toks), self.m_max))
        return toks

    def _forward_masks(self, prefix_tokens, clock_bin=None, n_mask=None):
        """前缀 [vis(+kept)] 送入模型，返回 (MASK 槽 token logits, 数量头 logits)。
        clock_bin：I1 进阶时钟条件。
        2026-09-30 固定形状修复：输入恒长 2*m_max（[前缀][MASK×n_mask][PAD…]），与训练
        批长一致 → 只编译一次；否则每事件新形状重编译冻结 cvrp 图（部署 10s 预算外）。"""
        n_mask = self.m_max if n_mask is None else int(n_mask)
        if len(prefix_tokens) + n_mask > 2 * self.m_max:
            raise ValueError('prefix %d + n_mask %d 超出固定批长 2*m_max=%d'
                             % (len(prefix_tokens), n_mask, 2 * self.m_max))
        toks = np.full(2 * self.m_max, PAD_TOKEN, dtype=np.int32)
        toks[:len(prefix_tokens)] = prefix_tokens
        toks[len(prefix_tokens):len(prefix_tokens) + n_mask] = MASK_TOKEN
        cb = jnp.asarray([clock_bin], dtype=jnp.int32) if clock_bin is not None else None
        tl, cl = self.model(jnp.asarray(toks)[None], clock_bin=cb)
        return (np.asarray(tl[0][len(prefix_tokens):len(prefix_tokens) + n_mask]),
                np.asarray(cl[0][:self.m_max + 1]))

    def sample(self, snapshot: VisibleSnapshot, rng, k: int) -> list[FutureScenario]:
        if self.iterative:
            return self.iterative_sample(snapshot, rng, k,
                                         remask_frac=self.remask_frac,
                                         rounds=self.rounds)
        vis = self._visible_tokens(snapshot)
        clock = float(snapshot.clock)
        cb = clock_bin_of(clock)
        tok_logits, cnt_logits = self._forward_masks(vis, clock_bin=cb)
        return sample_from_logits(tok_logits, cnt_logits, clock,
                                  rng, k, self.pools)

    def iterative_sample(self, snapshot: VisibleSnapshot, rng, k: int,
                         remask_frac: float = 0.5, rounds: int = 1) -> list[FutureScenario]:
        """I2（2026-09-30）：多轮保留/重掩码重构（初版规则 = 按采样概率保留高置信槽）。

        第一轮与单步一致（显式数量头 + 槽条件分布）；随后每轮：保留顶部
        (1−remask_frac) 概率的已生成槽作为条件，重掩码其余槽，在
        [vis + kept][MASK×rest] 上再生成。数量 n 跨轮固定（首轮决定）。
        因果保证：条件只含可见订单与**本方已生成**的槽，不读真实未来。"""
        vis = self._visible_tokens(snapshot)
        clock = float(snapshot.clock)
        clock_bin = clock_bin_of(clock)
        support = clock_support_mask(clock)
        tl, cl = self._forward_masks(vis, clock_bin=clock_bin)
        probs_tok = _cond_probs(tl, support)
        probs_cnt = _norm_probs(jax.nn.softmax(cl[None], axis=-1))[0]
        out = []
        for _ in range(k):
            n = min(int(np.argmax(rng.multinomial(1, probs_cnt))), self.m_max)
            idx = rng.multinomial(1, probs_tok).argmax(-1)
            scen_toks = [int(idx[j]) for j in range(n)]
            scen_probs = [float(probs_tok[j, int(idx[j])]) for j in range(n)]
            for _r in range(int(rounds)):
                if not scen_toks:
                    break
                keep = keep_mask_by_prob(scen_probs, remask_frac, rng)
                kept = [t for j, t in enumerate(scen_toks) if keep[j]]
                n_remask = n - len(kept)
                if n_remask <= 0:
                    break
                tl2, _ = self._forward_masks(vis + kept, clock_bin=clock_bin,
                                             n_mask=self.m_max - len(kept))
                p2 = _cond_probs(tl2[:n_remask], support)
                idx2 = rng.multinomial(1, p2).argmax(-1)
                new_toks, new_probs, ki = [], [], 0
                for j in range(n):
                    if keep[j]:
                        new_toks.append(scen_toks[j])
                        new_probs.append(scen_probs[j])
                    else:
                        new_toks.append(int(idx2[ki]))
                        new_probs.append(float(p2[ki, int(idx2[ki])]))
                        ki += 1
                scen_toks, scen_probs = new_toks, new_probs
            scen = []
            for j, t in enumerate(scen_toks):
                # 边界桶截断解码：reveal ∈ (clock, 桶尾] → 实际保留数 = 采样数量
                o = self.pools.decode_token(t, rng, clock=clock)
                o = replace(o, oid=-(1_000_000 + j + 1))
                if o.reveal > clock + 1e-6:      # 兜底安全过滤（正常不触发）
                    scen.append(o)
            out.append(scen)
        return out


# --------------------------------------------------------------------------- #
# 训练数据切片（历史日 → (可见 token 序列, 掩码未来 token 目标)）
# --------------------------------------------------------------------------- #
def save_model(model, path):
    import flax
    # 2026-10-01 关键修复：必须用 nnx.Param 过滤——无过滤 nnx.state(model) 含 Rngs 等
    # 混合类型，to_bytes/from_bytes 往返后 Params 回到初始值（实测保存的是未训练权重）。
    with open(path, "wb") as f:
        f.write(flax.serialization.to_bytes(
            nnx.state(model, nnx.Param).to_pure_dict()))


def load_model(model_template, path):
    import flax
    target = nnx.state(model_template, nnx.Param).to_pure_dict()
    with open(path, "rb") as f:
        b = f.read()
    restored = flax.serialization.from_bytes(target, b)
    # 2026-10-01：本 flax 版本 State.replace_by_pure_dict 不回写模块（实测回 init）；
    # nnx.update(model, pure_dict) 实测正确（探针往返一致）。
    nnx.update(model_template, restored)
    return model_template


def check_model_generation(model_bin, allow_mismatch=False):
    """模型代码世代校验（2026-09-30）：checkpoint 的 config.json 记录训练时
    maskco_scenario.py/train_maskco_scenario.py 的 sha256；与当前磁盘源码不一致 →
    SystemExit（--allow-code-mismatch 可放行并显式标注）。旧 config 无该字段 →
    返回 'legacy_unknown'（如 pre-clock 原型）。返回 'ok'/'legacy_unknown'。"""
    import hashlib
    import json
    cfg_path = os.path.join(os.path.dirname(os.path.abspath(model_bin)), "config.json")
    if not os.path.exists(cfg_path):
        return 'legacy_unknown'
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    stored = cfg.get("model_source_sha256")
    if not stored:
        return 'legacy_unknown'
    problems = []
    for rel, want in stored.items():
        p = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)))), *rel.split('/'))
        with open(p, 'rb') as fh:
            got = hashlib.sha256(fh.read()).hexdigest()
        if got != want:
            problems.append(rel)
    if problems:
        msg = ("模型代码世代不匹配（checkpoint 训练于旧版源码，当前磁盘已更新）：%s。"
               "旧权重不得静默加载进新模板——请用对应旧版代码评估，或用新版重新训练；"
               "确需放行加 --allow-code-mismatch。" % ', '.join(problems))
        if allow_mismatch:
            print('WARNING:', msg)
        else:
            raise SystemExit(msg)
    return 'ok' if not problems else 'mismatch_allowed'


def make_training_slices(history, cuts_per_day=4, rng=None):
    """每个历史日取若干时钟切面：可见 = reveal<=cut，目标 = reveal>cut 的 token。
    返回 (vis, tgt, n_future, clock_bin)（I1 进阶：clock_bin = 切面时钟所在 0.5h 桶，
    作为数量/token 头的显式条件；clock_bin_of 截断到 [0,15.999]）。
    2026-09-30：切面时钟改为在当日 reveal 事件时间上随机采样（对齐在线事件时钟分布，
    替代旧固定步长切面）。"""
    rng = np.random.default_rng(0) if rng is None else rng
    slices = []
    for day in history:
        if not day:
            continue
        day_sorted = sorted(day, key=lambda o: o.reveal)
        rev_times = [min(float(o.reveal), 15.99) for o in day_sorted]
        n_cuts = min(cuts_per_day, len(rev_times))
        cuts = sorted(rng.choice(rev_times, size=n_cuts, replace=False))
        for cut in cuts:
            vis = [order_to_token(o) for o in day_sorted if o.reveal <= cut + 1e-6]
            tgt = [order_to_token(o) for o in day_sorted if o.reveal > cut + 1e-6]
            if vis and tgt:
                slices.append((vis, tgt, len(tgt), clock_bin_of(float(cut))))
    return slices
