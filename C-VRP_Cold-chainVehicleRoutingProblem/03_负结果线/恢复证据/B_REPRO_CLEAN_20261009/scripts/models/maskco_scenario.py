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
MASK_TOKEN = VOCAB_SIZE    # [MASK] 槽
N_TOKENS = VOCAB_SIZE + 1  # 含 MASK（不含 NULL 预测外的…MASK 只出现在输入槽）

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
        self.demands = defaultdict(list)     # (class, spot) -> [demand]
        self.leads = defaultdict(list)       # (time_bin,) -> [lead]
        self.times = defaultdict(list)       # (time_bin,) -> [time in bin]
        for day in history:
            for o in day:
                b = int(np.argmin(((SPOT_CENTERS - [o.x, o.y]) ** 2).sum(1)))
                self.coords[b].append((o.x, o.y))
                self.demands[(int(o.temp_class), b)].append(o.demand)
                t = int(min(float(o.reveal), 16.0 - 1e-9) / TIME_BIN_H)
                self.leads[t].append(float(o.tw_end - o.reveal))
                self.times[t].append(float(o.reveal))
        self.coords = {k: np.array(v) for k, v in self.coords.items()}
        self.demands = {k: np.array(v) for k, v in self.demands.items()}
        self.leads = {k: np.array(v) for k, v in self.leads.items()}
        self.times = {k: np.array(v) for k, v in self.times.items()}

    def decode_token(self, token: int, rng: np.random.Generator) -> ScenarioOrder:
        t, c, b, d = token_to_bucket(token)
        if self.coords.get(b) is not None and len(self.coords[b]):
            x, y = self.coords[b][int(rng.integers(0, len(self.coords[b])))]
        else:
            x, y = SPOT_CENTERS[b]
        lead = (float(self.leads[t][int(rng.integers(0, len(self.leads[t])))])
                if self.leads.get(t) is not None and len(self.leads[t]) else 2.5)
        lo, hi = t * TIME_BIN_H, (t + 1) * TIME_BIN_H
        reveal = float(rng.uniform(lo, hi))
        dp = self.demands.get((c, b))
        demand = (float(dp[int(rng.integers(0, len(dp)))])
                  if dp is not None and len(dp) else (1.5 if d == 0 else 2.5))
        return ScenarioOrder(oid=-(1_000_000 + int(rng.integers(1 << 30))),
                             reveal=reveal, x=float(x), y=float(y),
                             demand=demand, temp_class=c, tw_start=0.0,
                             tw_end=reveal + lead, service_time=0.05)


def snapshot_tokens(snap: VisibleSnapshot) -> list[int]:
    return [order_to_token(o) for o in snap.orders]


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
        # 与 encode_cvrp 同口径归一化（序列内全 token mean/L2，MASK 占位确定 → 训练/部署一致）
        mean = coords.mean(axis=1, keepdims=True)
        l2 = jnp.sqrt((coords ** 2).sum(axis=-1, keepdims=True).mean(axis=1, keepdims=True))
        coords_n = (coords - mean) / jnp.maximum(l2, 1e-9)
        dem_n = (raw[..., 2:] / self._capacity)
        x3 = jnp.concatenate([coords_n, dem_n], axis=-1)
        h = jax.lax.stop_gradient(self._cvrp.encode(x3))   # [B, L, 512]（冻结）
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
    """[visible tokens][MASK×M_max] -> 每槽 token logits（softmax 重建）。"""

    def __init__(self, dim=128, arm='random', cvrp_model=None,
                 rngs: nnx.Rngs | None = None):
        rngs = rngs or nnx.Rngs(0)
        self.arm = arm
        if arm == 'explicit':
            self.encoder = ExplicitEncoder(dim, rngs)
        elif arm == 'pretrained':
            if cvrp_model is None:
                raise ValueError("pretrained arm requires cvrp_model")
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
        self.dim = dim

    def __call__(self, tokens, attn_mask=None):
        x = self.encoder(tokens)
        x = self._block(x, self.attn0, self.mlp0, attn_mask)
        x = self._block(x, self.attn1, self.mlp1, attn_mask)
        return self.head(x)

    @staticmethod
    def _block(x, attn, mlp, mask):
        h = attn(x, x, mask=mask)
        x = x + h
        x = x + jax.nn.gelu(mlp(x))
        return x


# --------------------------------------------------------------------------- #
# 采样器（接口与步骤 2 一致）
# --------------------------------------------------------------------------- #
class MaskCOScenarioSampler:
    """sample(snapshot, rng, k) -> list[FutureScenario]。

    v1：单步 softmax 采样；因果保证 = 只用 snapshot 可见 token + 训练历史解码池。
    """

    name = "maskco"

    def __init__(self, model: MaskCOScenarioModel, pools: HistoryPools, m_max: int = M_MAX):
        self.model = model
        self.pools = pools
        self.m_max = m_max

    def _visible_tokens(self, snap):
        toks = [order_to_token(o) for o in snap.orders]
        return toks

    def sample(self, snapshot: VisibleSnapshot, rng, k: int) -> list[FutureScenario]:
        vis = self._visible_tokens(snapshot)
        vis = vis[:self.m_max]  # 防御性截断
        L = len(vis) + self.m_max
        toks = np.array(vis + [MASK_TOKEN] * self.m_max, dtype=np.int32)
        logits = np.asarray(self.model(jnp.asarray(toks)[None])[0])   # (L, VOCAB)
        mask_logits = logits[len(vis):]                              # 只取 MASK 槽
        probs = jax.nn.softmax(mask_logits, axis=-1)
        out = []
        for _ in range(k):
            idx = rng.multinomial(1, np.asarray(probs)).argmax(-1)   # 每槽独立采样
            scen = []
            for j, tok in enumerate(idx):
                if tok == NULL_TOKEN:
                    continue
                o = self.pools.decode_token(int(tok), rng)
                o = replace(o, oid=-(1_000_000 + j + 1))
                if o.reveal > snapshot.clock + 1e-6:
                    scen.append(o)
            out.append(scen)
        return out


# --------------------------------------------------------------------------- #
# 训练数据切片（历史日 → (可见 token 序列, 掩码未来 token 目标)）
# --------------------------------------------------------------------------- #
def save_model(model, path):
    import flax
    with open(path, "wb") as f:
        f.write(flax.serialization.to_bytes(nnx.state(model).to_pure_dict()))


def load_model(model_template, path):
    import flax
    target = nnx.state(model_template).to_pure_dict()
    with open(path, "rb") as f:
        b = f.read()
    restored = flax.serialization.from_bytes(target, b)
    nnx.state(model_template).replace_by_pure_dict(restored)
    return model_template


def make_training_slices(history, cuts_per_day=4, rng=None):
    """每个历史日取若干时钟切面：可见 = reveal<=cut，目标 = reveal>cut 的 token。"""
    rng = np.random.default_rng(0) if rng is None else rng
    slices = []
    for day in history:
        if not day:
            continue
        day_sorted = sorted(day, key=lambda o: o.reveal)
        cuts = sorted(set(
            min(float(o.reveal), 15.99) for o in day_sorted[:: max(1, len(day_sorted) // cuts_per_day)]))
        for cut in cuts[:cuts_per_day]:
            vis = [order_to_token(o) for o in day_sorted if o.reveal <= cut + 1e-6]
            tgt = [order_to_token(o) for o in day_sorted if o.reveal > cut + 1e-6]
            if vis and tgt:
                slices.append((vis, tgt))
    return slices
