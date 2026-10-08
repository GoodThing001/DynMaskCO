"""LIH（Learn-to-Improve-Heuristics，L2I，TNNLS 2022）A-v1 排序 provider。

把 LIH-CVRP50（AM 编码器 + NeuRewriter 成对局部算子改进步）接到通用
OrderingAcceptReplanner 的 provider 接口：order(sp) -> tuple[int,...]。

官方评测入口（CVRP/CVRP50/test.py + Test with model.ipynb）：
- 模型 outputs/cvrp_50/run/epoch-199.pt（官方 CVRP50 权重；args.json：
  embedding 128 / hidden 128 / n_encode_layers 3 / normalization batch /
  steps 100）。
- 测试流水线：官方把实例规范化为 100-token 循环序列（≤50 客户 token +
  50 depot token，depot token 位于 depot 坐标、demand 0；seq_tensor2 硬编码
  token>50 为 depot），每步 NeuRewriter（test=False 采样档，n_heads=1）选一个
  成对 2-opt 反转区间并应用，共 steps 步；上报轨迹最优长度。本 provider 忠实
  复刻该流水线（包括 get_costs/seq_tensor2/emdedding 的精确语义），输出最优
  轨迹序列的客户序。**初始解桥接**：官方初始 rec=[1..100]（全体客户一段）在
  其自身数据上容量不可行（段和 >1 → che_mask 全 False → 官方 softmax 全 -inf
  → 采样 NaN），本 provider 改为容量可行的贪心装箱循环序列（见 _run_l2i）。
- 改进步数：官方 test.py 硬编码 range(1000)；CPU 实测 1000 步远超 10s/决策
  预算 → 取官方参数空间内最小改进步数 steps=100（官方 options.py 默认值，
  亦为 epoch-199.pt 训练配置 args.json 的 steps 值；RESULTS.md 记录 1000 步
  实测耗时与选择理由）。

任务口径（诚实零样本跨任务）：
- 原生约束集 = CVRP（容量 + 单一 depot，无 TW/冷链/多温区/在途锚点）。本
  provider 只按 CVRP 求解排序，忽略 TW；真实 TW/C0 可行性由下游冻结
  dcc_rh_v4 协调器 + certify_plan 硬认证兜底。
- 输入规范化与官方 VRPDataset 一致：坐标 [0,1]（A-v1 原生即 [0,1]），
  demand/capacity；客户行序 = pool 排序序（官方实例行序 = 生成器路线序，
  任意确定行序均合法）。
- 规模：官方模型硬编码 50 客户 token + 50 depot token（seq_tensor2 的
  token>50 判定）；pool > 50 时按 50 分块、逐块 LIH 求解后拼接（模型规模
  上限的必要桥接，非超参数），pool ≤ 50 单块。
- CPU 兼容桥接（进程内 monkey-patch，不改上游源码）：
  (a) 上游为 CUDA 时代代码，多处硬编码 tensor.cuda() → 进程内把
      torch.Tensor.cuda 替换为恒等（本臂纯 CPU）；
  (b) torch>=1.13 移除 Tensor.byte → 补回（返回 bool）；
  (c) MultiHeadAttention_to_attn.forward 用 `1 - mask.byte()` 做 uint8 掩码
      索引，torch 2.x 下会退化成整数索引 → 进程内替换为该函数的 bool 掩码
      等价实现（语义逐行相同）。
"""
from __future__ import annotations

import hashlib
import os
import sys
from itertools import combinations

import numpy as np

_LIH_ROOT = os.path.normpath(os.path.join(     # CC_Compare/Learn-Improvement-Heuristics/CVRP/CVRP50
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'CVRP', 'CVRP50'))
if _LIH_ROOT not in sys.path:
    sys.path.append(_LIH_ROOT)

STEPS = int(os.environ.get('LIH_STEPS', '100'))   # 官方 options 默认 100（test.py 硬编码 1000，见 docstring）
G2S = 100          # 50 客户 token + 50 depot token（官方 seq_tensor2 硬编码）
EMB = 128


_POS_ENC_CACHE = {}


def _position_encoding_init(n_position, emb_dim):
    """官方 position_encoding_init（缓存：每步调用，避免重复构造）。"""
    key = (int(n_position), int(emb_dim))
    enc = _POS_ENC_CACHE.get(key)
    if enc is not None:
        return enc
    enc = np.array([[pos / np.power(10000, 2 * (j // 2) / emb_dim) for j in range(emb_dim)]
                    if pos != 0 else np.zeros(emb_dim) for pos in range(n_position)])
    enc[1:, 0::2] = np.sin(enc[1:, 0::2])
    enc[1:, 1::2] = np.cos(enc[1:, 1::2])
    enc = torch_from_np(enc).float()
    _POS_ENC_CACHE[key] = enc
    return enc


def torch_from_np(x):
    import torch
    return torch.from_numpy(np.asarray(x))


def _emdedding(input, seq_tensor):
    """官方 test_function.emdedding 的 CPU 等价向量化版（数值逐元素一致，去
    .cuda() 与逐 token 的 Python 循环）。input: dict(loc (bs,100,2), demand
    (bs,100))；seq_tensor: (bs,100) 为 token 1..100 的排列。
    返回 (cor_3_demand (bs,100,7), pos_enc (bs,100,128))，行序 = token 序。"""
    import torch
    loc = input['loc']          # (bs, 100, 2)
    dem = input['demand']       # (bs, 100)
    bs, g2s = seq_tensor.size()
    pos = seq_tensor.argsort(dim=1)          # (bs, 100): token t 的位置（0-based）
    pre = seq_tensor.gather(1, (pos - 1) % g2s)      # 循环前驱 token
    mid = torch.arange(1, g2s + 1, device=seq_tensor.device).expand(bs, g2s)
    las = seq_tensor.gather(1, (pos + 1) % g2s)      # 循环后继 token
    dem_row = dem.gather(1, mid - 1)                 # token t → 行 t-1 的需求
    cor_indice = torch.stack((pre - 1, mid - 1, las - 1), dim=-1)   # (bs,100,3)
    xy_pre = loc.gather(1, (pre - 1).unsqueeze(-1).expand(bs, g2s, 2))
    xy_mid = loc.gather(1, (mid - 1).unsqueeze(-1).expand(bs, g2s, 2))
    xy_las = loc.gather(1, (las - 1).unsqueeze(-1).expand(bs, g2s, 2))
    cor_single = torch.stack((xy_pre, xy_mid, xy_las), dim=2)       # (bs,100,3,2)
    cor_3_demand = torch.cat((cor_single.reshape(bs, g2s, 6),
                              dem_row.unsqueeze(-1)), dim=-1)       # (bs,100,7)
    enc = _position_encoding_init(g2s, EMB)      # (100, 128)
    pos_enc = enc[pos]                           # (bs,100,128)，按 token 位置取
    return cor_3_demand, pos_enc


def _to_attn_forward(self, q, exchange, che_mask, h=None, mask=None):
    """官方 MultiHeadAttention_to_attn.forward 的 bool-mask 等价版（torch 2.x 兼容）。
    原版 `1 - mask.byte()` 在旧 torch 中是 uint8 掩码索引，torch>=1.13 退化为整数
    索引 —— 此处仅把三处掩码索引改为 bool，其余逐行一致。"""
    import math
    import torch
    import torch.nn.functional as F
    if h is None:
        h = q
    batch_size, graph_size, input_dim = h.size()
    gs = graph_size // 2
    n_query = q.size(1)
    hflat = h.contiguous().view(-1, input_dim)
    qflat = q.contiguous().view(-1, input_dim)
    shp = (self.n_heads, batch_size, graph_size, -1)
    shp_q = (self.n_heads, batch_size, n_query, -1)
    Q = torch.matmul(qflat, self.W_query).view(shp_q)
    K = torch.matmul(hflat, self.W_key).view(shp)
    V = torch.matmul(hflat, self.W_val).view(shp)
    compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))
    compatibility = F.tanh(compatibility) * 10.
    mask_dia = torch.tril(torch.ones(graph_size, graph_size)).view(
        1, 1, graph_size, graph_size).expand_as(compatibility)
    mask_dia_b = mask_dia.to(torch.bool)
    compatibility[mask_dia_b] = -np.inf
    tt = compatibility[~mask_dia_b].clone()
    tt[~che_mask.to(torch.bool)] = -np.inf
    compatibility[~mask_dia_b] = tt
    compatibility[:, :, -gs:, :] = -np.inf
    if exchange is not None:
        compatibility[0][torch.arange(batch_size), exchange[:, 1],
                          exchange[:, 0]] = -np.inf
        compatibility[0][torch.arange(batch_size), exchange[:, 0],
                          exchange[:, 1]] = -np.inf
    im = compatibility.view(self.n_heads, batch_size, -1)
    im_l = F.log_softmax(im, dim=-1)
    im_s = F.softmax(im, dim=-1)
    return im_l, im_s


def _ga_encoder_forward(self, x, test, exchange, che_mask, action, mask=None):
    """官方 GraphAttentionEncoder.forward 的 batch=1 安全等价版（逐行同语义）。
    上游 sample 分支 `att.squeeze().gather(1, inde)` 在 batch=1、n_heads=1 时会把
    batch 维挤掉而崩溃（上游 eval_batch_size=128 才成立）——此处显式取 head-0 分布
    （n_heads=1，官方 test.py 构造即 n_heads=1），其余逐行一致。"""
    import torch
    assert mask is None, "TODO mask not yet supported!"
    bs, gs, in_d = x.size()
    h_em = self.layers(x)
    graph_embed = h_em.max(1)[0]
    fixed_context = self.project_graph(graph_embed)[:, None, :]
    node_feature = self.project_node(h_em)
    fusion = node_feature + fixed_context.expand_as(node_feature)
    att, att_s = self.one_attn(fusion, exchange, che_mask)
    if action is None:
        if test:
            atten = att_s.view(self.heads, bs, gs, gs)   # for max selection
            softmax_max = atten.max(-1)[0].max(-1)[0]
            row = atten.max(-1)[0].max(-1)[1].unsqueeze(-1)
            col = atten.max(-1)[1].gather(2, row)
            exc = torch.cat((row, col), -1)
        else:
            att0 = att[0]           # (bs, gs*gs) head-0（n_heads=1）
            atts0 = att_s[0]
            inde = atts0.multinomial(1)
            softmax_max = att0.gather(1, inde)
            while not (softmax_max > -1e10).data.all():
                inde = atts0.multinomial(1)
                softmax_max = att0.gather(1, inde)
            col = inde % gs
            row = inde // gs
            exc = torch.cat((row, col), -1)
            exc = exc[None, :, :]
        assert (softmax_max > -1e10).data.all(), \
            print(softmax_max[softmax_max < -1e10])
        return softmax_max, exc, inde
    else:
        if test:
            atten = att_s.view(self.heads, bs, gs, gs)
            softmax_max = atten.max(-1)[0].max(-1)[0]
            row = atten.max(-1)[0].max(-1)[1].unsqueeze(-1)
            col = atten.max(-1)[1].gather(2, row)
            exc = torch.cat((row, col), -1)
        else:
            softmax_max = att[0].gather(1, action)
        assert (softmax_max > -1e10).data.all(), \
            print(softmax_max[softmax_max < -1e10])
        return softmax_max


def _install_shims():
    """进程内兼容桥接（不改上游文件；见模块 docstring）。"""
    import torch
    from graph_encoder import (MultiHeadAttention_to_attn, GraphAttentionEncoder)
    if not hasattr(torch.Tensor, 'byte'):
        torch.Tensor.byte = lambda self: self.to(torch.bool)
    else:
        torch.Tensor.byte = lambda self: self.to(torch.bool)
    torch.Tensor.cuda = lambda self, *a, **k: self   # CPU 恒等
    MultiHeadAttention_to_attn.forward = _to_attn_forward
    GraphAttentionEncoder.forward = _ga_encoder_forward


class LIHProvider:
    def __init__(self, checkpoint_path, device='cpu', num_loc=100):
        self.checkpoint_path = os.path.abspath(checkpoint_path)
        self.device = device
        self.num_loc = int(num_loc)
        self.steps = STEPS
        self._model = None
        self._problem = None
        self._dic = None
        self._cl = None
        self._cache = {}
        self.n_calls = 0
        self.n_cache_hits = 0

    # ------------------------------------------------------------------ #
    def _load(self):
        """懒加载模型 + 常量（只加载不推理；driver 在 10s 决策预算外先调用一次）。"""
        import torch
        try:
            torch.set_num_threads(max(1, min(8, os.cpu_count() or 8)))
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
        _install_shims()
        from utils import load_problem
        from attention_model import AttentionModel
        problem = load_problem('cvrp')
        load_data = torch.load(self.checkpoint_path, map_location='cpu',
                               weights_only=False)
        sd = load_data.get('model', load_data)
        model = AttentionModel(128, 128, problem, n_encode_layers=3,
                               mask_inner=True, mask_logits=True,
                               normalization='batch', tanh_clipping=10.)
        model.load_state_dict({**model.state_dict(), **sd})
        model.to(self.device).eval()
        # 官方 test.py 常量：所有 2-opt 位置对（graph_size=100 → C(100,2)=4950）
        pairs = torch.tensor(list(combinations(range(G2S), 2)), dtype=torch.long)
        i0, i1 = pairs[:, 0], pairs[:, 1]
        idx = torch.arange(G2S).expand(pairs.size(0), G2S)
        rev = torch.where((idx >= i0[:, None]) & (idx <= i1[:, None]),
                          (i1 + i0)[:, None] - idx, idx)
        self._model = model
        self._problem = problem
        self._dic = rev.to(self.device)
        self._cl = pairs.to(self.device)

    # ------------------------------------------------------------------ #
    @staticmethod
    def _pool_key(sp):
        pool = tuple(int(c) for c in sp.pool_customer_ids)
        nodes = [0] + list(pool)
        payload = [str(pool),
                   str(tuple(sp.coords[sp.node_index(c)] for c in nodes)),
                   str(tuple(sp.demands[sp.node_index(c)] for c in nodes)),
                   str(float(sp.capacity))]
        return hashlib.sha256('|'.join(payload).encode('utf-8')).hexdigest()

    def _run_l2i(self, sp, chunk):
        """官方 validate() 流水线的忠实复刻（batch=1，chunk ≤ 50 客户）。

        初始解桥接（RESULTS.md 记录）：官方 validate() 把实例行按需求排序后令
        rec=[1..100]（全体客户一段 + depot 段）——在其自身数据上该初始解容量
        不可行（段和 >1 → seq_tensor2 che_mask 全 False → softmax 全 -inf →
        官方采样路径 NaN 崩溃）。本 provider 桥接为容量可行的循环序列：按 chunk
        序贪心装箱（段和 ≤1），段满插入 depot token（51..100，seq_tensor2 的
        token>50 判定域内），尾部补 depot token + n<50 时的零需求「假客户」
        token（depot 行）凑满 100 位。NeuRewriter 改进步与官方一致（test=False
        采样档、steps 步、轨迹最优）。
        """
        import torch
        n = len(chunk)
        idx = {c: sp.node_index(c) for c in chunk}
        depot = [float(x) for x in sp.coords[sp.node_index(0)]]
        loc = np.zeros((1, G2S, 2), dtype=np.float32)
        dem = np.zeros((1, G2S), dtype=np.float32)
        dems = []
        for k, c in enumerate(chunk):          # 行 k ↔ token k+1（官方 reorder 后行序 = 客户序）
            loc[0, k] = [float(x) for x in sp.coords[idx[c]]]
            d = float(sp.demands[idx[c]]) / float(sp.capacity)
            dem[0, k] = d
            dems.append(d)
        for k in range(n, G2S):
            loc[0, k] = depot                  # 行 n..99 = depot 行（token n+1..100）
        split_depots = list(range(51, G2S + 1))      # 全 >50：seq_tensor2 视为 depot
        fake_customers = list(range(n + 1, 51))      # n<50 时的零需求假客户 token
        rec_tokens, di, seg = [], 0, 0.0
        for k in range(n):
            if seg > 0 and seg + dems[k] > 1.0 + 1e-6:
                rec_tokens.append(split_depots[di])
                di += 1
                seg = 0.0
            rec_tokens.append(k + 1)
            seg += dems[k]
        rec_tokens += split_depots[di:]              # 尾部 depot token（空段收尾）
        rec_tokens += fake_customers                 # 凑满 100 位（depot 行，0 demand）
        input_ = {
            'loc': torch.from_numpy(loc).to(self.device),
            'demand': torch.from_numpy(dem).to(self.device),
        }
        rec = torch.tensor([rec_tokens], dtype=torch.long).to(self.device)
        pre_length, rec, che_mask = self._problem.get_costs(
            input_, rec, self._dic, self._cl)
        best_length = pre_length.clone()
        best_rec = rec.clone()
        exchange = None
        with torch.no_grad():
            for _step in range(self.steps):
                info, pos = _emdedding(input_, rec)
                (_ll, now_length, rec, exchange, _act,
                 che_mask) = self._model(input_, rec, info, pos, exchange,
                                         che_mask, self._dic, self._cl,
                                         action=None, test=False)
                if float(now_length[0]) < float(best_length[0]):
                    best_length = now_length.clone()
                    best_rec = rec.clone()
        seq = [int(t) for t in best_rec[0].tolist()]
        real = set(range(1, n + 1))
        p = next((i for i, t in enumerate(seq) if t > 50), None)
        rot = seq[p + 1:] + seq[:p + 1] if p is not None else seq
        tokens = [t for t in rot if t in real]
        return tuple(chunk[t - 1] for t in tokens)

    def order(self, sp):
        """SubProblem → LIH-CVRP50 排序（真实节点 id，不含 depot/anchor）。"""
        if self._model is None:
            self._load()
        self.n_calls += 1
        pool = [int(c) for c in sp.pool_customer_ids]
        if len(pool) == 0:
            return ()
        key = self._pool_key(sp)
        if key in self._cache:
            self.n_cache_hits += 1
            return self._cache[key]
        order = ()
        for k in range(0, len(pool), 50):          # 官方模型硬编码 50 客户槽 → 分块桥接
            order += self._run_l2i(sp, pool[k:k + 50])
        if len(self._cache) > 512:
            self._cache.clear()
        self._cache[key] = order
        return order
