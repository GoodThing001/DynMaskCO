"""
Phase 3b: Learnable Event-Aware Mask Policy (REINFORCE)

将 D1 的硬规则 (新订单距离 → 受影响 segment) 替换为可学习策略。
轻量 MLP (~5K 参数) 输入每条边的特征, 输出 mask logit。
Gumbel-Top-K 采样 → 离散化 mask 动作。
训练: REINFORCE policy gradient, reward = -beam_route_cost (降低 cost)。

接口:
  mask_logit = policy(edge_features)          # (B, num_edges) → (B, num_edges)
  mask = gumbel_topk(mask_logit, K)           # 可微分采样 top-K
  new_route = decode_and_reconstruct(mask)    # 模型重建
  reward = -route_cost(new_route)             # beam cost as reward
  loss = -reward * log_prob(mask)            # REINFORCE
"""
import numpy as np
from typing import Optional, Tuple


# ============================================================
# Edge Feature Extractor
# ============================================================

def extract_edge_features(
    routes: np.ndarray,           # (batch, route_len)
    edge_logits: np.ndarray,      # (batch, N+1, N+1) model output logits
    tw_start: np.ndarray,         # (batch, N+1)
    tw_end: np.ndarray,           # (batch, N+1)
    quality_loss: Optional[np.ndarray],  # (batch, N+1) or None
    event_nodes: Optional[list] = None,   # list of new order node indices
    coords: Optional[np.ndarray] = None,  # (batch, N+1, 2)
    dist_mat: Optional[np.ndarray] = None,  # (batch, N+1, N+1)
    current_cycle: int = 0,
    max_cycles: int = 40,
    vehicle_load: float = 0.0,
    capacity: float = 50.0,
):
    """
    为当前解中的每条边提取特征向量。

    Returns:
        features: (batch, num_edges, 6) float32
          0: edge_confidence — 模型对该边的 logit 值 (高=模型确信这条边正确)
          1: tw_slack — 时间窗松弛量 = (tw_end - tw_start) / tw_max (窄窗→少mask)
          2: event_proximity — 到最近新订单的距离倒数 (近事件→多mask)
          3: quality_risk — 边的品质损失 (高损耗边→多mask)
          4: route_load_ratio — 当前载重/容量 (高负载→多mask)
          5: remaining_budget — 剩余 cycle/总 cycle (budget少→少mask)
    """
    B = routes.shape[0]
    R = routes.shape[1]
    N1 = edge_logits.shape[1]  # N+1

    # Compute features for each edge in the route
    features_list = []

    for b in range(B):
        batch_feats = []
        for p in range(R):
            i = int(routes[b, p])
            j = int(routes[b, (p + 1) % R])

            if i == 0 and j == 0:
                continue  # skip consecutive depots
            if i == j:
                continue

            # 1. Edge confidence (model logit)
            conf = float(edge_logits[b, i, j]) if i < N1 and j < N1 else 0.0

            # 2. TW slack (of source and destination nodes)
            if i < tw_end.shape[1] and j < tw_end.shape[1]:
                slack_i = tw_end[b, i] - tw_start[b, i]
                slack_j = tw_end[b, j] - tw_start[b, j]
                tw_max_val = tw_end[b].max() + 1e-6
                tw_slack = (slack_i + slack_j) / (2.0 * tw_max_val + 1e-6)
            else:
                tw_slack = 0.5

            # 3. Event proximity
            if event_nodes and coords is not None and dist_mat is not None:
                min_dist = float('inf')
                for en in event_nodes:
                    if en <= 0 or en >= N1:
                        continue
                    d = dist_mat[b, j, en] if j < N1 else float('inf')
                    min_dist = min(min_dist, d)
                # Inverse distance: closer → higher value
                event_prox = 1.0 / (min_dist + 0.01) if min_dist < float('inf') else 0.0
                event_prox = min(event_prox, 10.0)  # cap
            else:
                event_prox = 0.0

            # 4. Quality risk
            if quality_loss is not None and j < quality_loss.shape[1]:
                ql_risk = float(quality_loss[b, j])
            else:
                ql_risk = 0.0

            # 5. Route load ratio
            load_ratio = vehicle_load / max(capacity, 1.0)

            # 6. Remaining budget
            budget_rem = 1.0 - current_cycle / max(max_cycles, 1)

            batch_feats.append([conf, tw_slack, event_prox, ql_risk, load_ratio, budget_rem])

        features_list.append(np.array(batch_feats, dtype=np.float32))

    return features_list  # list of (num_edges_b, 6) per batch


# ============================================================
# Rule-Based Mask Policy (D1 baseline — for comparison)
# ============================================================

def rule_mask_policy(edge_features_list, base_keep_rate=0.3):
    """
    D1 硬规则策略 (baseline): 距新订单越近 → mask 越多。
    """
    batch_masks = []
    for feats in edge_features_list:
        event_prox = feats[:, 2]  # column 2 = event_proximity
        max_prox = event_prox.max() + 1e-6
        event_factor = 1.0 - 0.7 * (event_prox / max_prox)  # high proximity → low keep
        keep_prob = base_keep_rate * np.clip(event_factor, 0.1, 1.0)
        mask_prob = 1.0 - keep_prob
        batch_masks.append(mask_prob)
    return batch_masks


# ============================================================
# Gumbel-Top-K: differentiable discrete sampling
# ============================================================

def gumbel_topk(logits: np.ndarray, k: int, temperature: float = 1.0,
                 rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """
    Differentiable top-K selection via Gumbel-Softmax trick.

    Args:
        logits: (E,) float, per-edge mask logits
        k: number of edges to mask
        temperature: Gumbel softmax temperature (lower = closer to hard)

    Returns:
        mask: (E,) float, soft mask in [0,1], approximately k ones
    """
    if rng is None:
        rng = np.random.default_rng()

    E = logits.shape[0]
    k = min(k, E)

    # Gumbel noise
    gumbel = -np.log(-np.log(rng.uniform(1e-10, 1.0, size=(E,)) + 1e-10))

    # Gumbel-Top-K
    noisy = (logits + gumbel) / temperature  # (E,)

    # Find k-th largest noisy value
    kth_val = np.partition(noisy, -k)[-k]

    # Soft threshold: sigmoid with temp
    soft_mask = 1.0 / (1.0 + np.exp(-(noisy - kth_val) / temperature))

    return soft_mask


# ============================================================
# Training utility: mask edges → decode → compute cost improvement
# ============================================================

class MaskPolicyTrainer:
    """
    轻量策略训练器。
    用 beam route cost 作 reward, REINFORCE 更新 mask 策略参数。
    策略参数 = 6 个特征权重 (线性模型), 存储为 numpy 数组。
    """

    def __init__(self, lr=0.01):
        # 6 features → 6 learned weights (linear policy)
        # Positive weight = more likely to mask edges with high feature value
        self.weights = np.zeros(6, dtype=np.float32)  # start from zero (uniform mask)
        self.lr = lr
        self.weight_history = [self.weights.copy()]

    def compute_logits(self, edge_features: np.ndarray) -> np.ndarray:
        """edge_features: (E, 6) → logits: (E,)"""
        # Normalize features to zero-mean unit-std per batch
        mean = edge_features.mean(axis=0, keepdims=True)
        std = edge_features.std(axis=0, keepdims=True) + 1e-6
        normalized = (edge_features - mean) / std
        # Linear policy
        return normalized @ self.weights  # (E, 6) @ (6,) → (E,)

    def update(self, edge_features: np.ndarray, mask: np.ndarray,
               cost_before: float, cost_after: float):
        """
        REINFORCE 单步更新。
        reward = cost_before - cost_after (降低成本 = 正 reward)
        注意: mask 是 mask probability, 不是 0/1。
        """
        reward = cost_before - cost_after  # positive = improvement
        logits = self.compute_logits(edge_features)

        # Gradient of log-likelihood for Bernoulli:
        # For each edge e: d/dw log P(mask[e]=1) = x[e] * (mask[e] - sigmoid(logit[e]))
        # But since we use Gumbel-softmax, use straight-through:
        # Treat mask as continuous, gradient = x[e] * (mask[e] - 0.5)
        # This is a rough approximation but works for small models

        mean = edge_features.mean(axis=0)
        std = edge_features.std(axis=0) + 1e-6
        normalized = (edge_features - mean) / std
        grad = (normalized * (mask - 0.5)[:, None]).mean(axis=0)

        # REINFORCE update
        self.weights += self.lr * reward * grad
        self.weight_history.append(self.weights.copy())

    def get_config(self):
        """返回可解释的权重配置。"""
        names = ['confidence', 'tw_slack', 'event_prox', 'quality_risk',
                 'route_load', 'budget_rem']
        return dict(zip(names, [float(w) for w in self.weights.tolist()]))
