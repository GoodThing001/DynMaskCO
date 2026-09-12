"""
Phase 3b: Beam-Guided RL Fine-Tuning — 直接用 beam 路线成本作为 reward 信号。
不做复杂的 policy gradient，用 pairwise preference (beam-best vs beam-random):
  对每个训练实例: 取 beam 中的最低成本路线 vs 随机可行路线
  → 增加低成本路线的边 logits，减少高成本路线的边 logits
  → 等价于 DPO (Direct Preference Optimization) 的简化版

原理: 已有 beam decoder 100% 可行但 cost 高
      → 用 beam 生成 pairwise preference pairs
      → 微调模型使低 cost 路线的边概率上升
      → 保持可行性 (beam 的硬约束不变)
"""
import time, numpy as np


class BeamDPOTrainer:
    """Pairwise preference training on beam-generated feasible routes."""

    def __init__(self, model, graphdef, dataloader, beam_searcher_factory,
                 tx, savedir, logdir, encoder_input_dim=7, beta=0.1):
        self.model = model
        self.graphdef = graphdef
        self.params = None  # set after nnx.split
        self.dataloader = dataloader
        self.make_searcher = beam_searcher_factory
        self.tx = tx
        self.beta = beta         # DPO temperature
        self.savedir = savedir
        self.logdir = logdir
        self.encoder_input_dim = encoder_input_dim

    def _get_beam_routes(self, raw_features, num_pairs=2):
        """对每个 batch element: 用 beam 生成 K 条可行路线及其 cost。"""
        import jax, jax.numpy as jnp
        from modules.functional import coord_normalize

        # Encode
        @jax.jit
        def encode_fn(f):
            f = f.at[..., :2].set(coord_normalize(f[..., :2]))
            return self.model.encode(f[..., :self.encoder_input_dim])

        @jax.jit
        def logit_fn(feats, timestep):
            adjmat = jnp.zeros((1, 51, 51), dtype=jnp.int8)
            return self.model.decode(feats, timestep, adjmat.astype(jnp.float32))

        feats = np.array(encode_fn(jnp.array(raw_features)))
        logits = np.array(logit_fn(jnp.array(feats), jnp.full(feats.shape[0], 0.5)))

        routes_per_batch = []
        costs_per_batch = []

        for b in range(feats.shape[0]):
            logits_b = logits[b].copy()
            logits_b[:, 0] = -1e9
            np.fill_diagonal(logits_b, -1e9)
            searcher = self.make_searcher(b)
            routes = []
            costs = []
            for _ in range(num_pairs * 2):
                route, score = searcher.generate(logits_b, max_steps=200)
                if route and len(route) > 2:
                    prev, cost = 0, 0.0
                    for nd in route:
                        if nd <= 0 or nd >= 51:
                            continue
                        cost += searcher.dist[prev, nd]
                        prev = nd
                    routes.append(route)
                    costs.append(cost)
            routes_per_batch.append(routes)
            costs_per_batch.append(costs)

        return routes_per_batch, costs_per_batch, logits

    def train_step(self, step, max_steps=10000):
        """单步 DPO 训练。"""
        import jax, jax.numpy as jnp
        from flax import nnx
        import optax

        # This is a simplified version — full implementation would be JIT-compiled
        print(f"  RL fine-tuning: conceptual framework ready.")
        print(f"  Implementation: beam → K feasible routes → pairwise preference → DPO loss → gradient update")
        print(f"  Requires nnx.split/merge + JIT loop. Production code ~200 lines.")

    def run(self, num_steps=10000):
        print(f"Beam DPO Fine-Tuning: {num_steps} steps")
        print(f"  beta={self.beta}")

        for step in range(num_steps):
            for batch in self.dataloader:
                raw_features, target, timestep, visible_mask = batch
                raw_features = raw_features[..., :self.encoder_input_dim]
                self.train_step(step, num_steps)
                if step >= num_steps:
                    break
            if step >= num_steps:
                break
