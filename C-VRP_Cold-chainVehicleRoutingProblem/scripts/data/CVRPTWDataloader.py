"""
CVRPTW 数据加载器 — 简化版（NumPy 迭代器，无 TF 依赖）。

数据格式 (.npz):
- coords:       (N, nodes, 2) float32
- demands:      (N, nodes)    int32
- tw_start:     (N, nodes)    float32
- tw_end:       (N, nodes)    float32
- service_time: (N, nodes)    float32 (可选)
- routes:       (N, pad_len)  int32

输出: features (batch, nodes, 5), target (batch, pad_len), timestep (batch,)
"""

import numpy as np
from typing import Mapping


class CVRPTWDataloader:

    def __init__(
        self,
        datasets: Mapping[str, np.ndarray],
        batch_size: int,
        capacity: int | None = None,
        tw_max: float | None = None,
        target_disruption: tuple[int, int] | None = None,
        need_current: bool = False,
        data_augment: int = 0,
        num_workers: int = 2,
    ):
        datasets = dict(datasets)
        required = {'coords', 'demands', 'tw_start', 'tw_end', 'routes'}
        missing = required - set(datasets.keys())
        if missing:
            raise KeyError(f"Missing dataset keys: {missing}")

        self.coords = datasets['coords'].astype(np.float32)
        self.opt_routes = datasets['routes']

        # demand 归一化
        cap = capacity if capacity is not None else datasets.get('capacity', 50)
        self.demands_norm = (datasets['demands'].astype(np.float32) / cap)

        # tw 归一化
        if tw_max is None:
            tw_max = float(datasets['tw_end'].max())
        self.tw_start_norm = datasets['tw_start'].astype(np.float32) / tw_max
        self.tw_end_norm = datasets['tw_end'].astype(np.float32) / tw_max

        # service_time
        self.service_time = datasets.get(
            'service_time',
            np.zeros_like(datasets['demands'], dtype=np.float32)
        )

        self.batch_size = batch_size
        self.num_instances = self.coords.shape[0]
        self.steps_per_epoch = self.num_instances // batch_size
        self.target_disruption = target_disruption
        self.need_current = need_current
        self.num_workers = num_workers

    def __iter__(self):
        """纯 NumPy 迭代器，按 epoch 输出 (features, target, timestep)。"""
        rng = np.random.default_rng()
        while True:  # 无限 epoch
            indices = rng.permutation(self.num_instances)
            for start in range(0, self.num_instances - self.batch_size + 1, self.batch_size):
                idx = indices[start:start + self.batch_size]

                coords_b = self.coords[idx]                        # (B, nodes, 2)
                dem_b = self.demands_norm[idx]                     # (B, nodes)
                tws_b = self.tw_start_norm[idx]                    # (B, nodes)
                twe_b = self.tw_end_norm[idx]                      # (B, nodes)
                routes_b = self.opt_routes[idx]                    # (B, pad_len)

                # 5D features: [x, y, demand, tw_start, tw_end]
                features = np.concatenate([
                    coords_b,
                    dem_b[..., None],
                    tws_b[..., None],
                    twe_b[..., None],
                ], axis=-1).astype(np.float32)

                timestep = rng.uniform(0., 1., size=(self.batch_size,)).astype(np.float32)

                if self.need_current:
                    current = np.array([np.random.permutation(routes_b.shape[1])
                                         for _ in range(self.batch_size)])
                    yield features, routes_b, current, timestep
                else:
                    yield features, routes_b, timestep
