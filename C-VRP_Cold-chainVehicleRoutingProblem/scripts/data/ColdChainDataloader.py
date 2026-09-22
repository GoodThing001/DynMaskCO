"""
冷链数据加载器 — 扩展 CVRPTWDataloader 增加 temp_class。
输出: features (batch, nodes, 6), target (batch, pad_len), timestep (batch,)
"""

import numpy as np
from typing import Mapping


class ColdChainDataloader:
    def __init__(
        self, datasets: Mapping[str, np.ndarray],
        batch_size: int, capacity: int | None = None,
        tw_max: float | None = None,
        target_disruption=None, need_current=False,
        data_augment=0, num_workers=2,
        need_edge_feat=False,
    ):
        datasets = dict(datasets)
        required = {'coords', 'demands', 'tw_start', 'tw_end', 'temp_class', 'routes'}
        missing = required - set(datasets.keys())
        if missing:
            raise KeyError(f"Missing: {missing}")

        self.coords = datasets['coords'].astype(np.float32)
        self.opt_routes = datasets['routes']
        cap = capacity if capacity is not None else datasets.get('capacity', 50)
        self.demands_norm = datasets['demands'].astype(np.float32) / cap

        if tw_max is None:
            tw_max = float(datasets['tw_end'].max())
        self.tw_start_norm = datasets['tw_start'].astype(np.float32) / tw_max
        self.tw_end_norm = datasets['tw_end'].astype(np.float32) / tw_max
        self.temp_class = datasets['temp_class'].astype(np.float32) / 2.0  # normalize to [0,1]

        self.service_time = datasets.get('service_time',
            np.zeros_like(datasets['demands'], dtype=np.float32))
        # Phase B: reveal_time for dynamic scenarios (7D)
        self.has_reveal_time = 'reveal_time' in datasets
        if self.has_reveal_time:
            self.reveal_time = datasets['reveal_time'].astype(np.float32) / tw_max
        self.has_visible_mask = 'visible_mask' in datasets
        if self.has_visible_mask:
            self.visible_mask = datasets['visible_mask'].astype(np.float32)
        self.has_zone_density = 'zone_density' in datasets
        if self.has_zone_density:
            self.zone_density = datasets['zone_density'].astype(np.float32)
        self.has_quality_loss = 'quality_loss' in datasets
        if self.has_quality_loss:
            self.quality_loss = datasets['quality_loss'].astype(np.float32)
        self.has_energy_mat = 'energy_mat' in datasets
        if self.has_energy_mat:
            self.energy_mat = datasets['energy_mat'].astype(np.float32)

        self.batch_size = batch_size
        self.num_instances = self.coords.shape[0]
        self.steps_per_epoch = self.num_instances // batch_size
        self.target_disruption = target_disruption
        self.need_current = need_current
        self.num_workers = num_workers
        self.need_edge_feat = need_edge_feat

    def __iter__(self):
        # 修复 2026-08-26：用全局 np.random（受 np.random.seed 控制），
        # 原 np.random.default_rng() 是独立随机源，不受训练入口的 seed 控制。
        while True:
            indices = np.random.permutation(self.num_instances)
            for start in range(0, self.num_instances - self.batch_size + 1, self.batch_size):
                idx = indices[start:start + self.batch_size]
                # 6D or 7D: [x, y, demand, tw_start, tw_end, temp_class, ?reveal_time]
                feat_list = [
                    self.coords[idx],
                    self.demands_norm[idx][..., None],
                    self.tw_start_norm[idx][..., None],
                    self.tw_end_norm[idx][..., None],
                    self.temp_class[idx][..., None],
                ]
                if self.has_reveal_time:
                    feat_list.append(self.reveal_time[idx][..., None])
                if self.has_zone_density:
                    feat_list.append(self.zone_density[idx][..., None])
                if self.has_quality_loss:
                    feat_list.append(self.quality_loss[idx][..., None])
                features = np.concatenate(feat_list, axis=-1).astype(np.float32)

                # 掩码不再在 dataloader 做，而是在训练器/解码器里按当前可见性（vis_k）
                # 逐步掩码。这样 progressive reveal 才能真正揭示新订单的特征。
                # （2026-08-27 修复 zeroed-features bug：原 t=0 掩码导致刚揭示订单的
                #   demand/TW/temp 永不被恢复，动态训练信号退化。）
                if self.has_visible_mask:
                    visible_b = self.visible_mask[idx]
                else:
                    visible_b = None

                # 未掩码的 reveal_time（归一化），供训练器 online_seq 计算 vis_k
                reveal_b = self.reveal_time[idx] if self.has_reveal_time else None

                timestep = np.random.uniform(0., 1., size=(self.batch_size,)).astype(np.float32)
                routes_b = self.opt_routes[idx]

                edge_mat_b = (self.energy_mat[idx] if (self.need_edge_feat and self.has_energy_mat)
                              else None)
                if self.need_current:
                    current = np.array([np.random.permutation(routes_b.shape[1])
                                         for _ in range(self.batch_size)])
                    yield features, routes_b, current, timestep, visible_b, reveal_b
                elif self.need_edge_feat:
                    yield features, routes_b, timestep, visible_b, edge_mat_b, reveal_b
                else:
                    yield features, routes_b, timestep, visible_b, reveal_b
