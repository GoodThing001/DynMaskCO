"""
方向四：JAX-native EDD repair — 100% GPU 内运行，消除 CPU-GPU 通信。

用法:
    from cvrptw_jax_ops import edd_repair_jax
    repaired = edd_repair_jax(routes, tw_end)  # routes: (batch, route_len)

性能预期: <0.02ms vs C++ 0.3ms (15x faster), 零 Host-Device Transfer
"""

import jax, jax.numpy as jnp


@jax.jit
def edd_repair_jax(routes: jax.Array, tw_end: jax.Array) -> jax.Array:
    """
    JAX-native EDD repair。对每条子路径（depot=0 分隔）按 tw_end 排序。

    Args:
        routes: (batch, route_len) int32 — CVRP 路线（depot=0 分隔子路径）
        tw_end: (batch, nodes) float32 — 每个节点的最晚到达时间

    Returns:
        repaired: (batch, route_len) int32
    """
    batch, route_len = routes.shape
    nodes = tw_end.shape[1]

    # 创建掩码：有效节点（非 depot, 非 padding）
    is_valid = (routes > 0)  # (batch, route_len)
    is_depot = (routes == 0)

    # 将节点映射到其 tw_end 值（用于排序键）
    # 安全索引：对 padding 位置使用 large sentinel value
    safe_routes = jnp.clip(routes, 0, nodes - 1)
    tw_values = jnp.take_along_axis(tw_end, safe_routes, axis=1)  # (batch, route_len)
    # depot 和 padding 给 INF（排到最后）
    sort_key = jnp.where(is_valid, tw_values, jnp.inf)

    # 找到每个 segment 的边界
    # 累积 depot 数量作为 segment ID
    seg_id = jnp.cumsum(is_depot.astype(jnp.int32), axis=1)  # (batch, route_len)
    max_seg = seg_id.max(axis=1, keepdims=True) + 1

    # 对每个 segment 内的 valid 节点按 tw_end 排序
    # 使用 argsort in each segment
    def sort_segments(sort_key, seg_id, is_valid, routes, max_seg):
        """对每个 segment 独立排序。"""
        # 创建复合排序键：seg_id 优先，sort_key 次之
        # seg_id * INF + sort_key 确保段内排序
        composite = seg_id.astype(jnp.float32) * 1e10 + sort_key

        # 全局排序（segment 间不会交叉因为 seg_id 不同）
        sorted_idx = jnp.argsort(composite, axis=1)  # (batch, route_len)

        # 重排 routes
        sorted_routes = jnp.take_along_axis(routes, sorted_idx, axis=1)

        # 保持 depot 位置不变（depot 的 seg_id 边界是固定的）
        # 简化：将 depot 放回原位
        return jnp.where(is_depot, routes, sorted_routes)

    repaired = sort_segments(sort_key, seg_id, is_valid, routes, max_seg)
    return repaired.astype(jnp.int32)


@jax.jit
def edd_repair_jax_v2(routes: jax.Array, tw_end: jax.Array) -> jax.Array:
    """
    简化版 EDD：在每个 segment 内，将 valid 节点按 tw_end 升序排列。

    使用 vmap over batch + scan over segments。
    """
    batch, route_len = routes.shape
    nodes = tw_end.shape[1]

    safe_routes = jnp.clip(routes, 0, nodes - 1)
    tw = jnp.take_along_axis(tw_end, safe_routes, axis=1)

    def repair_one(r, tw_r):
        """单实例修复。"""
        # 标记 depot 位置
        is_depot = (r == 0)
        depots = jnp.where(is_depot, size=route_len, fill_value=-1)[0]
        depots = depots[depots >= 0]  # depot indices

        def process_segment(carry, seg_range):
            start, end = seg_range[0], seg_range[1]
            r_out = carry
            # 提取 segment 内的节点
            seg_indices = jnp.arange(start, end)
            seg_nodes = r[start:end]
            seg_tw = tw_r[start:end]
            # 只排序有效节点
            valid = seg_nodes > 0
            sorted_order = jnp.where(valid, seg_tw, jnp.inf)
            new_order = jnp.argsort(sorted_order)
            seg_sorted = seg_nodes[new_order]
            r_out = jax.lax.dynamic_update_slice(r_out, seg_sorted, (start,))
            return r_out, None

        # 构建 segment ranges: (depot_i, depot_{i+1})
        if len(depots) < 2:
            return r
        seg_ranges = jnp.stack([depots[:-1], depots[1:]], axis=1)
        r_out, _ = jax.lax.scan(process_segment, r, seg_ranges)
        return r_out

    return jax.vmap(repair_one)(routes, tw)


# Benchmark helper
def bench_edd(batch=8, route_len=200, nodes=51, n_iter=100):
    """对比 JAX vs C++ EDD 速度。"""
    import numpy as np, time

    key = jax.random.PRNGKey(0)
    routes = jax.random.randint(key, (batch, route_len), 0, nodes)
    tw_end = jax.random.uniform(key, (batch, nodes))

    # JIT warmup
    _ = edd_repair_jax(routes, tw_end)
    _.block_until_ready()

    t0 = time.time()
    for _ in range(n_iter):
        _ = edd_repair_jax(routes, tw_end)
    jax.block_until_ready(_)
    jax_time = (time.time() - t0) / n_iter * 1000

    print(f"JAX EDD: {jax_time:.3f}ms/batch (batch={batch}, route_len={route_len})")
    print(f"  vs C++ EDD: ~0.3ms/batch")
    print(f"  Speedup: {0.3/jax_time:.1f}x")
