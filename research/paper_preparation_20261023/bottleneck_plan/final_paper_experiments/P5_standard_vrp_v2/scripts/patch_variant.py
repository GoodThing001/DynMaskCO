# -*- coding: utf-8 -*-
"""为 pool_search 增加 inst/bugfix 与 variant 分支支持。"""
from pathlib import Path

p = Path(r"D:\PyCharm_\MASKCO-Main\research\paper_preparation_20261023\bottleneck_plan\final_paper_experiments\P5_standard_vrp_v2\standard_vrp\solver\pool_search.py")
t = p.read_text(encoding="utf-8")

# 1) 函数开头补 inst/bugfix
old = "    n = inst.dimension\n    started = time.perf_counter()"
new = "    inst = adapter.inst\n    n = inst.dimension\n    bugfix = variant == \"bugfix\"\n    started = time.perf_counter()"
assert old in t, "header pattern"
t = t.replace(old, new)

# 2) reset 仅 full
old = "        if epochs % RESET_PERIOD == 0:\n            state = [list(r) for r in best]"
new = "        if (not bugfix) and epochs % RESET_PERIOD == 0:\n            state = [list(r) for r in best]"
assert old in t, "reset pattern"
t = t.replace(old, new)

# 3) 跨车选择：bugfix = 2 条随机路线且不并入掩码
old = """        selected = ()
        if split_enabled and epochs % SPLIT_PERIOD == 0 and len(state) >= 2:
            selected = bundle(state, dist, int(nrng.integers(len(state))))"""
new = """        selected = ()
        if bugfix:
            # STD_BUGFIX：V1 语义——每 3 epoch 随机 2 条路线（总客户 ≤24），不并入掩码
            if split_enabled and epochs % SPLIT_PERIOD == 0 and len(state) >= 2:
                idx = int(nrng.integers(len(state)))
                cand = [k for k in range(len(state)) if k != idx and state[k]
                        and len(state[idx]) + len(state[k]) <= BUNDLE_CAP]
                if cand:
                    selected = tuple(sorted((idx, int(nrng.choice(cand)))))
        elif split_enabled and epochs % SPLIT_PERIOD == 0 and len(state) >= 2:
            selected = bundle(state, dist, int(nrng.integers(len(state))))"""
assert old in t, "bundle pattern"
t = t.replace(old, new)

# 4) 跨车顺序：bugfix = 单条最近邻链（V1 语义）
old = """            weights = heat if heat is not None else None
            for seq in propose_orders(parts, weights, seed * 100 + epochs, 100000):"""
new = """            weights = heat if heat is not None else None
            if bugfix:
                # V1 语义：热图/距离最近邻单链
                orders = [_nearest_chain(parts, heat, dist)]
            else:
                orders = propose_orders(parts, weights, seed * 100 + epochs, 100000)
            for seq in orders:"""
assert old in t, "orders pattern"
t = t.replace(old, new)

# 5) 冲突池/自适应仅 full
old = """        # ---- 自适应双模式 ----
        if epochs < 2 or epochs % RESET_PERIOD == 0:
            active_mode = int(np.argmin(uses))
        else:
            active_mode = int(np.argmax(rewards + 0.1 / np.sqrt(1 + uses)))
        uses[active_mode] += 1
        working_pool = conflict_pool(state, pool, heat) if (active_mode and heat is not None) else pool
        if not working_pool:
            working_pool = pool"""
new = """        # ---- 自适应双模式（仅 full） ----
        if bugfix:
            active_mode = 0
            uses[0] += 1
            working_pool = pool
        else:
            if epochs < 2 or epochs % RESET_PERIOD == 0:
                active_mode = int(np.argmin(uses))
            else:
                active_mode = int(np.argmax(rewards + 0.1 / np.sqrt(1 + uses)))
            uses[active_mode] += 1
            working_pool = conflict_pool(state, pool, heat) if (active_mode and heat is not None) else pool
            if not working_pool:
                working_pool = pool"""
assert old in t, "adaptive pattern"
t = t.replace(old, new)

# 6) 六配置：bugfix 全量（无时间片），否则 0.04s 时间片
old = """        if mode_name in ("hybrid", "choice") and end - time.perf_counter() > 0.30:
            trials = []
            for rk, opt in enumerate(options):
                for op in ("greedy", "regret2"):
                    if time.perf_counter() >= end - 0.005:
                        break
                    allowance = min(CONFIG_TRIAL_S, end - time.perf_counter() - 0.005)
                    if allowance <= 0:
                        continue"""
new = """        if (bugfix or mode_name in ("hybrid", "choice")) and end - time.perf_counter() > 0.30:
            trials = []
            for rk, opt in enumerate(options):
                for op in ("greedy", "regret2"):
                    if time.perf_counter() >= end - 0.005:
                        break
                    if bugfix:
                        allowance = max(0.0, end - time.perf_counter() - 0.005)
                        child, _ = masked_repair(adapter, [tuple(r) for r in state], set(working_pool),
                                                 ranking_for(adapter, state, working_pool, rk, opt),
                                                 op, heat=opt, slot=slot, deadline=None)
                        trials.append((score_of(child), op, rk, opt, child))
                        continue
                    allowance = min(CONFIG_TRIAL_S, end - time.perf_counter() - 0.005)
                    if allowance <= 0:
                        continue"""
assert old in t, "trial pattern"
t = t.replace(old, new)

# 7) 组内重构：bugfix 严格改善（无 SA）
old = """            if child_score >= cur_score - 1e-9 or \\
                    (child_score > best_score - 6 * temperature and
                     rng.random() < math.exp((child_score - cur_score) / max(temperature, 1e-9))):
                state = [list(r) for r in child]"""
new = """            if bugfix:
                if child_score > cur_score + 1e-9:
                    state = [list(r) for r in child]
            elif child_score >= cur_score - 1e-9 or \\
                    (child_score > best_score - 6 * temperature and
                     rng.random() < math.exp((child_score - cur_score) / max(temperature, 1e-9))):
                state = [list(r) for r in child]"""
assert old in t, "accept pattern"
t = t.replace(old, new)

# 8) 追加 _nearest_chain 辅助
t += '''

def _nearest_chain(parts, heat, dist):
    """V1 语义的跨车顺序：把路线块拼接后按热图（或距离）最近邻链排序。"""
    ids = [i for r in parts for i in r]
    left = set(ids)
    seq, cur = [], 0
    while left:
        if heat is not None:
            nxt = max(left, key=lambda i: (heat[cur][i] if math.isfinite(heat[cur][i]) else -math.inf))
            if not math.isfinite(heat[cur][nxt]):
                nxt = min(left, key=lambda i: dist[cur][i])
        else:
            nxt = min(left, key=lambda i: dist[cur][i])
        seq.append(nxt)
        left.remove(nxt)
        cur = nxt
    return seq
'''
p.write_text(t, encoding="utf-8")
print("patched variant support")
