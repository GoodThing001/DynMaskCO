import ast
from pathlib import Path

p = Path("standard_vrp/solver/pool_search.py")
t = p.read_text(encoding="utf-8")

start = t.index("def nn_witness(inst):")
end = t.index("def _ranks(heat):")
new = '''def nn_witness(inst):
    """共同初解（确定性构造，必须覆盖全部客户）。

    CVRP：最近邻链 + 容量断链（原语义）。
    VRPTW：按 ready time 升序做"最优可行插位"构造（缺此步骤时紧时窗实例会被漏客，
    造成共同初解不可行——2026-11-26 开发面板复验发现，见 BUGFIX_REPORT D7）。
    两条路径都不使用随机数，保证所有版本共用同一初解。
    """
    n = inst.dimension
    if not isinstance(inst, VRPTWInstance):
        limit = getattr(inst, "max_vehicles", None)
        remaining = set(range(1, n + 1))
        routes = []
        while remaining:
            route, cur = [], 0
            while remaining:
                nxt = min(remaining, key=lambda i: inst.dist[cur][i])
                trial = route + [nxt]
                if sum(inst.demands[i] for i in trial) > inst.capacity + 1e-9:
                    break
                route.append(nxt)
                remaining.remove(nxt)
                cur = nxt
            if route:
                routes.append(tuple(route))
            else:  # 容量不足以单独成线（异常实例）：单独成车并继续
                only = min(remaining)
                routes.append((only,))
                remaining.discard(only)
        return tuple(routes)

    order = sorted(range(1, n + 1),
                   key=lambda i: (inst.windows[i][0], inst.windows[i][1], i))
    routes = []          # list[list[int]]
    for cust in order:
        best = None
        for ri, route in enumerate(routes):
            for pos in range(len(route) + 1):
                trial = route[:pos] + [cust] + route[pos:]
                if route_schedule(inst, tuple(trial)) is None:
                    continue
                seq = [0] + route + [0]
                delta = (inst.edge_distance(seq[pos], cust)
                         + inst.edge_distance(cust, seq[pos + 1])
                         - inst.edge_distance(seq[pos], seq[pos + 1]))
                if best is None or delta < best[0] - 1e-12:
                    best = (delta, ri, pos)
        if best is not None:
            routes[best[1]].insert(best[2], cust)
        else:
            routes.append([cust])
    return tuple(tuple(r) for r in routes)


'''
t = t[:start] + new + t[end:]
p.write_text(t, encoding="utf-8")
ast.parse(t)
print("nn_witness rewritten; syntax ok")
