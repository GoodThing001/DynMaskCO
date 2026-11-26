import ast
from pathlib import Path

p = Path("standard_vrp/solver/pool_search.py")
t = p.read_text(encoding="utf-8")
old_tail = """        if best is not None:
            routes[best[1]].insert(best[2], cust)
        else:
            routes.append([cust])
    return tuple(tuple(r) for r in routes)"""
new_tail = """        if best is not None:
            routes[best[1]].insert(best[2], cust)
        else:
            routes.append([cust])
    limit = getattr(inst, "max_vehicles", None)
    if limit is not None:
        routes = _shrink_fleet(inst, routes, limit)
    return tuple(tuple(r) for r in routes)


def _shrink_fleet(inst, routes, limit):
    """车队压缩：把短线客户并入其他路线，直到车辆数 ≤ 限额或无法再并入。"""
    routes = [list(r) for r in routes if r]
    while len(routes) > limit:
        sizes = sorted(range(len(routes)), key=lambda k: len(routes[k]))
        moved_any = False
        for slot in sizes:
            source = routes[slot]
            plan, ok = [], True
            for cust in list(source):
                best = None
                for ri, route in enumerate(routes):
                    if ri == slot:
                        continue
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
                if best is None:
                    ok = False
                    break
                plan.append((cust, best[1], best[2]))
            if ok:
                for cust, ri, pos in plan:
                    routes[ri].insert(pos, cust)
                routes.pop(slot)
                moved_any = True
                break
        if not moved_any:
            break
    return routes"""
assert old_tail in t
t = t.replace(old_tail, new_tail)
p.write_text(t, encoding="utf-8")
ast.parse(t)
print("fleet shrink added; syntax ok")
