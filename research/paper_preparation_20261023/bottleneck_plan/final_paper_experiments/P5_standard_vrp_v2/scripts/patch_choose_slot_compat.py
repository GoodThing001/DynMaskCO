from pathlib import Path
p = Path("standard_vrp/solver/masked_repair.py")
t = p.read_text(encoding="utf-8")
t = t.replace("move, reg = choose_slot(adapter, rebuilt, values, cust, heat=heat, regret=True)",
              "move, reg = _choose_slot(adapter, rebuilt, values, cust, heat=heat, regret=True)")
t = t.replace("move, _ = choose_slot(adapter, rebuilt, values, cust, heat=heat, regret=False)",
              "move, _ = _choose_slot(adapter, rebuilt, values, cust, heat=heat, regret=False)")
# 兼容包装：旧签名 (adapter, routes, cust, heat=..., regret=..., slot=..., topk=...)
old = "def insert_customer(adapter, routes, values, cust, move):"
new = '''def choose_slot(adapter, routes, values=None, cust=None, heat=None, regret=False,
                topk=8, slot=False, greedy_distance_slots=6, greedy_heat_slots=6):
    """兼容包装：旧签名 choose_slot(adapter, routes, cust, ...) 仍可用。

    返回 (key, move, regret)；key 为 adapter 比较键（CVRP 为 -detour，VRPTW 为 (K标记,detour)）。
    """
    if cust is None:
        cust, values = values, None
    if values is None:
        values = [route_value(adapter, r) for r in routes]
    move, regret_value = _choose_slot(adapter, routes, values, cust, heat=heat, regret=regret,
                                      topk=topk, greedy_distance_slots=greedy_distance_slots,
                                      greedy_heat_slots=greedy_heat_slots)
    if move[0] >= len(routes):
        key = adapter.key(True, adapter.singleton_delta(cust))
    else:
        key = adapter.key(False, adapter.detour(routes[move[0]], move[1], cust))
    return key, move, regret_value


def insert_customer(adapter, routes, values, cust, move):'''
assert old in t
t = t.replace(old, new, 1)
p.write_text(t, encoding="utf-8")
print("compat wrapper added")
