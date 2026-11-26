import ast
from pathlib import Path

p = Path("standard_vrp/solver/masked_repair.py")
t = p.read_text(encoding="utf-8")

t = t.replace('''def _choose_slot(adapter, routes, values, cust, heat=None, regret=False,
                 topk=8, greedy_distance_slots=6, greedy_heat_slots=6):''',
'''def _choose_slot(adapter, routes, values, cust, heat=None, regret=False,
                 topk=8, greedy_distance_slots=6, greedy_heat_slots=6, deadline=None):''')

t = t.replace('''        n = len(route)
        for pos in range(n + 1):
            a = route[pos - 1] if pos else 0
            b = route[pos] if pos < n else 0
            detour = adapter.detour(route, pos, cust)
            slots.append((detour, _edge_score(heat, a, cust, b), ri, pos))''',
'''        n = len(route)
        for pos in range(n + 1):
            if deadline is not None and (pos & 7) == 0 and time.perf_counter() >= deadline:
                raise _TimeUp     # 大实例单次枚举可达数百毫秒：按槽位细粒度守卫
            a = route[pos - 1] if pos else 0
            b = route[pos] if pos < n else 0
            detour = adapter.detour(route, pos, cust)
            slots.append((detour, _edge_score(heat, a, cust, b), ri, pos))''')

t = t.replace("                move, reg = _choose_slot(adapter, rebuilt, values, cust, heat=heat, regret=True)",
              "                move, reg = _choose_slot(adapter, rebuilt, values, cust, heat=heat,\n"
              "                                         regret=True, deadline=deadline)")
t = t.replace("            move, reg = _choose_slot(adapter, rebuilt, values, cust, heat=heat, regret=False)",
              "            move, reg = _choose_slot(adapter, rebuilt, values, cust, heat=heat,\n"
              "                                     regret=False, deadline=deadline)")
p.write_text(t, encoding="utf-8")
ast.parse(t)

q = Path("standard_vrp/solver/pool_search.py")
t2 = q.read_text(encoding="utf-8")
t2 = t2.replace("    hard = started + seconds - 0.6",
                "    hard = started + seconds - (0.6 + 0.0008 * n)")
q.write_text(t2, encoding="utf-8")
ast.parse(t2)
print("per-slot deadline guard + adaptive margin")
