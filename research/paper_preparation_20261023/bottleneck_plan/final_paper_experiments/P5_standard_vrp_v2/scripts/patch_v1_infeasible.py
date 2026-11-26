import ast
from pathlib import Path

p = Path("scripts/p5v2_worker.py")
t = p.read_text(encoding="utf-8")

t = t.replace('''    allowed = ("AUDITED", "TRUNCATED_OK")
    if version == "STD_OLD":
        allowed += ("V1_CRASH",)   # 冻结 V1 代码在部分实例上崩溃：如实登记，不掩盖''',
'''    allowed = ("AUDITED", "TRUNCATED_OK")
    if version == "STD_OLD":
        # 冻结 V1 代码在部分实例上崩溃或产出不可行解：如实登记，不掩盖、不修复 V1
        allowed += ("V1_CRASH", "V1_INFEASIBLE")''')
t = t.replace('''    if row["status"] != "V1_CRASH" and not 0 <= row["online_s"] <= 60:''',
'''    if row["status"] not in ("V1_CRASH",) and not 0 <= row["online_s"] <= 60:''')

t = t.replace('''        routes = [tuple(r) for r in out["routes"]]
        ok, val, k = adapter.evaluate(routes)
        row = dict(case=case, seed=seed, version=version, objective=objective,
                   status="AUDITED" if ok else "INFEASIBLE", distance=val, vehicles=k,
                   routes=routes, online_s=online_s, stats=dict(epochs=out.get("epochs")))''',
'''        routes = [tuple(r) for r in out["routes"]]
        ok, val, k = adapter.evaluate(routes)
        row = dict(case=case, seed=seed, version=version, objective=objective,
                   status="AUDITED" if ok else "V1_INFEASIBLE", distance=val, vehicles=k,
                   routes=routes, online_s=online_s, stats=dict(epochs=out.get("epochs")))
        if not ok:
            row["violation"] = _violation(inst, routes)''')

t = t.replace('''def make_predictor():''',
'''def _violation(inst, routes):
    """VRPTW 不可行度（分钟）：最大时窗/仓库窗超限；CVRP 返回容量超限。"""
    try:
        from standard_vrp.core.standard_evaluator import CVRPInstance
        if isinstance(inst, CVRPInstance):
            worst = 0.0
            for r in routes:
                load = sum(inst.demands[i] for i in r)
                worst = max(worst, load - inst.capacity)
            return dict(kind="capacity", worst=round(worst, 6))
        worst = 0.0
        for r in routes:
            t = inst.depot_window[0]
            prev = 0
            for i in r:
                t += inst.dist[prev][i]
                if t < inst.windows[i][0]:
                    t = inst.windows[i][0]
                worst = max(worst, t - inst.windows[i][1])
                t += inst.service[i]
                prev = i
            t += inst.dist[prev][0]
            worst = max(worst, t - inst.depot_window[1])
        return dict(kind="window", worst=round(worst, 6))
    except Exception:  # noqa: BLE001
        return dict(kind="unknown", worst=None)


def make_predictor():''')
p.write_text(t, encoding="utf-8")
ast.parse(t)
print("V1_INFEASIBLE registered; syntax ok")
