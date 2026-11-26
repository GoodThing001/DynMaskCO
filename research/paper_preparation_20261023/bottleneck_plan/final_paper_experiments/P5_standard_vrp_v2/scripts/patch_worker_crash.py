from pathlib import Path
p = Path("scripts/p5v2_worker.py")
t = p.read_text(encoding="utf-8")

# 1) check：允许 STD_OLD 的 V1_CRASH 登记
t = t.replace("""    if row["status"] not in ("AUDITED", "TRUNCATED_OK"):
        raise ValueError("P5-V2 infeasible endpoint: " + str(row["status"]))""",
"""    allowed = ("AUDITED", "TRUNCATED_OK")
    if version == "STD_OLD":
        allowed += ("V1_CRASH",)   # 冻结 V1 代码在部分实例上崩溃：如实登记，不掩盖
    if row["status"] not in allowed:
        raise ValueError("P5-V2 infeasible endpoint: " + str(row["status"]))""")
t = t.replace("""    if not 0 <= row["online_s"] <= 60:
        raise ValueError("online budget exceeded: %.2f" % row["online_s"])""",
"""    if row["status"] != "V1_CRASH" and not 0 <= row["online_s"] <= 60:
        raise ValueError("online budget exceeded: %.2f" % row["online_s"])""")

# 2) STD_OLD 路径包 try/except 记录崩溃
old = """        heat = make_heatmap_v1(forward, inst, objective, v1fb_mod)
        t0 = time.perf_counter()
        out = v1.standard_search(inst, seconds=60, seed=seed, objective=objective,
                                 heatmap_fn=heat, split_enabled=True, slot=False)
        online_s = time.perf_counter() - t0
        adapter = adapter_for(inst)
        routes = [tuple(r) for r in out["routes"]]
        ok, val, k = adapter.evaluate(routes)
        row = dict(case=case, seed=seed, version=version, objective=objective,
                   status="AUDITED" if ok else "INFEASIBLE", distance=val, vehicles=k,
                   routes=routes, online_s=online_s, stats=dict(epochs=out.get("epochs")))
        init = [tuple(r) for r in nn_witness(inst)]"""
new = """        heat = make_heatmap_v1(forward, inst, objective, v1fb_mod)
        adapter = adapter_for(inst)
        from standard_vrp.solver.pool_search import nn_witness
        init = [tuple(r) for r in nn_witness(inst)]
        t0 = time.perf_counter()
        try:
            out = v1.standard_search(inst, seconds=60, seed=seed, objective=objective,
                                     heatmap_fn=heat, split_enabled=True, slot=False)
        except Exception as exc:  # noqa: BLE001 —— 冻结 V1 代码的缺陷如实登记
            import traceback
            online_s = time.perf_counter() - t0
            row = dict(case=case, seed=seed, version=version, objective=objective,
                       status="V1_CRASH", distance=None, vehicles=None, routes=[],
                       online_s=online_s, stats=dict(epochs=None),
                       error="%s: %s" % (type(exc).__name__, exc),
                       traceback=traceback.format_exc()[-2000:])
            ok0, val0, k0 = adapter.evaluate(init)
            row.update(initial_routes=init, initial_U=dict(distance=val0, vehicles=k0))
            check(row, version, block, sorted(os.sched_getaffinity(0)))
            (HERE / "out").mkdir(exist_ok=True)
            with (HERE / "out" / f"{idx}_{version}.json").open("x") as f:
                json.dump(_jsonable(row), f, allow_nan=False)
            return
        online_s = time.perf_counter() - t0
        routes = [tuple(r) for r in out["routes"]]
        ok, val, k = adapter.evaluate(routes)
        row = dict(case=case, seed=seed, version=version, objective=objective,
                   status="AUDITED" if ok else "INFEASIBLE", distance=val, vehicles=k,
                   routes=routes, online_s=online_s, stats=dict(epochs=out.get("epochs")))"""
assert old in t
t = t.replace(old, new)
# STD_OLD 分支里重复的 import 与 init 行清理
t = t.replace("""        from standard_vrp.adapters.objective import adapter_for
        from standard_vrp.solver.pool_search import nn_witness
        import importlib.util as iu""",
"""        from standard_vrp.adapters.objective import adapter_for
        import importlib.util as iu""")
p.write_text(t, encoding="utf-8")
print("worker crash handling added")
