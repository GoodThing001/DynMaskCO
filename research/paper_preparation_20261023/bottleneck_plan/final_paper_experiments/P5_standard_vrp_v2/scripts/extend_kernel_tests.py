from pathlib import Path
p = Path("standard_vrp/tests/test_kernels_v2.py")
t = p.read_text(encoding="utf-8")
t += '''

def test_deadline_large_instance_both_variants():
    """大实例（n=200）下两种变体都不得超出在线预算（含硬截止守卫）。"""
    inst = _cvrp(200, 60.0, 21)
    adapter = adapter_for(inst)
    for variant in ("full", "bugfix"):
        t0 = time.perf_counter()
        out = standard_search(adapter, seconds=4, seed=9, heatmap_fn=None,
                              split_enabled=True, slot=False, variant=variant)
        elapsed = time.perf_counter() - t0
        assert elapsed <= 4.0 + 1.0, (variant, elapsed)
        ok, val, k = adapter.evaluate([tuple(r) for r in out["routes"]])
        assert ok and k >= 1
        assert set(out["history"]) == {"5", "15", "30", "60"}


def test_repair_batch_related_and_acceptance():
    """related 模式（偶数次迭代亲和度掩码）与 SA 接受须可运行且不改坏记录解。"""
    from standard_vrp.solver.masked_repair import repair_batch
    inst = _cvrp(60, 40.0, 31)
    adapter = adapter_for(inst)
    from standard_vrp.solver.pool_search import nn_witness
    state = [list(r) for r in nn_witness(inst)]
    ok, val, k = adapter.evaluate([tuple(r) for r in state])
    assert ok
    res = repair_batch(adapter, state, state, list(range(1, 41)), heat=None, seed=5,
                       seconds=0.4, tolerance=1.5, related=True, regret=False)
    ok2, val2, k2 = adapter.evaluate([tuple(r) for r in res["best"]])
    assert ok2
    assert val2 <= val + 1e-6, (val2, val)
    assert res["iterations"] >= 1
    res2 = repair_batch(adapter, state, state, list(range(1, 41)), heat=None, seed=7,
                        seconds=0.4, tolerance=1.5, related=True, regret=True)
    ok3, val3, k3 = adapter.evaluate([tuple(r) for r in res2["best"]])
    assert ok3 and val3 <= val + 1e-6
'''
p.write_text(t, encoding="utf-8")
print("tests extended")
