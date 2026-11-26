# -*- coding: utf-8 -*-
"""worker 增加 numpy 标量清洗后写 JSON。"""
from pathlib import Path

p = Path(r"D:\PyCharm_\MASKCO-Main\research\paper_preparation_20261023\bottleneck_plan\final_paper_experiments\P5_standard_vrp_v2\scripts\p5v2_worker.py")
t = p.read_text(encoding="utf-8")

old = '''    check(row, version, block, sorted(os.sched_getaffinity(0)))
    (HERE / "out").mkdir(exist_ok=True)
    with (HERE / "out" / f"{idx}_{version}.json").open("x") as f:
        json.dump(row, f, allow_nan=False)'''
new = '''    check(row, version, block, sorted(os.sched_getaffinity(0)))
    (HERE / "out").mkdir(exist_ok=True)
    with (HERE / "out" / f"{idx}_{version}.json").open("x") as f:
        json.dump(_jsonable(row), f, allow_nan=False)'''
assert old in t
t = t.replace(old, new)

helper = '''

def _jsonable(obj):
    """递归把 numpy 标量/数组转为 Python 原生类型（JSON 兼容）。"""
    import numpy as np
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj
'''
t = t.replace('\n\ndef check(row, version, block, affinity):', helper + '\n\ndef check(row, version, block, affinity):')
p.write_text(t, encoding="utf-8")
print("patched jsonable")
