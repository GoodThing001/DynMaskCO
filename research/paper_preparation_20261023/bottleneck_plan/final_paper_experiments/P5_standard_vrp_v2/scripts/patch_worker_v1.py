# -*- coding: utf-8 -*-
"""修正 V2 worker：V2 包优先；STD_OLD 用 importlib 显式加载冻结 V1 模块。"""
from pathlib import Path

p = Path(r"D:\PyCharm_\MASKCO-Main\research\paper_preparation_20261023\bottleneck_plan\final_paper_experiments\P5_standard_vrp_v2\scripts\p5v2_worker.py")
t = p.read_text(encoding="utf-8")

old = '''V1_ROOT = P5  # 冻结的 P5-V1 包根（standard_vrp）
sys.path.insert(0, str(P5V2))
sys.path.insert(0, str(V1_ROOT))'''
new = '''V1_ROOT = P5  # 冻结的 P5-V1 包根（standard_vrp）
sys.path.insert(0, str(V1_ROOT))   # 供 V1 模块内部 import
sys.path.insert(0, str(P5V2))      # V2 包优先解析 standard_vrp'''
assert old in t
t = t.replace(old, new)

old2 = '''    if version == "STD_OLD":
        # 冻结 P5-V1 代码路径
        import standard_vrp.solver.standard_pool_search as v1
        fb = StandardFeatureBuilder(inst, objective, mode="v2_fix")
        heat = make_heatmap_v1(forward, inst, objective)'''
new2 = '''    if version == "STD_OLD":
        # 冻结 P5-V1 代码路径：显式按文件加载 V1 特征与搜索模块
        import importlib.util as iu
        v1feat = iu.spec_from_file_location(
            "v1_standard_features", str(V1_ROOT / "standard_vrp/core/standard_features.py"))
        v1fb_mod = iu.module_from_spec(v1feat)
        v1feat.loader.exec_module(v1fb_mod)
        v1search = iu.spec_from_file_location(
            "v1_standard_pool_search", str(V1_ROOT / "standard_vrp/solver/standard_pool_search.py"))
        v1 = iu.module_from_spec(v1search)
        v1search.loader.exec_module(v1)
        heat = make_heatmap_v1(forward, inst, objective, v1fb_mod)'''
assert old2 in t
t = t.replace(old2, new2)

old3 = '''def make_heatmap_v1(forward, inst, objective):
    """STD_OLD：冻结 P5-V1 特征构造器（v1 编码）。"""
    from standard_vrp.core.standard_features import StandardFeatureBuilder as V1FB, WindowedFeatureBuilder as V1WFB
    fb = V1FB(inst, objective)
    wfb = V1WFB(inst, objective)'''
new3 = '''def make_heatmap_v1(forward, inst, objective, mod):
    """STD_OLD：冻结 P5-V1 特征构造器（v1 编码，显式模块）。"""
    fb = mod.StandardFeatureBuilder(inst, objective)
    wfb = mod.WindowedFeatureBuilder(inst, objective)'''
assert old3 in t
t = t.replace(old3, new3)
p.write_text(t, encoding="utf-8")
print("patched worker imports")
