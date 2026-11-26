# -*- coding: utf-8 -*-
"""为 bugfix 变体的六配置试选加轮内截止（避免大实例溢出在线预算）。"""
from pathlib import Path

p = Path(r"D:\PyCharm_\MASKCO-Main\research\paper_preparation_20261023\bottleneck_plan\final_paper_experiments\P5_standard_vrp_v2\standard_vrp\solver\pool_search.py")
t = p.read_text(encoding="utf-8")

old = """                    if bugfix:
                        allowance = max(0.0, end - time.perf_counter() - 0.005)
                        child, _ = masked_repair(adapter, [tuple(r) for r in state], set(working_pool),
                                                 ranking_for(adapter, state, working_pool, rk, opt),
                                                 op, heat=opt, slot=slot, deadline=None)
                        trials.append((score_of(child), op, rk, opt, child))
                        continue"""
new = """                    if bugfix:
                        # V1 语义：全量修复，但必须有轮内截止（大实例单次修复可达数秒）
                        allowance = max(0.0, end - time.perf_counter() - 0.01)
                        if allowance <= 0:
                            break
                        try:
                            child, _ = masked_repair(adapter, [tuple(r) for r in state],
                                                     set(working_pool),
                                                     ranking_for(adapter, state, working_pool, rk, opt),
                                                     op, heat=opt, slot=slot,
                                                     deadline=time.perf_counter() + allowance)
                        except _TimeUp:
                            continue
                        trials.append((score_of(child), op, rk, opt, child))
                        continue"""
assert old in t, "bugfix trial pattern"
t = t.replace(old, new)
p.write_text(t, encoding="utf-8")
print("patched bugfix deadline")
