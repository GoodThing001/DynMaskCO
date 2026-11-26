from pathlib import Path
p = Path("standard_vrp/solver/pool_search.py")
t = p.read_text(encoding="utf-8")
# 1) 常量
old = "EPOCH_CAP_S"
anchor = t.index("# ---- 池搜索")
intro = t[:anchor]
assert 'CHECKPOINTS' not in t
t = t.replace("def standard_search(", "CHECKPOINTS = (5.0, 15.0, 30.0, 60.0)\n\n\ndef standard_search(", 1)
# 2) 初始化 history
t = t.replace("    split_blocks = split_orders = split_offers = 0\n    fallbacks = 0",
              "    split_blocks = split_orders = split_offers = 0\n    fallbacks = 0\n    history = {}", 1)
# 3) stats 加入 history
t = t.replace("                 fallbacks=fallbacks, predictor=pred.summary(),",
              "                 fallbacks=fallbacks, predictor=pred.summary(), history=history,", 1)
p.write_text(t, encoding="utf-8")
print("patched")
