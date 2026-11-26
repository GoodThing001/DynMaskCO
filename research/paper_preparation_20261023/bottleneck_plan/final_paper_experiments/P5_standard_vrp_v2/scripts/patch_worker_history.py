from pathlib import Path
p = Path("scripts/p5v2_worker.py")
t = p.read_text(encoding="utf-8")
old = '"mask_mode_counts", "fallbacks", "predictor")})'
new = '"mask_mode_counts", "fallbacks", "predictor",\n                                                 "history")})'
assert old in t
t = t.replace(old, new, 1)
p.write_text(t, encoding="utf-8")
print("worker patched")
