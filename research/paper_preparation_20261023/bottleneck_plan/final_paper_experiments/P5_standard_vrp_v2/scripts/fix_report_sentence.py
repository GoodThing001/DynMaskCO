from pathlib import Path
p = Path("FINAL_REPORT_V2.md")
t = p.read_text(encoding="utf-8")
old = "面板层面 STD_FIX 相对 STD_DIST 为 +0.74%（48 单元，24 优于 DIST），\n两者方向一致、量级不同（面板有搜索与跨车 Split 补偿）。"
new = "面板层面 STD_FIX 相对 STD_DIST 为 +1.26%（30/48 单元里 DIST 更优，中位 +0.41%），\n两者方向一致、量级不同（面板有搜索与跨车 Split 补偿）。"
assert old in t
t = t.replace(old, new)
p.write_text(t, encoding="utf-8")
print("fixed neural/distance sentence")
