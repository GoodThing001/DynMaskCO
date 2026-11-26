# -*- coding: utf-8 -*-
"""修正 Solomon 写入格式（name / 空行 / VEHICLE / NUMBER CAPACITY / k cap / 空行 / CUSTOMER / 表头 / 数据）。"""
from pathlib import Path

p = Path(r"D:\PyCharm_\MASKCO-Main\research\paper_preparation_20261023\bottleneck_plan\final_paper_experiments\P5_standard_vrp_v2\scripts\gen_development_instances.py")
t = p.read_text(encoding="utf-8")
old = '''    H = windows[0][1]
    lines = [name, "P5-V2DEV", "VEHICLE", "NUMBER CAPACITY", "%d %d" % (k, capacity),
             "CUSTOMER", "CUST NO.  XCOORD.   YCOORD.    DEMAND   READY TIME  DUE DATE   SERVICE TIME"]'''
new = '''    lines = [name, "", "VEHICLE", "NUMBER CAPACITY", "%d %d" % (k, capacity), "",
             "CUSTOMER",
             "CUST NO.  XCOORD.   YCOORD.    DEMAND   READY TIME  DUE DATE   SERVICE TIME"]'''
assert old in t, "solomon header block not found"
t = t.replace(old, new)
p.write_text(t, encoding="utf-8")
print("patched solomon header")
