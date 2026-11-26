import re
from pathlib import Path
p = Path("scripts/p5v2_bks_derive2.py")
t = p.read_text(encoding="utf-8")
old = """        if s.lower().startswith("number") and "capacity" in s.lower():
            cap=float(num.findall(s)[0]); continue"""
new = """        if s.lower().startswith("number") and "capacity" in s.lower():
            pend=True; continue
        if pend and cap is None:
            q=num.findall(s)
            if len(q)>=2:
                cap=float(q[-1]); pend=False; continue"""
assert old in t
t = t.replace(old, new).replace("rows=[]; cap=None", "rows=[]; cap=None; pend=False")
p.write_text(t, encoding="utf-8")
print("patched")
