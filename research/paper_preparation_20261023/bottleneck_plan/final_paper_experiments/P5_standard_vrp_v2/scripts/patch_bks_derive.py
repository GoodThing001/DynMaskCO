import re
from pathlib import Path
p = Path("scripts/p5v2_bks_derive2.py")
t = p.read_text(encoding="utf-8")
t = t.replace("""    cap=float(re.search(r"CAPACITY\\s*:\\s*(\\d+)",text,re.I).group(1))
    return coords, dem, cap""",
"""    cap=float(re.search(r"CAPACITY\\s*:\\s*(\\d+)",text,re.I).group(1))
    dep=coords[1]
    fixed={0: dep}
    fixed.update({k: v for k, v in coords.items() if k != 1})
    return fixed, dem, cap""")
p.write_text(t, encoding="utf-8")
print("ok")
