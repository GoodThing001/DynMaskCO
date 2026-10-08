# -*- coding: utf-8 -*-
"""Check relative markdown links under docs/当前规划 resolve to existing files."""
import os
import re
import sys

BASE = r"D:\PyCharm_\MASKCO-Main\C-VRP_Cold-chainVehicleRoutingProblem\docs\当前规划"
LINK = re.compile(r'\]\(([^()\s]+)\)')

broken = []
for root, dirs, files in os.walk(BASE):
    for fn in files:
        if not fn.endswith(".md"):
            continue
        p = os.path.join(root, fn)
        with open(p, encoding="utf-8") as f:
            s = f.read()
        for m in LINK.findall(s):
            t = m.partition("#")[0]
            if (not t or t.startswith(("http://", "https://", "mailto:"))
                    or re.match(r'^[A-Za-z]:', t)):
                continue
            target = os.path.normpath(os.path.join(os.path.dirname(p), t))
            if not os.path.exists(target):
                broken.append((os.path.relpath(p, BASE), t))
for rel, t in sorted(set(broken)):
    print("BROKEN  %s  ->  %s" % (rel, t))
print("total broken:", len(set(broken)))
sys.exit(0)
