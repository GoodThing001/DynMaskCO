import math, re, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
num=re.compile(r"-?\d+(?:\.\d+)?")
def parse_solomon(p):
    rows=[]; cap=None; pend=False
    for line in p.read_text(encoding="utf-8",errors="replace").splitlines():
        s=line.strip()
        if not s: continue
        if s.lower().startswith("number") and "capacity" in s.lower():
            pend=True; continue
        if pend and cap is None:
            q=num.findall(s)
            if len(q)>=2: cap=float(q[-1]); pend=False; continue
        q=num.findall(s)
        if cap is not None and len(q)==7: rows.append([float(x) for x in q])
    return rows, cap
def routes(p):
    return [[int(x) for x in re.findall(r"\d+", l.split(":",1)[1])] for l in p.read_text().splitlines() if ":" in l and re.findall(r"\d+", l.split(":",1)[1])]
for name in ("C101","R101","RC101"):
    rows,cap=parse_solomon(Path("data/solomon/instances")/(name+".vrp"))
    R=routes(Path("data/solomon/bks")/(name+".sol"))
    xy={i:(r[1],r[2]) for i,r in enumerate(rows)}
    def tot(f):
        t=0.0
        for r in R:
            prev=0
            for c in r+[0]:
                d=math.hypot(xy[prev][0]-xy[c][0], xy[prev][1]-xy[c][1])
                t+=f(d); prev=c
        return t
    print(name, "K",len(R),
          "float", round(tot(lambda d:d),3),
          "round0", round(tot(lambda d:int(d+0.5)),3),
          "floor0", round(tot(lambda d:math.floor(d)),3),
          "r1", round(tot(lambda d:round(d,1)),3),
          "r2", round(tot(lambda d:round(d,2)),3),
          "trunc1", round(tot(lambda d:math.floor(d*10)/10),3),
          "cost_line", re.search(r"Cost\s+(-?\d+(?:\.\d+)?)", (Path("data/solomon/bks")/(name+".sol")).read_text()).group(1))
