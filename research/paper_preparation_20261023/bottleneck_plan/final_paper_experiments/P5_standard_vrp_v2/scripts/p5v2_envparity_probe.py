import json, math, re, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
num = re.compile(r"-?\d+(?:\.\d+)?")
D = Path("data")
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
    return [[int(x) for x in re.findall(r"\d+", l.split(":",1)[1])] for l in p.read_text().splitlines()
            if ":" in l and re.findall(r"\d+", l.split(":",1)[1])]
def check(rows, R, mode):
    xy={i:(r[1],r[2]) for i,r in enumerate(rows)}
    worst=0.0
    for r in R:
        t=rows[0][4]; prev=0
        for c in r:
            d=math.hypot(xy[prev][0]-xy[c][0], xy[prev][1]-xy[c][1])
            if mode=="trunc1": d=math.floor(d*10)/10
            elif mode=="round1": d=round(d,1)
            t=max(t+d, rows[c][4])
            worst=max(worst, t-rows[c][5])
            t+=rows[c][6]; prev=c
        d=math.hypot(xy[prev][0]-xy[0][0], xy[prev][1]-xy[0][1])
        if mode=="trunc1": d=math.floor(d*10)/10
        elif mode=="round1": d=round(d,1)
        worst=max(worst, t+d-rows[0][5])
    return worst
res={}
for mode in ("double","trunc1","round1"):
    bad=[]
    for sol in sorted((D/"solomon/bks").glob("*.sol")):
        rows,cap=parse_solomon(D/"solomon/instances"/(sol.stem+".vrp"))
        w=check(rows, routes(sol), mode)
        if w>1e-9: bad.append((sol.stem, round(w,4)))
    res[mode]=bad
    print(mode, "infeasible", len(bad), bad[:8])
