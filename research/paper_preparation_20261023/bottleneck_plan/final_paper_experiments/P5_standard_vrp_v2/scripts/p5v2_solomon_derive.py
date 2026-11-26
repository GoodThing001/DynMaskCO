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

rows_out=[]
for sol in sorted((D/"solomon/bks").glob("*.sol")):
    name=sol.stem
    rows, cap = parse_solomon(D/"solomon/instances"/(name+".vrp"))
    xy={i:(r[1],r[2]) for i,r in enumerate(rows)}
    R=routes(sol)
    Dtot=0.0; maxviol=0.0; capviol=0.0; loadmax=0.0
    for r in R:
        t=rows[0][4]; prev=0; load=0.0
        for c in r:
            load+=rows[c][3]
            d=math.hypot(xy[prev][0]-xy[c][0], xy[prev][1]-xy[c][1])
            Dtot+=d
            t=max(t+d, rows[c][4])
            if t > rows[c][5]+1e-9: maxviol=max(maxviol, t-rows[c][5])
            t+=rows[c][6]; prev=c
        d=math.hypot(xy[prev][0]-xy[0][0], xy[prev][1]-xy[0][1])
        Dtot+=d
        if t+d > rows[0][5]+1e-6: maxviol=max(maxviol, t+d-rows[0][5])
        capviol=max(capviol, load-cap); loadmax=max(loadmax, load)
    cost=re.search(r"Cost\s+(-?\d+(?:\.\d+)?)", sol.read_text())
    rows_out.append(dict(instance=name, n=len(rows)-1, capacity=cap, k_bks=len(R),
                         distance_double=round(Dtot,4),
                         distance_trunc1=round(sum(int(math.hypot(xy[0][0]-xy[0][0],0)*0) for _ in []),4) or None,
                         cost_line=float(cost.group(1)) if cost else None,
                         max_window_violation=round(maxviol,6),
                         max_capacity_violation=round(capviol,6),
                         feasible_double=maxviol<=1e-9 and capviol<=1e-9))
feas=[r for r in rows_out if r["feasible_double"]]
print("feasible", len(feas), "/", len(rows_out))
print("infeasible:", [(r["instance"], r["max_window_violation"], r["max_capacity_violation"]) for r in rows_out if not r["feasible_double"]])
print("max viol overall", max(r["max_window_violation"] for r in rows_out))
json.dump(dict(convention="double precision euclidean, no rounding in evaluation",
               feasible=len(feas), total=len(rows_out), rows=rows_out),
          open("data/solomon_bks_derived.json","w",encoding="utf-8"), ensure_ascii=False, indent=1)
print("saved data/solomon_bks_derived.json")
