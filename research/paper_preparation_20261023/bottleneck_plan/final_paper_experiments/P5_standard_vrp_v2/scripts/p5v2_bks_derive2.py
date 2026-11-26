import re, math, sys, json
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
D = Path("data")

def parse_vrp(p):
    text = p.read_text(encoding="utf-8", errors="replace")
    sec=None; coords={}; dem={}
    for line in text.splitlines():
        s=line.strip()
        if not s: continue
        if s.endswith("_SECTION"): sec=s; continue
        if s=="EOF": break
        if sec=="NODE_COORD_SECTION":
            q=s.split()
            if len(q)==3: coords[int(q[0])]=(float(q[1]),float(q[2]))
        elif sec=="DEMAND_SECTION":
            q=s.split()
            if len(q)==2: dem[int(q[0])]=float(q[1])
    cap=float(re.search(r"CAPACITY\s*:\s*(\d+)",text,re.I).group(1))
    dep=coords[1]
    fixed={0: dep}
    fixed.update({k: v for k, v in coords.items() if k != 1})
    return fixed, dem, cap

def parse_solomon(p):
    num=re.compile(r"-?\d+(?:\.\d+)?")
    rows=[]; cap=None; pend=False
    for line in p.read_text(encoding="utf-8",errors="replace").splitlines():
        s=line.strip()
        if not s: continue
        if s.lower().startswith("number") and "capacity" in s.lower():
            pend=True; continue
        if pend and cap is None:
            q=num.findall(s)
            if len(q)>=2:
                cap=float(q[-1]); pend=False; continue
        q=num.findall(s)
        if cap is not None and len(q)==7: rows.append([float(x) for x in q])
    return rows, cap

def route_cost(coords, r, mode):
    prev=0; tot=0.0
    for c in r+[0]:
        a=coords[prev]; b=coords[c]
        d=math.hypot(a[0]-b[0],a[1]-b[1])
        tot += int(d+0.5) if mode=="round" else d
        prev=c
    return tot

def sol_routes(p):
    out=[]
    for line in p.read_text().splitlines():
        if ":" in line:
            ids=[int(x) for x in re.findall(r"\d+", line.split(":",1)[1])]
            if ids: out.append(ids)
    return out

def cost_line(p):
    m=re.search(r"Cost\s+(-?\d+(?:\.\d+)?)", p.read_text())
    return float(m.group(1)) if m else None

# CVRP
res={"cvrp":[], "vrptw":[]}
for sol in sorted((D/"cvrplib/bks").glob("*.sol")):
    coords, dem, cap = parse_vrp(D/"cvrplib/instances"/(sol.stem+".vrp"))
    routes=sol_routes(sol)
    rr=sum(route_cost(coords,[i+1 for i in r],"round") for r in routes)
    rf=sum(route_cost(coords,[i+1 for i in r],"float") for r in routes)
    res["cvrp"].append(dict(name=sol.stem, k=len(routes), rounded=int(rr), float=round(rf,3), cost=cost_line(sol)))
bad=[x for x in res["cvrp"] if x["cost"] is not None and int(x["cost"])!=x["rounded"]]
print("CVRP: cost line vs rounded mismatches:", len(bad), bad[:3])
print("CVRP without cost line:", sum(1 for x in res["cvrp"] if x["cost"] is None))

# VRPTW
for sol in sorted((D/"solomon/bks").glob("*.sol")):
    rows, cap = parse_solomon(D/"solomon/instances"/(sol.stem+".vrp"))
    coords={i:(r[1],r[2]) for i,r in enumerate(rows)}
    routes=sol_routes(sol)
    rf=sum(route_cost(coords,r,"float") for r in routes)
    rr=sum(route_cost(coords,r,"round") for r in routes)
    # feasibility
    ok=True
    for r in routes:
        t=rows[0][4]; prev=0; load=0.0
        for c in r:
            load+=rows[c][3]
            d=math.hypot(coords[prev][0]-coords[c][0],coords[prev][1]-coords[c][1])
            t=max(t+d, rows[c][4])
            if t>rows[c][5]+1e-9: ok=False
            t+=rows[c][6]; prev=c
        d=math.hypot(coords[prev][0]-coords[0][0],coords[prev][1]-coords[0][1])
        if t+d>rows[0][5]+1e-6: ok=False
        if load>cap+1e-9: ok=False
    res["vrptw"].append(dict(name=sol.stem, k=len(routes), float=round(rf,3), rounded=int(rr),
                             cost=cost_line(sol), feasible=ok, n=len(rows)-1))
json.dump(res, open("data/bks_derived.json","w",encoding="utf-8"), ensure_ascii=False, indent=1)
cmpf=[(x["name"],x["cost"],x["float"]) for x in res["vrptw"] if x["cost"] is not None and abs(x["cost"]-x["float"])>0.05]
cmpr=[(x["name"],x["cost"],x["rounded"]) for x in res["vrptw"] if x["cost"] is not None and abs(x["cost"]-x["rounded"])>0.05]
print("VRPTW: |cost-float|>0.05:", len(cmpf), cmpf[:6])
print("VRPTW: |cost-rounded|>0.05:", len(cmpr), cmpr[:6])
print("VRPTW infeasible bks:", [x["name"] for x in res["vrptw"] if not x["feasible"]])
print("C101:", [x for x in res["vrptw"] if x["name"]=="C101"])
