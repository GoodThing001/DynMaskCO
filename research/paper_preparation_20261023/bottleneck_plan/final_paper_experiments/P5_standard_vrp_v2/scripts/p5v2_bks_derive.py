import re, math, sys, json
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
D = Path("data/cvrplib")
names = sorted((D/"bks").glob("*.sol"))
out = []
for sol in names:
    name = sol.stem
    inst = D/"instances"/(name + ".vrp")
    text = inst.read_text(encoding="utf-8", errors="replace")
    sec=None; coords={}; dem={}
    for line in text.splitlines():
        s=line.strip()
        if not s: continue
        if s.endswith("_SECTION"): sec=s; continue
        if sec=="NODE_COORD_SECTION":
            p=s.split()
            if len(p)==3: coords[int(p[0])]=(float(p[1]),float(p[2]))
        elif sec=="DEMAND_SECTION":
            p=s.split()
            if len(p)==2: dem[int(p[0])]=float(p[1])
    cap = float(re.search(r"CAPACITY\s*:\s*(\d+)", text, re.I).group(1))
    routes=[[int(x) for x in re.findall(r"\d+",l.split(":",1)[1])] for l in sol.read_text().splitlines() if ":" in l and re.findall(r"\d+", l.split(":",1)[1])]
    dep = coords[1]
    def edge(a,b):
        pa = dep if a==0 else coords[a]
        pb = dep if b==0 else coords[b]
        return math.hypot(pa[0]-pb[0], pa[1]-pb[1])
    rf=0.0; rr=0
    viol=0
    for r in routes:
        nodes=[0]+[i+1 for i in r]+[0]
        load=sum(dem[i+1] for i in r)
        if load > cap + 1e-9: viol+=1
        for a,b in zip(nodes,nodes[1:]):
            dd=edge(a,b); rf+=dd; rr+=int(dd+0.5)
    k_name = int(name.split("-k")[-1])
    out.append(dict(name=name, k_name=k_name, k_sol=len(routes), rounded=rr, float=round(rf,3), cap_viol=viol))
diff = [o for o in out if o["k_sol"] != o["k_name"]]
print("instances", len(out), "k_sol != k_name:", len(diff))
print("viol", sum(1 for o in out if o["cap_viol"]))
print("examples", diff[:6])
vals = [o["rounded"] for o in out]
print("rounded sample", out[0]["name"], out[0]["rounded"])
json.dump(out, open("data/cvrplib/bks_recomputed.json","w",encoding="utf-8"), ensure_ascii=False, indent=1)
