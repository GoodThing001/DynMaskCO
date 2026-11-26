import json, glob, statistics as st, sys
from pathlib import Path
sys.stdout.reconfigure(encoding="utf-8")
rows = []
for f in glob.glob("external_results/pyvrp_vrptw/run*/**/*.json", recursive=True):
    try:
        rows.append(json.load(open(f, encoding="utf-8")))
    except Exception as exc:
        print("bad", f, exc)
keys = {(r["case"], r["seed"]) for r in rows}
print("endpoints", len(rows), "unique cells", len(keys), "methods", {r["method"] for r in rows})
import collections
print("status", collections.Counter(r["status"] for r in rows))
print("online_s max %.2f mean %.2f" % (max(r["online_s"] for r in rows), st.mean([r["online_s"] for r in rows])))
bks = {r["instance"]: r for r in json.load(open("data/solomon_bks_derived.json", encoding="utf-8"))["rows"]}
kd = []
for r in rows:
    b = bks.get(r["case"])
    if b:
        kd.append((r["vehicles"] - b["k_bks"], r["distance"] - b["distance_double"]))
print("K vs BKS: better %d tie %d worse %d" % (sum(1 for k,_ in kd if k<0), sum(1 for k,_ in kd if k==0), sum(1 for k,_ in kd if k>0)))
same = [d for k,d in kd if k == 0]
print("mean D delta on same-K %d: %.2f" % (len(same), st.mean(same) if same else float("nan")))
print("violations recorded:", [ (r["case"], r.get("max_violation")) for r in rows if r["status"] != "AUDITED"][:6])
