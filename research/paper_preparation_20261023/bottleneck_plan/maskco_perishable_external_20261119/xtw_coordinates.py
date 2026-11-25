"""Compare test clients, depots and complete request attributes with all previously acquired XTW instances."""
from collections import Counter
from pathlib import Path
import json
from strict_xtw_plan import HERE,PRIMARY,DEST,positions,panel
def read(path):return json.loads(path.read_text(encoding="utf8"))
def main():
    registry=read(DEST/"selection.json")
    files={}
    for p in HERE.parent.rglob("XTW-n*.vrp"):
        if not p.is_relative_to(DEST) and not p.is_relative_to(PRIMARY):files.setdefault(p.stem,p)
    named={n:panel.archives.xtw(p.read_bytes())[2] for n,p in files.items()}
    coords={n:Counter(tuple(row[1:3]) for row in rows[1:101]) for n,rows in named.items()}
    def compare(path,known):
        src=panel.archives.xtw(path.read_bytes())[2]
        query=Counter(tuple(r[1:3]) for r in src[1:101]);union=Counter()
        for m in known:union|=coords[m]
        count,closest=max((sum((query&coords[m]).values()),m) for m in known)
        nearest=named[closest]
        cells=Counter(tuple(r[1:7]) for r in src[1:101])
        exact=sum((cells&Counter(tuple(r[1:7]) for r in nearest[1:101])).values())
        return dict(closest=closest,individual_overlap=count,union_overlap=sum((union&query).values()),matching_raw_location_demand_window_service=exact,same_depot=tuple(nearest[0][1:3])==tuple(src[0][1:3]))
    historical=set(named)
    result={}
    for p in sorted((PRIMARY/"raw").glob("*.vrp")):result[p.stem]=compare(p,historical)
    for p in sorted((PRIMARY/"raw").glob("*.vrp")):
        named[p.stem]=panel.archives.xtw(p.read_bytes())[2];coords[p.stem]=positions(p)
    known=set(named)
    for rec in registry["screened"]:
        if not rec["included"]:continue
        p=DEST/"raw"/(rec["source"]+".vrp");result[p.stem]=compare(p,known)
        named[p.stem]=panel.archives.xtw(p.read_bytes())[2];coords[p.stem]=positions(p);known.add(p.stem)
    (DEST/"attribute_overlap.json").write_text(json.dumps(result,indent=2))
    print({k:(v["individual_overlap"],v["union_overlap"],v["matching_raw_location_demand_window_service"],v["same_depot"]) for k,v in result.items()})
if __name__=="__main__":main()
