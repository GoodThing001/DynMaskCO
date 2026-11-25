"""Cross-check every projected delivery against its published source fields and geometry."""
import json
import math
import re
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path
from zipfile import ZipFile
import prepare_panel as build

HERE=Path(__file__).resolve().parent
def read(p):return json.loads(p.read_text(encoding="utf8"))
def verify(complement=False):
    folder=HERE/"reserved_validation" if complement else HERE
    external=folder/"raw" if complement else HERE/"source_external"
    registry=read(folder/"heldout_sources.json")
    selected=read(HERE/"reserved_requests.json")["complete_parent_difference"] if complement else {}
    locations={}
    total=0
    with ZipFile(HERE/"source_perishable/93w8ynrk59-v2.zip") as archive:
        for case,entry in registry["derived_instances"].items():
            nodes=read(folder/entry["file"])["nodes"]
            dataset=registry["cases"][case]["dataset"]
            if dataset=="PDPCDPG-D100":
                letter=case[-1]
                stem=f"Dynamic PDPCDPG with Rolling-Horizon/Instances_static/Data{200 if complement else 100}_{letter}"
                rows=[[float(x) for x in l.split()] for l in archive.read(stem+".txt").decode().splitlines() if l.strip()]
                times=[[int(x) for x in l.split()] for l in archive.read(stem+"_tw.txt").decode().splitlines() if l.strip()]
                spec,orig=rows[0],rows[1:]
                if complement:
                    requested=selected[letter]["row_ids"]
                    oldstem=stem.replace("Data200_","Data100_")
                    previous=[[float(x) for x in l.split()] for l in archive.read(oldstem+".txt").decode().splitlines() if l.strip()][2:]
                    identity=lambda a:tuple(a[1:6])
                    remaining=Counter(map(identity,orig[1:]))-Counter(map(identity,previous))
                    if Counter(identity(orig[i]) for i in requested)!=remaining or len(requested)!=100:
                        raise ValueError("Published complementary request multiset differs")
                    orig=[orig[0]]+[orig[i] for i in requested]
                    times=[times[i-1] for i in requested]
                if nodes[0][4:6]!=[spec[2]*60,spec[3]*60]:raise ValueError("Depot hours differ")
                for row_id,(n,s,w) in enumerate(zip(nodes[1:],orig[1:],times,strict=True),1):
                    i,x,y,m,lo,hi,service=n
                    if s[0]!=(requested[row_id-1] if complement else row_id):raise ValueError("Original request ID differs")
                    if ([lo,hi]!=w[:2] or service!=0 or i!=row_id or not math.isclose(m/20000,s[5]/spec[1])
                        or not math.isclose(x*1000,s[1]) or not math.isclose(y*1000,s[2])):
                        raise ValueError(f"Delivery record differs: {case}/{i}")
                if not all(math.isclose(src[1]/1000,val[1]) and math.isclose(src[2]/1000,val[2]) for src,val in zip(orig,nodes,strict=True)):
                    raise ValueError("Depot or customer coordinates differ")
            else:
                if dataset=="XTW-p100":
                    raw=(external/("XTW-"+case.split("_")[1]+".vrp")).read_bytes()
                    _,cap,orig=build.archives.xtw(raw)
                else:
                    raw=(external/(case[3:-5].upper()+".TXT")).read_bytes()
                    cap,orig=build.archives.gh(raw)
                tscale=1440/orig[0][5]
                for src,val in zip(orig[:101],nodes,strict=True):
                    expected=[src[0],src[1]*tscale,src[2]*tscale,src[3]/cap*20000,*[s*tscale for s in src[4:7]]]
                    if any(not math.isclose(a,b,abs_tol=1e-9) for a,b in zip(expected,val,strict=True)):
                        raise ValueError(f"Benchmark row differs: {case}/{src[0]}")
            locations[case]=Counter(tuple(r[1:3]) for r in orig[1:101])
            total+=len(nodes)-1
    overlaps={}
    for dataset in sorted({r["dataset"] for r in registry["cases"].values()}):
        cases=[c for c,m in registry["cases"].items() if m["dataset"]==dataset]
        unique={registry["cases"][c]["geometry"]:c for c in cases}
        overlaps[dataset]=max(sum((locations[c]&locations[d]).values()) for c,d in combinations(unique.values(),2))
    prior={"GH-p100":[],"XTW-p100":[]}
    def customers(rows):
        return Counter(tuple(r[1:3]) for r in rows[1:101])
    with ZipFile(build.archives.OLD/"sources/homberger200.zip") as z:
        prior["GH-p100"].extend(customers(build.archives.gh(z.read(f"{g}_2_{i}.TXT"))[1]) for g in build.archives.GROUPS for i in (1,2))
    for src in (build.PREV/"validation_sources").glob("*.TXT"):
        prior["GH-p100"].append(customers(build.archives.gh(src.read_bytes())[1]))
    for src in list(build.archives.B.glob("**/originals/XTW-n*.vrp"))+list((build.PREV/"validation_sources").glob("*.vrp")):
        prior["XTW-p100"].append(customers(build.archives.xtw(src.read_bytes())[2]))
    if complement:
        prior["GH-p100"].extend(customers(build.archives.gh(f.read_bytes())[1]) for f in (HERE/"source_external").glob("*.TXT"))
        prior["XTW-p100"].extend(customers(build.archives.xtw(f.read_bytes())[2]) for f in (HERE/"source_external").glob("*.vrp"))
        with ZipFile(HERE/"source_perishable/93w8ynrk59-v2.zip") as z:
            prior["PDPCDPG-D100"]=[]
            for letter in "abcde":
                path=f"Dynamic PDPCDPG with Rolling-Horizon/Instances_static/Data100_{letter}.txt"
                rows=[[float(x) for x in l.split()] for l in z.read(path).decode().splitlines() if l.strip()]
                prior["PDPCDPG-D100"].append(customers(rows[1:]))
    by_case={c:max(sum((v&old).values()) for old in prior[registry["cases"][c]["dataset"]]) for c,v in locations.items()
             if registry["cases"][c]["dataset"] in prior}
    historical={s:max(n for c,n in by_case.items() if registry["cases"][c]["dataset"]==s) for s in prior}
    result=dict(status="PASS",checked_cases=len(locations),checked_customer_records=total,
                maximum_customer_coordinate_overlap_between_distinct_geometries_out_of_100=overlaps,
                maximum_customer_overlap_with_previous_same_source_cases=historical,
                overlap_with_most_similar_previous_case=by_case)
    if complement:
        with ZipFile(HERE/"source_perishable/93w8ynrk59-v2.zip") as z:
            rs={}
            for path in z.namelist():
                name=re.search(r"/(Instances_static|Instances_Rolling_Horizon)/Data(\d+)_([a-e])\.txt$",path)
                if name:
                    lines=[v for v in z.read(path).decode().splitlines() if v.strip()][2:]
                    rs[name.group(1),int(name.group(2)),name.group(3)]={tuple(map(float,l.split()[1:6])) for l in lines}
            parents=set.union(*(rs["Instances_static",200,c] for c in "abcde"))
            result["perishable_unique_location_demand_tuples_outside_five_200_request_parents"]={
                scope:len(set.union(*(orders for (kind,n,c),orders in rs.items() if kind==scope))-parents)
                for scope in ("Instances_static","Instances_Rolling_Horizon")}
    (folder/"source_audit.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    print(result)
if __name__=="__main__":verify(sys.argv[1:]==["reserve02"])
