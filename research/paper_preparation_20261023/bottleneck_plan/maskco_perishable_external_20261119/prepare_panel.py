"""Build the previously specified external source projections and audit their feasibility."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
from zipfile import ZipFile
from concurrent.futures import ThreadPoolExecutor

HERE=Path(__file__).resolve().parent
PREV=HERE.parent/"maskco_heatmap_calibration_20261119"
sys.path.insert(0,str(PREV))
import prepare_holdouts as archives
import audit_delivery
sys.path.insert(0,str(HERE.parents[3]/"C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_profit"))
import profit_core as core
from preflight_publication_model_v1 import nearest_neighbour_witness,independent_audit
XTW=[298,303,308,313,317,322,327,331,336,344]

def perishable():
    zpath=HERE/"source_perishable/93w8ynrk59-v2.zip"
    with ZipFile(zpath) as z:
        base="Dynamic PDPCDPG with Rolling-Horizon/Instances_static/"
        for letter in "abcde":
            stem="Data100_"+letter
            def read(name):return [[float(v) for v in l.split()] for l in z.read(base+name+".txt").decode().splitlines() if l.strip()]
            header,depot,*customers=read(stem)
            windows=read(stem+"_tw")
            if header[:5]!=[100.,33.,6.,22.,60.] or len(customers)!=100 or len(windows)!=100:
                raise ValueError("Author schema differs")
            if any(r[0]!=i for i,r in enumerate([depot]+customers)) or any(len(t)!=4 for t in windows):
                raise ValueError("Author customer order differs")
            nodes=[[0,depot[1]/1000,depot[2]/1000,0.,360.,1320.,0.]]
            for cust,tw in zip(customers,windows):
                i,x,y,_,_,pallets=cust
                nodes.append([int(i),x/1000,y/1000,20000*pallets/header[1],tw[0],tw[1],0.])
            yield "dk_"+stem.lower(),"PDPCDPG-D100",nodes,archives.coordinates([depot]+customers),"10.17632/93w8ynrk59.2#"+base+stem

def cvrplib(spec):
    kind,title,ident=spec
    url=f"https://galgos.inf.puc-rio.br/cvrplib/en/download/instance/{ident}"
    path=HERE/"source_external"/(title+(".vrp" if kind=="XTW-p100" else ".TXT"))
    data=archives.raw(path,url)
    if kind=="XTW-p100":
        name,cap,rows=archives.xtw(data)
        if name!=title or len(rows)!=int(title.split("n")[1]):raise ValueError("XTW identity mismatch")
        case="xtw_"+title.split("-")[1]+"_p100"
    else:
        cap,rows=archives.gh(data)
        if data.decode("ascii").split()[0].upper()!=title or len(rows)!=601:raise ValueError("GH600 identity mismatch")
        case="gh_"+title.lower()+"_p100"
    scale=1440/rows[0][5]
    nodes=[[i,r[1]*scale,r[2]*scale,r[3]*20000/cap,r[4]*scale,r[5]*scale,r[6]*scale] for i,r in enumerate(rows[:101])]
    return case,kind,nodes,archives.coordinates(rows),url

def write_instance(root,data,case,source,nodes,coords,origin):
    geo=hashlib.sha256(json.dumps(coords,separators=(",",":")).encode()).hexdigest()
    file=root/"inputs"/(case+".json")
    raw=json.dumps(dict(case=case,nodes=nodes),ensure_ascii=False,indent=2).encode()
    file.write_bytes(raw)
    data["cases"][case]=dict(dataset=source,geometry=geo)
    data["derived_instances"][case]=dict(file=file.relative_to(root).as_posix(),geometry_sha256=geo,sha256=hashlib.sha256(raw).hexdigest())
    data["sources"][case]=origin
    return geo


def keep(entries,history):
    inputs=HERE/"inputs";inputs.mkdir(exist_ok=True)
    data=dict(cases={},derived_instances={},sources={})
    admission={};geometry_by_source={}
    for case,source,nodes,coords,origin in entries:
        if coords in history:raise ValueError("Geometry previously used: "+case)
        geo=write_instance(HERE,data,case,source,nodes,coords,origin)
        geometry_by_source.setdefault(source,{}).setdefault(geo,[]).append(case)
        model=core.PublicationModel(dict(nodes=nodes,distance=[[math.dist(a[1:3],b[1:3]) for b in nodes] for a in nodes],name=case),core.load_config(),"main_Product1",.8)
        fails={i:model.route((i,)).reason for i in range(1,101) if not model.route((i,)).feasible}
        entry=dict(infeasible_singletons=fails)
        if not fails:
            solution=nearest_neighbour_witness(model,True)
            independent_audit(model,solution,True)
            entry["initializer_routes"]=len(solution)
        admission[case]=entry
    return data,admission,geometry_by_source

def save(name,value):
    (HERE/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf8")

def main():
    dk=list(perishable())
    if sys.argv[1:]==["perishable"]:
        _,report,geos=keep(dk,set())
        save("perishable_feasibility.json",report)
        print(len(geos["PDPCDPG-D100"]),"geometries",[(c,len(v["infeasible_singletons"])) for c,v in report.items()])
        return
    (HERE/"source_external").mkdir(exist_ok=True)
    specs=[("XTW-p100",f"XTW-n{n}",778+i) for i,n in enumerate(XTW)]+[("GH-p100",f"{g}_6_{i}",555+10*j+i) for j,g in enumerate(archives.GROUPS) for i in (1,2)]
    known=set()
    for path in list(archives.B.glob("**/originals/XTW-n*.vrp"))+list((PREV/"validation_sources").glob("*.vrp")):
        known.add(archives.coordinates(archives.xtw(path.read_bytes())[2]))
    with ZipFile(archives.OLD/"sources/homberger200.zip") as z:
        for g in archives.GROUPS:
            for i in (1,2):known.add(archives.coordinates(archives.gh(z.read(f"{g}_2_{i}.TXT"))[1]))
    for p in (PREV/"validation_sources").glob("*.TXT"):
        known.add(archives.coordinates(archives.gh(p.read_bytes())[1]))
    with ThreadPoolExecutor(max_workers=5) as executor:
        new=list(executor.map(cvrplib,specs))
    data,report,geos=keep(dk+new,known)
    save("heldout_sources.json",data)
    passed=all(not v["infeasible_singletons"] for v in report.values())
    save("heldout_admission.json",dict(status="PASS" if passed else "STOP_INFEASIBLE",cases=report,geometries={s:len(v) for s,v in geos.items()}))
    print("admitted",sum(not r["infeasible_singletons"] for r in report.values()),"/",len(report),{s:len(v) for s,v in geos.items()})
if __name__=="__main__":main()
