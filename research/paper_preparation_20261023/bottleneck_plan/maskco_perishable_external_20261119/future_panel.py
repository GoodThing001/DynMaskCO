"""Register source order and historical customer overlap for the prospective comparison."""
from collections import Counter
from pathlib import Path
from zipfile import ZipFile
import json
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import prepare_panel as panel
from strict_xtw_plan import ORDER
from freight_admission import DAYS, delivery_day

HERE=Path(__file__).resolve().parent
DEST=HERE/"future_validation"
archives=panel.archives
URL="https://galgos.inf.puc-rio.br/cvrplib/en/download/instance/"
XTW_ORDER=ORDER+(766,783,801,819,837,856,876,895,916,936,957,979,1001)


def coordinates(rows):
    return Counter(tuple(row[1:3]) for row in rows[1:101])


def closest(candidate,known):
    n,title=max((sum((candidate & group).values()),title) for title,group in known.items())
    all_positions=Counter()
    for group in known.values():all_positions|=group
    return dict(max_overlap=n,closest=title,union_overlap=sum((all_positions & candidate).values()))


def history():
    xtw={}
    for path in archives.B.rglob("XTW-n*.vrp"):
        if not path.is_relative_to(DEST) and path.stem not in xtw:
            xtw[path.stem]=coordinates(archives.xtw(path.read_bytes())[2])
    gh={}
    with ZipFile(archives.OLD/"sources/homberger200.zip") as z:
        for family in archives.GROUPS:
            for i in (1,2):
                name=f"{family}_2_{i}"
                gh[name]=coordinates(archives.gh(z.read(name+".TXT"))[1])
    for loc in (panel.PREV/"validation_sources",HERE/"source_external",HERE/"reserved_validation/raw"):
        for path in loc.glob("*.TXT"):
            if path.stem.upper().split("_")[0] in archives.GROUPS:
                gh[path.stem]=coordinates(archives.gh(path.read_bytes())[1])
    return xtw,gh


def projected(title,src,rows,capacity,origin):
    if [r[0] for r in rows]!=list(range(len(rows))):raise ValueError(title)
    scale=1440/rows[0][5]
    nodes=[[i,r[1]*scale,r[2]*scale,r[3]*20000/capacity,*[v*scale for v in r[4:7]]] for i,r in enumerate(rows[:101])]
    case="xtw_"+title[4:].lower()+"_p100" if src=="XTW-p100" else "gh_"+title.lower()+"_p100"
    return case,src,nodes,archives.coordinates(rows),origin


def add_freight(data,decision):
    receipt=json.loads((HERE/"freight_admission.json").read_text())
    inventory=json.loads((HERE/"freight_source_inventory.json").read_text())
    days=[f"{c}_{d}_0_F" for c in ("BE","NL") for d in DAYS]
    if receipt["status"]!="PASS" or days!=decision["freight_days"] or set(days)!=set(receipt["source_cases"]):
        raise ValueError("Freight source list or admission differs")
    for name in days:
        nodes,ids,ll,eligible=delivery_day(name)
        row=receipt["source_cases"][name]
        if ids!=row["source_node_ids"] or eligible!=row["eligible"] or row["unserviceable_ids"]:
            raise ValueError(f"Selected orders differ from admission for {name}")
        case="vrppdl_"+name.lower()+"_p100"
        origin=f"https://github.com/siemterbraake/vrppdl-algorithms-instances/blob/{inventory['archive_commit']}/instances/large/{name}.txt"
        panel.write_instance(DEST,data,case,"VRPPDL-p100",nodes,sorted(ll),origin)
        data["cases"][case].update(country=name.split("_")[0],source_day=name)
        decision["source_admission"][case]=dict(infeasible_singletons={},initializer_routes=row["common_initializer_routes"])


def save_panel(data,decision):
    add_freight(data,decision)
    cases=data["cases"]
    if (Counter(v["dataset"] for v in cases.values())!={"GH-p100":12,"XTW-p100":10,"VRPPDL-p100":10}
        or any(set(rows)!=set(cases) for rows in (data["derived_instances"],data["sources"],decision["source_admission"]))):
        raise ValueError("Expected the same 32 admitted cases at each evaluation entry")
    (DEST/"heldout_sources.json").write_text(json.dumps(data,indent=2),encoding="utf8")
    (DEST/"selection.json").write_text(json.dumps(decision,indent=2),encoding="utf8")
    save_admission(decision)


def save_admission(decision):
    admitted=decision["source_admission"]
    ready=all(not r["infeasible_singletons"] and r.get("initializer_routes",0)>0 for r in admitted.values())
    receipt=dict(status="PASS" if decision["status"]=="PASS" and ready else "STOP_ADMISSION",cases=admitted)
    (DEST/"heldout_admission.json").write_text(json.dumps(receipt,indent=2),encoding="utf8")


def main():
    raw=DEST/"raw";raw.mkdir(exist_ok=True,parents=True)
    session=requests.Session()
    session.mount("https://",HTTPAdapter(max_retries=Retry(total=3,backoff_factor=1,status_forcelist=(429,502,503,504))))
    def download(path,url):
        if not path.is_file():
            reply=session.get(url,timeout=(15,30));reply.raise_for_status()
            path.write_bytes(reply.content)
        return path.read_bytes()
    historical,gh=history()
    known=dict(historical);log=[];entries=[]
    for index,n in enumerate(XTW_ORDER):
        if n<=480:continue
        title=f"XTW-n{n}";url=URL+str(798+index)
        name,cap,rows=archives.xtw(download(raw/(title+".vrp"),url))
        if name!=title or len(rows)!=n:raise ValueError((title,name))
        coords=coordinates(rows);prior=closest(coords,historical);check=closest(coords,known)
        include=check["max_overlap"]<=10 and title not in historical
        log.append(dict(source=title,included=include,against_preselected=check,against_history=prior))
        if include:
            entries.append(projected(title,"XTW-p100",rows,cap,url));known[title]=coords
        if len(entries)==10:break
    gh_rows={};within={}
    for group,family in enumerate(archives.GROUPS):
        for i in (1,2):
            title=f"{family}_10_{i}";url=URL+str(675+10*group+i)
            content=download(raw/(title+".TXT"),url)
            cap,rows=archives.gh(content)
            if len(rows)!=1001 or content.decode("ascii").split()[0].upper()!=title:raise ValueError(title)
            coords=coordinates(rows);gh_rows[title]=closest(coords,gh);within[title]=coords
            entries.append(projected(title,"GH-p100",rows,cap,url))
    panel.HERE=DEST
    data,admission,_=panel.keep(entries,set())
    for title,coords in within.items():
        case="gh_"+title.lower()+"_p100"
        same=sorted(other for other,c in within.items() if c==coords)
        data["cases"][case]["geometry"]="GH:"+same[0]
        gh_rows[title]["same_customer_geometry"]=same
        gh_rows[title]["max_partial_overlap_in_panel"]=max((sum((coords&v).values()) for k,v in within.items() if k not in same),default=0)
    freight=json.loads((HERE/"freight_admission.json").read_text())
    fdays=[f"{c}_{d}_0_F" for c in ("BE","NL") for d in DAYS]
    if set(fdays)!=set(freight["source_cases"]):raise ValueError("Freight days differ")
    good=(len(entries)==22 and all(not v["infeasible_singletons"] for v in admission.values())
          and max(r["max_overlap"] for r in gh_rows.values())<=10 and freight["status"]=="PASS")
    decision=dict(status="PASS" if good else "STOP_ADMISSION",solver_endpoints=0,xtw=log,gh=gh_rows,
        xtw_historical_cases=len(historical),gh_historical_cases=len(gh),source_admission=admission,
        freight_days=fdays,gh_geometry_count=len({tuple(r["same_customer_geometry"]) for r in gh_rows.values()}))
    save_panel(data,decision)
    print(decision["status"],len(entries),sum(r["included"] for r in log),len(log),decision["gh_geometry_count"],max(x["max_overlap"] for x in gh_rows.values()))
if __name__=="__main__":main()
