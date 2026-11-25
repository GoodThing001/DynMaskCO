"""Admit the prespecified complementary panel after a new development lock."""
from pathlib import Path
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from zipfile import ZipFile
import prepare_panel as first
from prepare_panel import archives
HERE=Path(__file__).resolve().parent
DEST=HERE/"reserved_validation"
XTW=(351,359,367,376,384,393,401,411,420,429)
def read(p):return json.loads(p.read_text(encoding="utf8"))
def danish():
    selection=read(HERE/"reserved_requests.json")["complete_parent_difference"]
    archive=HERE/"source_perishable/93w8ynrk59-v2.zip"
    with ZipFile(archive) as z:
        for c in "abcde":
            stem=f"Dynamic PDPCDPG with Rolling-Horizon/Instances_static/Data200_{c}"
            def get(s):return [[float(v) for v in l.split()] for l in z.read(stem+s+".txt").decode().splitlines() if l.strip()]
            header,depot,*rows=get("");times=get("_tw")
            ids=selection[c]["row_ids"]
            if header[:5]!=[200,33,6,22,60] or len(set(ids))!=100:raise ValueError("Complement identity differs")
            raw=[depot];nodes=[[0,depot[1]/1000,depot[2]/1000,0.,360.,1320.,0.]]
            for n,k in enumerate(ids,1):
                row=rows[k-1];tw=times[k-1]
                if row[0]!=k:raise ValueError("Parent request differs")
                raw.append(row)
                nodes.append([n,row[1]/1000,row[2]/1000,20000*row[5]/33,tw[0],tw[1],0.])
            yield "dk_complement_"+c,"PDPCDPG-D100",nodes,archives.coordinates(raw),"10.17632/93w8ynrk59.2#"+stem
def cvrpl(spec):
    title,ident=spec;is_xtw=title.startswith("XTW")
    url=f"https://galgos.inf.puc-rio.br/cvrplib/en/download/instance/{ident}"
    path=DEST/"raw"/(title+(".vrp" if is_xtw else ".TXT"))
    raw=archives.raw(path,url)
    if is_xtw:
        name,cap,rows=archives.xtw(raw)
        if name!=title or len(rows)!=int(title[5:]):raise ValueError(title)
        source,case="XTW-p100","xtw_"+title[4:].lower()+"_p100"
    else:
        cap,rows=archives.gh(raw)
        if raw.decode("ascii").split()[0].upper()!=title or len(rows)!=801:raise ValueError(title)
        source,case="GH-p100","gh_"+title.lower()+"_p100"
    scale=1440/rows[0][5]
    nodes=[[i,r[1]*scale,r[2]*scale,r[3]*20000/cap,*[v*scale for v in r[4:7]]] for i,r in enumerate(rows[:101])]
    return case,source,nodes,archives.coordinates(rows),url
def main():
    prior=read(HERE/"results/replicate01/analysis.json")
    lock=read(DEST/"locked_method.json")
    dev=read(HERE/"results"/lock["development_phase"]/"analysis.json")
    audit=read(HERE/"results"/lock["development_phase"]/"independent_audit.json")
    if (prior["complete"]!=891 or prior["passes_numerical_criteria"]["HYBRID"] or
            not dev["passes_numerical_criteria"][lock["candidate"]] or dev["complete"]!=312 or
            audit["status"]!="PASS" or audit["endpoints"]!=312):raise ValueError("Prior gates do not trigger a new external study")
    (DEST/"raw").mkdir(exist_ok=True)
    known=set()
    for pr in list(archives.B.glob("**/originals/XTW-n*.vrp"))+list((archives.HERE/"validation_sources").glob("*.vrp"))+list((HERE/"source_external").glob("*.vrp")):
        known.add(archives.coordinates(archives.xtw(pr.read_bytes())[2]))
    for folder in (archives.HERE/"validation_sources",HERE/"source_external"):
        for pr in folder.glob("*.TXT"):known.add(archives.coordinates(archives.gh(pr.read_bytes())[1]))
    with ZipFile(archives.OLD/"sources/homberger200.zip") as z:
        for g in archives.GROUPS:
            for i in (1,2):known.add(archives.coordinates(archives.gh(z.read(f"{g}_2_{i}.TXT"))[1]))
    known.update(v[3] for v in first.perishable())
    specs=[(f"XTW-n{n}",788+i) for i,n in enumerate(XTW)]+[(f"{g}_8_{i}",615+10*j+i) for j,g in enumerate(archives.GROUPS) for i in (1,2)]
    with ThreadPoolExecutor(max_workers=5) as ex:external=list(ex.map(cvrpl,specs))
    records=list(danish())+external
    import tempfile
    first.HERE=DEST
    data,admission,geos=first.keep(records,known)
    first.save("heldout_sources.json",data)
    passed=len(admission)==27 and all(not x["infeasible_singletons"] for x in admission.values())
    first.save("heldout_admission.json",dict(status="PASS" if passed else "STOP_INFEASIBLE",cases=admission,geometry_counts={s:len(v) for s,v in geos.items()}))
    if not passed:raise ValueError("Fixed complementary panel failed common feasibility")
    print("Admitted",len(admission),{s:len(v) for s,v in geos.items()})
if __name__=="__main__":main()
