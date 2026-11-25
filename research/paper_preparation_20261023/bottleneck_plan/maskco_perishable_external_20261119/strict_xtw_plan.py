"""Admit the first three author-ordered XTW instances with low customer-coordinate overlap."""
from collections import Counter
from pathlib import Path
import json
import sys
import tarfile
import prepare_reserved as reserve
import prepare_panel as panel
from reserved_lock import paired,read,save
HERE=Path(__file__).resolve().parent
PRIMARY=HERE/"reserved_validation"
DEST=HERE/"xtw_low_overlap"
ORDER=(439,449,459,469,480,491,502,513,524,536,548,561,573,586,599,613,627,641,655,670,685,701,716,733,749)
def positions(path):
    _,_,rows=panel.archives.xtw(path.read_bytes())
    return Counter(tuple(row[1:3]) for row in rows[1:101])

def main():
    raw=DEST/"raw";raw.mkdir(exist_ok=True,parents=True)
    a=read(PRIMARY/"source_audit.json")
    low=[c for c,n in a["overlap_with_most_similar_previous_case"].items() if c.startswith("xtw_") and n<=10]
    if len(low)!=7:raise ValueError("Expected the seven registered low-overlap cases")
    prior_paths={}
    for path in HERE.parent.rglob("XTW-n*.vrp"):
        if not path.is_relative_to(DEST):
            prior_paths.setdefault(path.stem,path)
    previous={key:positions(path) for key,path in prior_paths.items()}
    original=set(path.stem for path in (PRIMARY/"raw").glob("XTW-n*.vrp"))
    before={k:v for k,v in previous.items() if k not in original}
    def closest(v,known):
        return max(((sum((v&w).values()),k) for k,w in known.items()),key=lambda x:(x[0],x[1]))
    previous_matches={title:closest(previous[title],before) for title in sorted(original)}
    retained={f"xtw_{title[4:].lower()}_p100" for title,(k,_) in previous_matches.items() if k<=10}
    if retained!=set(low):raise ValueError("Customer-only panel disagrees with all archived XTW sources")
    reserve.DEST=DEST
    entries=[];log=[]
    for index,n in enumerate(ORDER):
        title=f"XTW-n{n}";entry=reserve.cvrpl((title,798+index))
        locations=positions(raw/(title+".vrp"));overlap,name=closest(locations,previous)
        log.append(dict(source=title,most_similar=name,common_customer_positions=overlap,included=overlap<=10))
        if overlap<=10:
            entries.append(entry);previous[title]=locations
        if len(entries)==3:break
    if len(entries)!=3:raise ValueError("Published sequence gave fewer than three eligible geometries")
    panel.HERE=DEST
    data,admission,geos=panel.keep(entries,set())
    save(DEST/"selection.json",dict(original_customer_matches=previous_matches,retained=sorted(retained),screened=log,added=sorted(data["cases"])))
    save(DEST/"heldout_sources.json",data)
    eligible=len(admission)==3 and all(not z["infeasible_singletons"] for z in admission.values())
    save(DEST/"heldout_admission.json",dict(status="PASS" if eligible else "STOP_INFEASIBLE",cases=admission))
    if not eligible:raise ValueError("A preregistered case failed common feasibility")
    # Use the submitted external solver and all its old-version comparators byte for byte.
    cfg=read(PRIMARY/"reserve02.json");solver=DEST/"solver";solver.mkdir(exist_ok=True)
    with tarfile.open(HERE/"reserve02_inputs.tar.gz") as arc:
        unchanged=[n for n in cfg["files"] if n not in ("runner.py","heldout_sources.json","heldout_admission.json") and not n.startswith("inputs/")]
        for n in unchanged:
            (solver/n).write_bytes(arc.extractfile(n).read())
    (DEST/"locked_method.json").write_bytes((solver/"locked_method.json").read_bytes())
    paths={n:"xtw_low_overlap/solver/"+n for n in unchanged}
    paths.update({"runner.py":"development_runner.py","heldout_sources.json":"xtw_low_overlap/heldout_sources.json","heldout_admission.json":"xtw_low_overlap/heldout_admission.json"})
    for k in data["derived_instances"].values():paths[k["file"]]="xtw_low_overlap/"+k["file"]
    locked=read(DEST/"locked_method.json")
    arms=[locked["candidate"],locked["matched"],"DIST","DIST_SLOT","ALNS","VNS","NDS-Econ","CaR","DeepACO","POMO","AttentionModel","HYBRID","HYBRID_D"]
    cfg.update(files=list(paths),file_sources=paths,blocks=paired(data["cases"],locked["seeds"],arms))
    save(HERE/"xtwsupp01.json",cfg);save(DEST/"xtwsupp01.json",cfg)
    print("Seven retained",sorted(retained),"three additional",log)
if __name__=="__main__":main()
