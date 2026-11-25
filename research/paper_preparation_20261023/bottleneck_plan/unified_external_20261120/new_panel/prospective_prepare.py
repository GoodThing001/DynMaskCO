"""Package the admitted new projections, pairing schedule and single-core launcher."""
import ast
from collections import Counter
import json
from pathlib import Path
import random
import sys
import tarfile

HERE=Path(__file__).resolve().parent
STUDY=HERE.parent
sys.path.insert(0,str(STUDY))
from unified_plan import EXTERNAL, METHODS, read, write

SEEDS=(65001,65002,65003)


def main():
    admission=read(HERE/"heldout_admission.json")
    registry=read(HERE/"heldout_sources.json")
    inputs=registry["derived_instances"]
    cases=registry["cases"]
    if admission["status"]!="PASS" or set(cases)!=set(inputs) or set(cases)!=set(admission["cases"]):
        raise ValueError("Incomplete new-panel admission")
    previous=read(STUDY/"matrix.json")
    if set(cases)&{b["case"] for b in previous["blocks"]}:raise ValueError("Case reused from benchmark01")
    counts=Counter(row["dataset"] for row in cases.values())
    if counts!={"VRPPDL-p100":10,"XTW-p100":8,"GH-p100-block2":6}:raise ValueError(counts)
    rng=random.Random(65000)
    pairs=[(c,s) for c in sorted(cases) for s in SEEDS];rng.shuffle(pairs)
    arms=METHODS.copy();rng.shuffle(arms)
    blocks=[dict(case=c,seed=s,methods=arms[(i%16):]+arms[:(i%16)]) for i,(c,s) in enumerate(pairs)]
    schedule=dict(name="prospective01",training_only=False,seconds=60,methods=METHODS,external_methods=EXTERNAL,
                  max_cores=18,keep_free=4,exclude_active_stages=["benchmark01"],blocks=blocks,
                  interpreters=previous["interpreters"],reused_code=[
                      n for n in previous["reused_recent_files"] if n.endswith((".cpp",".py")) or n=="calibrated.json"
                  ]+["unified_worker.py","future_dispatch.py","locked_method.json"])
    write(HERE/"matrix.json",schedule)
    primary=[c for c,r in cases.items() if r["dataset"]!="GH-p100-block2"]
    extra={"GH-p100-block2":[c for c in cases if c not in primary]}
    for country,label in (("BE","比利时配送日"),("NL","荷兰配送日")):
        extra[label]=[c for c,r in cases.items() if r.get("country")==country]
    if set(pairs)!={(c,s) for c in cases for s in SEEDS} or any(len(b["methods"])!=16 or set(b["methods"])!=set(METHODS) for b in blocks):
        raise ValueError("Incomplete within-block pairing")
    write(HERE/"metadata.json",dict(cases=cases,primary=primary,stress=[c for c in cases if c not in primary],
                                  additional_panels=extra,panels={s:[c for c,r in cases.items() if r["dataset"]==s] for s in sorted(counts)}))
    package=HERE/"prospective01.tar.gz"
    items={"phase.json":HERE/"matrix.json","unified_runner.py":STUDY/"unified_runner.py",
           "stage_prospective.py":HERE/"stage_prospective.py",
           "heldout_sources.json":HERE/"heldout_sources.json","heldout_admission.json":HERE/"heldout_admission.json"}
    items.update({e["file"]:HERE/e["file"] for e in inputs.values()})
    for name,path in items.items():
        if name.endswith(".py"):ast.parse(path.read_text(encoding="utf8"))
        if not path.is_file():raise FileNotFoundError(path)
    with tarfile.open(package,"w:gz") as out:
        for name,path in items.items():out.add(path,arcname=name)
    write(HERE/"prospective_preflight.json",dict(status="PASS",cases=len(cases),blocks=len(blocks),
          endpoints=len(blocks)*16,external_methods=12,core_seconds=60*16*len(blocks),
          panel_sizes=dict(counts),package_bytes=package.stat().st_size,package_members=list(items),
          seeds=SEEDS,reserved_peer="benchmark01",max_cores=18,reserve_physical_cores=4,
          prospective_results_opened=False))
    print("PASS",dict(counts),len(blocks)*16,package.stat().st_size)


if __name__=="__main__":main()
