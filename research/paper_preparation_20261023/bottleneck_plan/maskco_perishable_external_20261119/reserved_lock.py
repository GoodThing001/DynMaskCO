"""Bind a development-qualified solver to the published complementary data and untouched seeds."""
from pathlib import Path
import json
import sys
import tarfile
HERE=Path(__file__).resolve().parent
PREV=HERE.parent/"maskco_heatmap_calibration_20261119"
DEST=HERE/"reserved_validation"
sys.path.insert(0,str(PREV))
from phase_plan import paired,read,save

def lock(phase):
    method={"disdev01":"DISAGREE","staticdev01":"STATIC","orderdev01":"ORDER_SPLIT","workdev01":"WORKMASK","refreshdev01":"REFRESH"}[phase]
    r=read(HERE/f"results/{phase}/analysis.json")
    a=read(HERE/f"results/{phase}/independent_audit.json")
    original=read(HERE/"results/replicate01/analysis.json")
    if (original["complete"]!=891 or original["passes_numerical_criteria"]["HYBRID"] or
        not r["passes_numerical_criteria"][method] or r["complete"]!=312 or a["status"]!="PASS" or a["endpoints"]!=312):raise ValueError("External lock requires a new passing method")
    if (DEST/"locked_method.json").exists():raise ValueError("A complementary validation is already locked")
    directory=DEST/"solver";directory.mkdir(parents=True,exist_ok=True)
    cfg=read(HERE/(phase+".json"))
    with tarfile.open(HERE/(phase+"_inputs.tar.gz")) as tar:
        for item in cfg["files"]:
            if item=="runner.py":continue
            target="pool_search.py" if item==cfg["worker"] else item
            (directory/target).write_bytes(tar.extractfile(item).read())
    manifest=dict(candidate=method,matched=method+"_D",development_phase=phase,development_decision="COMPLETE_PASS",development=r["sources"],
                  seeds=[33001,33002,33003],checkpoint="repeat02 f0_r0 maskco_base 1440",external_role="external_test")
    save(DEST/"locked_method.json",manifest)
    print("Locked",method)
def plan():
    source=read(DEST/"heldout_sources.json");a=read(DEST/"heldout_admission.json");lock=read(DEST/"locked_method.json")
    if a["status"]!="PASS" or len(source["cases"])!=27 or len(a["cases"])!=27:raise ValueError("Complementary sources incomplete")
    method=lock["candidate"]
    arms=[method,method+"_D","DIST","DIST_SLOT","ALNS","VNS","NDS-Econ","CaR","DeepACO","POMO","AttentionModel","HYBRID","HYBRID_D"]
    files=[p.name for p in sorted((DEST/"solver").iterdir())]
    paths={n:"reserved_validation/solver/"+n for n in files}
    paths.update({"runner.py":"../"+PREV.name+"/runner.py","external_common.py":"independent.py","independent.py":"reserved_entry.py",
                  "locked_method.json":"reserved_validation/locked_method.json","heldout_sources.json":"reserved_validation/heldout_sources.json",
                  "heldout_admission.json":"reserved_validation/heldout_admission.json"})
    with tarfile.open(PREV/"confirm01_inputs.tar.gz") as old_solver:
        for n in ("masked_repair.cpp","calibration.py","calibrated.json"):
            if old_solver.extractfile(n).read()!=(DEST/"solver"/n).read_bytes():raise ValueError("Prior solver dependencies differ")
    paths["previous_pool.py"]="../"+PREV.name+"/lock/pool_search.py"
    general=files+["runner.py","external_common.py","independent.py","previous_pool.py","locked_method.json","heldout_sources.json","heldout_admission.json"]
    derived=[q["file"] for q in source["derived_instances"].values()]
    for p in derived:paths[p]="reserved_validation/"+p
    cfg=dict(worker="independent.py",training_only=False,purpose="locked independent comparison",seconds=60,block_timeout_s=1500,files=general+derived,
             file_sources=paths,blocks=paired(source["cases"],lock["seeds"],arms))
    save(HERE/"reserve02.json",cfg)
    # Reporting tools use the same source and method registry as this manifest.
    save(DEST/"reserve02.json",cfg)
    old=read(HERE/"foodqual01.json")
    qual_sources=dict(paths)
    for n in ("heldout_sources.json","heldout_admission.json"):
        qual_sources[n]=old["file_sources"][n]
    extra=read(HERE/"qualification/heldout_sources.json")["derived_instances"]["gh_c1_4_1_p100"]["file"]
    qual_sources[extra]=old["file_sources"][extra]
    save(HERE/"reservequal01.json",dict(cfg,seconds=4,block_timeout_s=300,archived_case_acceptance=True,purpose="previously tested case, solver acceptance",
         files=general+[extra],file_sources=qual_sources,blocks=[dict(case="gh_c1_4_1_p100",seed=998,arms=[m]) for m in arms]))
if __name__=="__main__":
    if sys.argv[1]=="plan":plan()
    else:lock(sys.argv[1])
