"""Prepare a fixed-method external replication, qualify the external loader and recover the same job."""
import json
from pathlib import Path
import sys
import tarfile

HERE=Path(__file__).resolve().parent
PREV=HERE.parent/"maskco_heatmap_calibration_20261119"
sys.path.insert(0,str(PREV))
import phase_plan
import transport_heat

def read(p):return json.loads(p.read_text(encoding="utf8"))
def put(p,value):
    with p.open("x",encoding="utf8") as f:json.dump(value,f,indent=2,ensure_ascii=False)
ARMS=["HYBRID","HYBRID_D","DIST","DIST_SLOT","ALNS","VNS","NDS-Econ","CaR","DeepACO","POMO","AttentionModel"]
def plan():
    source=read(HERE/"heldout_sources.json")
    admitted=read(HERE/"heldout_admission.json")
    if admitted["status"]!="PASS" or len(source["cases"])!=27 or set(admitted["cases"])!=set(source["cases"]):raise ValueError("Incomplete external admission")
    lock=read(PREV/"locked_method.json")
    if lock["candidate"]!="HYBRID":raise ValueError("Recipe differs from previous confirmation")
    lock.update(seeds=[32001,32002,32003],original_role="archived VRPTWQ result retained",replicated_from=PREV.name)
    put(HERE/"locked_method.json",lock)
    cfg=dict(training_only=False,worker="independent.py",block_timeout_s=1500,seconds=60,purpose="locked independent comparison",blocks=phase_plan.paired(source["cases"],lock["seeds"],ARMS))
    names=["pool_search.py","masked_repair.cpp","calibrated.json","calibration.py"]
    with tarfile.open(PREV/"confirm01_inputs.tar.gz") as orig:
        for name in names:
            if orig.extractfile(name).read()!=(PREV/"lock"/name).read_bytes():raise ValueError("Stored solver differs")
    cfg["files"]=["independent.py","runner.py","heldout_sources.json","heldout_admission.json","locked_method.json"]+names
    external_paths={name:"../"+PREV.name+"/lock/"+name for name in names}
    external_paths["runner.py"]="../"+PREV.name+"/runner.py"
    cfg["file_sources"]=external_paths
    entries=[e["file"] for e in source["derived_instances"].values()]
    put(HERE/"replicate01.json",dict(cfg,files=cfg["files"]+entries))
    tech=HERE/"qualification"
    tech.mkdir(exist_ok=True)
    c="gh_c1_4_1_p100"
    old=read(PREV/"heldout_sources.json")
    unit=dict(cases={c:old["cases"][c]},derived_instances={c:old["derived_instances"][c]})
    put(tech/"heldout_sources.json",unit)
    put(tech/"heldout_admission.json",dict(status="PASS",cases={c:"previously audited"}))
    sample=unit["derived_instances"][c]["file"]
    sources=dict(external_paths,**{f:"qualification/"+f for f in ("heldout_sources.json","heldout_admission.json")})
    sources[sample]="../"+PREV.name+"/"+sample
    put(HERE/"foodqual01.json",dict(cfg,purpose="previously tested external case, software acceptance",seconds=4,block_timeout_s=240,
           archived_case_acceptance=True,files=cfg["files"]+[sample],file_sources=sources,blocks=[dict(case=c,seed=998,arms=[a]) for a in ARMS]))
    print("11 acceptance endpoints, 891 replication endpoints")
def run(action,phase):
    if phase not in ("foodqual01","replicate01","encqual01","cachequal01","cachescreen01","cachedev01","edgeprobe01","staticprobe01","staticqual01","staticscreen01","staticdev01","disqual01","disscreen01","disdev01","splitprobe01","facqual01","factorial01","reservequal01","reserve02","xtwsupp01","orderqual01","orderqual02","orderscreen01","orderdev01","trialclock01","workqual01","workscreen01","workdev01","economicprobe01","directedprobe01","maskvalue01","refreshqual01","refreshscreen01","refreshdev01","placeboqual01","placebodev01","oppqual01","opprobe01","loadqual01","loadprobe01","tailqual01","tailprobe01","cyclequal01","cycleprobe01","fpqual01","fpprobe01","regionqual01","regionprobe01","headqual01","headcapture01","mixpath01","mixscreen01","mixdev01","jointqual01","jointcapture01","utilitycapture01","skeletonqual01","skeletoncontext01","groupcapture01","viewqual01","viewqual02","viewprobe01","viewpairqual01","viewpair01","carryqual01","carryprobe01","remaskqual01","remaskprobe01","selectqual01","selectprobe01","contextqual01","contextprobe01"):raise ValueError(phase)
    if action=="submit" and phase=="contextprobe01":
        q=read(HERE/"results/contextqual01/out/0.json")
        if (q["forwards"],q["audited_candidates"])!=(6,20) or len(q["rows"])!=2 or any(set(z["arms"])!={"distance","true20","pad40a","pad40b","shuffle20"} for z in q["rows"]):
            raise ValueError("Matched 20-versus-40 context qualification incomplete")
    if action=="submit" and phase=="selectprobe01":
        q=read(HERE/"results/selectqual01/out/0.json")
        if (q["forwards"],q["audited_candidates"])!=(2,20) or any(set(z["arms"])!={"distance","neural"} or any(set(a["second"])!={"distance","neural","random","shuffled"} for a in z["arms"].values()) for z in q["rows"]):
            raise ValueError("Matched second insertion qualification incomplete")
    if action=="submit" and phase=="remaskprobe01":
        q=read(HERE/"results/remaskqual01/out/0.json")
        if (q["forwards"],q["repairs"],q["audited_candidates"])!=(6,14,14) or any(len(set(z["neural_mask"]))!=20 or set(z["neural_mask"])-set(z["mask"]) or set(z["arms"])!={"distance","updated","stale","random","shuffled"} for z in q["rows"]):
            raise ValueError("Conditional remask qualification incomplete")
    if action=="submit" and phase=="carryprobe01":
        q=read(HERE/"results/carryqual01/out/0.json")
        if q["forwards"]!=8 or q["strict_suffix_calls"]!=16 or any(set(v["checkpoints"])!={"0","4"} for v in q["seeds"][0]["arms"].values()):raise ValueError("Common continuation acceptance incomplete")
    if action=="submit" and phase in ("viewpairqual01","viewpair01"):
        prior=read(HERE/"results/viewprobe01/analysis.json")
        if prior["selected"] or any(v["means"]["oracle"]<=v["means"]["distance"]+1.e-6 for v in prior["sources"].values()):raise ValueError("Insufficient observed orientation diversity")
    if action=="submit" and phase=="viewpair01":
        q=read(HERE/"results/viewpairqual01/out/0.json")
        if q["forwards"]!=8 or len(q["rows"])!=4 or any(set(z["arms"])!={"distance","v0","v3","shuffled"} for z in q["rows"]):raise ValueError("Second-candidate acceptance incomplete")
    if action=="submit" and phase=="viewprobe01":
        q=read(HERE/"results/viewqual02/out/0.json")
        if q["forwards"]!=16 or q["identity_error"]>.002 or any(len(r["arms"])!=12 for r in q["rows"]):raise ValueError("Axis and repair acceptance incomplete")
    if action=="submit" and phase=="groupcapture01":
        earlier=read(HERE/"results/jointqual01/out/0.json")
        if earlier["contexts"]!=2 or earlier["max_logit_error"]>.002:raise ValueError("Regional encoder acceptance incomplete")
    if action=="submit" and phase=="skeletoncontext01":
        receipt=read(HERE/"results/skeletonqual01/out/0.json")
        if receipt["forwards"]!=2 or receipt["max_error"]>.002:raise ValueError("Conditional model acceptance")
    if action=="submit" and phase=="utilitycapture01":
        measurement=read(HERE/"results/jointcapture01/analysis.json")
        if measurement["requested"]!=500 or measurement["passed"]:raise ValueError("Initial whole-region diagnosis incomplete or already promoted")
    if action=="submit" and phase=="jointcapture01":
        trial=read(HERE/"results/jointqual01/out/0.json")
        if trial["contexts"]!=2 or trial["max_logit_error"]>.002:raise ValueError("Whole-region embedding acceptance failed")
    if action=="submit" and phase=="mixpath01" and not read(HERE/"results/headcapture01/customer_mixture.json")["passed"]:
        raise ValueError("Conditional customer model did not qualify")
    if action=="submit" and phase=="mixscreen01":
        q=read(HERE/"results/mixpath01/out/0.json")["arms"]
        if (set(q)!={"HEAD","PHYSICAL","ORDER_SPLIT","ORDER_SPLIT_D"} or
                any(r["status"]!="AUDITED" or not r["endpoint"]["independently_audited"] for r in q.values()) or
                q["HEAD"]["forwards"]==0 or q["PHYSICAL"]["readouts"]==0 or q["PHYSICAL"]["forwards"]!=0):
            raise ValueError("Conditional customer search paths not verified")
    if action=="submit" and phase=="mixdev01" and not read(HERE/"results/mixscreen01/analysis.json")["passed"]:
        raise ValueError("Customer-mixture screening failed")
    if action=="submit" and phase=="headcapture01":
        q=read(HERE/"results/headqual01/out/0.json")
        if q["contexts"]!=2 or q["max_logit_error"]>.002:
            raise ValueError("Frozen embedding did not reproduce checkpoint outputs")
    if action=="submit" and phase=="regionprobe01":
        q=read(HERE/"results/regionqual01/out/0.json")
        if q["forwards"]!=6 or len(q["rows"])!=2 or any(r["covered"]!=r["customers"]**2 or
                min(x["orders"] for x in r["arms"].values())!=12 for r in q["rows"]):
            raise ValueError("Overlapping regional inference or partitioning incomplete")
    if action=="submit" and phase=="fpprobe01":
        q=read(HERE/"results/fpqual01/out/0.json")
        if q["masks"]!=2 or q["max_error"]>.002 or q["neighbor_overlap"]<.995 or q["slot_overlap"]<.99:
            raise ValueError("Frozen checkpoint reconstruction failed forward equivalence")
    if action=="submit" and phase=="cycleprobe01":
        q=read(HERE/"results/cyclequal01/out/0.json")
        if (q["forwards"]!=2 or len(q["rows"])!=2 or
            any(r["cycles"]<=32 or r["feasible"]<=0 or r["feasible"]>r["cycles"] or
                max(r["gains"].values())>r["gains"]["oracle"]+1.e-7 for r in q["rows"])):
            raise ValueError("Three-route candidates or independent audit missing")
    if action=="submit" and phase=="tailprobe01":
        q=read(HERE/"results/tailqual01/out/0.json")
        if q["forwards"]!=2 or any(r["pairs"]<=4 or r["cuts"]<=r["pairs"] or max(r["gains"].values())>r["gains"]["oracle"]+1e-7 for r in q["rows"]):
            raise ValueError("Whole-route selection or economic upper bound missing")
    if action=="submit" and phase=="loadprobe01":
        import numpy as np
        q=read(HERE/"results/loadqual01/out/0.json")
        with np.load(HERE/"results/loadqual01/out/0.npz") as x:
            if (q["contexts"]!=20 or q["forwards"]!=2 or len(x["gain"])!=sum(x["lengths"])
                    or q["slots"]<=0 or q["route_requests"]<=q["slots"]
                    or not np.any(x["real"]!=x["shuffled"]) or not np.all(np.isfinite(x["gain"]) & (x["gain"]>=0))):
                raise ValueError("Strict load-conditioned labels or neural control missing")
    if action=="submit" and phase=="opprobe01":
        trial=read(HERE/"results/oppqual01/out/0.json")
        if trial["forwards"]!=6 or len(trial["rows"])!=12 or not all(set(r["options"])=={"common","distance","heat"} for r in trial["rows"]):
            raise ValueError("Matched insertion and order probe qualification incomplete")
    if action=="submit" and phase=="placebodev01":
        q=read(HERE/"results/placeboqual01/out/0.json")["arms"]
        if (set(q)!={"ALIGNED","SHUFFLED","ALIGNED_D"} or any(v["status"]!="AUDITED" for v in q.values())
                or any(q[m]["changed_top8"]<=.001 or q[m]["directional_logit_recovery_error"]>.002 for m in ("ALIGNED","SHUFFLED"))
                or q["ALIGNED_D"]["forwards"] or any(v["split_orders"]<=0 for v in q.values())):
            raise ValueError("Paired placebo qualification failed")
    if action=="submit" and phase=="refreshscreen01":
        q=read(HERE/"results/refreshqual01/out/0.json")["arms"]
        if (set(q)!={"REFRESH","REFRESH_D","HYBRID"} or any(r["status"]!="AUDITED" for r in q.values())
                or not q["REFRESH"]["refresh_comparisons"] or not q["REFRESH"]["skipped_predictions"]
                or q["REFRESH_D"]["forwards"] or q["HYBRID"]["skipped_predictions"]):
            raise ValueError("Economic refresh or full baseline path missing")
    if action=="submit" and phase=="refreshdev01":
        if not read(HERE/"results/refreshscreen01/gate.json")["passed"]:raise ValueError("Refresh screening failed")
        if read(HERE/"results/orderdev01/analysis.json")["passes_numerical_criteria"]["ORDER_SPLIT"]:raise ValueError("Earlier candidate qualified")
    if action=="submit" and phase in ("orderqual01","orderqual02"):
        d=read(HERE/"results/disdev01/analysis.json")
        if d["complete"]!=312 or d["passes_numerical_criteria"]["DISAGREE"] or not read(HERE/"results/splitprobe01/analysis.json")["advance"]:
            raise ValueError("Full development and order diagnosis did not authorize this branch")
    if action=="submit" and phase=="orderscreen01":
        q=read(HERE/"results/orderqual02/out/0.json")["arms"]
        if (set(q)!={"ORDER_SPLIT","ORDER_SPLIT_D","HYBRID"} or any(r["status"]!="AUDITED" for r in q.values()) or
            min(q[a]["split_orders"] for a in ("ORDER_SPLIT","ORDER_SPLIT_D"))<=0 or q["HYBRID"]["split_orders"]!=0):
            raise ValueError("Order splitting or baseline path missing")
    if action=="submit" and phase=="orderdev01" and not read(HERE/"results/orderscreen01/gate.json")["passed"]:
        raise ValueError("Order-split screening failed")
    if action=="submit" and phase=="workqual01" and not read(HERE/"results/trialclock01/analysis.json")["advance"]:
        raise ValueError("Training trial-clock diagnosis missing")
    if action=="submit" and phase=="workscreen01":
        q=read(HERE/"results/workqual01/out/0.json")["arms"]
        if (set(q)!={"WORKMASK","WORKMASK_D","MASK_ONLY","MASK_ONLY_D"} or any(r["status"]!="AUDITED" for r in q.values())
            or not all(q[m]["repeat_three_epochs_equal"] for m in ("WORKMASK","WORKMASK_D"))):
            raise ValueError("Repeated work-prefix acceptance failed")
    if action=="submit" and phase=="workdev01":
        if not read(HERE/"results/workscreen01/gate.json")["passed"]:raise ValueError("Work-count screening failed")
        order=read(HERE/"results/orderscreen01/gate.json")
        if order["passed"] and read(HERE/"results/orderdev01/analysis.json")["passes_numerical_criteria"]["ORDER_SPLIT"]:
            raise ValueError("An earlier candidate qualified")
    if action=="submit" and phase=="factorial01":
        q=read(HERE/"results/facqual01/out/0.json")["arms"]
        if len(q)!=8 or any(r["status"]!="AUDITED" or r["selected_s"]>8 for r in q.values()):
            raise ValueError("Factorial solver qualification failed")
        if any(q[m]["targeted_masks"] for m in ("CROSS_ONLY","CROSS_ONLY_D","HYBRID","HYBRID_D")):
            raise ValueError("Mask switch did not turn off")
        if any(q[m]["cross_offers"] for m in ("MASK_ONLY","MASK_ONLY_D","HYBRID","HYBRID_D")):
            raise ValueError("Cross switch did not turn off")
    if action=="submit" and phase in ("reserve02","xtwsupp01"):
        root=HERE/"reserved_validation"
        q=read(root/"results/reservequal01/status.json");cfg=read(HERE/"reservequal01.json")
        if q["failed"] or q["complete"]!=13 or not q["end"]:raise ValueError("External solver acceptance incomplete")
        for i,b in enumerate(cfg["blocks"]):
            arm=b["arms"][0]
            r=read(root/f"results/reservequal01/out/{i}.json")["arms"][arm]
            if r["status"]!="AUDITED" or r["role"]!="external_test" or r["selected_s"]>4:raise ValueError("External solver or comparator differs")
    if action=="submit" and phase=="disqual01":
        prior=read(HERE/"results/replicate01/analysis.json")
        if prior["complete"]!=891 or prior["passes_numerical_criteria"]["HYBRID"] or read(HERE/"results/staticscreen01/gate.json")["passed"]:
            raise ValueError("Earlier branches did not trigger disagreement development")
    if action=="submit" and phase=="disscreen01":
        q=read(HERE/"results/disqual01/out/0.json")["arms"]
        if (set(q)!={"DISAGREE","DISAGREE_D","HYBRID"} or any(r["status"]!="AUDITED" for r in q.values()) or
            q["DISAGREE"]["targeted_masks"]<1 or q["DISAGREE_D"]["targeted_masks"]<1):raise ValueError("Disagreement path incomplete")
    if action=="submit" and phase=="disdev01" and not read(HERE/"results/disscreen01/gate.json")["passed"]:
        raise ValueError("Disagreement screening failed")
    if action=="submit" and phase=="cachequal01" and not read(HERE/"results/encqual01/out/0.json")["gate"]:
        raise ValueError("Training inference comparison failed")
    if action=="submit" and phase=="staticqual01" and not read(HERE/"results/staticprobe01/gate.json")["passed"]:
        raise ValueError("Instance encoding training gate failed")
    if action=="submit" and phase=="staticscreen01":
        q=read(HERE/"results/staticqual01/out/0.json")["arms"]
        if (set(q)!={"STATIC","STATIC_D","HYBRID"} or any(r["status"]!="AUDITED" for r in q.values()) or
            not (q["STATIC"]["encodings"]==1 and q["STATIC"]["forwards"]>1 and q["STATIC_D"]["forwards"]==0)):
            raise ValueError("Instance encoder full path missing")
    if action=="submit" and phase=="staticdev01" and not read(HERE/"results/staticscreen01/gate.json")["passed"]:
        raise ValueError("Instance encoder development screening failed")
    if action=="submit" and phase=="cachescreen01":
        q=read(HERE/"results/cachequal01/out/0.json")["arms"]
        if set(q)!={"CACHEHYBRID","CACHEHYBRID_D","CYCLE_FULL"} or any(x["status"]!="AUDITED" or x["selected_s"]>24 for x in q.values()):
            raise ValueError("Full-path training acceptance failed")
        if not (q["CACHEHYBRID"]["encodings"]==10 and q["CACHEHYBRID"]["forwards"]>10 and q["CACHEHYBRID_D"]["forwards"]==0):
            raise ValueError("Mask reuse or distance ablation did not run")
    if action=="submit" and phase=="cachedev01":
        if not read(HERE/"results/cachescreen01/screen_gate.json")["passed"]:raise ValueError("Development screening gate failed")
    if action=="submit" and phase=="replicate01":
        q=read(HERE/"results/foodqual01/status.json")
        if q["failed"] or q["complete"]!=len(ARMS) or not q["end"]:raise ValueError("Full entrypoint acceptance missing")
        for i,arm in enumerate(ARMS):
            a=read(HERE/f"results/foodqual01/out/{i}.json")["arms"][arm]
            if a["role"]!="external_test" or a["status"]!="AUDITED" or a["selected_s"]>4:raise ValueError("Comparator acceptance differs")
    transport_heat.HERE=HERE/"reserved_validation" if phase in ("reservequal01","reserve02") and action!="submit" else HERE
    if phase=="xtwsupp01" and action!="submit":transport_heat.HERE=HERE/"xtw_low_overlap"
    transport_heat.DEST=transport_heat.PROJECT+"/_maskco_perishable_external_20261119"
    transport_heat.transport.LOCAL=HERE
    transport_heat.action(action,phase)
if __name__=="__main__":
    if sys.argv[1:]==["plan"]:plan()
    else:run(*sys.argv[1:])
