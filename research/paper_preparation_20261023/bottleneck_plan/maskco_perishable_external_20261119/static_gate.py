"""Apply the prespecified instance-encoding training diagnosis or paired development gate."""
from pathlib import Path
import json
from statistics import mean
import sys
HERE=Path(__file__).resolve().parent
def read(f):return json.loads(f.read_text(encoding="utf8"))
def main(phase):
    out=HERE/"results"/phase
    b=read(HERE/(phase+".json"))["blocks"];state=read(out/"status.json")
    if not state["end"] or state["failed"] or len(b)!=state["complete"]:raise ValueError("Incomplete panel")
    if phase=="staticprobe01":
        rows=[read(out/f"out/{i}.json") for i in range(len(b))]
        if not all(r["case"]==t["case"] and r["customers"]==120 and r["encodes"]==1 and r["forwards"]==10 for r,t in zip(rows,b)):raise ValueError("Training opportunity count differs")
        gs={}
        for group in ("r","rc"):
            rs=[r for r in rows if ("rc" if r["case"].startswith("rc") else "r")==group]
            gs[group]={m:mean(r["shortlist_regret_U"][m] for r in rs) for m in ("full","static","distance")}
        save=1-sum(r["cpu"]["static"] for r in rows)/sum(r["cpu"]["full"] for r in rows)
        gate=dict(geometries=gs,saved_cpu_fraction=save,top8_overlap=mean(r["top8_overlap"] for r in rows),
                  passed=save>=.20 and all(z["static"]<z["distance"] for z in gs.values()) and mean(z["static"]-z["full"] for z in gs.values())<=1.)
    else:
        import cache_assess
        cache_assess.main(phase)
        s=read(out/"analysis.json")["sources"]
        m={"staticscreen01":"STATIC","disscreen01":"DISAGREE","orderscreen01":"ORDER_SPLIT","workscreen01":"WORKMASK","refreshscreen01":"REFRESH"}[phase]
        heat={k:v["scores"][m]-v["scores"][m+"_D"] for k,v in s.items()}
        reference="MASK_ONLY" if m=="WORKMASK" else "HYBRID"
        vs_full={k:v["scores"][m]-v["scores"][reference] for k,v in s.items()}
        gate=dict(heat_minus_distance=heat,candidate_minus_full=vs_full,passed=len(s)==3 and min(heat.values())>1e-6 and mean(vs_full.values())>1e-6)
    (out/"gate.json").write_text(json.dumps(gate,indent=2))
    print(gate)
if __name__=="__main__":main(sys.argv[1])
