"""Cross-fit stage-specific heatmap calibration across the two training geometries."""
import json
from pathlib import Path
from statistics import mean

HERE=Path(__file__).resolve().parent
study=HERE.parent/"maskco_heatmap_calibration_20261119"
def read(p):return json.loads(p.read_text(encoding="utf8"))

def main():
    path=study/"results/traincal01"
    plan=read(study/"traincal01.json")
    rows=[read(path/"out"/f"{i}.json") for i in range(len(plan["blocks"]))]
    if read(path/"status.json")["complete"]!=12 or [w["case"] for w in rows]!=[b["case"] for b in plan["blocks"]]:
        raise ValueError("Missing original training opportunity labels")
    groups={"R":[r for r in rows if r["case"].startswith("r") and not r["case"].startswith("rc")],
            "RC":[r for r in rows if r["case"].startswith("rc")]}
    alphas=rows[0]["alphas"]
    def best(records,key):
        return min(range(len(alphas)),key=lambda i:(mean(r[key][i] for r in records),i))
    out={}
    for origin,target in (("R","RC"),("RC","R")):
        tr,te=groups[origin],groups[target]
        fixed=best(tr,"loss")
        idx={k:best(tr,k) for k in ("initial","improved")}
        fixed_regret=mean(r["loss"][fixed] for r in te)
        staged=mean(r[k][idx[k]] for r in te for k in idx)
        out[origin+"_to_"+target]=dict(fixed_alpha=alphas[fixed],stage_alphas={k:alphas[i] for k,i in idx.items()},
                                         fixed_regret_U=fixed_regret,staged_regret_U=staged,gain_U=fixed_regret-staged)
    result=dict(source="traincal01",tasks=sum(len(v) for v in groups.values()),opportunities=sum(r["customers"] for r in rows),
                heldout_training_geometries=out,advance=all(x["gain_U"]>1e-6 for x in out.values()))
    (HERE/"stage_calibration.json").write_text(json.dumps(result,indent=2),encoding="utf8")
    print(json.dumps(result,indent=2))
if __name__=="__main__":main()
