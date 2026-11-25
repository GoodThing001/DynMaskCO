"""Score the transport of neural calibration beyond an economic-only edge prior."""
from pathlib import Path
from statistics import mean
import json
import sys
from economic_edges import ALPHAS
HERE=Path(__file__).resolve().parent
def read(f):return json.loads(f.read_text(encoding="utf8"))
def main(action):
    if action=="plan":
        q=read(HERE/"staticprobe01.json")
        q["files"]+=["economic_edges.py","economic_probe.py"];q.update(worker="economic_probe.py",purpose="training-only directional savings and conditional MASKCO calibration")
        with (HERE/"economicprobe01.json").open("x",encoding="utf8") as f:json.dump(q,f,indent=2)
        return
    folder=HERE/"results/economicprobe01";t=read(folder/"status.json")
    if t["failed"] or t["complete"]!=12 or not t["end"]:raise ValueError("Incomplete training calibration")
    rs=[read(folder/f"out/{i}.json") for i in range(12)];out={}
    if any(r["customers"]!=120 or r["forwards"]!=10 for r in rs):raise ValueError("Training observation count differs")
    def loss(records,key):return mean(r["loss"][key] for r in records)
    for src in ("R","RC"):
        tr=[r for r in rs if r["case"].startswith("rc")==(src=="RC")]
        te=[r for r in rs if r not in tr]
        alpha=min(ALPHAS,key=lambda a:(loss(tr,str(a)),a))
        vals={k:loss(te,k) for k in ("0.0","distance","legacy")}
        vals.update(alpha=alpha,chosen=loss(te,str(alpha)))
        out[src]=vals
    advance=all(v["chosen"]<min(v["0.0"],v["distance"])-1e-6 for v in out.values()) and mean(v["legacy"]-v["chosen"] for v in out.values())>0
    doc=dict(crossfit_train_to_other_geometry=out,advance=advance,examples=sum(r["customers"] for r in rs))
    (folder/"analysis.json").write_text(json.dumps(doc,indent=2));print(doc)
if __name__=="__main__":main(sys.argv[1])
