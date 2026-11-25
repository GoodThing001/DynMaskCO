"""Cross-fit matched neural and distance readouts, preserving true economic gains for evaluation."""
from pathlib import Path
import json
import sys
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
HERE=Path(__file__).resolve().parent

def read(p):return json.loads(p.read_text(encoding="utf8"))
def plan():
    q=read(HERE/"economicprobe01.json")
    q["files"]=[n for n in q["files"] if n not in ("economic_probe.py","static_probe.py","static_encoder.py","encoded_mask.py","calibration.py")]
    q["files"]+=["directional_head.py","directional_head_probe.py"]
    q["file_sources"]={k:v for k,v in q["file_sources"].items() if k in q["files"]}
    q.update(worker="directional_head_probe.py",purpose="training-only paired direction-sensitive masked readout",
             block_timeout_s=480)
    with (HERE/"directedprobe01.json").open("x",encoding="utf8") as f:json.dump(q,f,indent=2)

def fit(records,kind):
    fs=[];os=[];ws=[]
    for case,row in records:
        y=row["gains"];v=row["offset"];features=row[kind]
        if not len(y) or max(y)<1e-7:continue
        winners=np.flatnonzero(y>=max(y)-1e-6);win=winners[np.argmax(v[winners])]
        selected=np.argsort(-v,kind="stable")[:18]
        neg=selected[y[selected]<y[win]-1e-6]
        if not len(neg):continue
        fs.extend(features[win]-features[neg]);os.extend(v[win]-v[neg])
        ws.extend(np.minimum(y[win]-y[neg],50.)/len(neg))
    if not fs:raise ValueError("Missing ranked training contrasts")
    x=np.asarray(fs);off=np.asarray(os);w=np.asarray(ws);w=w/w.sum()
    def objective(beta):
        value=off+x@beta
        return (w@np.logaddexp(0,-value)+.01*(beta@beta),-(w*expit(-value))@x+.02*beta)
    sol=minimize(objective,np.zeros(x.shape[-1]),method="L-BFGS-B",jac=True)
    if not sol.success:raise ValueError(sol.message)
    return sol.x

def report():
    folder=HERE/"results/directedprobe01";q=read(HERE/"directedprobe01.json");t=read(folder/"status.json")
    if not t["end"] or t["failed"] or t["complete"]!=12:raise ValueError("Incomplete paired training readout")
    allrows=[]
    for i,b in enumerate(q["blocks"]):
        case=b["case"];h=read(folder/f"out/{i}.json")
        if h["case"]!=case or h["opportunities"]!=120 or h["forwards"]!=10:raise ValueError("Training metadata mismatch")
        with np.load(folder/f"out/{i}.npz") as z:
            data={k:z[k] for k in ("gains","offset","neural","spatial")}
            if not all(len(a)==int(z["lengths"].sum()) and np.isfinite(a).all() for a in data.values()):raise ValueError("Missing training slots")
            start=0
            for count in z["lengths"]:
                end=start+count;allrows.append((case,{k:v[start:end] for k,v in data.items()}));start=end
    out={}
    for group in ("R","RC"):
        train=[r for r in allrows if r[0].startswith("rc")==(group=="RC")]
        test=[r for r in allrows if r[0].startswith("rc")!=(group=="RC")]
        beta={k:fit(train,k) for k in ("spatial","neural")}
        percase={}
        for case,row in test:
            vs=percase.setdefault(case,{k:[] for k in ("prior","spatial","neural")})
            for k in vs:
                g=row["gains"]
                if not len(g):vs[k].append(0.);continue
                score=row["offset"]+(row[k]@beta[k] if k in beta else 0.)
                selected=np.argsort(-score,kind="stable")[:6]
                vs[k].append(float(max(g)-max(g[selected])))
        means={c:{k:float(np.mean(v)) for k,v in scores.items()} for c,scores in percase.items()}
        out[group]=dict(test_cases=means,loss={k:float(np.mean([d[k] for d in means.values()])) for k in ("prior","spatial","neural")},
                        neural_beats_spatial_cases=sum(d["neural"]<d["spatial"]-1e-6 for d in means.values()),weights={k:v.tolist() for k,v in beta.items()})
    advance=all(x["loss"]["neural"]<min(x["loss"]["prior"],x["loss"]["spatial"])-1e-6 and x["neural_beats_spatial_cases"]>=4 for x in out.values())
    (folder/"analysis.json").write_text(json.dumps(dict(opportunities=len(allrows),crossfit=out,advance=advance),indent=2))
    print([(g,v["loss"],v["neural_beats_spatial_cases"]) for g,v in out.items()],"advance",advance)
if __name__=="__main__":plan() if sys.argv[1]=="plan" else report()
