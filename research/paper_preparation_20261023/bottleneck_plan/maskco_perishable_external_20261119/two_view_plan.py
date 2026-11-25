from pathlib import Path
import json
from directed_decoder_fit import HERE,read
def main():
    prior=read(HERE/"results/viewprobe01/analysis.json")
    if prior["selected"] is not None or any(s["means"]["oracle"]<=s["means"]["distance"]+1.e-6 for s in prior["sources"].values()):
        print("Second-pass condition did not trigger")
        return
    cfg=read(HERE/"viewprobe01.json")
    for i,b in enumerate(cfg["blocks"]):
        r=read(HERE/f"results/viewprobe01/out/{i}.json");b["first_index"]=i
        b["contexts"]=[{k:z[k] for k in ("state","rep","group","mask","original_U")} for z in r["rows"]]
    cfg.update(worker="two_view_probe.py",purpose="paired second independent reconstruction");cfg["files"].append("two_view_probe.py")
    for name,b in (("viewpairqual01",cfg["blocks"][:1]),("viewpair01",cfg["blocks"])):
        (HERE/(name+".json")).write_text(json.dumps(dict(cfg,blocks=b),indent=2),encoding="utf8")
if __name__=="__main__":main()
