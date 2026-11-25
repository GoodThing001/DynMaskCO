from directed_decoder_fit import HERE,read
import json
def main():
    before=read(HERE/"results/viewpair01/analysis.json");view=read(HERE/"results/viewprobe01/analysis.json")
    if before["passed"] or any(s["by_state"]["initial"]["eight"]<=s["by_state"]["initial"]["distance"] for s in view["sources"].values()):raise ValueError("Initial ensemble headroom missing")
    cfg=read(HERE/"viewprobe01.json");cfg.update(worker="prefix_persistence.py",purpose="common distance search after masked initial repair")
    cfg["files"].append("prefix_persistence.py")
    for ph,b,seeds,steps in (("carryqual01",cfg["blocks"][:1],[51291],4),("carryprobe01",cfg["blocks"],[51291,51292],256)):
        (HERE/(ph+".json")).write_text(json.dumps(dict(cfg,blocks=b,seeds=seeds,steps=steps),indent=2),encoding="utf8")
if __name__=="__main__":main()
