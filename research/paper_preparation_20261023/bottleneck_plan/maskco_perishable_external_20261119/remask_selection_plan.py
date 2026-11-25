import json
from directed_decoder_fit import HERE, SOURCES, read


def main():
    previous=read(HERE/"results/remaskprobe01/analysis.json")
    if previous["passed"] or not all(previous["sources"][s]["mean"]>0 for s in SOURCES):
        raise ValueError("Separate selection diagnostic not triggered")
    cfg=read(HERE/"remaskprobe01.json")
    cfg.update(worker="remask_selection.py",purpose="matched distance insertion after reopening selected customers")
    cfg["files"].append("remask_selection.py")
    for name,blocks,seeds in (("selectqual01",cfg["blocks"][:1],[53491]),("selectprobe01",cfg["blocks"],[53491,53492,53493])):
        with (HERE/(name+".json")).open("x",encoding="utf8") as out:
            json.dump(dict(cfg,blocks=blocks,seeds=seeds),out,indent=2)


if __name__=="__main__":
    main()
