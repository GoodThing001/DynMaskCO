import json
from directed_decoder_fit import HERE, read


def main():
    if read(HERE/"results/global_split01/analysis.json")["passed"]:
        raise ValueError("Global masked order already qualified")
    cfg=read(HERE/"remaskprobe01.json")
    cfg.update(worker="mask_context_size.py",purpose="identical 20 movers with and without hidden bystanders")
    cfg["files"]=[p for p in cfg["files"] if p!="iterative_remask.py"]+["mask_context_size.py"]
    for name,blocks,seeds in (("contextqual01",cfg["blocks"][:1],[55231]),("contextprobe01",cfg["blocks"],[55231,55232,55233])):
        with (HERE/(name+".json")).open("x",encoding="utf8") as out:
            json.dump(dict(cfg,blocks=blocks,seeds=seeds),out,indent=2)


if __name__=="__main__":
    main()
