"""Pair new random masks with independent archived distance parent states."""
import json
from directed_decoder_fit import HERE, read


def main():
    carry=read(HERE/"carryprobe01.json")
    design=read(HERE/"viewprobe01.json")
    if read(HERE/"results/trajectory_choice01/analysis.json")["passed"]:
        raise ValueError("Two-trajectory selection already qualified")
    design.update(worker="iterative_remask.py",purpose="confidence-remasked second reconstruction")
    design["files"].append("iterative_remask.py")
    for i,block in enumerate(design["blocks"]):
        r=read(HERE/f"results/carryprobe01/out/{i}.json")
        if block["case"]!=r["case"] or [s["seed"] for s in r["seeds"]]!=carry["seeds"]:
            raise ValueError("Archived development identities differ")
        stages=[s["arms"]["distance"]["checkpoints"]["64"] for s in r["seeds"]]
        block["distance64"]=[z["routes"] for z in stages]
        block["U64"]=[z["U"] for z in stages]
    for name,blocks,seeds in (("remaskqual01",design["blocks"][:1],[53101]),("remaskprobe01",design["blocks"],[53101,53102])):
        path=HERE/(name+".json")
        with path.open("x",encoding="utf8") as f:
            json.dump(dict(design,blocks=blocks,seeds=seeds,states=["initial","distance64"]),f,indent=2)


if __name__=="__main__":
    main()
