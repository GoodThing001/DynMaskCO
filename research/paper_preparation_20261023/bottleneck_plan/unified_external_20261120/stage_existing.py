"""Install only previously tested baseline code and registered inputs already on this host."""
from pathlib import Path
import json
import shutil

HERE=Path(__file__).resolve().parent
PROJECT=HERE.parents[1]

def main():
    cfg=json.loads((HERE/"phase.json").read_text())
    recent=PROJECT/"_maskco_recent5_20261120"/"benchmark01"
    reserve=PROJECT/"_maskco_perishable_external_20261119"/"reserve02"
    assets=recent.parent/"assets"
    if not assets.is_dir():raise RuntimeError("Reference assets not available")
    for executable in set(cfg["interpreters"].values()):
        if not Path(executable).is_file():raise FileNotFoundError(executable)
    for base,expected in ((recent,540),(reserve,81)):
        state=json.loads((base/"status.json").read_text())
        if not state["end"] or state["failed"] or (state["cells"] if base==recent else state["complete"])!=expected:
            raise ValueError(f"Incomplete reference {base.name}")
    for n in cfg["reused_recent_files"]:
        shutil.copy2(recent/n,HERE/n)
    catalog=json.loads((recent/"heldout_sources.json").read_text())
    wanted={b["case"] for b in cfg["blocks"]}
    if wanted!=set(catalog["derived_instances"]):raise ValueError("Different comparison panel")
    for c in sorted(wanted):
        name=catalog["derived_instances"][c]["file"]
        dest=HERE/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(recent/name,dest)
    shutil.copy2(reserve/"locked_method.json",HERE/"locked_method.json")
    print("30 registered input projections and deployed solver dependencies staged")

if __name__=="__main__":main()

