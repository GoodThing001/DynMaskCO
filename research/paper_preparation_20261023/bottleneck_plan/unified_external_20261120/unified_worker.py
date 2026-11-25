"""Execute a registered endpoint through its original interpreter and solver."""
import json
import os
from pathlib import Path
import sys
from future_dispatch import execute, check

HERE=Path(__file__).resolve().parent

def main():
    i,method=sys.argv[1:]
    cfg=json.loads((HERE/"phase.json").read_text())
    b=cfg["blocks"][int(i)]
    if method not in b["methods"] or cfg["seconds"]!=60:raise ValueError((i,method))
    os.environ["RECENT5_ASSETS"]=str(HERE.parents[1]/"_maskco_recent5_20261120"/"assets")
    row=execute(method,b,cfg)
    check(row,method,b,sorted(os.sched_getaffinity(0)))
    row.setdefault("budget_s",cfg["seconds"])
    with (HERE/"out"/f"{i}_{method}.json").open("x") as f:
        json.dump(row,f,allow_nan=False)

if __name__=="__main__":main()

