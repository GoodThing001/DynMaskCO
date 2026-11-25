# -*- coding: utf-8 -*-
"""远端目录布局只读探测 v2。"""
import json
import shlex
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

PYTHON = "/home/hzeng/envs/MASKCO_env/bin/python"

PROBE = r'''
import json
from pathlib import Path
r = Path("/home/hzeng/project/MASKCO-Main")
def tree(p, depth=2, limit=60):
    out=[]
    if not p.is_dir(): return out
    try:
        for x in sorted(p.iterdir()):
            if len(out)>=limit: out.append("..."); break
            if x.is_dir():
                out.append(x.name+"/")
                if depth>1: out += ["  "+y for y in tree(x, depth-1, limit-len(out))]
            else:
                out.append(x.name)
    except PermissionError:
        out.append("<permission denied>")
    return out
print(json.dumps({
 "perishable": tree(r/"_maskco_perishable_external_20261119", 2),
 "unified": tree(r/"_maskco_unified_external_20261120", 1),
 "cross": tree(r/"_maskco_cross_dataset_20261119", 1),
 "distance_slot": tree(r/"_maskco_distance_slot_20261119", 1),
 "recent5": tree(r/"_maskco_recent5_20261120", 1),
 "ofs01r": tree(r/"_ofs01r", 2, 25),
}))
'''

def main():
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        out = transport.remote_exec(client, PYTHON + " -B -c " + shlex.quote(PROBE))
    finally:
        client.close()
    print(out)


if __name__ == "__main__":
    main()
