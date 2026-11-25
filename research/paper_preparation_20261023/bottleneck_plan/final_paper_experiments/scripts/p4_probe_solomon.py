# -*- coding: utf-8 -*-
"""P4 预检：Solomon C1/C2 案例在冻结工作区中的身份、fold 角色与冷链转换存在性。"""
import json
import shlex
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

PYTHON = "/home/hzeng/envs/MASKCO_env/bin/python"
WS = "/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/fixed_adapt_repeat_20261022/repeat02/workspace"

PROBE = r'''
import json, sys
from pathlib import Path
ws = Path("__WS__")
sp = ws/"C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_profit"
sys.path.insert(0, str(sp))
from profit_core import fold_definition, load_original, load_config
print("fold0:", json.dumps(fold_definition(0)))
print("fold3:", json.dumps(fold_definition(3)))
for fold in (0, 3):
    fd = fold_definition(fold)
    for role in ("train", "validation", "test"):
        cases = fd.get(role, [])
        print("fold" + str(fold) + " " + role + ":", sorted(c for c in cases if c.startswith(("c1","c2"))))
for case in ("c105", "c106", "c107", "c108", "c109", "c205", "c206", "c207", "c208"):
    try:
        ins = load_original(case, load_config())
        rows = ins["nodes"]
        print("LOAD", case, "nodes:", len(rows), "schema:", rows[0])
    except Exception as e:
        print("FAIL", case, type(e).__name__, str(e)[:120])
'''.replace("__WS__", WS)


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
