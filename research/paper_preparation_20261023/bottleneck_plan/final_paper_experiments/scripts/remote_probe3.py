# -*- coding: utf-8 -*-
"""抓取 repeat02 工作区 experiment_scope/profit_core 与输入目录清单。"""
import json
import shlex
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

PYTHON = "/home/hzeng/envs/MASKCO_env/bin/python"
PROJECT = "/home/hzeng/project/MASKCO-Main"
WS = PROJECT + "/C-VRP_Cold-chainVehicleRoutingProblem/research/fixed_adapt_repeat_20261022/repeat02/workspace"

PROBE = r'''
import json
from pathlib import Path
ws = Path(%(ws)r)
sp = ws/"C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_profit"
print("workspace:", ws.is_dir())
print("scripts:", sp.is_dir())
if sp.is_dir():
    for name in ("experiment_scope.py","profit_core.py","timed_search.py","profit_search.py"):
        p = sp/name
        print("==", name, p.is_file())
        if p.is_file():
            text = p.read_text()
            import re
            m = re.search(r"def identity\(.*?(?=\ndef )", text, re.S)
            print(m.group(0)[:800] if m else "identity not found")
    regs = [p for p in sp.glob("*.json") if p.name in ("case_registry.json","inputs.json","registry.json")]
    for p in regs[:5]:
        print("REG", p.name, p.stat().st_size)
inputs = ws/"C-VRP_Cold-chainVehicleRoutingProblem/research/fixed_adapt_repeat_20261022/repeat02/outputs/panel"
print("panel dir:", inputs.is_dir())
if inputs.is_dir():
    print("panel entries:", sorted(x.name for x in inputs.iterdir())[:20])
''' % dict(ws=WS)


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
