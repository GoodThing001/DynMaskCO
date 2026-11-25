# -*- coding: utf-8 -*-
"""诊断：工作区 profit_core.ProfitTask 类型与 external_common.install 失败原因。"""
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
import sys
from pathlib import Path
ws = Path("__WS__")
sp = ws/"C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_profit"
sys.path.insert(0, str(sp))
import profit_core as core
print("module:", core.__file__)
print("ProfitTask type:", type(core.ProfitTask).__name__, repr(core.ProfitTask)[:80])
import inspect
print("is class:", inspect.isclass(core.ProfitTask))
'''

def main():
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        out = transport.remote_exec(client, PYTHON + " -B -c " + shlex.quote(PROBE.replace("__WS__", WS)))
    finally:
        client.close()
    print(out)


if __name__ == "__main__":
    main()
