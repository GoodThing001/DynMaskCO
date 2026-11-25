# -*- coding: utf-8 -*-
"""复现 P4 admission 的 install 失败并打印中间类型。"""
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
import sys, traceback
from pathlib import Path
sp = Path("__WS__")/"C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_profit"
sys.path.insert(0, str(sp))
from profit_core import fold_definition, load_original, load_config
import profit_core as first_core
print("after dump import: ProfitTask =", type(first_core.ProfitTask).__name__)
here = Path("/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p4/smoke01")
sys.path.insert(0, str(here))
import external_common
import external_common as ec
import inspect
src = inspect.getsource(ec.install)
print("install source lines:", src.splitlines()[0])
# 直接复现 install 前半段
import json
manifest = json.loads((here/"locked_method.json").read_text())
sys.path.insert(0, str(Path("/home/hzeng/project/MASKCO-Main/_maskco_cross_dataset_20261119")))
import compare as study
print("study.REPEAT:", study.REPEAT)
prefix = study.REPEAT if "SLOT" in study.MASK_ARMS else study.RUN10
sys.path.insert(0, str(prefix/study.EXT/"scripts/static_profit"))
import profit_core as core
print("after install-style import: core file =", core.__file__)
print("core.ProfitTask =", type(core.ProfitTask).__name__, repr(core.ProfitTask)[:60])
base = core.ProfitTask
try:
    class LockedTask(base):
        pass
    print("class OK")
except TypeError as e:
    print("CLASS FAIL:", e)
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
