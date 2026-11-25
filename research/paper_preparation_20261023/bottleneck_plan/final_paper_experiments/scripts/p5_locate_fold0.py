# -*- coding: utf-8 -*-
"""远端读取冻结项目 fold0 训练案例清单（repeat02 workspace 内）。"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

SHELL = r"""
W=/home/hzeng/project/MASKCO-Main/research/static_profit_20261021/fixed_adapt_repeat_20261022/repeat02/workspace
ls $W 2>/dev/null | head -5
echo ===find split/catalog===
find $W -maxdepth 6 -type f \( -name '*.json' -o -name '*.py' \) 2>/dev/null | grep -iE 'split|fold|catalog|instance|train' | head -20
"""


def main():
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        print(transport.remote_exec(client, SHELL))
    finally:
        client.close()


if __name__ == "__main__":
    main()
