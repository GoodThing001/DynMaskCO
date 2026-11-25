# -*- coding: utf-8 -*-
"""远端清理：删除已本地回收工作区的 out/、__pycache__ 与 tar 归档，释放磁盘给剩余格。

仅删除 P5 实验工作区内可重算/已回收产物；不触碰项目冻结链、作者仓库、其它用户数据。
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

P5 = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5"
SAFE = ["stage01/out", "stage02/out", "vrptw_full01/out", "smoke01", "external_precheck",
        "external_pyvrp2/out", "external01/out", "vrptw_smoke01"]

SHELL = r"""
cd /home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5
for d in stage01/out stage02/out vrptw_full01/out smoke01 external_precheck external_pyvrp2/out external01/out vrptw_smoke01; do
  rm -rf "$d" 2>/dev/null && echo "removed $d"
done
find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null
rm -f */receipts.tar.gz */dispatch.tar.gz stage01/*.tar.gz 2>/dev/null
df -h /home | tail -1
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
