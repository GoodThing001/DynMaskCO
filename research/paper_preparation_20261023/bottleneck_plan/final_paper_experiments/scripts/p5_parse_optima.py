# -*- coding: utf-8 -*-
"""从提取的 PDF 文本中解析 Solomon 最优表（Global Optima 列）。"""
import re
import shlex
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

PYTHON = "/home/hzeng/envs/MASKCO_env/bin/python"

PROBE = r'''
import re
text = open("/tmp/solomon_table.txt", encoding="utf-8", errors="replace").read()
i = text.find("Global Optima")
chunk = text[max(0, i-4000):i+200]
# 打印实例行（c1xx/r1xx/rc1xx 后跟数字）
for m in re.finditer(r"\b([cr]c?\d{3})\b[^\n]*", chunk):
    print(m.group(0)[:120])
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
