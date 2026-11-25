# -*- coding: utf-8 -*-
"""检查单个 BKS 端点响应格式。"""
import shlex
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

PYTHON = "/home/hzeng/envs/MASKCO_env/bin/python"

PROBE = r'''
import urllib.request, ssl
ctx = ssl.create_default_context()
req = urllib.request.Request("https://galgos.inf.puc-rio.br/cvrplib/en/download/bks/158", headers={"User-Agent": "Mozilla/5.0"})
raw = urllib.request.urlopen(req, timeout=60, context=ctx).read()
print(raw[:600].decode("utf-8", "replace"))
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
