# -*- coding: utf-8 -*-
"""探测 CVRPLIB VRPTW BKS 挑战页面（含车辆数与距离的官方最优记录）。"""
import shlex
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

PYTHON = "/home/hzeng/envs/MASKCO_env/bin/python"

PROBE = r'''
import re, urllib.request, ssl
from pathlib import Path
ctx = ssl.create_default_context()
def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=90, context=ctx).read().decode("utf-8", "replace")
for u in ("https://galgos.inf.puc-rio.br/cvrplib/en/challenges/vrptw/overview",
          "https://galgos.inf.puc-rio.br/cvrplib/en/challenges/vrptw/score/instances"):
    try:
        html = get(u)
        print("URL:", u, "bytes:", len(html), "C101:", html.count("C101"))
        i = html.find("C101")
        if i >= 0:
            print(html[i-300:i+800].replace("\n", " "))
        break
    except Exception as e:
        print("FAIL:", u, type(e).__name__, str(e)[:100])
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
