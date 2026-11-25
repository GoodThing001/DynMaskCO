# -*- coding: utf-8 -*-
"""查找 SINTEF Solomon BKS 下载链接。"""
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
ctx = ssl.create_default_context()
def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=60, context=ctx).read().decode("utf-8", "replace")
html = get("https://galgos.inf.puc-rio.br/cvrplib/en/vrptw/instances")
print("bytes:", len(html), "solomon mentions:", html.count("Solomon"), "c101:", html.count("C101"))
i = html.find("C101")
print(html[i-400:i+300].replace("\n", " ")[:700])
sets = re.findall(r'instance-set/(\d+)', html)
print("vrptw set ids:", sorted(set(sets))[:30])
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
