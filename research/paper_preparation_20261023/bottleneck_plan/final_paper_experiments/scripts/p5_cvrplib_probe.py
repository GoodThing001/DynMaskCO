# -*- coding: utf-8 -*-
"""远端联网检查：CVRPLIB 可达性与 X 集合下载端点探测。"""
import json
import shlex
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

PYTHON = "/home/hzeng/envs/MASKCO_env/bin/python"

PROBE = r'''
import json, re, urllib.request, ssl
from pathlib import Path
ctx = ssl.create_default_context()
try:
    req = urllib.request.Request("https://galgos.inf.puc-rio.br/cvrplib/en/instances", headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(req, timeout=60, context=ctx).read().decode("utf-8", "replace")
    Path("/tmp/cvrplib_instances.html").write_text(html)
    print("PAGE bytes:", len(html))
    sets = re.findall(r'instance-set/(\d+)', html)
    print("set ids:", sorted(set(sets))[:40])
    for name in ("X-n101-k25", "X-n1001-k43", "Uchoa"):
        print(name, "found:", html.count(name))
    if html.count("X-n101-k25"):
        i = html.find("X-n101-k25")
        print(html[i-600:i+200].replace("\n", " ")[-700:])
except Exception as e:
    print("NETWORK FAIL:", type(e).__name__, str(e)[:200])
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
