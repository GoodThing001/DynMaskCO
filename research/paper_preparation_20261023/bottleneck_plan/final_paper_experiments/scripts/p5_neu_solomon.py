# -*- coding: utf-8 -*-
"""探测 NEU Solomon 原始主页的解文件链接。"""
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
try:
    html = get("http://w.cba.neu.edu/~msolomon/problems.htm")
    print("bytes:", len(html))
    links = re.findall(r'href="([^"]+)"', html)
    print([l for l in links if "sol" in l.lower() or "opt" in l.lower()][:30])
    i = html.find("c101")
    print(html[i-200:i+300])
except Exception as e:
    print("FAIL:", type(e).__name__, str(e)[:150])
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
