# -*- coding: utf-8 -*-
"""经 Wayback Machine 获取 SINTEF Solomon 100 官方最优解文件。"""
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
ctx = ssl.create_default_context()
def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=90, context=ctx).read().decode("utf-8", "replace")
# 1) 查询 wayback 快照
api = "http://archive.org/wayback/available?url=sintef.no/projectweb/top/vrptw/100-customers/"
try:
    info = json.loads(get(api))
    print(json.dumps(info))
    snap = info.get("archived_snapshots", {}).get("closest", {})
    if snap.get("url"):
        html = get(snap["url"])
        print("SNAP bytes:", len(html))
        links = re.findall(r'href="([^"]+\.(?:zip|txt))"[^>]*>([^<]*)', html)
        seen = set()
        for u, t in links:
            if ("sol" in u.lower() or "solution" in t.lower()) and u not in seen:
                seen.add(u)
                print(u, "|", t.strip()[:60])
except Exception as e:
    print("FAIL:", type(e).__name__, str(e)[:200])
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
