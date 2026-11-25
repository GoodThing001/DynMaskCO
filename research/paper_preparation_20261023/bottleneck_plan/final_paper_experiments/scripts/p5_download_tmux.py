# -*- coding: utf-8 -*-
"""P5-A 数据下载（远端 tmux 长任务版）：逐实例下载 X 集 100 例 + BKS。"""
import json
import shlex
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

PYTHON = "/home/hzeng/envs/MASKCO_env/bin/python"
DST = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5/cvrplib"

SCRIPT = r'''
import hashlib, json, re, time, urllib.request, ssl
from pathlib import Path
ctx = ssl.create_default_context()
dst = Path("__DST__")
(dst/"instances").mkdir(exist_ok=True)
(dst/"bks").mkdir(exist_ok=True)
def fetch(url, retries=4):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research; benchmark download)"})
            return urllib.request.urlopen(req, timeout=60, context=ctx).read()
        except Exception as e:
            if i == retries - 1:
                raise
            time.sleep(3)
inst, bks, problems = {}, {}, []
id_name = {}
for i in range(158, 258):
    try:
        raw = fetch(f"https://galgos.inf.puc-rio.br/cvrplib/en/download/instance/{i}")
        text = raw.decode("utf-8", "replace")
        m = re.search(r"NAME\s*:?\s*(\S+)", text)
        name = m.group(1) if m else "unknown-%d" % i
        if not name.startswith("X-"):
            problems.append(["instance", i, name, len(raw)])
        id_name[i] = name
        (dst/"instances"/(name + ".vrp")).write_bytes(raw)
        inst[name] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), instance_id=i)
    except Exception as e:
        problems.append(["instance", i, type(e).__name__, str(e)[:80]])
    time.sleep(0.1)
for i in range(158, 258):
    try:
        raw = fetch(f"https://galgos.inf.puc-rio.br/cvrplib/en/download/bks/{i}")
        text = raw.decode("utf-8", "replace")
        name = id_name.get(i)
        m = re.search(r"NAME\s*:?\s*(\S+)", text)
        if name is None:
            name = m.group(1) if m else None
        if name is None:
            problems.append(["bks", i, "no-name", len(raw)])
            continue
        routes = [tuple(int(x) for x in line.split(":")[1].split()) for line in text.splitlines()
                  if line.startswith("Route #")]
        mc = re.search(r"Cost\s+(\d+)", text)
        cost = int(mc.group(1)) if mc else None
        (dst/"bks"/(name + ".sol")).write_bytes(raw)
        bks[name] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), instance_id=i,
                         routes=routes, cost=cost)
    except Exception as e:
        problems.append(["bks", i, type(e).__name__, str(e)[:80]])
    time.sleep(0.1)
manifest = json.loads((dst/"source_manifest.json").read_text())
manifest["individual_download"] = dict(status="OK" if not problems else "PARTIAL",
                                       problems=problems, instances=inst, bks=bks)
(dst/"source_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
print("DONE instances:", len(inst), "bks:", len(bks), "problems:", problems)
'''

def main():
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        with client.open_sftp() as s:
            try:
                s.mkdir("/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5")
            except OSError:
                pass
            with s.open("/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5/dl_cvrplib.py", "w") as f:
                f.write(SCRIPT.replace("__DST__", DST))
        cmd = "cd /home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5 && " + \
              PYTHON + " -B dl_cvrplib.py > dl_cvrplib.log 2>&1"
        out = transport.remote_exec(client, "tmux new-session -d -s cvrplib_download " + shlex.quote(cmd))
        print("tmux:", out)
    finally:
        client.close()


if __name__ == "__main__":
    main()
