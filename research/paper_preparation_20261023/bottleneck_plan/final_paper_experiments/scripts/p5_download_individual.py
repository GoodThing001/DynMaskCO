# -*- coding: utf-8 -*-
"""P5-A：远端逐实例下载 CVRPLIB X 集 100 例 + BKS，逐件 SHA 登记。"""
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
import hashlib, json, re, time, urllib.request, ssl
from pathlib import Path
ctx = ssl.create_default_context()
dst = Path("/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5/cvrplib")
(dst/"instances").mkdir(exist_ok=True)
(dst/"bks").mkdir(exist_ok=True)
def fetch(url, retries=3):
    for i in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research; benchmark download)"})
            return urllib.request.urlopen(req, timeout=90, context=ctx).read()
        except Exception as e:
            if i == retries - 1:
                raise
            time.sleep(2)
inst = {}
bks = {}
problems = []
# X 集合实例 id 158..257（X-n101-k25 .. X-n1001-k43，共 100 例）
for i in range(158, 258):
    raw = fetch(f"https://galgos.inf.puc-rio.br/cvrplib/en/download/instance/{i}")
    text = raw.decode("utf-8", "replace")
    m = re.search(r"NAME\s*:\s*(\S+)", text)
    name = m.group(1) if m else "unknown-%d" % i
    if not name.startswith("X-"):
        problems.append((i, name, len(raw)))
    p = dst/"instances"/(name + ".vrp")
    p.write_bytes(raw)
    inst[name] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), instance_id=i)
    time.sleep(0.15)
for i in range(158, 258):
    raw = fetch(f"https://galgos.inf.puc-rio.br/cvrplib/en/download/bks/{i}")
    text = raw.decode("utf-8", "replace")
    m = re.search(r"NAME\s*:\s*(\S+)", text)
    name = m.group(1) if m else None
    if name is None:
        problems.append(("bks", i, len(raw)))
        continue
    p = dst/"bks"/(name + ".sol")
    p.write_bytes(raw)
    bks[name] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), instance_id=i)
    time.sleep(0.15)
print("instances:", len(inst), "bks:", len(bks), "problems:", problems)
print("names:", sorted(inst)[:5], "...", sorted(inst)[-3:])
manifest = json.loads((dst/"source_manifest.json").read_text())
manifest["individual_download"] = dict(status="OK" if not problems else "PARTIAL",
                                       problems=problems,
                                       instances=inst, bks=bks)
(dst/"source_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
print("DONE")
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
