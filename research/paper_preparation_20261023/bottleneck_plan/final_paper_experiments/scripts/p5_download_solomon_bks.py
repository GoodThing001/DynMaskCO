# -*- coding: utf-8 -*-
"""P5-B：远端下载 CVRPLIB Solomon VRPTW 集合 BKS（逐实例）并与本地 SINTEF 原件比对。"""
import shlex
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

PYTHON = "/home/hzeng/envs/MASKCO_env/bin/python"

SCRIPT = r'''
import hashlib, json, re, time, urllib.request, ssl
from pathlib import Path
ctx = ssl.create_default_context()
dst = Path("/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5/solomon")
(dst/"instances").mkdir(parents=True, exist_ok=True)
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
# Solomon VRPTW 集：C101=实例380；56 例 id 380..435
inst, bks, problems, id_name = {}, {}, [], {}
for i in range(380, 436):
    try:
        raw = fetch(f"https://galgos.inf.puc-rio.br/cvrplib/en/download/instance/{i}")
        text = raw.decode("utf-8", "replace")
        m = re.search(r"NAME\s*:?\s*(\S+)", text)
        name = m.group(1) if m else (text.splitlines()[0].strip() if text.splitlines() else None)
        if name is None:
            problems.append(["instance", i, "no-name", len(raw)])
            continue
        id_name[i] = name
        (dst/"instances"/(name + ".vrp")).write_bytes(raw)
        inst[name] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), instance_id=i)
    except Exception as e:
        problems.append(["instance", i, type(e).__name__, str(e)[:80]])
    time.sleep(0.1)
for i in range(380, 436):
    try:
        raw = fetch(f"https://galgos.inf.puc-rio.br/cvrplib/en/download/bks/{i}")
        text = raw.decode("utf-8", "replace")
        name = id_name.get(i) or (re.search(r"NAME\s*:?\s*(\S+)", text).group(1) if re.search(r"NAME\s*:?\s*(\S+)", text) else None)
        if name is None:
            problems.append(["bks", i, "no-name", len(raw)])
            continue
        routes = [tuple(int(x) for x in line.split(":")[1].split()) for line in text.splitlines()
                  if line.startswith("Route #")]
        mc = re.search(r"Cost\s+([0-9.]+)", text)
        cost = float(mc.group(1)) if mc else None
        (dst/"bks"/(name + ".sol")).write_bytes(raw)
        bks[name] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), instance_id=i,
                         routes=routes, cost=cost, n_vehicles=len(routes))
    except Exception as e:
        problems.append(["bks", i, type(e).__name__, str(e)[:80]])
    time.sleep(0.1)
manifest = dict(status="OK" if not problems else "PARTIAL", problems=problems,
                instances=inst, bks=bks,
                source="https://galgos.inf.puc-rio.br/cvrplib/ (VRPTW Solomon set, ids 380-435)",
                note="BKS 用于评价器校验与 Gap 计算，不参与搜索初始化")
(dst/"source_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
print("DONE instances:", len(inst), "bks:", len(bks), "problems:", problems[:5])
'''

def main():
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        with client.open_sftp() as s:
            with s.open("/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5/dl_solomon_bks.py", "w") as f:
                f.write(SCRIPT)
        cmd = "cd /home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5 && " + \
              PYTHON + " -B dl_solomon_bks.py > dl_solomon_bks.log 2>&1"
        out = transport.remote_exec(client, "tmux new-session -d -s solomon_bks " + shlex.quote(cmd))
        print("tmux:", out)
    finally:
        client.close()


if __name__ == "__main__":
    main()
