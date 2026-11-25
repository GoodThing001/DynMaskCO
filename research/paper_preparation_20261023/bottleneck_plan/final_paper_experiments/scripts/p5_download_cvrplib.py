# -*- coding: utf-8 -*-
"""P5-A 数据登记：远端下载 CVRPLIB Uchoa X 实例与 BKS，SHA 锁定。

来源：https://galgos.inf.puc-rio.br/cvrplib/（X 集合，set-17；实例 158–257）
论文：New benchmark instances for the Capacitated Vehicle Routing Problem,
      European Journal of Operational Research 2017, DOI 10.1016/j.ejor.2016.08.012
"""
import hashlib
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
import hashlib, json, os, urllib.request, ssl, zipfile, io
from pathlib import Path
ctx = ssl.create_default_context()
dst = Path("/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5")
(dst/"cvrplib").mkdir(parents=True, exist_ok=True)
def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research; one-time benchmark download)"})
    return urllib.request.urlopen(req, timeout=120, context=ctx).read()
meta = {}
for kind, url in [("instances", "https://galgos.inf.puc-rio.br/cvrplib/en/download/instance-set/17"),
                  ("bks", "https://galgos.inf.puc-rio.br/cvrplib/en/download/instance-set-bks/17")]:
    raw = fetch(url)
    print(kind, "bytes:", len(raw), "head:", raw[:120])
    path = dst/"cvrplib"/(kind + ".zip")
    path.write_bytes(raw)
    meta[kind] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw), url=url)
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            names = z.namelist()
            print(kind, "members:", len(names), "sample:", names[:6])
            meta[kind]["members"] = names
    except Exception as e:
        print(kind, "zip error:", type(e).__name__, str(e)[:120])
(dst/"cvrplib"/"source_manifest.json").write_text(json.dumps(dict(
    source="https://galgos.inf.puc-rio.br/cvrplib/",
    paper="Uchoa et al., New benchmark instances for the Capacitated Vehicle Routing Problem, EJOR 2017",
    doi="10.1016/j.ejor.2016.08.012",
    set_name="X", set_id=17, first_instance_id=158, n_instances=100,
    files=meta), indent=2, ensure_ascii=False), encoding="utf-8")
print("MANIFEST", meta)
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
