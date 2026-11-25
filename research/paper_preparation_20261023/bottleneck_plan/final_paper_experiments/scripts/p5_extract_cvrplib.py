# -*- coding: utf-8 -*-
"""P5-A：远端解压 CVRPLIB 7z 包并登记逐文件 SHA。"""
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
import hashlib, json, subprocess, shutil
from pathlib import Path
dst = Path("/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5/cvrplib")
out = {}
for kind in ("instances", "bks"):
    arc = dst/(kind + ".zip")
    extract = dst/kind
    extract.mkdir(exist_ok=True)
    tool = shutil.which("7z") or shutil.which("7za") or shutil.which("7zr")
    if tool:
        r = subprocess.run([tool, "x", "-y", "-o" + str(extract), str(arc)], capture_output=True, text=True)
        print(kind, "7z exit:", r.returncode, r.stderr[-200:])
        out[kind] = dict(tool=tool, exit=r.returncode)
    else:
        try:
            import py7zr
            with py7zr.SevenZipFile(arc) as z:
                z.extractall(extract)
            print(kind, "py7zr OK")
            out[kind] = dict(tool="py7zr", exit=0)
        except ImportError:
            print(kind, "NO 7z TOOL")
            out[kind] = dict(tool=None, exit=-1)
files = {}
for p in sorted((dst/"instances").rglob("*")):
    if p.is_file():
        files[p.name] = dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size, path=str(p.relative_to(dst)))
bks = {}
for p in sorted((dst/"bks").rglob("*")):
    if p.is_file():
        bks[p.name] = dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size, path=str(p.relative_to(dst)))
print("instance files:", len(files), "bks files:", len(bks))
print("sample:", list(files)[:8])
manifest = json.loads((dst/"source_manifest.json").read_text())
manifest["extraction"] = out
manifest["instance_files"] = files
manifest["bks_files"] = bks
(dst/"source_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
print("MANIFEST updated; total files:", len(files) + len(bks))
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
