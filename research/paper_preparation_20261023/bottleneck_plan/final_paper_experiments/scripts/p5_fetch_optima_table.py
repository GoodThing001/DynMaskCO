# -*- coding: utf-8 -*-
"""远端下载 Maastricht 论文 PDF 并提取 Solomon Global Optima 表。"""
import shlex
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

PYTHON = "/home/hzeng/envs/MASKCO_env/bin/python"

PROBE = r'''
import urllib.request, ssl, re
ctx = ssl.create_default_context()
url = "https://cris.maastrichtuniversity.nl/ws/portalfiles/portal/669165/guid-9f287323-4672-421b-9cef-fa9de04f5a91-ASSET1.0.pdf"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
raw = urllib.request.urlopen(req, timeout=120, context=ctx).read()
Path("/tmp/solomon_table.pdf").write_bytes(raw)
print("pdf bytes:", len(raw))
# 尝试 pdftotext / pdfminer / pypdf
text = None
import shutil
for tool in ("pdftotext",):
    if shutil.which(tool):
        import subprocess
        subprocess.run([tool, "/tmp/solomon_table.pdf", "/tmp/solomon_table.txt"], check=True)
        text = Path("/tmp/solomon_table.txt").read_text(encoding="utf-8", errors="replace")
        break
if text is None:
    try:
        import pdfminer.high_level as hl
        text = hl.extract_text("/tmp/solomon_table.pdf")
    except Exception as e1:
        try:
            import pypdf
            r = pypdf.PdfReader("/tmp/solomon_table.pdf")
            text = "\n".join(p.extract_text() or "" for p in r.pages)
        except Exception as e2:
            print("no pdf tool:", type(e1).__name__, type(e2).__name__)
if text:
    Path("/tmp/solomon_table.txt").write_text(text)
    i = text.find("Global Optima")
    print(text[i:i+2000] if i >= 0 else text[:2000])
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
