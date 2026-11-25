# -*- coding: utf-8 -*-
"""远端 tmux：下载 Maastricht PDF 并 pdftotext 提取 Solomon 最优表。"""
import shlex
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

PYTHON = "/home/hzeng/envs/MASKCO_env/bin/python"

SCRIPT = r'''
import subprocess, urllib.request, ssl
from pathlib import Path
ctx = ssl.create_default_context()
url = "https://cris.maastrichtuniversity.nl/ws/portalfiles/portal/669165/guid-9f287323-4672-421b-9cef-fa9de04f5a91-ASSET1.0.pdf"
req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
raw = urllib.request.urlopen(req, timeout=180, context=ctx).read()
Path("/tmp/solomon_table.pdf").write_bytes(raw)
subprocess.run(["pdftotext", "/tmp/solomon_table.pdf", "/tmp/solomon_table.txt"], check=True)
text = Path("/tmp/solomon_table.txt").read_text(encoding="utf-8", errors="replace")
i = text.find("Global Optima")
print(text[i:i+2500] if i >= 0 else text[:2500])
'''

def main():
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        with client.open_sftp() as s:
            with s.open("/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5/fetch_optima.py", "w") as f:
                f.write(SCRIPT)
        out = transport.remote_exec(client, "cd /home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5 && tmux new-session -d -s fetch_optima '" +
                                    PYTHON + " -B fetch_optima.py > fetch_optima.log 2>&1'")
        print("tmux:", out)
    finally:
        client.close()


if __name__ == "__main__":
    main()
