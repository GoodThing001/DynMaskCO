# -*- coding: utf-8 -*-
"""检查远端 PDF 工具与已下载文件。"""
import shlex
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

PYTHON = "/home/hzeng/envs/MASKCO_env/bin/python"

PROBE = r'''
import shutil, subprocess
from pathlib import Path
print("pdf exists:", Path("/tmp/solomon_table.pdf").is_file(),
      Path("/tmp/solomon_table.pdf").stat().st_size if Path("/tmp/solomon_table.pdf").is_file() else 0)
for tool in ("pdftotext", "7z"):
    print(tool, shutil.which(tool))
try:
    import pdfminer
    print("pdfminer OK")
except ImportError as e:
    print("pdfminer missing")
try:
    import pypdf
    print("pypdf OK")
except ImportError as e:
    print("pypdf missing")
try:
    import PyPDF2
    print("PyPDF2 OK")
except ImportError as e:
    print("PyPDF2 missing")
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
