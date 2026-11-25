# -*- coding: utf-8 -*-
"""P5-A 冒烟/批次监视器：状态 + 已完成单元摘要 + 失败日志尾部。"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

WORKSPACE = "/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5/{ws}"

SHELL = r"""
date -u
echo ===STATUS===
cat {base}/status.json 2>/dev/null | head -c 600 || true
echo
echo ===CELLS===
python3 - <<'EOF'
import json, glob, os
rows = []
for f in sorted(glob.glob('{base}/out/*.json')):
    try:
        r = json.load(open(f))
    except Exception:
        continue
    rows.append((r.get('case'), r.get('seed'), r.get('method'),
                 r.get('objective'), r.get('vehicles'), r.get('distance'), r.get('online_s')))
for c, s, m, o, k, d, t in rows:
    print('%-14s %d %-13s %-5s K=%-3s D=%-10s online=%.2f' % (c, s, m, o, k, (('%.2f' % d) if d is not None else d), t))
print('json cells:', len(rows))
EOF
echo ===FAILED===
cat {base}/status.json 2>/dev/null | python3 -c "import json,sys; s=json.load(sys.stdin); print(s.get('failed'))" 2>/dev/null || true
"""


def main(ws):
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        base = WORKSPACE.format(ws=ws)
        out = transport.remote_exec(client, SHELL.format(base=base))
        print(out)
    finally:
        client.close()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "smoke01")
