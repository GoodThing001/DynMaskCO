"""Run tools/_bench.py on both configured servers and print their JSON side by side.

Uploads the benchmark first (it is tiny), then executes it with each host's MASKCO environment
interpreter. Linux and Windows hosts need different command syntax, which is why this lives in a
script rather than being typed at a shell where quoting would be mangled.

Usage:
    python tools/_run_bench.py
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _srv import run  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

SERVERS = {
    '5090-linux': {
        'config': '.vscode/sftp.json',
        'cmd': ('cd /home/hzeng/project/MASKCO-Main && '
                '/home/hzeng/miniconda3/envs/MASKCO_env/bin/python tools/_bench.py'),
    },
    '3080-windows': {
        'config': '.vscode/sftp.3080.json',
        'cmd': ('cd C:\\Users\\zh\\project\\MASKCO-Main; '
                '& "D:\\anaconda\\envs\\MASKCO_env\\python.exe" tools\\_bench.py'),
    },
}


def main():
    subprocess.run([sys.executable, os.path.join(HERE, '_sftp_put.py'), 'tools/_bench.py'],
                   cwd=ROOT, check=False)
    subprocess.run([sys.executable, os.path.join(HERE, '_sftp_put.py'), 'tools/_bench.py',
                    '--config', '.vscode/sftp.3080.json'], cwd=ROOT, check=False)
    results = {}
    for name, cfg in SERVERS.items():
        try:
            raw = run(cfg['cmd'], timeout=900, config=cfg['config'])
            start = raw.find('{')
            results[name] = json.loads(raw[start:]) if start >= 0 else {'raw': raw}
        except Exception as exc:                      # noqa: BLE001
            results[name] = {'PROBE_FAILED': '%s: %s' % (type(exc).__name__, exc)}
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
