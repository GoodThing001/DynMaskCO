"""Bootstrap the Windows 3080 host: create the personal/project/repo folder layout.

Mirrors the 5090 Linux layout (`/home/<user>/project/MASKCO-Main`) on the Windows box, where the
same structure is `C:\\Users\\zh\\project\\MASKCO-Main`. Everything is created with -Force so the
script is idempotent and safe to re-run.

The 3080 was found to be a Windows host whose SSH shell is PowerShell, so its commands must be
PowerShell and its SFTP root is of the form /C:/Users/zh/... - the unix commands in _probe.py do
not apply to it and their "command not found" output is not a server fault.

Usage:
    python tools/_setup_3080.py [--config .vscode/sftp.3080.json] [--dirs-only]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _srv import run  # noqa: E402

TARGET_BASE = r'C:\Users\zh\project\MASKCO-Main'
# Mirrors the 5090 repo top level so later uploads land in the expected places.
SUBDIRS = ['', 'MASKCO_code', 'MASKCO_code/ckpts', 'C-VRP_Cold-chainVehicleRoutingProblem',
           'CC_Compare', 'tools', 'data', 'B_REPRO_CLEAN']


def main():
    args = sys.argv[1:]
    cfg = '.vscode/sftp.3080.json'
    if '--config' in args:
        cfg = args[args.index('--config') + 1]
    dirs_only = '--dirs-only' in args
    out = {}

    def ps(cmd):
        return run(cmd, timeout=300, config=cfg)

    paths = [TARGET_BASE] + [TARGET_BASE + '\\' + s.replace('/', '\\')
                             for s in SUBDIRS if s]
    quoted = ','.join("'%s'" % p for p in paths)
    out['mkdir'] = ps('$p=@(%s); foreach($x in $p){ New-Item -ItemType Directory -Force '
                      '-Path $x | Out-Null }; "created"' % quoted)
    out['tree'] = ps('Get-ChildItem -Force -Recurse -Depth 1 "%s" | '
                     'Select-Object -ExpandProperty FullName' % TARGET_BASE)
    out['free_space'] = ps('Get-PSDrive C,D | Select-Object Name,'
                           '@{n="free_gb";e={[math]::Round($_.Free/1GB,1)}} | '
                           'Format-Table -AutoSize | Out-String')
    out['python'] = ps('python -V; python -m pip -V')
    out['long_paths'] = ps('(Get-ItemProperty '
                           '"HKLM:\\SYSTEM\\CurrentControlSet\\Control\\FileSystem").'
                           'LongPathsEnabled')
    if not dirs_only:
        out['conda_envs'] = ps('conda env list')
        # One package per line, no nested quoting hazards.
        out['packages'] = ps(
            'foreach($m in @("numpy","scipy","jax","flax","optax","pandas",'
            '"matplotlib","paramiko","pytest")){'
            ' $v = python -c "import importlib,sys;'
            ' m=importlib.import_module(sys.argv[1]);'
            ' print(getattr(m,\'__version__\',\'?\'))" $m 2>$null;'
            ' if($LASTEXITCODE -ne 0){ $v = "MISSING" };'
            ' Write-Output ("{0,-12} {1}" -f $m, $v) }')
        out['cuda'] = ps('nvidia-smi | Select-Object -First 12 | Out-String')
        # The box turned out to already have MASKCO envs; find out which is usable.
        out['existing_maskco_envs'] = {}
        for env_name in ('MASKCO_env', 'MASKCO_py310'):
            py = r'D:\anaconda\envs\%s\python.exe' % env_name
            out['existing_maskco_envs'][env_name] = ps(
                'if (Test-Path "%s") { & "%s" -V;'
                ' & "%s" -c "import jax,flax,optax,numpy;'
                ' print(\'jax\',jax.__version__);'
                ' print(\'flax\',flax.__version__);'
                ' print(\'optax\',optax.__version__);'
                ' print(\'numpy\',numpy.__version__);'
                ' print(\'devices\',jax.devices())" } else { "env python not found" }'
                % (py, py, py))
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
