"""One-shot environment probe for either configured server (no shell quoting involved).

Usage:
    python tools/_probe.py [config-json]      # default .vscode/sftp.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _srv import run  # noqa: E402

PROBES = [
    ('uname', 'uname -a'),
    ('whoami', 'whoami; id'),
    ('home', 'echo HOME=$HOME; pwd'),
    ('project_listing', 'ls -la $HOME/project 2>&1 | head -30'),
    ('repo_listing', 'ls -1 $HOME/project/MASKCO-Main 2>&1 | head -40'),
    ('python', 'command -v python python3; python3 -V 2>&1; python -V 2>&1'),
    ('conda', 'ls -d $HOME/miniconda3 $HOME/anaconda3 /opt/conda 2>&1'),
    ('conda_envs', '$HOME/miniconda3/bin/conda env list 2>&1 || true'),
    ('gpu', 'nvidia-smi --query-gpu=index,name,memory.total,driver_version '
            '--format=csv 2>&1'),
    ('cpu_mem', 'nproc; free -g 2>&1 | head -3'),
    ('disk', 'df -h $HOME 2>&1 | head -3'),
]

# The 3080 host turned out to be a Windows box whose SSH shell is PowerShell, so the unix probe
# set above produces "command not found" for uname/head/nproc/free/df and must not be read as a
# broken server. These probes ask the same questions in PowerShell.
WINDOWS_PROBES = [
    ('whoami', 'whoami'),
    ('home', 'echo $env:USERPROFILE'),
    ('os', '(Get-CimInstance Win32_OperatingSystem).Caption; $PSVersionTable.PSVersion.ToString()'),
    ('user_dir', 'Get-ChildItem -Force $env:USERPROFILE | Select-Object -ExpandProperty Name'),
    ('project_dir', 'if (Test-Path $env:USERPROFILE\\project) '
                    '{ Get-ChildItem $env:USERPROFILE\\project | '
                    'Select-Object -ExpandProperty Name } else { "NO_PROJECT_DIR" }'),
    ('python', 'Get-Command python,python3,conda -ErrorAction SilentlyContinue | '
               'Select-Object -ExpandProperty Source; python -V'),
    ('pip_list_head', 'python -m pip list 2>&1 | Select-Object -First 25'),
    ('gpu', 'nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv'),
    ('cpu_mem', 'Write-Output ("cores=" + (Get-CimInstance Win32_ComputerSystem).NumberOfLogicalProcessors); '
                'Write-Output ("ram_gb=" + [math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,1))'),
    ('disk', 'Get-PSDrive -PSProvider FileSystem | '
             'Select-Object Name,@{n="free_gb";e={[math]::Round($_.Free/1GB,1)}}'),
    ('long_paths', '(Get-ItemProperty "HKLM:\\SYSTEM\\CurrentControlSet\\Control\\FileSystem").LongPathsEnabled'),
    ('git', 'Get-Command git -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source'),
]


def main():
    cfg = sys.argv[1] if len(sys.argv) > 1 else None
    shell = sys.argv[2] if len(sys.argv) > 2 else 'unix'
    probes = WINDOWS_PROBES if shell == 'windows' else PROBES
    out = {}
    for name, cmd in probes:
        try:
            out[name] = run(cmd, timeout=60, config=cfg)
        except Exception as exc:                      # noqa: BLE001
            out[name] = 'PROBE_FAILED: %s: %s' % (type(exc).__name__, exc)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
