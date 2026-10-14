"""Loaded-file identity for small, no-model engineering stages."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
import psutil

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(262144), b''):
            h.update(b)
    return h.hexdigest()

def save(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def initialize():
    psutil.Process().cpu_affinity([7])
    ctypes.WinDLL('SHCore.dll')
    psutil.Process().username()
    psutil.Process().cmdline()
    psutil.Process().memory_maps()
    json.loads(json.dumps({'initialize': True}))

def identity():
    p = psutil.Process()
    return {'pid': p.pid, 'create_time': p.create_time(), 'cmdline': p.cmdline(), 'owner': p.username(), 'cpu_affinity': p.cpu_affinity()}

def closure(extra=()):
    files = {Path(sys.executable).resolve(), Path(__file__).resolve()}
    modules = {}
    for name, m in list(sys.modules.items()):
        raw = getattr(m, '__file__', None)
        if raw and Path(raw).is_file():
            p = Path(raw).resolve()
            modules[name] = str(p)
            files.add(p)
        cached = getattr(m, '__cached__', None)
        if cached and Path(cached).is_file():
            files.add(Path(cached).resolve())
    for row in psutil.Process().memory_maps():
        p = Path(row.path)
        if p.is_file():
            files.add(p.resolve())
    files.update(Path(p).resolve() for p in extra)
    return {'files': {str(p): sha(p) for p in sorted(files)}, 'modules': modules, 'identity': identity(), 'scope': 'actual loaded files for this no-model workload, not full scientific chain'}

def same(a, b):
    return a['files'] == b['files'] and a['modules'] == b['modules']
