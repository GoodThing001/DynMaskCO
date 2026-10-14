"""Identity helpers for the new CPU action-head engineering workload."""
import hashlib, importlib, json, sys
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def save(base,name,value):
    with (base/name).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def closure(extra,trees):
    import psutil
    files={Path(sys.executable).resolve()}
    files.update(Path(p).resolve() for p in extra)
    for m in list(sys.modules.values()):
        for attr in ('__file__','__cached__'):
            p=Path(getattr(m,attr,'') or '')
            if p.is_file():files.add(p.resolve())
    for name in ('numpy','jax','jaxlib','flax','optax','chex','scipy','ml_dtypes','etils','ortools','google.protobuf','psutil'):
        m=importlib.import_module(name)
        files.update(p.resolve() for p in Path(m.__file__).parent.rglob('*') if p.is_file())
    roots=list(trees)+[Path(sys.base_prefix)/'DLLs']
    for child in (Path(sys.base_prefix)/'Lib').iterdir():
        if child.name=='site-packages':continue
        if child.is_file():files.add(child.resolve())
        elif child.is_dir():roots.append(child)
    for root in roots:files.update(p.resolve() for p in Path(root).rglob('*') if p.is_file())
    files.update(p.resolve() for p in Path(sys.base_prefix).glob('python*.dll'))
    for m in psutil.Process().memory_maps():
        p=Path(m.path)
        if p.is_file():files.add(p.resolve())
    maps={}
    for name,m in list(sys.modules.items()):
        p=Path(getattr(m,'__file__','') or '')
        if p.is_file():maps[name]=str(p.resolve())
    return {'files':{str(p):sha(p) for p in sorted(files)},'module_files':maps}

def delta(start,end):
    return dict(changed=[p for p,h in start['files'].items() if end['files'].get(p)!=h],
                new_files=sorted(set(end['files'])-set(start['files'])))
