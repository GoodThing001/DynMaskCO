"""Read-only identities of this process's loaded modules and native mapped files."""
import hashlib,os,sys
from pathlib import Path
import psutil
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def capture(extra=()):
    paths={str(Path(sys.executable).resolve())};modules={};absent_cached=[];unbacked=[]
    for name,m in tuple(sys.modules.items()):
        f=getattr(m,'__file__',None);cached=getattr(m,'__cached__',None)
        if f and Path(f).is_file():
            p=str(Path(f).resolve());paths.add(p);modules[name]=p
            if Path(p).suffix.lower()=='.pyc':
                try:
                    import importlib.util
                    source=Path(importlib.util.source_from_cache(p))
                    if source.is_file():paths.add(str(source.resolve()))
                except ValueError:pass
        if cached:
            if Path(cached).is_file():paths.add(str(Path(cached).resolve()))
            else:absent_cached.append(str(cached))
    for item in psutil.Process().memory_maps(grouped=False):
        path=item.path
        if path and Path(path).is_file():paths.add(str(Path(path).resolve()))
        elif path:unbacked.append(path)
    for f in extra:
        p=Path(f)
        if not p.is_file():raise FileNotFoundError(p)
        paths.add(str(p.resolve()))
    return dict(pid=os.getpid(),files={p:sha(p) for p in sorted(paths)},module_files=modules,absent_optional_cache_paths=sorted(set(absent_cached)),unbacked_mapping_labels=sorted(set(unbacked)))
