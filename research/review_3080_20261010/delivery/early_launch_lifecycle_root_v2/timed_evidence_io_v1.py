"""Exclusive atomic evidence names; killed partial writes retain their .part bytes."""
import json
from pathlib import Path
def save(base,name,value):
    target=Path(base)/name;part=target.with_name(target.name+'.part')
    if target.exists():raise FileExistsError(target)
    with part.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)
    part.rename(target)
def save_npz(path,**arrays):
    import numpy as np
    target=Path(path);part=target.with_name(target.name+'.part')
    if target.exists():raise FileExistsError(target)
    with part.open('xb') as f:np.savez(f,**arrays)
    part.rename(target)
def save_bytes(path,value):
    target=Path(path);part=target.with_name(target.name+'.part')
    if target.exists():raise FileExistsError(target)
    with part.open('xb') as f:f.write(value)
    part.rename(target)
