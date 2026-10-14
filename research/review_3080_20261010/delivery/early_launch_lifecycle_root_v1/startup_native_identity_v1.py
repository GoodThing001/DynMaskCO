"""Explicit Windows native setup before an actual starting identity snapshot.

This does not identify the loader of the old missing SHCore dependency and
does not certify a future model workload or arbitrary native libraries.
"""
import ctypes
import os
from pathlib import Path
import psutil


def prime_native_identity():
    if os.name != 'nt':
        raise RuntimeError('This registered component requires Windows')
    path = Path(os.environ['SystemRoot']) / 'System32' / 'SHCore.dll'
    native = ctypes.WinDLL(str(path.resolve()))
    owner = psutil.Process()
    owner.create_time()
    owner.cmdline()
    owner.username()
    mapped = {os.path.normcase(str(Path(x.path).resolve()))
              for x in owner.memory_maps() if x.path}
    if os.path.normcase(str(path.resolve())) not in mapped:
        raise RuntimeError('Explicit native dependency is not actually mapped')
    return native
