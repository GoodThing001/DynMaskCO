"""Fail-closed identity of *actual* Python/native dependencies and explicit files.  r21.

r20 -> r21 is a defect fix forced by a real server crash.  The r1 CPU-only diagnostic
(run_r1/DIAG_FAILURE.traceback.txt) aborted at the P1 capture with

    File "tools/actual_loaded_closure_candidate_r20.py", line 42, in capture
        cache = _real_file(getattr(module, "__cached__", None))
    File ".../pathlib.py", line 578, in _parse_args
        a = os.fspath(a)
    TypeError: expected str, bytes or os.PathLike object, not _OpNamespace

Root cause: r20 assumed `module.__file__` / `module.__cached__` are path-like whenever
they are truthy.  A module in the real maskco import stack (a module with a module-level
`__getattr__` that fabricates attributes) returns a non-path object instead, so the
deployed closure gate crashed on the real module set -- it was never exercised against it.

Two changes, both strictly fail-closed:

  1. `_real_file` accepts only `str` / `os.PathLike`; anything else is treated as
     "no real file" instead of being handed to `Path()`.
  2. the module map records **every** loaded module name, mapped to its file or to
     `None`.  r20 only recorded modules that had a file, so a module appearing or
     disappearing without a file (exactly the class of module that crashed r20) was
     invisible to the closure comparison.

A module whose file cannot be resolved is still *seen* (its name is in the map), so
lazy loading during a timed region changes `module_files` and is rejected.
"""
import hashlib
import os
import sys
from pathlib import Path

import psutil


def _sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _real_file(raw):
    """Return the resolved path for a path-like value, else None.

    Never raises on a non-path value: module-level `__getattr__` hooks can hand back
    arbitrary objects for `__file__` / `__cached__`.
    """
    if not raw:
        return None
    if not isinstance(raw, (str, bytes, os.PathLike)):
        return None
    try:
        path = Path(raw)
        if not path.is_file():
            return None
        return str(path.resolve())
    except (OSError, ValueError, TypeError):
        return None


def capture(explicit_files):
    explicit = []
    for raw in explicit_files:
        path = _real_file(raw)
        if path is None:
            raise FileNotFoundError(raw)
        explicit.append(path)
    module_files = {}
    loaded = set()
    executable = _real_file(sys.executable)
    if executable:
        loaded.add(executable)
    loaded.update(explicit)
    for name, module in tuple(sys.modules.items()):
        try:
            primary = _real_file(getattr(module, "__file__", None))
            cache = _real_file(getattr(module, "__cached__", None))
        except Exception:
            # a module whose attribute access itself raises is still recorded by name
            primary, cache = None, None
        module_files[name] = primary          # None when the module has no real file
        if primary:
            loaded.add(primary)
        if cache:
            loaded.add(cache)
    mapped = set()
    for item in psutil.Process().memory_maps():
        path = _real_file(item.path)
        if path:
            mapped.add(path)
    loaded.update(mapped)
    loaded.discard(None)
    return {
        "file_sha256": {p: _sha(p) for p in sorted(loaded)},
        "module_files": module_files,
        "module_count": len(module_files),
        "modules_without_file": sorted(n for n, f in module_files.items() if f is None),
        "native_paths": sorted(mapped),
        "explicit_paths": sorted(explicit),
        "includes_unloaded_package_trees": False,
    }


def compare(start, end):
    a, b = start["file_sha256"], end["file_sha256"]
    changed = sorted(p for p in a.keys() & b.keys() if a[p] != b[p])
    missing = sorted(a.keys() - b.keys())
    added = sorted(b.keys() - a.keys())
    module_change = start["module_files"] != end["module_files"]
    native_change = start["native_paths"] != end["native_paths"]
    explicit_change = start["explicit_paths"] != end["explicit_paths"]
    module_added, module_dropped, module_remapped = [], [], []
    if module_change:
        sa, sb = start["module_files"], end["module_files"]
        module_added = sorted(sb.keys() - sa.keys())
        module_dropped = sorted(sa.keys() - sb.keys())
        module_remapped = sorted(k for k in sa.keys() & sb.keys() if sa[k] != sb[k])
    return {
        "qualified": not (changed or missing or added or module_change or native_change
                          or explicit_change),
        "changed_files": changed,
        "missing_files": missing,
        "new_files": added,
        "module_map_changed": module_change,
        "module_added": module_added,
        "module_dropped": module_dropped,
        "module_remapped": module_remapped,
        "native_map_changed": native_change,
        "explicit_map_changed": explicit_change,
    }
