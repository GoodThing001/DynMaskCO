"""Fail-closed identity of *actual* Python/native dependencies and explicit files.

Engineering candidate only. It cannot replace a registered scientific source gate
until the full model request independently proves startup completeness and timing.
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
    if not raw:
        return None
    path = Path(raw)
    if not path.is_file():
        return None
    return str(path.resolve())


def capture(explicit_files):
    explicit = []
    for raw in explicit_files:
        path = _real_file(raw)
        if path is None:
            raise FileNotFoundError(raw)
        explicit.append(path)
    module_files = {}
    loaded = {_real_file(sys.executable), *explicit}
    for name, module in tuple(sys.modules.items()):
        primary = _real_file(getattr(module, "__file__", None))
        cache = _real_file(getattr(module, "__cached__", None))
        if primary:
            module_files[name] = primary
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
    return {
        "qualified": not (changed or missing or added or module_change or native_change or explicit_change),
        "changed_files": changed,
        "missing_files": missing,
        "new_files": added,
        "module_map_changed": module_change,
        "native_map_changed": native_change,
        "explicit_map_changed": explicit_change,
    }
