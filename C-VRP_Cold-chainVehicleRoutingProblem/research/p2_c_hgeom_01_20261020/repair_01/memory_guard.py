"""Unqualified repair candidate. No native case was run; preflight BLOCKED.

The candidate adds an independent per-process committed-memory quota. It NEVER
accepts a margin on JobMemoryLimit or discards PeakJobMemoryUsed. Construction
requires a hash-bound, compiled SDK checker; none was produced in this repair.
"""
from __future__ import annotations
import ctypes as C
from ctypes import wintypes as W
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent


def load_frozen_native():
    identity = json.loads((ORIGINAL/'run_identity.json').read_text(encoding='utf-8'))
    path = ORIGINAL/'windows_resources.py'
    rel = path.relative_to(ORIGINAL.parents[2]).as_posix()
    if hashlib.sha256(path.read_bytes()).hexdigest() != identity['actual_package_files'][rel]:
        raise RuntimeError('IDENTITY_MISMATCH: original native module')
    spec = importlib.util.spec_from_file_location('_hgeom_original_native', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class SDKChecker:
    def __init__(self, module, binary, expected_sha256):
        binary = Path(binary)
        if not binary.is_file():
            raise RuntimeError('RESOURCE_MONITOR_UNAVAILABLE: compiled SDK checker missing')
        if hashlib.sha256(binary.read_bytes()).hexdigest() != expected_sha256:
            raise RuntimeError('IDENTITY_MISMATCH: SDK checker')
        self.dll = C.WinDLL(str(binary))
        self.dll.sdk_layout.argtypes = [C.POINTER(C.c_ulonglong)]
        self.dll.sdk_layout.restype = None
        self.dll.sdk_snapshot.argtypes = [W.HANDLE, C.POINTER(C.c_ubyte), W.DWORD,
                                         C.POINTER(C.c_ulonglong)]
        self.dll.sdk_snapshot.restype = W.DWORD
        layout = (C.c_ulonglong*9)()
        self.dll.sdk_layout(layout)
        ext = module.Extended
        expected = [C.sizeof(module.Basic), C.sizeof(ext), ext.basic.offset, ext.io.offset,
                    ext.process_memory.offset, ext.job_memory.offset,
                    ext.peak_process.offset, ext.peak_job.offset, C.sizeof(C.c_size_t)]
        if list(layout) != expected:
            raise RuntimeError('RESOURCE_MONITOR_UNAVAILABLE: SDK ABI mismatch')
        self.module = module

    def snapshot(self, handle):
        raw = (C.c_ubyte*C.sizeof(self.module.Extended))()
        native = (C.c_ulonglong*7)()
        error = self.dll.sdk_snapshot(handle, raw, len(raw), native)
        if error:
            raise OSError(error, 'SDK snapshot failed')
        decoded = self.module.Extended.from_buffer_copy(bytes(raw))
        fields = [int(decoded.basic.flags), int(decoded.basic.active_processes),
                  int(decoded.process_memory), int(decoded.job_memory),
                  int(decoded.peak_process), int(decoded.peak_job)]
        if fields != list(native)[:6] or native[6] != len(raw):
            raise RuntimeError('RESOURCE_MONITOR_UNAVAILABLE: same-snapshot ABI disagreement')
        return decoded


def proposed_tree(limit_bytes, checker_binary, checker_sha256):
    """NOT VALIDATED. Call only after all preflight/identity gates pass."""
    original = load_frozen_native()
    checker = SDKChecker(original, checker_binary, checker_sha256)
    tree = original.NativeTree(limit_bytes)
    info = checker.snapshot(tree.handle)
    info.basic.flags |= 0x100  # JOB_OBJECT_LIMIT_PROCESS_MEMORY
    info.process_memory = int(limit_bytes)//2
    original.NativeTree._ok(tree.k.SetInformationJobObject(
        tree.handle, 9, C.byref(info), C.sizeof(info)))
    readback = checker.snapshot(tree.handle)
    if (readback.job_memory != limit_bytes or readback.process_memory != limit_bytes//2
            or readback.basic.active_processes != 2 or not readback.basic.flags & 0x100
            or not readback.basic.flags & 0x200 or readback.peak_job > limit_bytes):
        raise RuntimeError('RESOURCE_MONITOR_UNAVAILABLE: exact quota/peak gate failed')
    return tree, checker
