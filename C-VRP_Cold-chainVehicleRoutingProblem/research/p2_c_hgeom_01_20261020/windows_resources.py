"""Native Windows job + HARDWS_MAX limits for the entire two-process runner.

The supervisor is itself in the memory job. The job permits at most two active
processes: the supervisor and ONE worker; further child creation is refused.
Each gets a hard RSS maximum of half the aggregate ceiling. The aggregate
private commit ceiling is enforced by the job separately from resident memory.
Never import this module's control into an interactive host: call from the
dedicated supervisor process, whose noninheritable job handle closes on death.
"""
from __future__ import annotations
import ctypes as C
from ctypes import wintypes as W
import os


class Basic(C.Structure):
    _fields_ = [('process_time', C.c_longlong), ('job_time', C.c_longlong),
                ('flags', W.DWORD), ('min_ws', C.c_size_t), ('max_ws', C.c_size_t),
                ('active_processes', W.DWORD), ('affinity', C.c_size_t),
                ('priority', W.DWORD), ('scheduling', W.DWORD)]


class IO(C.Structure):
    _fields_ = [(x, C.c_ulonglong) for x in ('r', 'w', 'o', 'rb', 'wb', 'ob')]


class Extended(C.Structure):
    _fields_ = [('basic', Basic), ('io', IO), ('process_memory', C.c_size_t),
                ('job_memory', C.c_size_t), ('peak_process', C.c_size_t), ('peak_job', C.c_size_t)]


class NativeTree:
    def __init__(self, limit_bytes):
        if os.name != 'nt':
            raise OSError('RESOURCE_MONITOR_UNAVAILABLE: Windows required')
        self.k = C.WinDLL('kernel32', use_last_error=True)
        specs = {
            'CreateJobObjectW': ([C.c_void_p, W.LPCWSTR], W.HANDLE),
            'SetInformationJobObject': ([W.HANDLE, C.c_int, C.c_void_p, W.DWORD], W.BOOL),
            'QueryInformationJobObject': ([W.HANDLE, C.c_int, C.c_void_p, W.DWORD, C.c_void_p], W.BOOL),
            'AssignProcessToJobObject': ([W.HANDLE, W.HANDLE], W.BOOL),
            'GetCurrentProcess': ([], W.HANDLE),
            'OpenProcess': ([W.DWORD, W.BOOL, W.DWORD], W.HANDLE),
            'CloseHandle': ([W.HANDLE], W.BOOL),
            'SetHandleInformation': ([W.HANDLE, W.DWORD, W.DWORD], W.BOOL),
            'SetProcessWorkingSetSizeEx': ([W.HANDLE, C.c_size_t, C.c_size_t, W.DWORD], W.BOOL),
            'GetProcessWorkingSetSizeEx': ([W.HANDLE, C.POINTER(C.c_size_t), C.POINTER(C.c_size_t), C.POINTER(W.DWORD)], W.BOOL),
        }
        for name, (args, ret) in specs.items():
            fn = getattr(self.k, name)
            fn.argtypes, fn.restype = args, ret
        self.handle = self.k.CreateJobObjectW(None, None)
        self._ok(self.handle)
        self._ok(self.k.SetHandleInformation(self.handle, 1, 0))  # never inherited
        self.limit = int(limit_bytes)
        self.rss_per_process = (self.limit // 2) // 4096 * 4096
        info = Extended()
        info.basic.flags = 0x200 | 0x2000 | 0x8
        info.basic.active_processes = 2
        info.job_memory = self.limit
        self._ok(self.k.SetInformationJobObject(self.handle, 9, C.byref(info), C.sizeof(info)))
        self._ok(self.k.AssignProcessToJobObject(self.handle, self.k.GetCurrentProcess()))
        self.supervisor_ws = self.hard_ws(self.k.GetCurrentProcess())

    @staticmethod
    def _ok(value):
        if not value:
            raise C.WinError(C.get_last_error())

    def hard_ws(self, process):
        # 0x4 = QUOTA_LIMITS_HARDWS_MAX_ENABLE, no soft maximum substitution.
        self._ok(self.k.SetProcessWorkingSetSizeEx(process, 4096*16, self.rss_per_process, 0x4))
        low, high, flags = C.c_size_t(), C.c_size_t(), W.DWORD()
        self._ok(self.k.GetProcessWorkingSetSizeEx(process, C.byref(low), C.byref(high), C.byref(flags)))
        if not flags.value & 0x4 or high.value > self.rss_per_process:
            raise OSError('RESOURCE_MONITOR_UNAVAILABLE: HARDWS_MAX readback mismatch')
        return dict(min=low.value, max=high.value, flags=flags.value)

    def constrain_suspended(self, pid):
        handle = self.k.OpenProcess(0x100 | 0x400 | 0x800, False, int(pid))
        self._ok(handle)
        try:
            receipt = self.hard_ws(handle)
        finally:
            self.k.CloseHandle(handle)
        # Popen(CREATE_SUSPENDED) already inherited this job before first instruction.
        # CPython closes the primary thread handle; resume by process, pinned to NT ABI.
        nt = C.WinDLL('ntdll')
        nt.NtResumeProcess.argtypes, nt.NtResumeProcess.restype = [W.HANDLE], W.LONG
        handle = self.k.OpenProcess(0x800, False, int(pid))
        self._ok(handle)
        try:
            if nt.NtResumeProcess(handle) != 0:
                raise OSError('RESOURCE_MONITOR_UNAVAILABLE: NtResumeProcess failed')
        finally:
            self.k.CloseHandle(handle)
        return receipt

    def readback(self):
        info = Extended()
        self._ok(self.k.QueryInformationJobObject(self.handle, 9, C.byref(info), C.sizeof(info), None))
        return dict(job_limit=int(info.job_memory), active_process_limit=int(info.basic.active_processes),
                    flags=int(info.basic.flags), peak_job_commit=int(info.peak_job),
                    supervisor_hard_ws=self.supervisor_ws, worker_hard_ws_max=self.rss_per_process)

    # Do not CloseHandle here: this process belongs to the job. Kernel cleanup at
    # process termination closes this last handle and kills any remaining worker.
