import ctypes
from ctypes import wintypes as W
from pathlib import Path

class FileInfo(ctypes.Structure):
    _fields_ = [('attributes',W.DWORD),('creation',W.FILETIME),('access',W.FILETIME),('write',W.FILETIME),('volume',W.DWORD),('size_high',W.DWORD),('size_low',W.DWORD),('links',W.DWORD),('index_high',W.DWORD),('index_low',W.DWORD)]

K=ctypes.WinDLL('kernel32',use_last_error=True)
K.CreateFileW.argtypes=[W.LPCWSTR,W.DWORD,W.DWORD,W.LPVOID,W.DWORD,W.DWORD,W.HANDLE]; K.CreateFileW.restype=W.HANDLE
K.CloseHandle.argtypes=[W.HANDLE]; K.CloseHandle.restype=W.BOOL
K.GetFileInformationByHandle.argtypes=[W.HANDLE,ctypes.POINTER(FileInfo)]; K.GetFileInformationByHandle.restype=W.BOOL

def owned(path,base):
    p=Path(path).resolve(); b=Path(base).resolve()
    if not p.is_relative_to(b): raise ValueError('Lease target outside owned copy tree')
    return p

def acquire(path,base):
    p=owned(path,base)
    h=K.CreateFileW(str(p),0x80000000,1,None,3,0x80,None)
    if h==ctypes.c_void_p(-1).value: raise ctypes.WinError(ctypes.get_last_error())
    return h

def info(h):
    x=FileInfo()
    if not K.GetFileInformationByHandle(h,ctypes.byref(x)): raise ctypes.WinError(ctypes.get_last_error())
    return {'volume':int(x.volume),'file_index':(int(x.index_high)<<32)|int(x.index_low),'size':(int(x.size_high)<<32)|int(x.size_low),'links':int(x.links)}

def release(h):
    if not K.CloseHandle(h): raise ctypes.WinError(ctypes.get_last_error())
