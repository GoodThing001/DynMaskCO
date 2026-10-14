"""Preserve missing optional psutil failure and use Windows native affinity."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];D=ROOT/'research/review_3080_20261010/delivery'
old=ROOT/'tools/qualify_delivery_retained_teacher_root_v1.py';module=D/'retained_edge_teacher_engineering_root_v1/retained_gls.py'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
with (module.parent/'ENGINEERING_ENTRY_FAILURE.json').open('x') as f:
    json.dump({'exit':1,'error':'ModuleNotFoundError: psutil','phase':'before registration and before starting any worker or search','source_sha256':sha(old),'module_sha256':sha(module),'science_parameters_changed':False,'recovery_directory':'retained_edge_teacher_engineering_root_v2'},f,indent=2)
dest=D/'retained_edge_teacher_engineering_root_v2';dest.mkdir(exist_ok=False);shutil.copy2(module,dest/module.name)
source=old.read_text().replace("OUT=D/'retained_edge_teacher_engineering_root_v1'","OUT=D/'retained_edge_teacher_engineering_root_v2'")
source=source.replace('import psutil;psutil.Process().cpu_affinity([7]);boot()','actual_cpu=affinity();boot()')
source=source.replace("'cpu':[7]","'cpu':actual_cpu").replace("'actual_parent_worker_cpu':[7]","'actual_parent_worker_cpu':actual_cpu")
helper='''def affinity():
    import ctypes
    api=ctypes.WinDLL('kernel32',use_last_error=True)
    api.GetCurrentProcess.restype=ctypes.c_void_p
    api.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
    api.GetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]
    h=api.GetCurrentProcess()
    if not api.SetProcessAffinityMask(h,1<<7):raise ctypes.WinError(ctypes.get_last_error())
    own=ctypes.c_size_t();system=ctypes.c_size_t()
    if not api.GetProcessAffinityMask(h,ctypes.byref(own),ctypes.byref(system)):raise ctypes.WinError(ctypes.get_last_error())
    result=[i for i in range(64) if own.value&(1<<i)]
    if result!=[7]:raise RuntimeError('Actual charged CPU affinity mismatch')
    return result

'''
source=source.replace('def worker():',helper+'def worker():')
with (ROOT/'tools/qualify_delivery_retained_teacher_root_v2.py').open('x',encoding='utf-8') as f:f.write(source)
print('Preserved pre-run dependency failure; isolated native-affinity recovery prepared.')
