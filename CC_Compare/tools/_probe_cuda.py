"""Probe torch/CUDA availability in whatever conda env runs this file.

Used to decide whether the 4 torch providers of the GPU leg (mvmoe / routefinder /
cada / rrnco) can run with --provider-device cuda on the 3080 box, instead of the
--provider-device cpu that the generator currently hardcodes.
"""
import sys

print("PY", sys.executable)
try:
    import torch
except Exception as exc:  # noqa: BLE001
    print("TORCH_IMPORT_FAIL", type(exc).__name__, exc)
    raise SystemExit(0)

print("TORCH", torch.__version__)
print("CUDA_BUILD", torch.version.cuda)
avail = torch.cuda.is_available()
print("CUDA_AVAIL", avail)
if avail:
    print("DEV", torch.cuda.get_device_name(0))
    print("MEM_GB", round(torch.cuda.get_device_properties(0).total_memory / 1024 ** 3, 2))
