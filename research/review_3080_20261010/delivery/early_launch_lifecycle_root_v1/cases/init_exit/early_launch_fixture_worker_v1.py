"""Registered startup-only real child. No model or trace kernel invocation."""
import json
import mmap
import os
import struct
import sys
import time
from pathlib import Path
import psutil

PACKAGE = Path(__file__).resolve().parent
sys.path.insert(0, str(PACKAGE.parents[3]/'tools'))
from pickup_loaded_closure_v1 import capture
from startup_native_identity_v1 import prime_native_identity


def save(base, name, value):
    with (base/name).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2)


def main():
    specpath=Path(sys.argv[1]);base=specpath.parent
    reg=json.loads((base/'EXECUTION_REGISTRATION.json').read_text(encoding='utf-8'))
    if not reg.get('early_launch_component_only') or reg.get('model_execution_authorized') is not False:
        raise RuntimeError('Fixture only; model execution prohibited')
    spec=json.loads(specpath.read_text(encoding='utf-8'))
    psutil.Process().cpu_affinity([7])
    native=prime_native_identity()
    proc=psutil.Process()
    identity=dict(pid=os.getpid(),create_time=proc.create_time(),cmdline=proc.cmdline(),username=proc.username())
    ctrl=mmap.mmap(-1,128,tagname=spec['control'])
    telemetry=mmap.mmap(-1,64+64*4096,tagname=spec['trace'])
    telemetry[:17]=b'EARLY-LAUNCH-ONLY!'
    extra=[Path(__file__),PACKAGE/'startup_native_identity_v1.py',Path(sys.argv[1]),base/'EXECUTION_REGISTRATION.json']
    capture(extra)
    source=capture(extra)
    save(base,'CHILD_SOURCE_START.json',source)
    save(base,'CHILD_READY.json',dict(identity=identity,request_sequence=struct.unpack_from('<Q',ctrl,0)[0],
        marker_sha256=__import__('hashlib').sha256(bytes(telemetry)).hexdigest(),source_files=len(source['files'])))
    if spec['arm']=='init_exit':
        end=capture(extra);save(base,'CHILD_SOURCE_END.json',end)
        telemetry.close();ctrl.close()
        raise SystemExit(7)
    struct.pack_into('<I',ctrl,12,1)
    time.sleep(20)
    raise AssertionError('Caller must reap the exact fixture child before 20 seconds')


if __name__=='__main__':
    main()
