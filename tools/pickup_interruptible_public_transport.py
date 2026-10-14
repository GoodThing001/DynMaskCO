"""Windows bounded opaque transport. No bulk parent serialization/deserialization."""
import ctypes, msvcrt, os, struct, subprocess, time
from multiprocessing.shared_memory import SharedMemory

HEADER=struct.Struct('<BI')
CONTROL=struct.Struct('<QIdI')
MAX_PAYLOAD=2*1024*1024

def available(stream):
    count=ctypes.c_ulong()
    handle=msvcrt.get_osfhandle(stream.fileno())
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.PeekNamedPipe.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_ulong,ctypes.c_void_p,ctypes.POINTER(ctypes.c_ulong),ctypes.c_void_p]
    kernel.PeekNamedPipe.restype=ctypes.c_int
    if not kernel.PeekNamedPipe(handle,None,0,None,ctypes.byref(count),None):
        error=ctypes.get_last_error()
        if error in (109,232):return 0
        raise OSError(error,'PeekNamedPipe')
    return count.value

class PublicTransport:
    def __init__(self,program,variant=0):
        self.memory=SharedMemory(create=True,size=64)
        self.memory.buf[:]=bytes(64)
        self.process=subprocess.Popen([os.sys.executable,'-B',str(program),'--worker',self.memory.name,str(variant)],
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,stdin=subprocess.DEVNULL,
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
        self.buffer=bytearray();self.frames=[]
        try:self.ready=self.read_frame(time.perf_counter()+30)
        except BaseException:self.close();raise
        if self.ready[0]!=0:self.close();raise ValueError('Worker not ready')
    def read_frame(self,deadline):
        expected=None
        while time.perf_counter()<deadline:
            if len(self.buffer)>=HEADER.size:
                kind,size=HEADER.unpack_from(self.buffer)
                if kind not in (0,1,2,3) or size>MAX_PAYLOAD:raise ValueError('Invalid bounded frame')
                expected=HEADER.size+size
                if len(self.buffer)>=expected:
                    payload=bytes(self.buffer[HEADER.size:expected]);del self.buffer[:expected]
                    if time.perf_counter()>=deadline:raise TimeoutError('Late frame')
                    return kind,payload
            count=available(self.process.stdout)
            if count:
                chunk=os.read(self.process.stdout.fileno(),min(count,65536))
                self.buffer.extend(chunk)
                if len(self.buffer)>MAX_PAYLOAD+HEADER.size:raise ValueError('Bounded receive exceeded')
            elif self.process.poll() is not None:raise RuntimeError('Worker ended before complete frame')
            else:time.sleep(min(.001,max(0.,deadline-time.perf_counter())))
        raise TimeoutError('Opaque receive deadline')
    def ask(self,index,cap_s=10.,fault=0):
        started=time.perf_counter();deadline=started+cap_s
        # Fixed 24-byte control. The generation is published last; no pipe write.
        CONTROL.pack_into(self.memory.buf,0,0,index,deadline,fault)
        struct.pack_into('<Q',self.memory.buf,0,1)
        stages=[]
        try:
            while True:
                kind,payload=self.read_frame(deadline)
                if kind==1:stages.append(payload.decode('ascii'))
                elif kind==2:
                    return {'kind':'opaque_public_packet','bytes':payload,'stages':stages,
                            'started':started,'finished':time.perf_counter(),'deadline':deadline}
                else:raise RuntimeError('Worker error '+repr(payload[:500]))
        except (TimeoutError,ValueError,RuntimeError) as exc:
            self.close()
            return {'kind':'cancelled','error':type(exc).__name__+':'+str(exc),'stages':stages,
                    'started':started,'finished':time.perf_counter(),'deadline':deadline,
                    'worker_reaped':self.process.poll() is not None,'partial_bytes_discarded':len(self.buffer)}
    def close(self):
        if self.process.poll() is None:self.process.kill()
        self.process.wait(timeout=5)
        self.process.stdout.close();self.process.stderr.close()
        self.memory.close()
        try:self.memory.unlink()
        except FileNotFoundError:pass
