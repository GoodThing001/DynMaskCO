"""Fixed shared-memory request/response; pipe I/O only in declared setup/audit."""
import os,struct,subprocess,time
from multiprocessing.shared_memory import SharedMemory
from pickup_interruptible_public_transport import HEADER,CONTROL,MAX_PAYLOAD,available

class JointTransport:
    def __init__(self,program,variant=0):
        self.memory=SharedMemory(create=True,size=64)
        self.memory.buf[:]=bytes(64)
        self.output=SharedMemory(create=True,size=MAX_PAYLOAD+HEADER.size)
        self.output.buf[:HEADER.size]=bytes(HEADER.size)
        self.process=subprocess.Popen([os.sys.executable,'-B',str(program),'--worker',self.memory.name,str(variant),self.output.name],
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,stdin=subprocess.DEVNULL,
            env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
        self.buffer=bytearray()
        try:self.ready=self.read_frame(time.perf_counter()+30)
        except BaseException:self.close();raise
        if self.ready[0]!=0:self.close();raise ValueError('Worker not ready')
    def read_frame(self,deadline):
        # Setup or offline source audit only. Never called by ask().
        while time.perf_counter()<deadline:
            if len(self.buffer)>=HEADER.size:
                kind,size=HEADER.unpack_from(self.buffer)
                if kind not in (0,3) or size>MAX_PAYLOAD:raise ValueError('Invalid setup/audit frame')
                length=HEADER.size+size
                if len(self.buffer)>=length:
                    payload=bytes(self.buffer[HEADER.size:length]);del self.buffer[:length]
                    return kind,payload
            count=available(self.process.stdout)
            if count:self.buffer.extend(os.read(self.process.stdout.fileno(),min(count,65536)))
            elif self.process.poll() is not None:raise RuntimeError('Worker ended before audit frame')
            else:time.sleep(.001)
        raise TimeoutError('Setup/audit deadline')
    def ask(self,index,cap_s=10.,fault=0):
        started=time.perf_counter();deadline=started+cap_s
        CONTROL.pack_into(self.memory.buf,0,0,index,deadline,fault)
        struct.pack_into('<Q',self.memory.buf,0,1)
        stages=[];labels={1:'private_projection',2:'admission',3:'proposal',4:'scoring',5:'certification',6:'compile',7:'result_encoding',8:'opaque_transfer'}
        try:
            while time.perf_counter()<deadline:
                phase=struct.unpack_from('<I',self.memory.buf,32)[0]
                for i in range(1,phase+1):
                    if labels[i] not in stages:stages.append(labels[i])
                kind,size=HEADER.unpack_from(self.output.buf)
                if kind:
                    if kind!=2 or size>MAX_PAYLOAD:raise ValueError('Invalid bounded shared frame')
                    payload=bytes(self.output.buf[HEADER.size:HEADER.size+size])
                    if time.perf_counter()>=deadline:raise TimeoutError('Late shared-memory copy')
                    return {'kind':'opaque_joint_response','bytes':payload,'stages':stages,'started':started,
                            'finished':time.perf_counter(),'deadline':deadline,'timed_request_pipe_IO':False}
                if self.process.poll() is not None:raise RuntimeError('Worker ended before complete shared frame')
                time.sleep(min(.001,max(0.,deadline-time.perf_counter())))
            raise TimeoutError('Shared response deadline')
        except (TimeoutError,ValueError,RuntimeError) as exc:
            partial=struct.unpack_from('<I',self.memory.buf,40)[0]
            self.close()
            return {'kind':'cancelled','error':type(exc).__name__+':'+str(exc),'stages':stages,
                    'started':started,'finished':time.perf_counter(),'deadline':deadline,
                    'worker_reaped':self.process.poll() is not None,'partial_bytes_discarded':partial,'timed_request_pipe_IO':False}
    def close(self):
        if self.process.poll() is None:self.process.kill()
        self.process.wait(timeout=5)
        self.process.stdout.close();self.process.stderr.close()
        for memory in (self.memory,self.output):
            memory.close()
            try:memory.unlink()
            except FileNotFoundError:pass
