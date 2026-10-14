"""Resident public-input shared transfer; no timed pipe I/O. Parent encode/decode remain synchronous."""
import json,os,pickle,queue,struct,subprocess,sys,threading,time
from multiprocessing.shared_memory import SharedMemory
CONTROL=struct.Struct('<QdII');REPLY=struct.Struct('<QII');MAX_INPUT=2*1024*1024;MAX_OUTPUT=2*1024*1024
class LiveShared:
    def __init__(self,program):
        self.control=SharedMemory(create=True,size=64);self.input=SharedMemory(create=True,size=MAX_INPUT);self.output=SharedMemory(create=True,size=MAX_OUTPUT+REPLY.size)
        self.control.buf[:]=bytes(64);self.output.buf[:REPLY.size]=bytes(REPLY.size);self.seq=0
        self.process=subprocess.Popen([sys.executable,'-B',str(program),'--worker',self.control.name,self.input.name,self.output.name],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,encoding='utf-8',env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
        self.q=queue.Queue();self.errors=[]
        def pump():
            for line in self.process.stdout:
                try:self.q.put(json.loads(line))
                except Exception:self.errors.append(line)
            self.q.put(dict(kind='closed'))
        def err():
            for line in self.process.stderr:self.errors.append(line)
        threading.Thread(target=pump,daemon=True).start();threading.Thread(target=err,daemon=True).start()
        try:self.ready=self.q.get(timeout=30)
        except queue.Empty:self.close();raise TimeoutError('setup')
        if self.ready.get('kind')!='ready':self.close();raise RuntimeError(str(self.errors))
    def ask(self,packet,deadline,reserve_s=2.,fault=0):
        # Caller started the clock before actual projection. All parent work here
        # is charged but synchronous serialization is not claimed preemptible.
        self.seq+=1;payload=pickle.dumps(packet,protocol=5)
        if len(payload)>MAX_INPUT:raise ValueError('bounded public input')
        compute_deadline=deadline-reserve_s
        if time.perf_counter()>=compute_deadline:raise TimeoutError('encode exhausted compute window')
        self.output.buf[:REPLY.size]=bytes(REPLY.size);struct.pack_into('<I',self.control.buf,32,0)
        self.input.buf[:len(payload)]=payload;CONTROL.pack_into(self.control.buf,0,0,compute_deadline,len(payload),fault)
        struct.pack_into('<Q',self.control.buf,0,self.seq)
        while time.perf_counter()<compute_deadline:
            seq,kind,size=REPLY.unpack_from(self.output.buf)
            if seq==self.seq:
                if size>MAX_OUTPUT:raise ValueError('bounded output')
                opaque=bytes(self.output.buf[REPLY.size:REPLY.size+size]);decoded=pickle.loads(opaque)
                if kind!=1:raise RuntimeError(decoded)
                decoded['deadline']=deadline
                return decoded,dict(sequence=self.seq,input_bytes=len(payload),output_bytes=size,received=time.perf_counter(),stage=struct.unpack_from('<I',self.control.buf,32)[0],timed_pipe_IO=False)
            if self.process.poll() is not None:raise RuntimeError('worker exited '+str(self.errors))
            time.sleep(min(.001,max(0.,compute_deadline-time.perf_counter())))
        stage=struct.unpack_from('<I',self.control.buf,32)[0];self.close()
        raise TimeoutError('cancelled stage='+str(stage))
    def close(self):
        if getattr(self,'closed',False):return
        self.closed=True
        if self.process.poll() is None:self.process.kill()
        self.process.wait(timeout=5)
        for memory in (self.control,self.input,self.output):
            memory.close()
            try:memory.unlink()
            except FileNotFoundError:pass
