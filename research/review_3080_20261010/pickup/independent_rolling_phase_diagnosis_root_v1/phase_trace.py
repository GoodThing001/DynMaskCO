"""Bounded one-writer telemetry; no memory atomicity or durability claim."""
import mmap,os,struct,time
from pathlib import Path
HEADER=struct.Struct('<8sIIQQ32s');RECORD=struct.Struct('<QQ16siiQQ8s');CAP=512;SIZE=HEADER.size+CAP*RECORD.size
_file=None;_buffer=None;_count=0
def initialize(path):
    global _file,_buffer,_count
    _file=Path(path).open('r+b');_buffer=mmap.mmap(_file.fileno(),SIZE);_count=0
    _buffer[:64]=HEADER.pack(b'PKPHv001',CAP,0,os.getpid(),time.perf_counter_ns(),b'\0'*32)
def emit(stage,index=-1,decision='none',value0=0,value1=0):
    global _count
    if _buffer is None:raise RuntimeError('trace not initialized')
    if _count>=CAP:raise OverflowError('bounded trace capacity')
    label=stage.encode('ascii')
    if len(label)>16:raise ValueError('phase label exceeds width')
    b={'reject':0,'accept':1,'none':-1}[decision];offset=64+64*_count
    _buffer[offset:offset+64]=RECORD.pack(_count+1,time.perf_counter_ns(),label.ljust(16,b'\0'),int(index),b,int(value0),int(value1),b'\0'*8)
    _count+=1;struct.pack_into('<I',_buffer,12,_count)
def close():
    global _buffer,_file
    if _buffer is not None:_buffer.flush();_buffer.close();_buffer=None
    if _file is not None:_file.close();_file=None
