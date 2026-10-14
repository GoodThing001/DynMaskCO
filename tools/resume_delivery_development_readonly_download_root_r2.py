"""Bound SFTP read-ahead to32 chunks; unchanged read-only recovery and raw bytes."""
from pathlib import Path
import ast
import resume_delivery_development_readonly_download_root_v1 as base
SOURCE=Path(base.__file__).read_text()
OLD="""        with s.open(remote,'rb') as f:
            if size>=65536:f.prefetch(file_size=size)
            return f.read()"""
NEW="""        with s.open(remote,'rb') as f:
            chunks=[]
            for start in range(0,size,1024*1024):
                end=min(size,start+1024*1024)
                vectors=[(offset,min(32768,end-offset)) for offset in range(start,end,32768)]
                data=list(f.readv(vectors))
                if sum(map(len,data))!=end-start:raise ValueError('Incomplete remote read block')
                chunks.extend(data)
                if size>16*1024*1024 and (start//(1024*1024))%16==0:
                    print(json.dumps({'readonly_large_file_bytes_read':end,'total_bytes':size}),flush=True)
            return b''.join(chunks)"""
assert SOURCE.count(OLD)==1
SOURCE=SOURCE.replace(OLD,NEW)
SOURCE=SOURCE.replace("'READONLY_TRANSFER_RECOVERY.json'","'READONLY_TRANSFER_RECOVERY_R2.json'")
ast.parse(SOURCE)
if __name__=='__main__':
    namespace={'__name__':'bounded_readonly_transfer','__file__':str(Path(__file__).resolve())}
    exec(compile(SOURCE,str(Path(__file__).resolve()),'exec'),namespace)
    raise SystemExit(namespace['main']())
