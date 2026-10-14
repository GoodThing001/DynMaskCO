"""Read-only SFTP evidence recovery; keep original server/local archives untouched."""
from pathlib import Path,PurePosixPath
import sys,json,hashlib,paramiko
E=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(E.parent/'tools'))
from _srv import _cfg
host,user,password=_cfg()
client=paramiko.SSHClient();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(host,port=22,username=user,password=password,timeout=30)
sftp=client.open_sftp()
root=PurePosixPath('/home/hzeng/project/MASKCO-Main/B_REPRO_CLEAN')
dest=E/'03_负结果线/恢复证据/B_REPRO_CLEAN_20261009'
manifest=[]
def walk(folder):
    import stat
    for attr in sftp.listdir_attr(str(folder)):
        p=folder/attr.filename
        if stat.S_ISDIR(attr.st_mode):
            if attr.filename not in ('__pycache__','.git','node_modules'):walk(p)
        elif p.suffix in ('.py','.json','.txt') and attr.st_size<=2_000_000:
            rel=p.relative_to(root)
            if '..' in rel.parts:raise ValueError('Unsafe path')
            with sftp.open(str(p),'rb') as f:data=f.read()
            out=dest/str(rel);out.parent.mkdir(parents=True,exist_ok=True)
            with out.open('xb') as f:f.write(data)
            manifest.append({'remote':str(p),'relative':str(rel),'bytes':len(data),
                             'sha256':hashlib.sha256(data).hexdigest()})
try:walk(root)
finally:sftp.close();client.close()
with (dest/'RECOVERY_MANIFEST.json').open('x',encoding='utf-8') as f:json.dump({'files':manifest,
 'original_training_source_hash_match':False,'scope':'small source/JSON/text runtime reconstruction evidence; not from-zero training identity'},f,ensure_ascii=False,indent=2)
print(json.dumps({'files':len(manifest),'bytes':sum(x['bytes'] for x in manifest),
 'repro_check_recovered':any(x['relative']=='repro_check.txt' for x in manifest),'archive':str(dest)},ensure_ascii=False))
