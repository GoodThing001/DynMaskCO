"""Resume missing immutable bytes with SFTP prefetch; never execute remote work."""
import hashlib,json,posixpath
from datetime import datetime,timezone
import run_delivery_fresh_development_root_v1 as run
def main():
    out=run.LOCAL/'downloaded'
    cfg=json.loads(run.t.CONFIG.read_text());base=posixpath.join(cfg['remotePath'].rstrip('/'),run.REL)
    c=run.t.connect();s=c.open_sftp();kept=[];added=[]
    def raw(remote):
        size=s.stat(remote).st_size
        with s.open(remote,'rb') as f:
            if size>=65536:f.prefetch(file_size=size)
            return f.read()
    try:
        if json.loads(raw(posixpath.join(base,'RUN_EXIT.json')))['exit_code']!=0:raise ValueError('Nonzero run cannot be silently accepted')
        names=[(name,posixpath.join(base,name)) for name in ('RUN_EXIT.json','LAUNCH.json','WORKER_IDENTITY.json','run.stdout.log','run.stderr.log')]
        names +=[(name,posixpath.join(base,'output',name)) for name in s.listdir(posixpath.join(base,'output')) if name.endswith('.json')]
        for name,remote in names:
            local=out/name
            if local.exists():
                if local.stat().st_size!=s.stat(remote).st_size:raise ValueError('Incomplete prior file retained; recover separately:'+name)
                kept.append(name);continue
            payload=raw(remote)
            with local.open('xb') as f:f.write(payload)
            added.append(name)
        manifest=json.loads((run.LOCAL/'REMOTE_ENGINEERING_REGISTRATION.json').read_text());checks={}
        for name,expected in manifest['source_sha256'].items():checks[name]=hashlib.sha256(raw(posixpath.join(base,name))).hexdigest()==expected
        with (out/'ACTUAL_END_SOURCE_CHECK.json').open('x') as f:json.dump(checks,f,indent=2)
        receipt={'read_at_utc':datetime.now(timezone.utc).isoformat(),'previous_complete_files_preserved':kept,'new_files_downloaded':added,
          'source_check_failed':sum(not v for v in checks.values()),'experiment_reexecuted':False,'remote_bytes_overwritten':False}
        with (run.LOCAL/'READONLY_TRANSFER_RECOVERY.json').open('x') as f:json.dump(receipt,f,indent=2)
        print(json.dumps({'previous_files_preserved':len(kept),'missing_files_downloaded':len(added),'source_failed':receipt['source_check_failed']}));return int(not all(checks.values()))
    finally:s.close();c.close()
if __name__=='__main__':raise SystemExit(main())
