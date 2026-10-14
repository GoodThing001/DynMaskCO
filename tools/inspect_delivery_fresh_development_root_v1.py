"""Read only complete raw files from sole active development; no mutable means."""
import json,posixpath
from datetime import datetime,timezone
import run_delivery_fresh_development_root_v1 as run
def main():
    cfg=json.loads(run.t.CONFIG.read_text());base=posixpath.join(cfg['remotePath'].rstrip('/'),run.REL)
    c=run.t.connect();s=c.open_sftp()
    try:
        names=s.listdir(posixpath.join(base,'output'));complete=[];records=[]
        for name in names:
            if not name.endswith('.json'):continue
            if name.startswith('progress_') or any(name.startswith(a+'_') for a in ('paired_objective','reconstruction_control','pretrained','random','nonlearning','gls')):
                with s.open(posixpath.join(base,'output',name),'rb') as f:
                    try:row=json.loads(f.read())
                    except json.JSONDecodeError:continue
                if name.startswith('progress_'):complete.append(row['complete_instances'])
                elif 'arm' in row:records.append(row)
        result={'observed_utc':datetime.now(timezone.utc).isoformat(),'complete_instances':max(complete,default=0),
          'complete_arm_rows':len(records),'timeouts_seen':sum(r['timeout'] for r in records),'remote_result_exists':'RESULT.json' in names,
          'science_means_not_computed_on_incomplete_rows':True}
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        with (run.LOCAL/('READONLY_PROGRESS_'+stamp+'.json')).open('x') as f:json.dump(result,f,indent=2)
        print(json.dumps(result))
    finally:s.close();c.close()
if __name__=='__main__':main()
