"""Read only own bounded training files; never launch, select or overwrite."""
import hashlib, json, posixpath, sys
from datetime import datetime, timezone
from pathlib import Path
import remote_actual_fullstep_3080_v2 as t
ROOT=t.ROOT
B=ROOT/'research/review_3080_20261010/delivery/remote_finite_research_root_v2'
REL='research/review_3080_20261010/remote_finite_research_root_v2'
def main():
    c=t.connect();s=c.open_sftp()
    cfg=json.loads(t.CONFIG.read_text());base=posixpath.join(cfg['remotePath'].rstrip('/'),REL)
    def read(name):
        try:
            with s.open(posixpath.join(base,name),'rb') as f:return f.read()
        except FileNotFoundError:return None
    try:
        exit_bytes=read('training_RUN_EXIT.json')
        result={'observed_utc':datetime.now(timezone.utc).isoformat(),'finished':exit_bytes is not None,'arms':{}}
        if exit_bytes:result['run_exit']=json.loads(exit_bytes)
        for arm in ('reconstruction_control','paired_objective'):
            raw=read('training_output/'+arm+'_STEPS.jsonl')
            if raw is None:
                result['arms'][arm]={'complete_rows':0};continue
            rows=[]
            for line in raw.splitlines():
                try:rows.append(json.loads(line))
                except json.JSONDecodeError:break  # live writer partial final line
            result['arms'][arm]={'complete_rows':len(rows),'last_step':rows[-1]['step'] if rows else None,
                'continuous_prefix':[r['step'] for r in rows]==list(range(1,len(rows)+1)),
                'full_gradient_reports':all(r['actual_gradient_leaf_count']==234 and r['gradient_finite'] is True for r in rows),
                'last_actual_parameter_digest':rows[-1]['actual_after_parameter_digest'] if rows else None,
                'live_read_sha256':hashlib.sha256(raw).hexdigest()}
        error=read('training_output/FAILURE.json')
        result['failure']=json.loads(error) if error else None
        result['qualification']='live observation only; incomplete training is not accepted'
        name='READONLY_TRAINING_PROGRESS_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json'
        with (B/name).open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
        print(json.dumps(result))
        return int(result['failure'] is not None or (exit_bytes is not None and result['run_exit']['exit_code']!=0))
    finally:s.close();c.close()
if __name__=='__main__':sys.exit(main())
