"""Read actual normalization source without imports, model execution or upload."""
import hashlib,json,traceback
from pathlib import Path
from datetime import datetime,timezone
from _srv_exec_v2 import build,run_capture
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'research/review_3080_20261010/delivery/action_head_independent_replay_preparation_root_v1'

def main():
    OUT.mkdir(exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    code=r'''import json,hashlib
from pathlib import Path
p=Path(r'C:\Users\zh\project\MASKCO-Main\research\review_3080_20261010\remote_finite_research_root_v2\upstream\modules\functional\coord_transform.py')
b=p.read_bytes()
print(json.dumps(dict(path=str(p),sha256=hashlib.sha256(b).hexdigest(),source=b.decode('utf-8'),model_imported=False)))'''
    receipt=dict(created_utc=datetime.now(timezone.utc).isoformat(),readonly=True,model_executed=False,uploaded=False)
    try:
        cfg=str(ROOT/'.vscode/sftp.3080.json')
        raw=run_capture(build(cfg,r'C:\Users\zh\project\MASKCO-Main',r'D:\anaconda\envs\MASKCO_env\python.exe','-B',['-c',code],'windows'),30,cfg)
        receipt['transport']=raw
        if raw['exit_code']==0:
            payload=json.loads(raw['stdout']);source=payload.pop('source')
            target=OUT/'OBSERVED_functional.py'
            b=source.encode('utf-8')
            assert hashlib.sha256(b).hexdigest()==payload['sha256']
            with target.open('xb') as f:f.write(b)
            receipt['observation']=payload
    except Exception:receipt['failure']=traceback.format_exc()
    with (OUT/('SOURCE_READ_'+stamp+'.json')).open('x',encoding='utf-8') as f:json.dump(receipt,f,indent=2)
    print(json.dumps(dict(success='observation' in receipt,observation=receipt.get('observation'),failure=receipt.get('failure')),ensure_ascii=True))

if __name__=='__main__':main()
