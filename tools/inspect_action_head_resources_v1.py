"""Read-only current CPU/RAM and process ownership, no model import or launch."""
import json, os, traceback
from pathlib import Path
from datetime import datetime, timezone
import psutil
from _srv_exec_v2 import build, run_capture

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'research/review_3080_20261010/delivery/action_head_execution_chain_preparation_root_v1'

def main():
    OUT.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    dest = OUT / ('RESOURCE_OBSERVATION_' + stamp + '.json')
    code = '''import os,json,psutil,sys,importlib.metadata as m
p=psutil.Process();p.cpu_affinity([15]);psutil.process_iter.cache_clear()
rows=[]
for x in psutil.process_iter(['pid','ppid','name','cmdline','username']):
 try:
  if x.pid!=os.getpid() and 'python' in (x.info['name'] or '').lower():
   rows.append(dict(pid=x.pid,ppid=x.info['ppid'],name=x.info['name'],username=x.info['username'],argv=x.info['cmdline'],rss=x.memory_info().rss,affinity=x.cpu_affinity()))
 except (psutil.NoSuchProcess,psutil.AccessDenied):pass
print(json.dumps(dict(available_ram_gib=psutil.virtual_memory().available/2**30,cpu15_busy_percent=psutil.cpu_percent(interval=2,percpu=True)[15],processes=rows,executable=sys.executable,python=sys.version,versions={n:m.version(n) for n in ['numpy','jax','jaxlib','flax','psutil']},model_imported=False,gpu_used=False)))'''
    result = dict(created_utc=datetime.now(timezone.utc).isoformat(),minimum_available_ram_gib=8,
                  local_available_ram_gib=psutil.virtual_memory().available/2**30,
                  model_executed=False,server_upload=False)
    try:
        cfg=str(ROOT/'.vscode/sftp.3080.json')
        raw=run_capture(build(cfg,r'C:\Users\zh\project\MASKCO-Main',r'D:\anaconda\envs\MASKCO_env\python.exe','-B',['-c',code],'windows'),30,cfg)
        result['transport']=raw
        if raw['exit_code']==0:result['server']=json.loads(raw['stdout'])
    except Exception:
        result['failure']=traceback.format_exc()
    with dest.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,ensure_ascii=False)
    print(json.dumps(dict(saved=str(dest),server=result.get('server'),failure=result.get('failure')),ensure_ascii=True))

if __name__=='__main__':main()
