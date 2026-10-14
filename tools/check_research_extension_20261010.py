"""Read-only resource and research-ownership snapshot; never prints credentials."""
import base64
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _srv

REMOTE = r'''
import json, os, subprocess, datetime
rows=[]
for p in subprocess.check_output(['ps','-eo','pid,ppid,user,psr,etime,pcpu,args']).decode().splitlines()[1:]:
    if any(s in p for s in ('research_1010', 'research_20261010', 'i3_online_recovery', 'delivery_any', 'delivery_gls', 'python', 'tmux')):
        rows.append(p)
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],capture_output=True,text=True)
print(json.dumps({'time':datetime.datetime.now().astimezone().isoformat(),'load':os.getloadavg(),'process_rows':rows,'gpu':gpu.stdout,'gpu_exit':gpu.returncode}))
'''
if __name__ == '__main__':
    code = base64.b64encode(REMOTE.encode()).decode()
    reply = _srv.run("/home/hzeng/miniconda3/envs/MASKCO_env/bin/python -c \"import base64;exec(base64.b64decode('%s'))\"" % code, timeout=60)
    print(reply)
