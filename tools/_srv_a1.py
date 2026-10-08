import subprocess
cmd = ("/home/hzeng/miniconda3/envs/MASKCO_env/bin/python -c \"import json; d=json.load(open('/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/results/a1_step3_dev/random/gate.json')); print(list(d.keys())); import sys; s=json.dumps({k:v for k,v in d.items() if not isinstance(v,(list,dict))}, ensure_ascii=False); print(s[:1200])\"")
out = subprocess.run(['python', 'tools/_srv.py', cmd], capture_output=True)
print(out.stdout.decode('utf-8', 'replace'))
