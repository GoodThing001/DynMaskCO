#!/bin/bash
set -u
umask 077
ulimit -c 0
SECONDS=0
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.35
export XLA_FLAGS=--xla_gpu_autotune_level=4
export JAX_COMPILATION_CACHE_DIR=/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/static_profit_20261021/dev10/compile_cache
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/static_profit_20261021/dev10/workspace
mkdir -p /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/static_profit_20261021/dev10/outputs
/home/hzeng/envs/MASKCO_env/bin/python -B -c 'import pathlib,json,hashlib
w=pathlib.Path('"'"'/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/static_profit_20261021/dev10/workspace'"'"');m=json.loads((w/'"'"'payload_manifest.json'"'"').read_text())
assert all(hashlib.sha256((w/n).read_bytes()).hexdigest()==h for n,h in m.items())' || exit 91
(
timeout 900s nice -n 10 taskset -c 83,85,86,89 /home/hzeng/envs/MASKCO_env/bin/python -B /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/static_profit_20261021/dev10/workspace/C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_profit/recover_validation.py --parent /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/static_profit_20261021/dev09 --out /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/static_profit_20261021/dev10/outputs/recovery || exit $?
) > /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/static_profit_20261021/dev10/run.log 2>&1
result=$?
/home/hzeng/envs/MASKCO_env/bin/python -B -c 'import pathlib,json,hashlib
w=pathlib.Path('"'"'/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/static_profit_20261021/dev10/workspace'"'"');m=json.loads((w/'"'"'payload_manifest.json'"'"').read_text())
assert all(hashlib.sha256((w/n).read_bytes()).hexdigest()==h for n,h in m.items())' || result=92
/home/hzeng/envs/MASKCO_env/bin/python -B -c 'import pathlib,json,hashlib
r=pathlib.Path('"'"'/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/static_profit_20261021/dev10'"'"');m={}
for p in (r/'"'"'outputs'"'"').rglob('"'"'*'"'"'):
 if p.is_file():
  digest=hashlib.sha256()
  with p.open('"'"'rb'"'"') as f:
   for block in iter(lambda:f.read(1024*1024),b'"'"''"'"'): digest.update(block)
  m[str(p.relative_to(r))]=digest.hexdigest()
(r/'"'"'output_manifest.json'"'"').write_text(json.dumps(m,indent=2))' || result=93
printf "%s\n" "$result" > /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/static_profit_20261021/dev10/run.exit
exit "$result"
