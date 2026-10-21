#!/bin/bash
set -u
umask 077
ulimit -c 0
SECONDS=0
export JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.35
export XLA_FLAGS=--xla_gpu_autotune_level=4
export JAX_COMPILATION_CACHE_DIR=/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/static_profit_20261021/dev09/compile_cache
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/static_profit_20261021/dev09/workspace
mkdir -p /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/static_profit_20261021/dev09/outputs
/home/hzeng/envs/MASKCO_env/bin/python -B -c 'import pathlib,json,hashlib
w=pathlib.Path('"'"'/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/static_profit_20261021/dev09/workspace'"'"');m=json.loads((w/'"'"'payload_manifest.json'"'"').read_text())
assert all(hashlib.sha256((w/n).read_bytes()).hexdigest()==h for n,h in m.items())' || exit 91
(
remaining=$((2400-SECONDS))
if [ "$remaining" -le 0 ]; then exit 124; fi
if [ "$remaining" -gt 1800 ]; then remaining=1800; fi
timeout "${remaining}"s nice -n 10 taskset -c 86,87,90,91 /home/hzeng/envs/MASKCO_env/bin/python -B /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/static_profit_20261021/dev09/workspace/C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_profit/run.py train --fold 0 --seed 1000 --profile maskco_256 --backend cuda --threads 4 --variants maskco_base,directed_only --cases r101,r102,r103,r104,r105,r106,r107,r108,r109,r110,r111,r112,r201,r202,r203,r204,r205,r206,r207,r208,r209,r210,r211,rc101,rc102,rc103,rc104,rc105,rc106,rc107,rc108,rc201,rc202,rc203,rc204,rc206,rc207,rc208 --steps 1440 --sample-order shuffled --teacher-file /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/static_profit_20261021/dev09/workspace/C-VRP_Cold-chainVehicleRoutingProblem/results/static_delivery/algorithm_dev_20261021/remote/dev04/outputs/train/teachers.json --out /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/static_profit_20261021/dev09/outputs/old_teachers/train || exit $?
remaining=$((2400-SECONDS))
if [ "$remaining" -le 0 ]; then exit 124; fi
if [ "$remaining" -gt 1800 ]; then remaining=1800; fi
timeout "${remaining}"s nice -n 10 taskset -c 86,87,90,91 /home/hzeng/envs/MASKCO_env/bin/python -B /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/static_profit_20261021/dev09/workspace/C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_profit/run.py train --fold 0 --seed 1000 --profile maskco_256 --backend cuda --threads 4 --variants maskco_base,directed_only --cases r101,r102,r103,r104,r105,r106,r107,r108,r109,r110,r111,r112,r201,r202,r203,r204,r205,r206,r207,r208,r209,r210,r211,rc101,rc102,rc103,rc104,rc105,rc106,rc107,rc108,rc201,rc202,rc203,rc204,rc206,rc207,rc208 --steps 1440 --sample-order shuffled --teacher-file /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/static_profit_20261021/dev09/workspace/C-VRP_Cold-chainVehicleRoutingProblem/results/static_delivery/algorithm_dev_20261021/remote_3080/dev03/outputs/teacher_upgrade/teachers.json --out /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/static_profit_20261021/dev09/outputs/upgraded_teachers/train || exit $?
remaining=$((2400-SECONDS))
if [ "$remaining" -le 0 ]; then exit 124; fi
if [ "$remaining" -gt 1800 ]; then remaining=1800; fi
timeout "${remaining}"s nice -n 10 taskset -c 86,87,90,91 /home/hzeng/envs/MASKCO_env/bin/python -B /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/static_profit_20261021/dev09/workspace/C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_profit/run.py replay --fold 0 --seed 1000 --profile maskco_256 --backend cuda --threads 4 --variants maskco_base,directed_only --role validation --case c205 --checkpoint-root /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/static_profit_20261021/dev09/outputs/old_teachers/train --predictor compiled --matmul-precision highest --repair-implementation vectorized --seconds 10 --iterations 100000 --out /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/static_profit_20261021/dev09/outputs/old_teachers/validation/c205 || exit $?
remaining=$((2400-SECONDS))
if [ "$remaining" -le 0 ]; then exit 124; fi
if [ "$remaining" -gt 1800 ]; then remaining=1800; fi
timeout "${remaining}"s nice -n 10 taskset -c 86,87,90,91 /home/hzeng/envs/MASKCO_env/bin/python -B /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/static_profit_20261021/dev09/workspace/C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_profit/run.py replay --fold 0 --seed 1000 --profile maskco_256 --backend cuda --threads 4 --variants maskco_base,directed_only --role validation --case c206 --checkpoint-root /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/static_profit_20261021/dev09/outputs/old_teachers/train --predictor compiled --matmul-precision highest --repair-implementation vectorized --seconds 10 --iterations 100000 --out /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/static_profit_20261021/dev09/outputs/old_teachers/validation/c206 || exit $?
remaining=$((2400-SECONDS))
if [ "$remaining" -le 0 ]; then exit 124; fi
if [ "$remaining" -gt 1800 ]; then remaining=1800; fi
timeout "${remaining}"s nice -n 10 taskset -c 86,87,90,91 /home/hzeng/envs/MASKCO_env/bin/python -B /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/static_profit_20261021/dev09/workspace/C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_profit/run.py replay --fold 0 --seed 1000 --profile maskco_256 --backend cuda --threads 4 --variants maskco_base,directed_only --role validation --case c207 --checkpoint-root /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/static_profit_20261021/dev09/outputs/old_teachers/train --predictor compiled --matmul-precision highest --repair-implementation vectorized --seconds 10 --iterations 100000 --out /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/static_profit_20261021/dev09/outputs/old_teachers/validation/c207 || exit $?
remaining=$((2400-SECONDS))
if [ "$remaining" -le 0 ]; then exit 124; fi
if [ "$remaining" -gt 1800 ]; then remaining=1800; fi
timeout "${remaining}"s nice -n 10 taskset -c 86,87,90,91 /home/hzeng/envs/MASKCO_env/bin/python -B /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/static_profit_20261021/dev09/workspace/C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_profit/run.py replay --fold 0 --seed 1000 --profile maskco_256 --backend cuda --threads 4 --variants maskco_base,directed_only --role validation --case c208 --checkpoint-root /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/static_profit_20261021/dev09/outputs/old_teachers/train --predictor compiled --matmul-precision highest --repair-implementation vectorized --seconds 10 --iterations 100000 --out /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/static_profit_20261021/dev09/outputs/old_teachers/validation/c208 || exit $?
remaining=$((2400-SECONDS))
if [ "$remaining" -le 0 ]; then exit 124; fi
if [ "$remaining" -gt 1800 ]; then remaining=1800; fi
timeout "${remaining}"s nice -n 10 taskset -c 86,87,90,91 /home/hzeng/envs/MASKCO_env/bin/python -B /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/static_profit_20261021/dev09/workspace/C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_profit/run.py replay --fold 0 --seed 1000 --profile maskco_256 --backend cuda --threads 4 --variants maskco_base,directed_only --role validation --case c205 --checkpoint-root /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/static_profit_20261021/dev09/outputs/upgraded_teachers/train --predictor compiled --matmul-precision highest --repair-implementation vectorized --seconds 10 --iterations 100000 --out /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/static_profit_20261021/dev09/outputs/upgraded_teachers/validation/c205 || exit $?
remaining=$((2400-SECONDS))
if [ "$remaining" -le 0 ]; then exit 124; fi
if [ "$remaining" -gt 1800 ]; then remaining=1800; fi
timeout "${remaining}"s nice -n 10 taskset -c 86,87,90,91 /home/hzeng/envs/MASKCO_env/bin/python -B /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/static_profit_20261021/dev09/workspace/C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_profit/run.py replay --fold 0 --seed 1000 --profile maskco_256 --backend cuda --threads 4 --variants maskco_base,directed_only --role validation --case c206 --checkpoint-root /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/static_profit_20261021/dev09/outputs/upgraded_teachers/train --predictor compiled --matmul-precision highest --repair-implementation vectorized --seconds 10 --iterations 100000 --out /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/static_profit_20261021/dev09/outputs/upgraded_teachers/validation/c206 || exit $?
remaining=$((2400-SECONDS))
if [ "$remaining" -le 0 ]; then exit 124; fi
if [ "$remaining" -gt 1800 ]; then remaining=1800; fi
timeout "${remaining}"s nice -n 10 taskset -c 86,87,90,91 /home/hzeng/envs/MASKCO_env/bin/python -B /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/static_profit_20261021/dev09/workspace/C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_profit/run.py replay --fold 0 --seed 1000 --profile maskco_256 --backend cuda --threads 4 --variants maskco_base,directed_only --role validation --case c207 --checkpoint-root /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/static_profit_20261021/dev09/outputs/upgraded_teachers/train --predictor compiled --matmul-precision highest --repair-implementation vectorized --seconds 10 --iterations 100000 --out /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/static_profit_20261021/dev09/outputs/upgraded_teachers/validation/c207 || exit $?
remaining=$((2400-SECONDS))
if [ "$remaining" -le 0 ]; then exit 124; fi
if [ "$remaining" -gt 1800 ]; then remaining=1800; fi
timeout "${remaining}"s nice -n 10 taskset -c 86,87,90,91 /home/hzeng/envs/MASKCO_env/bin/python -B /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/static_profit_20261021/dev09/workspace/C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_profit/run.py replay --fold 0 --seed 1000 --profile maskco_256 --backend cuda --threads 4 --variants maskco_base,directed_only --role validation --case c208 --checkpoint-root /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/static_profit_20261021/dev09/outputs/upgraded_teachers/train --predictor compiled --matmul-precision highest --repair-implementation vectorized --seconds 10 --iterations 100000 --out /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/static_profit_20261021/dev09/outputs/upgraded_teachers/validation/c208 || exit $?
) > /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/static_profit_20261021/dev09/run.log 2>&1
result=$?
/home/hzeng/envs/MASKCO_env/bin/python -B -c 'import pathlib,json,hashlib
w=pathlib.Path('"'"'/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/static_profit_20261021/dev09/workspace'"'"');m=json.loads((w/'"'"'payload_manifest.json'"'"').read_text())
assert all(hashlib.sha256((w/n).read_bytes()).hexdigest()==h for n,h in m.items())' || result=92
/home/hzeng/envs/MASKCO_env/bin/python -B -c 'import pathlib,json,hashlib
r=pathlib.Path('"'"'/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/static_profit_20261021/dev09'"'"');m={}
for p in (r/'"'"'outputs'"'"').rglob('"'"'*'"'"'):
 if p.is_file():
  digest=hashlib.sha256()
  with p.open('"'"'rb'"'"') as f:
   for block in iter(lambda:f.read(1024*1024),b'"'"''"'"'): digest.update(block)
  m[str(p.relative_to(r))]=digest.hexdigest()
(r/'"'"'output_manifest.json'"'"').write_text(json.dumps(m,indent=2))' || result=93
printf "%s\n" "$result" > /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/research/static_profit_20261021/dev09/run.exit
exit "$result"
