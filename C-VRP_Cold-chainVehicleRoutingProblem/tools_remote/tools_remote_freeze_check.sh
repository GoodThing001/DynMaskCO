#!/bin/bash
# rr3 冻结链复建 + 服务器侧逐文件核验 + import 证据测试（不启动批次）
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 9
PY=/home/hzeng/miniconda3/envs/MASKCO_env/bin/python

rm -rf results/rr3_src_frozen
bash tools_remote/tools_remote_freeze_rr3_src.sh || exit 9

echo "=== per-file verify (server side, sha256sum -c) ==="
(cd results/rr3_src_frozen && sha256sum -c SOURCE_MANIFEST.sha256)

echo "=== import evidence test (--help 不启动批次) ==="
RR3_FROZEN_SRC=results/rr3_src_frozen \
  RR3_EVIDENCE_OUT=/tmp/rr3_import_evidence.txt \
  $PY results/rr3_src_frozen/rr3_run_frozen.py --help > /tmp/rr3_help.txt 2>&1
cat /tmp/rr3_import_evidence.txt
echo "=== frozen tree perms ==="
ls -ld results/rr3_src_frozen
