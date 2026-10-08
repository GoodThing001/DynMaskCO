#!/bin/bash
# rr3 落盘后一键裁决链（助手侧，2026-10-01）：只读核查 + 门摘要 + 四臂差分。
# 前置：tmux 链已自动产出 run.exit / adjudication.json / verify_end.txt。
# 用法：bash tools_remote/tools_remote_post_rr3.sh
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 9
PY=/home/hzeng/miniconda3/envs/MASKCO_env/bin/python
TS=1001_234935
OUT_R=results/a1_rr3_reveal_40_$TS
OUT_D=results/a1_rr3_density_40_$TS
POST=$OUT_D/post_rr3_landing_$(date +%H%M%S).txt

{
  echo "=== POST-RR3 LANDING $(date '+%F %T') ==="
  echo "--- 工具身份（工作目录版，自动链实际使用） ---"
  sha256sum scripts/evaluation/diag_adjudicate.py scripts/evaluation/run_exp_reserve.py \
             scripts/evaluation/diag_gate_summary.py scripts/evaluation/diag_s3_density_analysis.py
  for OUT in $OUT_R $OUT_D; do
    echo "--- $OUT ---"
    echo "[run.exit]";        cat $OUT/run.exit 2>/dev/null || echo MISSING
    echo "[verify_end.txt]"; cat $OUT/verify_end.txt 2>/dev/null || echo MISSING
    echo "[adjudication.log tail]"; tail -n 12 $OUT/adjudication.log 2>/dev/null || echo MISSING
    echo "[adjudication.json 关键字段]"
    if test -f $OUT/adjudication.json; then
      $PY -c "import json;d=json.load(open('$OUT/adjudication.json',encoding='utf-8'));
print('formal_adjudicable =', d.get('formal_adjudicable'));
print('identity_ok =', d.get('identity_ok'), ' gate_seed =', d.get('gate_seed'));
print('problems =', d.get('problems')[:8]);
print('zero_timeout_ok =', d.get('zero_timeout_ok'), ' labels =', d.get('zero_timeout_sensitivity_labels')[:6]);
print('numeric_per_tier =', d.get('numeric_per_tier'))"
    else
      echo MISSING
    fi
  done
  echo "--- gate 摘要 ---"
  if test -f $OUT_R/gate.json; then $PY scripts/evaluation/diag_gate_summary.py $OUT_R/gate.json 40; fi
  if test -f $OUT_D/gate.json; then $PY scripts/evaluation/diag_gate_summary.py $OUT_D/gate.json 40; fi
  echo "--- 四臂差分（仅归因） ---"
  if test -f $OUT_R/gate.json && test -f $OUT_D/gate.json; then
    $PY scripts/evaluation/diag_s3_density_analysis.py $OUT_R/gate.json $OUT_D/gate.json \
      --out $OUT_D/s3_density_analysis.json && echo "S3_DENSITY_ANALYSIS_EXIT=0"
  else
    echo "S3_DENSITY_SKIP(gate.json 缺失)"
  fi
  echo "=== POST-RR3 LANDING END ==="
} 2>&1 | tee $POST
echo "POST_SAVED=$POST"
