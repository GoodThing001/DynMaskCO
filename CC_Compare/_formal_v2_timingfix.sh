#!/bin/bash
# A-v1 正式批次 v2 · **时间混淆臂的干净重跑**（2026-10-14）
#
# 动因（实测证据，非猜测）：
#   v2 批次是在 `load average ≈ 152`（96 核，7 个用户）下跑的，其中 6 个臂的
#   `超时/天 > 0`，最严重的是 `lih_51015`：**54.225 超时/天、最大单决策 50.79 s**
#   （协议上限 10 s，即 5 倍超限），其 v2 效用 −335.15 vs v1 −105.01（**偏 219%**）。
#   `sgbs_0` 也有 12.0 超时/天，且其 v2 1266.49 恰好压在 `routefinder_0_f` 1266.2 之上
#   ——「最好的学习型外部方法是谁」因此成了刀锋并列，这对交付表的可比性有实质影响。
#   机器负载随后降到 ~23 ⇒ 现在具备重跑出**协议干净**数字的条件。
#
# ⚠️ 关键设计（绝不破坏已有交付物）：
#   重跑写进**独立目录** `results/a1_ext_formal2_timingfix/`，**绝不碰**
#   `results/a1_ext_formal2/`（那份正被采集与收口读取）。采集器只扫描
#   `a1_ext_formal2` / `a1_ext_a05e` / `a1_ext_extra` 三组（见 collect_compare_table
#   `scan_arm_gates`），因此 timingfix 目录**默认不进表**。
#   ⇒ 采用与否是一个**显式的、可复核的**后续动作，而不是隐式覆盖：
#      把某个 `timingfix/<arm>/gate.json` 与 `a1_ext_formal2/<arm>/gate.json`
#      的超时表与效用比对后，再决定是否替换并重采集。
#
# 顺序按**影响大小**排（先跑最影响结论的）：sgbs_0 → sgbs_51015 → lih_51015 → lih_0
#   → symnco_51015 → symnco_0。任何一步失败都如实写进 MARK，不静默跳过。
#
# 用法（服务器，仓库根）：
#   tmux new -d -s xf2_tfix 'bash /home/hzeng/project/MASKCO-Main/CC_Compare/_formal_v2_timingfix.sh > /tmp/xformal_v2/timingfix.log 2>&1'
set -u
ROOT=/home/hzeng/project/MASKCO-Main
EXT=$ROOT/C-VRP_Cold-chainVehicleRoutingProblem
PY=/home/hzeng/envs/cc_compare/bin/python
RUN=/tmp/xformal_v2
MARK=$RUN/timingfix.done

cd "$EXT" || exit 1
OUT=results/a1_ext_formal2_timingfix
BASE=results/a1_c1fresh_reveal_40/gate.json
mkdir -p "$OUT" "$RUN"; : > "$MARK"

SGBS=../CC_Compare/SGBS/CVRP/1_pre_trained_model/Saved_CVRP100_Model/checkpoint-30500.pt
LIH=../CC_Compare/Learn-Improvement-Heuristics/CVRP/CVRP50/outputs/cvrp_50/run/epoch-199.pt
SYMNCO=../CC_Compare/Sym-NCO/Sym-NCO-POMO/CVRP/pretrained_model/Sym-NCO/checkpoint-8000.pt

if [ ! -f "$BASE" ]; then echo "FATAL: 缺基线 $BASE" >> "$MARK"; exit 1; fi

echo "== [1/2] 等主流水线收口（否则会和 pchgs/MAPT 抢 CPU，把新跑的臂又跑脏）==" >> "$MARK"
for i in $(seq 1 2400); do
  if grep -q '收口完成' "$RUN/after.log" 2>/dev/null; then
    echo "主流水线已收口（第 $i 次检查）" >> "$MARK"; break
  fi
  if [ $((i % 20)) -eq 0 ]; then echo "  ... 等收口（第 $i 次检查）" >> "$MARK"; fi
  sleep 60
done
echo "START_timingfix $(date '+%F %T') load=$(cut -d' ' -f1 /proc/loadavg)" >> "$MARK"

run_one() {  # provider ckpt penalty sfx
  local prov=$1 ckpt=$2 pen=$3 sfx=$4
  local dest="$OUT/${prov}_${sfx}"
  if [ -f "$dest/gate.json" ]; then
    echo "SKIP_done ${prov}_${sfx}" >> "$MARK"; return 0
  fi
  $PY scripts/evaluation/run_a1_external_accept.py --solver ordering \
    --provider "$prov" --provider-ckpt "$ckpt" --provider-device cpu \
    --baseline-from "$BASE" --dev-instances 40 --workers 4 \
    --penalty "$pen" --out "$dest"
  echo "DONE_${prov}_${sfx} $? load=$(cut -d' ' -f1 /proc/loadavg)" >> "$MARK"
}

echo "== [2/2] 重跑 6 个时间混淆臂 ==" >> "$MARK"
run_one sgbs   "$SGBS"   "p_c=0"         "0"
run_one sgbs   "$SGBS"   "p_c=(5,10,15)" "51015"
run_one lih    "$LIH"    "p_c=(5,10,15)" "51015"
run_one lih    "$LIH"    "p_c=0"         "0"
run_one symnco "$SYMNCO" "p_c=(5,10,15)" "51015"
run_one symnco "$SYMNCO" "p_c=0"         "0"
echo "ALLDONE" >> "$MARK"
