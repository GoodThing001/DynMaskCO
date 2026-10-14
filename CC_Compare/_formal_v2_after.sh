#!/bin/bash
# F1 正式批次 v2 完成后的**一键收口**（可提前挂起，幂等）：
#   1) 等 F1 四条腿全部 ALLDONE
#   2) 幂等接线追加 provider（coenrich / mapt）——必须在四条腿之后，否则在跑臂 source_stable=False
#   3) 起 CO-enriched 腿（2 模式 × 2 档）并等它结束
#   4) 若 MAPT checkpoint 已就位则起 MAPT 腿并等它结束（否则提示稍后单独跑）
#   5) 重新采集数据集 + 打印各臂身份/裁决状态
#
# 用法（服务器仓库根）：
#   tmux new -d -s xf2_after 'bash CC_Compare/_formal_v2_after.sh > /tmp/xformal_v2/after.log 2>&1'
set -u
ROOT=/home/hzeng/project/MASKCO-Main
EXT=$ROOT/C-VRP_Cold-chainVehicleRoutingProblem
PY=/home/hzeng/envs/cc_compare/bin/python
RUN=/tmp/xformal_v2
mkdir -p "$RUN"

wait_for() {  # donefile label timeout_iters
  local f=$1 label=$2 iters=${3:-2000} i
  for i in $(seq 1 "$iters"); do
    if grep -q ALLDONE "$f" 2>/dev/null; then
      echo "$label 完成（第 $i 次检查）"; return 0
    fi
    if [ $((i % 20)) -eq 0 ]; then echo "  ... 等 $label（第 $i 次检查）"; fi
    sleep 60
  done
  echo "WARN: 等 $label 超时"; return 1
}

echo "== [1/5] 等 F1 四条腿 =="
for leg in gpu cpuA cpuB cpuC; do
  wait_for "$RUN/$leg.done" "F1:$leg" 2000 || true
done
for f in "$RUN"/gpu.done "$RUN"/cpuA.done "$RUN"/cpuB.done "$RUN"/cpuC.done; do
  echo "--- $f"; cat "$f" 2>/dev/null
done

echo "== [2/5] 接线追加 provider =="
cd "$EXT" || exit 1
$PY ../CC_Compare/tools/wire_extra_providers.py
$PY - <<'PYEOF'
import sys
sys.path.insert(0, 'scripts/evaluation')
import ordering_providers as op
print('registry now:', sorted(op._REGISTRY))
PYEOF

echo "== [3/5] CO-enriched 腿 =="
# ⚠️ 必须用**绝对路径**：第 39 行已 `cd "$EXT"`（扩展根），而 `CC_Compare` 在**仓库根**，
# 扩展根下没有它。2026-10-13 实测：写成 `bash CC_Compare/_formal_v2_coenrich.sh` 会让
# xf2_coe 会话立刻 "No such file or directory" 退出 ⇒ `coenrich.done` 永不出现 ⇒
# `wait_for` 白等满 40 h（2400×60s）才继续，且**全程看起来像「还在跑」**。
tmux kill-session -t xf2_coe 2>/dev/null
tmux new -d -s xf2_coe "bash $ROOT/CC_Compare/_formal_v2_coenrich.sh > $RUN/coenrich.log 2>&1"
# 实测（1 天探针，workers=6）：纯排序 ≈1–2 min/天；+PC-HGS ≈13.6 min/天（**timeouts=0**，
# 协议内可行）⇒ 4 臂合计约 18–20 h。上限给到 40 h 再收口。
wait_for "$RUN/coenrich.done" "CO-enriched" 2400 || true
cat "$RUN/coenrich.done" 2>/dev/null

echo "== [4/5] MAPT 腿（需显式批准的 marker 文件）=="
# ✅ 2026-10-14 实测更正：**MAPT 两档已实际跑完，且协议干净** —— 单档 ≈57 min（不是曾经的 2.5–3 h，
#   更不是过时测量推出的 ≈1300 h）。第 1 档实测 `util=1215.40`、**`超时/天=0.0`**、
#   `identity_verified=true`、`formal_adjudication=true`；主档 `-102.22` / 同样 `超时/天=0.0`。
#   ⇒ **「预算不可行、无法产出有效臂」的旧结论已作废**（成因是旧测量用了未限线程的 `full` 档，
#     现行默认 `config=fast`：`n_frame_eval=8`、`n_phantom_vehicles=0`、`cpu_threads=4`）。
#   marker（2256 B）与 `mapt_a1.pt`（5,821,920 B）都在位 ⇒ 本分支按预期执行。
# ⚠️ 这两个路径必须是**绝对**的：cwd 是 $EXT（扩展根），
#   用相对的 `CC_Compare/...` 会让 `[ -f … ]` 永远为假 ⇒ **MAPT 被静默 SKIP**。
MAPT_MARK=$ROOT/CC_Compare/MAPT_code/a1_accept_v1/RUN_ARM_APPROVED
MAPT_CKPT_FILE=$ROOT/CC_Compare/MAPT_code/a1_accept_v1/mapt_a1.pt
if [ -f "$MAPT_MARK" ] && [ -f "$MAPT_CKPT_FILE" ]; then
  tmux kill-session -t xf2_mapt 2>/dev/null
  tmux new -d -s xf2_mapt "bash $ROOT/CC_Compare/_formal_v2_mapt.sh > $RUN/mapt.log 2>&1"
  # 实测单档 ≈57 min（1003，负载 ~23）⇒ 两档 ≈2 h；上限仍给 10 h 作为兜底，
  # 避免「还没跑完就采集」导致表里缺 MAPT 行。
  wait_for "$RUN/mapt.done" "MAPT" 600 || true
  cat "$RUN/mapt.done" 2>/dev/null
else
  echo "SKIP MAPT：未批准或 ckpt 缺失（marker=$MAPT_MARK exists=$([ -f "$MAPT_MARK" ] && echo yes || echo no)）"
  echo "  注意：这不是因为预算不可行 —— 实测两档各 ≈57 min、超时=0（2026-10-14）。"
  echo "SKIP_MAPT_MARKER_OR_CKPT_MISSING" >> "$RUN/mapt.done"
fi

echo "== [5/5] 采集 + 身份/裁决状态 =="
$PY ../CC_Compare/tools/collect_compare_table.py \
    --json ../CC_Compare/results/a1_compare_table/dataset.json
echo "== 服务端 Markdown 总表（Excel 由本地渲染）=="
$PY ../CC_Compare/tools/render_compare_table.py \
    --data ../CC_Compare/results/a1_compare_table/dataset.json \
    --outdir ../CC_Compare/results/a1_compare_table --md-only 2>&1 | tail -3
$PY - <<'PYEOF'
import glob, json, os
for p in sorted(glob.glob('results/a1_ext_formal2/*/gate.json')):
    name = os.path.basename(os.path.dirname(p))
    try:
        d = json.load(open(p, encoding='utf-8'))
    except Exception as e:
        print('%-26s LOAD_FAIL %s' % (name, e)); continue
    for yr, yb in (d.get('results') or {}).items():
        pid = yb.get('_pairing_identity') or {}
        for pname, blk in yb.items():
            if pname.startswith('_') or not isinstance(blk, dict):
                continue
            for an, ab in (blk.get('arms') or {}).items():
                if an in ('cond_hist', 'uncond_hist', 'explicit_feat'):
                    continue
                u = ab.get('utility_all_days') or {}
                pd = blk.get('%s_minus_cond_hist' % an) or {}
                print('%-26s %-16s %-18s util=%-9s to=%-6s paired=%s formal=%s' % (
                    name, pname, an,
                    ('%.1f' % u['mean']) if u.get('mean') is not None else 'n/a',
                    ab.get('mean_timeouts'), pid.get('identity_verified'),
                    pd.get('formal_adjudication')))
PYEOF
echo "== 收口完成；本地执行 render_compare_table.py 重出表格 =="
