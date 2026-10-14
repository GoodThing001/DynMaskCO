#!/bin/bash
# 正式批次 v2 · 追加腿：MAPT（2 档）
#
# ⚠️ 2026-10-13 **口径更正（重要）**：本文件原先的头注写「默认不要跑本脚本」，并据
#   `MAPTProvider.order()` ≈ **40.6 s/次** 推出「单决策 ≈10 分钟、一条 40 天臂 ≈1300 h、
#   无法产出有效臂」。**那个 40.6 s 是过时且被取代的测量值**，结论随之作废。
#
#   更正依据（`模型参与证据.md` §2.1，同日在 1003 实测）：
#     - 40.6 s 来自「**未限线程 + `full` 档**（`n_frame_eval=32`、`n_phantom_vehicles=14`）」；
#     - 现行默认档是 **`config=fast`**（`n_frame_eval=8`、`n_phantom_vehicles=0`、
#       `cpu_threads=4`），实测单次 `order()` = **0.536 s**，快约两个数量级；
#     - `rrnco_guided_adapter` 的不变量是「每辆 replan 车调一次 provider」（≤15 次/决策），
#       故单决策 ≈ 0.50 s × 15 ≈ **7.6 s ≤ 10 s 协议上限**（pool 30/90 实测区间 3.8–7.6 s）。
#   ⇒ 因此 **MAPT 两档是「已批准、应当产出」的正式臂**，本脚本就是产出它们的入口。
#
#   批准凭据：`CC_Compare/MAPT_code/a1_accept_v1/RUN_ARM_APPROVED` 存在（2256 B），
#   且已静态核验「marker 批准的档位 = 实跑档位」（本脚本**不设** `MAPT_PROVIDER_CONFIG`
#   ⇒ `mapt_provider.py:74` 的默认 `fast` 生效，与 marker 的 5 项逐项一致）。
#
#   预算与兜底：≈2.5–3 h/档 ⇒ 两档 5–6 h，编排给的**上限 10 h**。
#   若 10 h 内未写出 `ALLDONE` ⇒ 采集时会缺 MAPT 行，**必须如实标注为「部分/未完成」**，
#   不得静默跳过（见 README §12 R4）；若某档出现 `mean_timeouts>0`，按 R3/§7 记为
#   「资源混淆」并连同 `max_decide_s` 一起报告。
#
# 前置：
#   1) results/a1_c1fresh_reveal_40/gate.json 已产出；
#   2) F1 四条腿全部 ALLDONE；
#   3) ordering_providers.py 已加 'mapt' 注册（wire_extra_providers.py）—— 已在 1003 实测在位（第 35–36 行）。
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 1
PY=/home/hzeng/envs/cc_compare/bin/python
BASE=results/a1_c1fresh_reveal_40/gate.json
OUT=results/a1_ext_formal2
CKPT=${MAPT_CKPT:-../CC_Compare/MAPT_code/a1_accept_v1/mapt_a1.pt}
MARK=/tmp/xformal_v2/mapt.done
mkdir -p "$OUT"; : > "$MARK"

if [ ! -f "$BASE" ]; then echo "FATAL: 缺基线 $BASE" | tee -a "$MARK"; exit 1; fi
if [ ! -f "$CKPT" ]; then echo "FATAL: 缺 MAPT checkpoint $CKPT" | tee -a "$MARK"; exit 1; fi

for tier in "p_c=0:0" "p_c=(5,10,15):51015"; do
  p="${tier%%:*}"; sfx="${tier##*:}"
  dest="$OUT/mapt_${sfx}"
  if [ -f "$dest/gate.json" ]; then
    echo "SKIP_done mapt_${sfx}" >> "$MARK"; continue
  fi
  $PY scripts/evaluation/run_a1_external_accept.py --solver ordering \
    --provider mapt --provider-ckpt "$CKPT" --provider-device cpu \
    --baseline-from "$BASE" --dev-instances 40 --workers 6 \
    --penalty "$p" --out "$dest"
  echo "DONE_mapt_${sfx} $?" >> "$MARK"
done
echo ALLDONE >> "$MARK"
