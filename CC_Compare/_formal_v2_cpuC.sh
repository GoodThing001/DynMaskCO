#!/bin/bash
# A-v1 正式批次 v2 · CPU 腿 C：经典求解器（OR-Tools / PyVRP，3 档）+ 规则臂
#
# ⚠️ 求解器必须用 **a1_ext** 环境：`cc_compare` 里**没有** ortools / pyvrp。
#   2026-10-13 实际踩到：用 cc_compare 跑这两个 solver → 每个实例 ModuleNotFoundError，
#   而 driver **仍然写出 gate.json**（60 KB 的失败记录）→ 污染 results/a1_ext_formal2/，
#   又因 render 的 GROUP_RANK 优先 formal2，坏结果会顶掉 a05e 里本来就好的结果。
#   已删除那两个污染目录；本脚本改用实测有 ortools 9.11.4210 + pyvrp 的解释器。
#
# 参数全部走 driver 默认值，与冻结的 a05e 配置逐项一致（solution_limit=30 /
# pyvrp_max_iterations=1000 / time_limit=10 / workers=8 / n_orders=200 / capacity=50 /
# num_vehicles=15 / penalty=null 即三档全跑），**只**把 baseline_from 换成身份合格的新基线
# （a05e 用的是旧的 a1_step2_gate_c1_20260926，无 identity 块 → 不可正式裁决，这正是 v2 要修的）。
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 1
PY=/home/hzeng/envs/cc_compare/bin/python        # 规则臂（纯 NumPy）
# ⚠️ 两个经典求解器**必须各用版本对的解释器**（2026-10-13 实测踩到，见下）：
#   ortools -> a1_ext（ortools 9.11.4210）  实测 p_c=0 util=1300.65，与 v1 的 1300.7 一致 ✅
#   pyvrp   -> MASKCO_env（pyvrp 0.11.3）   **不能用 a1_ext**：a1_ext 是 pyvrp 0.14.0，其签名
#              add_depot(location: Location, ...) 已不再接受 (x, y)，而
#              solver_accept_replanner._solve_pyvrp:307 按 0.11.3 传 (x, y, tw_early=..., tw_late=...)
#              ⇒ `TypeError: got multiple values for argument 'tw_early'`，每实例都抛
#              ⇒ driver 静默退化为「全部拒单」：p_c=0 util=**0.00**、主档 **−1943.12**，
#              而 rc=0 / hfr=1.0 / timeouts=0 / 身份全合格 —— **看起来正常，实为空臂**。
PY_ORTOOLS=/home/hzeng/envs/a1_ext/bin/python
PY_PYVRP=/home/hzeng/envs/MASKCO_env/bin/python
BASE=results/a1_c1fresh_reveal_40/gate.json
OUT=results/a1_ext_formal2
MARK=/tmp/xformal_v2/cpuC.done
mkdir -p "$OUT"; : > "$MARK"

# --- 经典求解器：一次调用跑三档（--penalty 留空 = 三档全跑） ---
for solver in ortools pyvrp; do
  case "$solver" in
    ortools) PYS="$PY_ORTOOLS" ;;
    pyvrp)   PYS="$PY_PYVRP" ;;
  esac
  $PYS scripts/evaluation/run_a1_external_accept.py --solver "$solver" \
    --baseline-from "$BASE" --dev-instances 40 --workers 8 \
    --out "$OUT/$solver"
  echo "DONE_${solver} $?" >> "$MARK"
done

# --- 规则臂：myopic / value-e λ2 / value-e λ3 ---
rule_run() {  # tag rule lambda
  local tag=$1 rule=$2 lam=$3
  for tier in "p_c=0:0" "p_c=(5,10,15):51015"; do
    p="${tier%%:*}"; sfx="${tier##*:}"
    $PY scripts/evaluation/run_a1_downstream_ablation.py \
      --sampler none --router greedy --vote on --rule "$rule" \
      --valuee-lambda "$lam" --baseline-from "$BASE" --dev-instances 40 \
      --workers 6 --penalty "$p" --out "$OUT/rule_${tag}_${sfx}"
    echo "DONE_rule_${tag}_${sfx} $?" >> "$MARK"
  done
}
rule_run myopic  myopic 0.0
rule_run valuee20 valuee 2.0
rule_run valuee30 valuee 3.0
echo ALLDONE >> "$MARK"
