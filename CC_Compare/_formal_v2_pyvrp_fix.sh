#!/bin/bash
# ============================================================================
# 一次性修复：用**版本对的 pyvrp**重跑 v2 的 PyVRP 臂。
#
# 背景（2026-10-13 实测）——这是本次批次第 4 个缺陷，属「静默空臂」类：
#   `_formal_v2_cpuC.sh` 让 ortools 与 pyvrp 共用 a1_ext（pyvrp **0.14.0**）。
#   pyvrp 0.14.0 的签名已变成
#       add_depot(self, location: Location, tw_early=0, tw_late=..., service_duration=0, *, name='')
#   而 `solver_accept_replanner._solve_pyvrp:307` 按 **0.11.3** 的
#       add_depot(self, x, y, tw_early=0, tw_late=..., *, name='')
#   传 `(x, y, tw_early=..., tw_late=...)` ⇒
#       TypeError: Model.add_depot() got multiple values for argument 'tw_early'
#   每个实例都抛（实测 40 天 × 3 档 = **120** 次 SOLVER_EXC）。
#
#   driver 对 SOLVER_EXC 是**静默降级**：该决策退化为「全部拒单」。于是该臂
#   p_c=0 util = **0.00**（p_c=0 时拒单损失为 0，全拒 = 恰好 0）、主档 util = **−1943.12**，
#   而 `hard_feasible_rate=1.0`、`mean_timeouts=0.0`、rc=0、身份/配对所有字段都合格
#   —— **看起来完全正常，其实是空臂**。v1 的同臂是 **909.2 / −569.8**。
#   ⇒ 所以「rc=0 + timeouts=0 + 身份合格」**不足以**判定一条臂有效。
#
# 修法：pyvrp 用 MASKCO_env（pyvrp 0.11.3，已实测 `CALL_OK`）；ortools 继续用 a1_ext
#   （a1_ext 的 ortools 9.11.4210 跑出来 p_c=0 util=1300.65，与 v1 的 1300.7 一致，无需改）。
#
# 运行时机：**等 cpuC 的规则臂跑完再跑**，让机器回到安静状态，避免 10 s 预算被负载吃掉。
# ============================================================================
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 1
PY_PYVRP=/home/hzeng/envs/MASKCO_env/bin/python
BASE=results/a1_c1fresh_reveal_40/gate.json
OUT=results/a1_ext_formal2
MARK=/tmp/xformal_v2/pyvrp_fix.done
: > "$MARK"

wait_quiet() {  # 等 cpuC（规则臂）ALLDONE，最多 4 h
  local i
  for i in $(seq 1 240); do
    if grep -q ALLDONE /tmp/xformal_v2/cpuC.done 2>/dev/null; then
      echo "cpuC 已 ALLDONE（第 $i 次检查）" >> "$MARK"; return 0
    fi
    sleep 60
  done
  echo "WARN cpuC 等待超时，仍继续（会带负载跑）" >> "$MARK"
}

echo "== 等 cpuC 结束（避免与规则臂抢核）==" >> "$MARK"
wait_quiet

# 删掉空臂，再重跑（否则 run 脚本会 SKIP 已有的 gate.json）
rm -rf "$OUT/pyvrp"
echo "== 重跑 pyvrp（MASKCO_env, pyvrp 0.11.3）==" >> "$MARK"
$PY_PYVRP scripts/evaluation/run_a1_external_accept.py --solver pyvrp \
  --baseline-from "$BASE" --dev-instances 40 --workers 8 \
  --out "$OUT/pyvrp"
echo "DONE_pyvrp_fix $?" >> "$MARK"
echo ALLDONE >> "$MARK"
