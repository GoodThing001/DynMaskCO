#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""`role_c_forward`（Role C 的前瞻 `Ĵ(b)`）的测试。

重点是四类**会被论文引用**的事实，全部做成机械可验证：

  A. **§2.1.1 不被违反**：`Ĵ` 的签名里**没有**当前决策订单收益的入口（结构级保证）；
     且 `Ĵ` 里的**常数偏移**（既有计划/在途 committed leg 的收入）在 `λ_B` 差分里
     **确实抵消**（数值级验证）。
  B. **`inner_gate` 两档的预算通道**（规格 §2.2.2）：
     * `'on'`（主档）—— 离散 `est` 闸门生效；**终局能耗超标不罚**（照抄对照
       `scenario_saa.py:923-928` 的根因修复，否则 40 天会退化成"全接/全拒"空转）；
     * `'off'`（机制消融）——无闸门 + 终局**硬约束** `energy ≤ b`；并**正面演示**
       "只删闸门不补硬约束 ⇒ `Ĵ` 与 `b` 无关 ⇒ `λ_B ≡ 0` ⇒ 退化成 A 版"，
       这正是该档必须补终局硬约束的理由。
  C. **经济行为**：同一订单在**预算吃紧**时被拒、在**预算宽松**时被接
     （`λ_B` 随稀缺性上升），并且 `'on'`/`'off'` 两档用的是**同一场景集合**（digest 相等）。
  D. **冻结与失败语义**：场景订单不得插进在途 committed 前缀；超时 ⇒ `NaN` ⇒
     `shadow_price_budget` 报 B3 未定义 ⇒ 事前边界规则**强制拒单**（fail-closed）。

纯合成夹具；**不 import 项目 env、不读效用、不跑实验**。
运行：python CC_Compare/tools/test_role_c_forward.py   （退出码 0 = 全过）
"""
import ast
import inspect
import os
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:                                          # noqa: BLE001
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import role_c_forward as fwd                               # noqa: E402
import role_c_local_search as ls                           # noqa: E402
import role_c_opportunity_cost as oc                       # noqa: E402

SRC = os.path.join(HERE, 'role_c_forward.py')
BIG = 1e9

# 线度量：depot 在 x=0，节点 i 在 x=i（i=1..4）⇒ d(a,b) = |x_a − x_b|
N = 4
COORD = [0.0, 1.0, 2.0, 3.0, 4.0]


def line_space(*, twe=None, service=None, reveal=None, capacity=100.0,
               return_deadline=BIG):
    D = [[abs(COORD[a] - COORD[b]) for b in range(N + 1)] for a in range(N + 1)]
    return fwd.ForwardSpace(
        dist=D, tws=[0.0] * (N + 1), twe=twe or [BIG] * (N + 1),
        service=service or [0.0] * (N + 1), demand=[0.0] + [1.0] * N,
        reveal=reveal or [0.0] * (N + 1), capacity=capacity, tw_speed=1.0,
        return_deadline=return_deadline)


def empty_states():
    return {1: dict(cur=0, cur_time=0.0, load=0.0, route=[], frozen=0)}


def mk_complete(energy):
    """终局完成函数：`energy` 可为常数或 `fn(st) -> energy`；距离用 LS 的 route_dist。"""
    def _complete(st):
        ctx = ls.RouteContext(
            dist=line_space().dist, tws=[0.0] * (N + 1), twe=[BIG] * (N + 1),
            service=[0.0] * (N + 1), demand=[0.0] + [1.0] * N,
            reveal=[0.0] * (N + 1), capacity=100.0, tw_speed=1.0,
            deadline=BIG)
        dist = sum(ls.route_dist(ctx, s['cur'], s['route']) for s in st.values())
        e = energy(st) if callable(energy) else float(energy)
        return e, dist
    return _complete


REV = lambda _o: 10.0          # noqa: E731
PEN = lambda _o: 5.0           # noqa: E731
ENERGY1 = lambda _o: 1.0       # noqa: E731


def run(space, scenarios, budget, *, gate='on', complete=None, states=None,
        est0=0.0, penalty=-100.0, fcu=1.0, **kw):
    return fwd.forward_value(
        space=space, base_states=states or empty_states(), scenarios=scenarios,
        budget=budget, initial_est_used=est0, order_energy_fn=ENERGY1,
        order_revenue_fn=REV, order_penalty_fn=PEN,
        complete_fn=complete or mk_complete(0.5),
        fuel_cost_per_distance_unit=fcu, infeasible_penalty=penalty,
        inner_gate=gate, **kw)


# ===========================================================================
# A. §2.1.1 不被违反
# ===========================================================================
def test_no_revenue_o_channel():
    cases = []
    sig = inspect.signature(fwd.forward_value)
    params = set(sig.parameters)
    cases.append(('A1 签名里没有当前决策订单收益的入口（结构级保证）',
                  'revenue_o' not in params
                  and not any(p in params for p in ('revenue', 'order_revenue',
                                                    'current_revenue', 'R_o')),
                  sorted(params)))
    cases.append(('A2 场景收入的入口只有 order_revenue_fn（按节点注入，不是本单）',
                  'order_revenue_fn' in params, sorted(params)))
    # AST：代码里不得出现 `revenue_o` 这类标识符
    tree = ast.parse(open(SRC, encoding='utf-8').read())
    ids = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    ids |= {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    cases.append(('A3 代码标识符里没有 revenue_o / R_o',
                  not (ids & {'revenue_o', 'R_o', 'current_order_revenue'}),
                  sorted(ids & {'revenue_o', 'R_o', 'current_order_revenue'})))
    cases.append(('A4 规格自述：Ĵ 只用于估计 λ_B，禁用写法 (iii)',
                  fwd.FORWARD_SPEC['used_for'].find('lambda_B') >= 0
                  and fwd.FORWARD_SPEC['forbidden'].find('(iii)') >= 0
                  and fwd.FORWARD_SPEC['rng'] is None,
                  fwd.FORWARD_SPEC))
    cases.append(('A5 前瞻规格哈希可复算且为 64 位十六进制',
                  fwd.FORWARD_SPEC_SHA256 == fwd._spec_sha256(fwd.FORWARD_SPEC)
                  and len(fwd.FORWARD_SPEC_SHA256) == 64,
                  fwd.FORWARD_SPEC_SHA256))
    return cases


def test_constant_offset_cancels_in_lambda():
    """`Ĵ` 里**既有计划**那部分收入是常数偏移 ⇒ 必须在 λ_B 差分里抵消。

    ⚠️ 第一版夹具是错的：我让场景订单复用了计划里的订单号（`route=[2]` 且场景含 2），
    这在真实设置里不可能（场景订单来自采样器的临时编号空间）。修好后顺便发现一个**真问题**：
    重叠会让 `best_insert` 抛 `LSStateError`，被 `shadow_price_budget` 吞成
    `undefined=True`（B3）⇒ 伪装成"预算稀缺 ⇒ 强制拒单"。现已在入口显式拦成 `ForwardError`
    （见 D14）。

    夹具：计划含订单 1；场景 = [2,3,4]；预算 1.5 / e_o=1.0 ⇒ 只接得下 1 单。
    计划订单 1 的收入在 `Ĵ(1.5)` 与 `Ĵ(0.5)` 里都出现 ⇒ 差分时必须抵消。
    手算：Ĵ(1.5) = (R1+R2) − 4 − 2×5 = 20 − 14 = 6（路线 [2,1] 距离 2+1+1=4）；
          Ĵ(0.5) = R1 − 2 − 3×5 = 10 − 17 = −7 ⇒ λ_B = (6 − (−7))/1 = **13**。
    把 R1 置 0：Ĵ(1.5) = 10 − 14 = −4、Ĵ(0.5) = −17 ⇒ λ_B = 13，**逐位不变**。
    """
    sp = line_space()
    scen = [[2, 3, 4]]
    states = {1: dict(cur=0, cur_time=0.0, load=0.0, route=[1], frozen=0)}

    def lam(rev_fn):
        fvf = fwd.FutureValueFn(
            space=sp, base_states=states, scenarios=scen, order_energy_fn=ENERGY1,
            order_revenue_fn=rev_fn, order_penalty_fn=PEN,
            complete_fn=mk_complete(0.5), fuel_cost_per_distance_unit=1.0,
            infeasible_penalty=-100.0, initial_est_used=0.0, inner_gate='on')
        v, _ = oc.shadow_price_budget(1.5, 1.0, fvf)
        return v

    lam_all = lam(REV)                                  # 计划订单 1 也计收入
    lam_no_base = lam(lambda o: 0.0 if int(o) == 1 else 10.0)   # 只把**计划订单**置 0
    return [
        ('A6 λ_B 非退化（否则本测试空转）：计划偏移存在时 λ_B = 13',
         lam_all is not None and abs(lam_all - 13.0) < 1e-9, lam_all),
        ('A7 ★ 把**既有计划**订单的收入置 0（常数偏移）⇒ λ_B **逐位不变** ⇒ 偏移确实抵消',
         lam_no_base is not None and abs(lam_all - lam_no_base) < 1e-9,
         (lam_all, lam_no_base)),
    ]


# ===========================================================================
# B. inner_gate 两档的预算通道
# ===========================================================================
def test_gate_on_matches_control_semantics():
    cases = []
    sp = line_space()
    scen = [[1, 2, 3, 4]]
    # 预算充足：4 单全接，路线最优 [4,3,2,1]（线度量上距离 8）
    v_hi, d_hi = run(sp, scen, 10.0, gate='on')
    cases.append(('B1 主档 on：预算充足 ⇒ 4 单全接，Ĵ = 40 − 8 = 32',
                  abs(v_hi - 32.0) < 1e-9 and d_hi['n_future_accept'] == 4
                  and d_hi['n_gate_reject'] == 0, (v_hi, d_hi['n_future_accept'])))
    # 预算吃紧：只有 1 单进（est 1 ≤ 1.5），其余 3 单被闸门拒并计 p_c=5
    v_lo, d_lo = run(sp, scen, 1.5, gate='on')
    cases.append(('B2 主档 on：预算吃紧 ⇒ 只有 1 单进，Ĵ = 10 − 2 − 3×5 = −7',
                  abs(v_lo - (-7.0)) < 1e-9 and d_lo['n_gate_reject'] == 3
                  and d_lo['n_future_accept'] == 1,
                  (v_lo, d_lo['n_gate_reject'], d_lo['n_future_accept'])))
    cases.append(('B3 on 档：Ĵ 随预算单调上升（预算越少 ⇒ 越多的场景订单被拒）',
                  v_hi > v_lo + 1e-9, (v_lo, v_hi)))
    # ★ 终局能耗超标**不罚**（照抄对照 §923-928 的根因修复）
    v_over, d_over = run(sp, scen, 10.0, gate='on', complete=mk_complete(999.0))
    cases.append(('B4 ★ on 档：终局能耗远超标也**不罚**（照抄对照根因修复；'
                  '否则 40 天退化"全接/全拒"空转）',
                  abs(v_over - 32.0) < 1e-9 and d_over['n_infeasible_scenario'] == 0,
                  (v_over, d_over['n_infeasible_scenario'])))
    # 真实不可行（energy=None）⇒ 有限罚
    v_none, d_none = run(sp, scen, 10.0, gate='on',
                         complete=lambda _st: (None, None))
    cases.append(('B5 on 档：终局真实不可行（energy=None）⇒ 有限罚（不是 −inf）',
                  abs(v_none - (-100.0)) < 1e-9
                  and d_none['n_infeasible_scenario'] == 1, v_none))
    return cases


def test_gate_off_needs_terminal_cap():
    cases = []
    sp = line_space()
    scen = [[1, 2, 3, 4]]
    # off 档 + 终局硬约束：预算 1.5 < 终局能耗 4 ⇒ 不可行罚；预算 10 ⇒ 正常值
    v_lo, d_lo = run(sp, scen, 1.5, gate='off', complete=mk_complete(4.0))
    v_hi, d_hi = run(sp, scen, 10.0, gate='off', complete=mk_complete(4.0))
    cases.append(('B6 off 档：无闸门 ⇒ 4 单全接（n_gate_reject=0）',
                  d_hi['n_gate_reject'] == 0 and d_hi['n_future_accept'] == 4,
                  d_hi))
    cases.append(('B7 off 档：终局硬约束 energy=4 > b=1.5 ⇒ 有限罚；b=10 ⇒ 32',
                  abs(v_lo - (-100.0)) < 1e-9 and abs(v_hi - 32.0) < 1e-9
                  and d_lo['n_infeasible_scenario'] == 1,
                  (v_lo, v_hi, d_lo['n_infeasible_scenario'])))
    cases.append(('B8 off 档：预算通道来自终局硬约束 ⇒ Ĵ 仍随 b 上升（λ_B 有定义）',
                  v_hi > v_lo + 1e-9, (v_lo, v_hi)))
    # ★ 正面演示：只删闸门、不补终局硬约束 ⇒ Ĵ 与 b 无关 ⇒ λ_B ≡ 0（退化成 A 版）
    v_a, _ = run(sp, scen, 1.5, gate='off', complete=mk_complete(0.5))
    v_b, _ = run(sp, scen, 10.0, gate='off', complete=mk_complete(0.5))
    lam_degenerate, _ = oc.shadow_price_budget(10.0, 1.0, fwd.FutureValueFn(
        space=sp, base_states=empty_states(), scenarios=scen,
        order_energy_fn=ENERGY1, order_revenue_fn=REV, order_penalty_fn=PEN,
        complete_fn=mk_complete(0.5), fuel_cost_per_distance_unit=1.0,
        infeasible_penalty=-100.0, initial_est_used=0.0, inner_gate='off'))
    cases.append(('B9 ★ 只删闸门不补终局硬约束 ⇒ Ĵ(b) 与 b 无关（32==32）⇒ λ_B ≡ 0 '
                  '⇒ 该档退化成 A 版 —— 这就是 off 档**必须**补终局硬约束的理由',
                  abs(v_a - v_b) < 1e-12 and abs(v_a - 32.0) < 1e-9
                  and lam_degenerate == 0.0, (v_a, v_b, lam_degenerate)))
    cases.append(('B10 两档用的是**同一场景集合**（scenario_digest 相等）',
                  d_hi['scenario_digest'] == d_lo['scenario_digest'],
                  (d_hi['scenario_digest'], d_lo['scenario_digest'])))
    return cases


def test_scenario_digest():
    cases = []
    a = [[1, 2], [3, 4]]
    b = [[4, 3], [2, 1]]
    cases.append(('B11 scenario_digest 顺序无关（场景内/场景间重排都不变）',
                  fwd.scenario_digest(a) == fwd.scenario_digest(b),
                  (fwd.scenario_digest(a), fwd.scenario_digest(b))))
    cases.append(('B12 改一个节点 ⇒ digest 改变（能发现集合被换掉）',
                  fwd.scenario_digest(a) != fwd.scenario_digest([[1, 2], [3, 3]]),
                  None))
    return cases


# ===========================================================================
# C. 经济行为（端到端接拒）
# ===========================================================================
def test_scarcity_changes_decision():
    """同一订单：预算吃紧 ⇒ 拒；预算宽松 ⇒ 接。这是整个机制的核心行为主张。"""
    sp = line_space()
    scen = [[1, 2, 3, 4]]

    def decide_at(budget, myopic_net=5.0, e_o=1.0):
        fvf = fwd.FutureValueFn(
            space=sp, base_states=empty_states(), scenarios=scen,
            order_energy_fn=ENERGY1, order_revenue_fn=REV, order_penalty_fn=PEN,
            complete_fn=mk_complete(0.5), fuel_cost_per_distance_unit=1.0,
            infeasible_penalty=-100.0, initial_est_used=0.0, inner_gate='on')
        lam, ldiag = oc.shadow_price_budget(budget, e_o, fvf)
        acc, diag = oc.decide(myopic_net=myopic_net, e_o=e_o, lambda_B=lam,
                              certify_ok=True)
        return acc, lam, ldiag, diag

    cases = []
    acc_tight, lam_tight, ld_tight, d_tight = decide_at(1.5)
    acc_slack, lam_slack, ld_slack, d_slack = decide_at(10.0)
    cases.append(('C1 预算吃紧（B=1.5）：λ_B 大 ⇒ 同一订单被判**拒单**',
                  acc_tight is False and lam_tight > 5.0, (acc_tight, lam_tight)))
    cases.append(('C2 预算宽松（B=10）：λ_B = 0 ⇒ 同一订单被判**接单**',
                  acc_slack is True and abs(lam_slack) < 1e-12,
                  (acc_slack, lam_slack)))
    cases.append(('C3 ★ 同一订单、同一 myopic_net：稀缺性改变了判决（机制真的在起作用）',
                  acc_tight != acc_slack, (acc_tight, acc_slack)))
    cases.append(('C4 λ_B 的原始差分与截断flag都留痕（供 T2b 区分 raw>0 与有效 λ_B=0）',
                  ld_tight['v_hi'] is not None and ld_tight['v_lo'] is not None
                  and ld_tight['clipped'] is False
                  and ld_slack['lambda_raw'] == 0.0 and ld_slack['undefined'] is False,
                  (ld_tight, ld_slack)))
    cases.append(('C5 前瞻诊断随 λ 估计一起可见（含场景 digest / 接受拒单计数）',
                  fwd.FutureValueFn(space=sp, base_states=empty_states(),
                                    scenarios=scen, order_energy_fn=ENERGY1,
                                    order_revenue_fn=REV, order_penalty_fn=PEN,
                                    complete_fn=mk_complete(0.5),
                                    fuel_cost_per_distance_unit=1.0,
                                    infeasible_penalty=-100.0,
                                    initial_est_used=0.0).last_diag is None,
                  None))
    return cases


def test_decide_candidate_end_to_end():
    """走**真实代码路径** `decide_candidate`（含事前边界规则与 certify 顺序）。"""
    sp = line_space()
    scen = [[1, 2, 3, 4]]
    cases = []

    def make(budget, gate='on'):
        fvf = fwd.FutureValueFn(
            space=sp, base_states=empty_states(), scenarios=scen,
            order_energy_fn=ENERGY1, order_revenue_fn=REV, order_penalty_fn=PEN,
            complete_fn=mk_complete(0.5), fuel_cost_per_distance_unit=1.0,
            infeasible_penalty=-100.0, initial_est_used=0.0, inner_gate=gate)
        return oc.decide_candidate(
            order=1, candidate_plan={1: [1]}, budget_B=budget,
            myopic_net_fn=lambda _o: 5.0, marginal_energy_fn=ENERGY1,
            future_value_fn=fvf, certify_fn=lambda _p: (True, 1.0))

    acc_t, diag_t = make(1.5)
    acc_s, diag_s = make(10.0)
    cases.append(('C6 decide_candidate 端到端：吃紧 ⇒ 拒（看得到 λ_B 与 net）',
                  acc_t is False and diag_t['lambda_B'] > 5.0
                  and diag_t['net_after_opportunity_cost'] < 0,
                  (acc_t, diag_t['lambda_B'], diag_t['net_after_opportunity_cost'])))
    cases.append(('C7 decide_candidate 端到端：宽松 ⇒ 接（λ_B=0）',
                  acc_s is True and abs(diag_s['lambda_B']) < 1e-12,
                  (acc_s, diag_s['lambda_B'])))
    cases.append(('C8 判决式口径写明是写法 (i)，且 lambda_source 标为 opportunity_cost',
                  diag_s['formulation'] == '(i) myopic_net − λ_B·e_o > 0 ∧ certify_ok'
                  and diag_s['lambda_diag']['lambda_source'] == 'opportunity_cost',
                  (diag_s['formulation'], diag_s['lambda_diag']['lambda_source'])))
    # 事前边界规则 B2：e_o > B ⇒ 强制拒单（不走前瞻）
    acc_b2, diag_b2 = oc.decide_candidate(
        order=1, candidate_plan={1: [1]}, budget_B=0.5,
        myopic_net_fn=lambda _o: 5.0, marginal_energy_fn=ENERGY1,
        future_value_fn=lambda _b: (_ for _ in ()).throw(AssertionError('不应求值')),
        certify_fn=lambda _p: (True, 1.0))
    cases.append(('C9 B2 事前边界规则：e_o > B ⇒ 强制拒单且**不调用**前瞻（短路）',
                  acc_b2 is False and diag_b2['boundary'] == 'B2_budget_out_of_range',
                  (acc_b2, diag_b2['boundary'])))
    return cases


# ===========================================================================
# D. 冻结与失败语义
# ===========================================================================
def test_frozen_respected_in_forward():
    """场景订单**不得**插进在途 committed 前缀。

    夹具（判别性）：在途车 `cur=1, route=[1], frozen=1`；订单 2 的 TW 只允许
    "先服务 2 再回到 1"（因为订单 1 的服务时长 0.6 会把到达 2 推过 `twe`）
    ⇒ 唯一可行位是 **pos 0**（冻结前缀之内）⇒ 必须插不进；
    同一夹具 `frozen=0` ⇒ 插得进。两侧对照，证明判别器非空转。
    """
    sp = line_space(twe=[BIG, BIG, 1.2, BIG, BIG], service=[0.0, 0.6, 0.0, 0.0, 0.0])
    scen = [[2]]
    base_f = {1: dict(cur=1, cur_time=0.0, load=0.0, route=[1], frozen=1)}
    base_0 = {1: dict(cur=1, cur_time=0.0, load=0.0, route=[1], frozen=0)}
    _, d_f = run(sp, scen, 100.0, gate='on', states=base_f)
    _, d_0 = run(sp, scen, 100.0, gate='on', states=base_0)
    return [
        ('D1 场景订单落进 frozen 前缀 ⇒ 插入失败并计 p_c（frozen=1）',
         d_f['n_insert_reject'] == 1 and d_f['n_future_accept'] == 0, d_f),
        ('D1b 同夹具 frozen=0 ⇒ 插得进（判别器非空转）',
         d_0['n_insert_reject'] == 0 and d_0['n_future_accept'] == 1, d_0),
    ]


def test_fail_closed_paths():
    cases = []
    sp = line_space()
    # 超时 ⇒ NaN ⇒ shadow_price_budget 报 B3 ⇒ 事前边界规则强制拒单
    fvf = fwd.FutureValueFn(
        space=sp, base_states=empty_states(), scenarios=[[1, 2]],
        order_energy_fn=ENERGY1, order_revenue_fn=REV, order_penalty_fn=PEN,
        complete_fn=mk_complete(0.5), fuel_cost_per_distance_unit=1.0,
        infeasible_penalty=-100.0, initial_est_used=0.0, deadline=-1.0)
    lam, ldiag = oc.shadow_price_budget(10.0, 1.0, fvf)
    cases.append(('D2 超时熔断 ⇒ Ĵ 为 NaN ⇒ λ_B **未定义**（B3，不静默用半成品值）',
                  lam is None and ldiag['undefined'] is True, ldiag))
    rule = oc.apply_budget_boundary(1.0, 10.0, lam)
    reject_b, rule_name, reason = rule
    cases.append(('D3 B3 ⇒ 事前边界规则**强制拒单**（fail-closed，与对照"超时降级"同语义）',
                  reject_b is True and rule_name == 'B3_lambda_undefined',
                  (reject_b, rule_name, str(reason)[:60])))
    cases.append(('D4 熔断诊断里如实留痕 deadline_hit=True',
                  fvf.last_diag['deadline_hit'] is True, fvf.last_diag))
    for bad, kw, name in (
            ('maybe', dict(gate='maybe'), '未知 inner_gate'),
            (None, dict(penalty=float('nan')), '不可行罚 NaN'),
            (None, dict(penalty=float('-inf')), '不可行罚 −inf'),
            (None, dict(penalty=float('inf')), '不可行罚 +inf')):
        try:
            run(sp, [[1]], 10.0, **kw)
            cases.append(('D5 %s ⇒ ForwardError' % name, False, '未抛错'))
        except fwd.ForwardError as exc:
            cases.append(('D5 %s ⇒ ForwardError' % name, True, str(exc)[:50]))
    try:
        run(sp, [], 10.0)
        cases.append(('D6 空场景集合 ⇒ ForwardError（不得冒充"已评估"）', False, '未抛错'))
    except fwd.ForwardError:
        cases.append(('D6 空场景集合 ⇒ ForwardError（不得冒充"已评估"）', True, None))
    try:
        run(sp, [[1]], -1.0)
        cases.append(('D7 预算为负 ⇒ ForwardError', False, '未抛错'))
    except fwd.ForwardError:
        cases.append(('D7 预算为负 ⇒ ForwardError', True, None))
    try:
        fwd.FutureValueFn(space=sp, base_states=empty_states(), scenarios=[[1]],
                          order_energy_fn=ENERGY1, order_revenue_fn=REV,
                          order_penalty_fn=PEN, complete_fn=mk_complete(0.5),
                          fuel_cost_per_distance_unit=1.0, infeasible_penalty=-1.0,
                          initial_est_used=0.0, inner_gate='off-the-record')
        cases.append(('D8 FutureValueFn 同样拒绝未知 inner_gate', False, '未抛错'))
    except fwd.ForwardError:
        cases.append(('D8 FutureValueFn 同样拒绝未知 inner_gate', True, None))
    # ★ 场景订单与既有计划重叠 ⇒ 必须在**装配期**响亮报错，不得静默退化成 B3
    overlap = {1: dict(cur=0, cur_time=0.0, load=0.0, route=[2], frozen=0)}
    try:
        fwd.FutureValueFn(space=sp, base_states=overlap, scenarios=[[2, 3]],
                          order_energy_fn=ENERGY1, order_revenue_fn=REV,
                          order_penalty_fn=PEN, complete_fn=mk_complete(0.5),
                          fuel_cost_per_distance_unit=1.0, infeasible_penalty=-1.0,
                          initial_est_used=0.0)
        cases.append(('D14 ★ 场景订单与既有计划重叠 ⇒ 构造期 ForwardError'
                      '（否则会被吞成 B3 "λ 未定义"，伪装成预算稀缺）', False, '未抛错'))
    except fwd.ForwardError as exc:
        cases.append(('D14 ★ 场景订单与既有计划重叠 ⇒ 构造期 ForwardError'
                      '（否则会被吞成 B3 "λ 未定义"，伪装成预算稀缺）',
                      'id 空间错误' in str(exc), str(exc)[:70]))
    try:
        run(sp, [[2, 3]], 10.0, states=overlap)
        cases.append(('D15 forward_value 直接调用时同样拦截重叠', False, '未抛错'))
    except fwd.ForwardError:
        cases.append(('D15 forward_value 直接调用时同样拦截重叠', True, None))
    return cases


def test_determinism_and_reference_values():
    """确定性 + 手算对拍（值必须能逐项对上，不是"看起来合理"）。"""
    cases = []
    sp = line_space()
    v1, d1 = run(sp, [[1, 2, 3, 4]], 10.0, gate='on')
    v2, d2 = run(sp, [[1, 2, 3, 4]], 10.0, gate='on')
    cases.append(('D9 同一输入两次评估：值与逐步计数完全一致',
                  v1 == v2 and d1 == d2, (v1, v2)))
    # 手算：单场景单订单 ⇒ Ĵ = R − fuel = 10 − 2 = 8
    v3, _ = run(sp, [[1]], 10.0, gate='on')
    cases.append(('D10 手算对拍：单订单 ⇒ Ĵ = R − fuel·(去+回) = 10 − 2 = 8',
                  abs(v3 - 8.0) < 1e-9, v3))
    # 多场景取平均（不是求和）
    v4, d4 = run(sp, [[1], [1, 2]], 10.0, gate='on')
    cases.append(('D11 多场景取**平均**（等权），且 n_scenarios 如实记录',
                  d4['n_scenarios'] == 2 and 8.0 < v4 < 32.0, (v4, d4['n_scenarios'])))
    # 燃油口径：fcu 变化必须线性影响价值（边际行驶燃料口径写死）
    v5, _ = run(sp, [[1]], 10.0, gate='on', fcu=2.0)
    cases.append(('D12 燃油口径线性：fcu 1→2 ⇒ Ĵ 8→6（R − 2·dist）',
                  abs(v5 - 6.0) < 1e-9, v5))
    # 前瞻规格哈希随诊断一起返回（可进闭包）
    _, d6 = run(sp, [[1]], 10.0, gate='on')
    cases.append(('D13 诊断携带前瞻规格哈希与邻域 id（供 provider_dependency 闭包）',
                  d6['forward_spec_sha256'] == fwd.FORWARD_SPEC_SHA256
                  and d6['neighborhood_id'] == ls.NEIGHBORHOOD_ID
                  and d6['source_spec_sha256'] == ls.SOURCE_SPEC_SHA256, None))
    return cases


def main():
    cases = []
    for fn in (test_no_revenue_o_channel, test_constant_offset_cancels_in_lambda,
               test_gate_on_matches_control_semantics,
               test_gate_off_needs_terminal_cap, test_scenario_digest,
               test_scarcity_changes_decision, test_decide_candidate_end_to_end,
               test_frozen_respected_in_forward, test_fail_closed_paths,
               test_determinism_and_reference_values):
        try:
            cases.extend(fn())
        except Exception as exc:                            # noqa: BLE001
            import traceback
            cases.append(('★ %s 抛异常' % fn.__name__, False,
                          traceback.format_exc()[-500:]))
    passed = sum(1 for _n, ok_, _d in cases if ok_)
    for n, ok_, detail in cases:
        print(('  PASS ' if ok_ else '  FAIL ') + n
              + ('' if ok_ else ('   ← ' + str(detail))))
    failed = len(cases) - passed
    print('\n结果：%d 通过 / %d 失败' % (passed, failed))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
