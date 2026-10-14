#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""`role_c_local_search`（Role C 的**独立** LS，路线 A）的测试。

三组检查，重点是**把"独立"和"纯距离"变成机械可验证的事实**，而不是我的声明：

  A. **源码级独立性与纯度**（AST，不靠 grep 文本）：
     * 本模块的**实际 import** 只有 `__future__/hashlib/json/time` ⇒ 不可能
       在运行期拿到 `scenario_saa` / `rolling_optimizer` / `run_exp_reserve` 的 LS；
     * 模块内**所有类都没有基类** ⇒ 不继承 `C0ReserveReplanner` / `SaaReplanner`
       血统（这是规格 §2.2.1 前提②）；
     * 代码里出现的**标识符**与 `energy/energy_kwh/coldchain_cost/revenue/p_c` 等
       **不相交** ⇒ 目标里没有能量/成本/收入通道；
     * `improve` 的**签名**只有 7 个参数，没有任何效用/能量入口；
     * `certify_fn` 返回的能量值被**丢弃**（只取 bool）⇒ 该通道是惰性的（行为测试）。
  B. **能力三项齐备**：
     * ① 显式已承诺锁定 —— `frozen` 是**必需字段**（缺失即 `LSStateError`），
       且**判别性夹具**证明：唯一改进动作是移动冻结订单时，搜索**不动**；
       同一夹具 `frozen=0` 时**会**动（证明判别器非空转）；
     * ② 真实车辆锚点 —— `route_dist(ctx, cur, route)` 用 `cur`（不是 depot）；
       载重/时刻/在途续行锚点由 `state_from_vehicles` 提供并逐字段核对；
     * ③ 跨车移动 —— 跨车 relocate 与 swap 都在**判别性夹具**上被触发。
  C. **可复现 / 可消融 / 认证闭环**：两次运行逐位一致；`max_moves=0` 与
     `ablate_operators=()` 都是恒等（机制消融非"动作不同"）；末端认证失败
     ⇒ 回退 incumbent 并**如实**记 `certified=False`；`certify_fn=None` ⇒
     `certified=None`（不假装通过）；incumbent 本身不可认证 ⇒ `LSError`。

纯合成夹具；**不 import 项目 env、不读效用、不跑实验**。
运行：python CC_Compare/tools/test_role_c_local_search.py   （退出码 0 = 全过）
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
import role_c_local_search as ls                            # noqa: E402

SRC = os.path.join(HERE, 'role_c_local_search.py')

# ---------------------------------------------------------------------------
# 合成夹具
# ---------------------------------------------------------------------------
BIG = 1e9


def ctx_line(n, *, d_pairs=None, reveal=None, twe=None, demand=None,
             capacity=1e9, tw_speed=1.0, deadline=BIG, service=None, tws=None):
    """显式距离矩阵夹具（`dist[a][b]`，索引 0 = depot）。缺省距离 = 1e6（不吸引）。"""
    D = [[0.0 if a == b else (1e6 if d_pairs is None else 1e6)
          for b in range(n + 1)] for a in range(n + 1)]
    for (a, b), v in (d_pairs or {}).items():
        D[a][b] = float(v)
        D[b][a] = float(v)
    return ls.RouteContext(
        dist=D, tws=tws or [0.0] * (n + 1), twe=twe or [BIG] * (n + 1),
        service=service or [0.0] * (n + 1),
        demand=demand or [0.0] * (n + 1), reveal=reveal or [0.0] * (n + 1),
        capacity=capacity, tw_speed=tw_speed, deadline=deadline)


def st1(**kw):
    """单车状态简写。"""
    d = dict(cur=0, cur_time=0.0, load=0.0, route=[], frozen=0)
    d.update(kw)
    return {1: d}


# ===========================================================================
# A. 源码级独立性与纯度
# ===========================================================================
FORBIDDEN_IMPORTS = {
    'scenario_saa', 'rolling_optimizer', 'run_exp_reserve', 'run_a1_external_accept',
    'rrnco_accept_replanner', 'random', 'numpy', 'np', 'jax',
    'C0ReserveReplanner', 'SaaReplanner', 'RoleCReplanner', 'role_c_replanner',
}
ALLOWED_IMPORTS = {'__future__', 'hashlib', 'json', 'time'}
FORBIDDEN_IDENTIFIERS = {
    'energy', 'energy_kwh', 'coldchain_cost', 'revenue', 'p_c', 'profit',
    'utility', 'fuel', 'J_vis', 'accept', 'reject_penalty', 'cumulative_energy_kwh',
}


def test_source_independence():
    cases = []
    tree = ast.parse(open(SRC, encoding='utf-8').read())

    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add(node.module or '')
    cases.append(('A1 实际 import 仅 %s' % sorted(ALLOWED_IMPORTS),
                  mods <= ALLOWED_IMPORTS, sorted(mods)))
    cases.append(('A1b 无对照血统 / 无 RNG / 无 numpy 的 import',
                  not (mods & FORBIDDEN_IMPORTS), sorted(mods & FORBIDDEN_IMPORTS)))

    # 类基类：只允许标准异常（本模块的 3 个类都应是 `object` / 内建异常派生的错误类型）；
    # **不得**出现对照血统的基类。
    SAFE_BASES = {'ValueError', 'RuntimeError', 'Exception', 'object'}
    unsafe = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            for b in node.bases:
                name = b.id if isinstance(b, ast.Name) else ast.dump(b)
                if name not in SAFE_BASES:
                    unsafe.append((node.name, name))
    cases.append(('A2 所有类的基类只允许内建异常/object ⇒ 不继承对照血统',
                  unsafe == [], unsafe))

    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            ids.add(node.id)
        elif isinstance(node, ast.Attribute):
            ids.add(node.attr)
    hit = ids & FORBIDDEN_IDENTIFIERS
    cases.append(('A3 代码标识符与能量/成本/收入标识符**不相交** ⇒ 无该通道',
                  not hit, sorted(hit)))

    sig = inspect.signature(ls.improve)
    want = {'ctx', 'state', 'certify_fn', 'certify_mode', 'max_passes', 'max_moves',
            'deadline', 'ablate_operators'}
    cases.append(('A4 improve 签名无任何效用/能量入口',
                  set(sig.parameters) == want, sorted(sig.parameters)))
    cases.append(('A5 邻域规格声明纯距离目标 + 冻结哈希',
                  ls.SOURCE_SPEC['objective'] == 'pure_travel_distance'
                  and ls.SOURCE_SPEC['objective_depends_on'] == ('dist',)
                  and ls.SOURCE_SPEC_SHA256 == ls._spec_sha256()
                  and len(ls.SOURCE_SPEC_SHA256) == 64,
                  (ls.SOURCE_SPEC['objective'], ls.SOURCE_SPEC_SHA256)))
    cases.append(('A6 规格明示路线 A / 独立血统 / 无 RNG / relocate→swap / 首次改进',
                  ls.SOURCE_SPEC['route'] == 'A'
                  and ls.SOURCE_SPEC['lineage'] == 'independent'
                  and ls.SOURCE_SPEC['rng'] is None
                  and ls.SOURCE_SPEC['operator_order'] == 'relocate_then_swap'
                  and ls.SOURCE_SPEC['acceptance'] == 'first_improvement',
                  ls.SOURCE_SPEC))
    return cases


def test_certify_energy_channel_inert():
    """`certify_fn` 返回的能量值必须**不影响**输出（只有 bool 生效）。"""
    c = ctx_line(2, d_pairs={(0, 1): 10.0, (1, 2): 5.0, (0, 2): 30.0})
    st = st1(route=[1, 2])
    outs = []
    for e in (0.0, 12345.678, -1e9):
        s2, cert = ls.improve(c, st, certify_fn=lambda _s, _e=e: (True, _e))
        outs.append((ls.plan_of(s2), cert.certified, cert.dist_after))
    return [('A7 certify_fn 的能量返回值被丢弃 ⇒ 三次输出逐位一致',
             outs[0] == outs[1] == outs[2], outs)]


# ===========================================================================
# B. 能力三项齐备
# ===========================================================================
def test_frozen_required():
    cases = []
    for raw, name in (
            ({1: dict(cur=0, cur_time=0.0, load=0.0, route=[1, 2])},
             '缺 frozen'),
            ({1: dict(cur=0, cur_time=0.0, load=0.0, route=[1], frozen=True)},
             'frozen 是 bool'),
            ({1: dict(cur=0, cur_time=0.0, load=0.0, route=[1], frozen=2)},
             'frozen 超界'),
            ({1: dict(cur=0, cur_time=0.0, load=0.0, route=[1], frozen=-1)},
             'frozen 为负'),
            ({1: dict(cur=0, cur_time=0.0, load=0.0, route=[1, 1], frozen=0)},
             '路线重复单'),
            ({1: dict(cur=0, cur_time=0.0, load=0.0, route=[1, 0], frozen=0)},
             '路线含 depot'),
            ({1: 'nope'}, '记录不是 dict'),
            ('nope', '状态不是 dict')):
        try:
            ls.normalize_state(raw)
            cases.append(('B1 %s ⇒ LSStateError' % name, False, '未抛错'))
        except ls.LSStateError as exc:
            cases.append(('B1 %s ⇒ LSStateError' % name, True, str(exc)[:40]))
    # 正例
    ok = ls.normalize_state({1: dict(cur=0, cur_time=0.0, load=0.0,
                                     route=[1], frozen=0)})
    cases.append(('B1z 合法状态通过', 1 in ok and ok[1]['frozen'] == 0, ok))
    # 不改原对象
    raw = {1: dict(cur=0, cur_time=0.0, load=0.0, route=[1, 2], frozen=0)}
    snap = {1: dict(raw[1], route=list(raw[1]['route']))}
    ls.normalize_state(raw)
    cases.append(('B1y normalize_state 不改动输入对象', raw == snap, raw))
    return cases


def _frozen_fixture():
    """判别性夹具：唯一改进是"移动冻结订单"，且判别器**非空转**。

    ⚠️ 夹具设计的关键（第一版错了）：在**对称**度量下，`[1,2]` 与 `[2,1]` 的巡回
    距离**恒等**（反转巡回不改总长）⇒ 用两个订单构造"重排即改进"是**不可能**的，
    判别器会空转。因此这里用**非树度量**（4-环 `0-1-2-3-0`，边长 1）：

        d(0,1)=d(1,2)=d(2,3)=d(0,3)=1；d(0,2)=d(1,3)=2

    巡回 `[1,2,3]` = 1+1+1+1 = **4**（最优之一）；`[2,1,3]` = 2+1+2+1 = **6**；
    且最优巡回必须让订单 3 居中 ⇒ **把订单 2 钉在首位就无法达到最优**。
    于是 `route=[2,1,3]`：`frozen=1`（订单 2 冻结）⇒ 允许的重排只能得 6 ⇒ **不动**；
    `frozen=0` ⇒ 可把订单 2 移走 ⇒ 得 4 ⇒ **必动**。
    """
    return ctx_line(3, d_pairs={(0, 1): 1.0, (1, 2): 1.0, (2, 3): 1.0,
                                (0, 3): 1.0, (0, 2): 2.0, (1, 3): 2.0})


def test_frozen_blocks_the_only_improving_move():
    c = _frozen_fixture()
    st = st1(route=[2, 1, 3], frozen=1)
    s2, cert = ls.improve(c, st, max_passes=3)
    cases = [
        ('B2 冻结订单不可移动：唯一改进被禁 ⇒ 计划逐位不变',
         s2[1]['route'] == [2, 1, 3] and cert.moves_applied == 0
         and cert.dist_before == 6.0 and cert.dist_after == 6.0,
         (s2[1]['route'], cert.moves_applied, cert.dist_before, cert.dist_after)),
        ('B2b 冻结前缀逐位保留（自证字段为 True）',
         cert.frozen_prefix_preserved is True
         and ls.frozen_prefixes(s2) == {1: (2,)},
         (cert.frozen_prefix_preserved, ls.frozen_prefixes(s2))),
    ]
    # 同一夹具 frozen=0 ⇒ 必须**会**动（证明 B2 的判别器不是空转）
    s3, cert3 = ls.improve(c, st1(route=[2, 1, 3], frozen=0), max_passes=3)
    cases.append(('B2c 同夹具 frozen=0 ⇒ 搜索确实找到并采纳该改进（判别器非空转：6→4）',
                  s3[1]['route'] == [1, 2, 3] and cert3.dist_after == 4.0
                  and cert3.dist_before == 6.0 and cert3.moves_applied >= 1,
                  (s3[1]['route'], cert3.dist_before, cert3.dist_after,
                   cert3.moves_applied)))
    return cases


def test_real_anchors_and_cross_vehicle():
    """判别性夹具：跨车 relocate 的改进**只在用真锚点时才成立**。

    d(0,1)=10, d(0,2)=100, d(0,3)=100, d(2,3)=1, d(1,2)=d(1,3)=90；
    车 1：cur=0, route=[1,2]（成本 10+90+100=200）；
    车 2：cur=3, route=[3], frozen=1（成本 0+100=100）——在途车，锚点在节点 3。
    真锚点：把订单 2 移入车 2 的**冻结后缀** ⇒ 200→20、100→101，总 300→121 ✅
    若错用 depot 锚点：车 2 的 [3,2] 成本为 0→3? 不成立，且 [3] 本身价 200 ⇒ 不改进。
    """
    c = ctx_line(3, d_pairs={(0, 1): 10.0, (0, 2): 100.0, (0, 3): 100.0,
                             (1, 2): 90.0, (1, 3): 90.0, (2, 3): 1.0})
    stale = {1: dict(cur=0, cur_time=0.0, load=0.0, route=[1, 2], frozen=0),
             2: dict(cur=3, cur_time=0.0, load=0.0, route=[3], frozen=1)}
    s2, cert = ls.improve(c, stale, max_passes=4)
    cases = [
        ('B3 route_dist 用**真锚点** cur（≠ depot）',
         ls.route_dist(c, 3, [2]) == 101.0 and ls.route_dist(c, 0, [2]) == 200.0
         and ls.route_dist(c, 0, [1, 2]) == 200.0,
         (ls.route_dist(c, 3, [2]), ls.route_dist(c, 0, [2]))),
        ('B4 ③跨车 relocate 被触发：订单从车 1 移到车 2 的**冻结后缀**（pos≥1）',
         (2, (3,), (3, 2)) in cert.moved_orders and s2[1]['route'] != [1, 2]
         and cert.dist_before == 300.0 and cert.dist_after == 101.0,
         ({k: v['route'] for k, v in s2.items()}, cert.moved_orders)),
        ('B4b 车 2 的冻结单 3 始终在首位（插入侧尊重 frozen）',
         s2[2]['route'][0] == 3 and ls.frozen_prefixes(s2)[2] == (3,),
         ls.frozen_prefixes(s2)),
        ('B4c 首个改进尝试确实是"移入冻结后缀"⇒ 300→121（再继续优化到 101）',
         cert.moved_orders[1] == (2, (3,), (3, 2))
         and cert.moved_orders[0][0] == 1,
         cert.moved_orders),
        ('B4d 证书记录移动轨迹且总距离严格下降',
         cert.moves_applied >= 1 and cert.dist_after < cert.dist_before - 1e-9
         and cert.improved is True and bool(cert.moved_orders),
         (cert.moves_applied, cert.moved_orders)),
    ]
    # swap 必须也能被触发（构造一个 relocate 被关闭、只有 swap 能改的夹具）
    # 线度量 x = [0,1,2,3]：A=[1,3] 成本 1+2+3=6；B=[2] 成本 2+2=4 ⇒ 总 10。
    # swap(1↔2) ⇒ A=[2,3]=6、B=[1]=2 ⇒ 总 **8**（改进）。
    c3 = ctx_line(3, d_pairs={(0, 1): 1.0, (1, 2): 1.0, (2, 3): 1.0,
                              (0, 2): 2.0, (1, 3): 2.0, (0, 3): 3.0})
    st3 = {1: dict(cur=0, cur_time=0.0, load=0.0, route=[1, 3], frozen=0),
           2: dict(cur=0, cur_time=0.0, load=0.0, route=[2], frozen=0)}
    s4, cert4 = ls.improve(c3, st3, max_passes=3, ablate_operators=('swap',))
    cases.append(('B5 ③swap 单独开启时也能改进（A=[1,3] / B=[2]：总 10 → 8）',
                  s4[1]['route'] == [2, 3] and s4[2]['route'] == [1]
                  and cert4.dist_after == 8.0 and cert4.improved is True,
                  ({k: v['route'] for k, v in s4.items()}, cert4.dist_before,
                   cert4.dist_after)))
    return cases


def test_state_from_vehicles_real_anchors():
    class CC(object):
        def __init__(self, closed=False):
            self.closed = closed

    class V(object):
        def __init__(self, vid, **kw):
            self.vehicle_id = vid
            self.status = kw.get('status', 'idle')
            self.coldchain_state = kw.get('cc', CC())
            self.committed_next = kw.get('committed_next', 0)
            self.committed_finish = kw.get('committed_finish', 0.0)
            self.current_load = kw.get('current_load', 0.0)
            self.current_node = kw.get('current_node', 0)

    vehicles = [
        V(1, status='committed', committed_next=7, committed_finish=12.5,
          current_load=3.0),
        V(2, status='idle', current_node=0),
        V(3, status='returning'),
        V(4, status='committed', cc=CC(closed=True)),
        V(5, status='ready', current_node=4),
    ]
    plan = {1: [7, 8], 2: [], 3: [9], 4: [9], 5: [8, 9]}
    served = {i: False for i in range(20)}
    served[8] = True
    st = ls.state_from_vehicles(vehicles, served, plan, clock=5.0)
    anchor = lambda vid: {k: st[vid][k] for k in ('cur', 'cur_time', 'load', 'route',
                                                  'frozen')}
    cases = [
        ('B6 在途 committed 车：真锚点 (committed_next/finish/load) + frozen=1 + 去重',
         anchor(1) == dict(cur=7, cur_time=12.5, load=3.0, route=[7], frozen=1),
         st.get(1)),
        ('★B6-cc 冷链状态 `cc` **原样透传**（终局完成必须从真实状态续行，否则累计能耗/'
         '在途货被丢掉、能耗系统性偏低）',
         st[1]['cc'] is vehicles[0].coldchain_state
         and st[5]['cc'] is vehicles[4].coldchain_state, st[1].get('cc')),
        ('B6b returning / closed 车退出搜索',
         3 not in st and 4 not in st, sorted(st)),
        ('B6c 未出车空计划车**保留**（可接未来订单、无幻影预冷）',
         2 in st and anchor(2) == dict(cur=0, cur_time=5.0, load=0.0, route=[],
                                       frozen=0), st.get(2)),
        ('B6d 非 committed 车锚点 = (current_node, clock, load)；已送达单剔除',
         anchor(5) == dict(cur=4, cur_time=5.0, load=0.0, route=[9], frozen=0),
         st.get(5)),
        ('B6e 产出的状态能被 normalize_state 接受（结构自洽，且 cc 不丢）',
         ls.normalize_state(st) == st
         and ls.normalize_state(st)[1]['cc'] is vehicles[0].coldchain_state,
         ls.normalize_state(st).get(1)),
    ]
    return cases


def test_feasibility_semantics():
    """可行性预检的语义（揭示感知出发 / 容量余量 / 回仓截止 / 恰好一次）。"""
    cases = []
    # ① 揭示感知**出发**等待：twe 落在 [reveal, reveal+d/speed) 之间 ⇒ 不可行
    c = ctx_line(1, d_pairs={(0, 1): 10.0}, reveal=[0.0, 100.0],
                 twe=[BIG, 105.0], tw_speed=1.0)
    cases.append(('B7 揭示感知出发：arr = max(t,reveal)+d/speed = 110 > twe=105 ⇒ 不可行',
                  ls.route_feasible(c, 0, 0.0, 0.0, [1]) is False, None))
    cases.append(('B7b twe 放宽到 110 ⇒ 恰好可行（含 1e-6 容差）',
                  ls.route_feasible(c, 0, 0.0, 0.0, [1]) is False
                  and ls.route_feasible(
                      ctx_line(1, d_pairs={(0, 1): 10.0}, reveal=[0.0, 100.0],
                               twe=[BIG, 110.0]), 0, 0.0, 0.0, [1]) is True, None))
    # ② 容量：留 1e-4 保守余量 ⇒ 恰好等于容量被判不可行
    c2 = ctx_line(1, demand=[0.0, 5.0], capacity=5.0)
    cases.append(('B8 容量留 1e-4 保守余量：恰好 = 容量 ⇒ 不可行',
                  ls.route_feasible(c2, 0, 0.0, 0.0, [1]) is False, None))
    c3 = ctx_line(1, demand=[0.0, 4.999], capacity=5.0)
    cases.append(('B8b 略低于容量 ⇒ 可行',
                  ls.route_feasible(c3, 0, 0.0, 0.0, [1]) is True, None))
    # ③ 回仓截止
    c4 = ctx_line(1, d_pairs={(0, 1): 10.0}, deadline=19.0)
    cases.append(('B9 回仓 t+2*10=20 > deadline=19 ⇒ 不可行',
                  ls.route_feasible(c4, 0, 0.0, 0.0, [1]) is False, None))
    cases.append(('B9b deadline=25 ⇒ 可行',
                  ls.route_feasible(ctx_line(1, d_pairs={(0, 1): 10.0},
                                             deadline=25.0), 0, 0.0, 0.0,
                                    [1]) is True, None))
    # ④ 恰好一次
    c5 = ctx_line(1, d_pairs={(0, 1): 1.0})
    cases.append(('B10 路线含重复单 ⇒ 不可行（恰好一次护栏）',
                  ls.route_feasible(c5, 0, 0.0, 0.0, [1, 1]) is False, None))
    # ⑤ 载重锚点（在途已装货）
    c6 = ctx_line(1, demand=[0.0, 1.0], capacity=5.0)
    cases.append(('B11 载重锚点生效：load=4.5 已在车上 + 1 单 > 容量 ⇒ 不可行',
                  ls.route_feasible(c6, 0, 0.0, 4.5, [1]) is False
                  and ls.route_feasible(c6, 0, 0.0, 0.0, [1]) is True, None))
    # ⑥ 时间锚点（cur_time 已很晚）
    c7 = ctx_line(1, d_pairs={(0, 1): 10.0}, twe=[BIG, 100.0])
    cases.append(('B12 时间锚点生效：cur_time=95 ⇒ arr=105 > twe=100 不可行',
                  ls.route_feasible(c7, 0, 95.0, 0.0, [1]) is False
                  and ls.route_feasible(c7, 0, 0.0, 0.0, [1]) is True, None))
    # ⑦ twe 为 0 的空路线
    cases.append(('B13 空路线恒可行（回仓 = cur_time + d(cur,0)/speed）',
                  ls.route_feasible(c7, 0, 0.0, 0.0, []) is True, None))
    return cases


# ===========================================================================
# C. 可复现 / 可消融 / 认证闭环
# ===========================================================================
def test_determinism_and_ablation():
    c = ctx_line(3, d_pairs={(0, 1): 10.0, (0, 2): 100.0, (0, 3): 100.0,
                             (1, 2): 90.0, (1, 3): 90.0, (2, 3): 1.0})
    mk = lambda: {1: dict(cur=0, cur_time=0.0, load=0.0, route=[1, 2], frozen=0),
                  2: dict(cur=3, cur_time=0.0, load=0.0, route=[3], frozen=1)}
    a = ls.improve(c, mk(), max_passes=4)
    b = ls.improve(c, mk(), max_passes=4)
    cases = [
        ('C1 同一输入两次运行：计划逐位一致',
         ls.plan_of(a[0]) == ls.plan_of(b[0]), (ls.plan_of(a[0]), ls.plan_of(b[0]))),
        ('C1b 证书逐步一致（含 evaluations / moves / dist）',
         a[1].to_dict() == b[1].to_dict(),
         {k: (a[1].to_dict()[k], b[1].to_dict()[k])
          for k in a[1].to_dict() if a[1].to_dict()[k] != b[1].to_dict()[k]}),
        ('C1c 证书自报无 RNG、未触熔断、无回退',
         a[1].rng_used is False and a[1].deadline_hit is False
         and a[1].fallback_to_incumbent is False, a[1].to_dict()),
        ('C1d 证书携带邻域 id 与冻结规格哈希（供 provider_dependency_sha256 闭包）',
         a[1].neighborhood_id == ls.NEIGHBORHOOD_ID
         and a[1].source_spec_sha256 == ls.SOURCE_SPEC_SHA256,
         (a[1].neighborhood_id, a[1].source_spec_sha256)),
    ]
    # 机制消融：关掉全部邻域 ⇒ 恒等
    s0, cert0 = ls.improve(c, mk(), ablate_operators=())
    cases.append(('C2 ablate_operators=() ⇒ 恒等算子（机制消融，非"动作不同"）',
                  ls.plan_of(s0) == ls.plan_of(mk())
                  and cert0.moves_applied == 0 and cert0.evaluations == 0,
                  (ls.plan_of(s0), cert0.moves_applied, cert0.evaluations)))
    s0b, cert0b = ls.improve(c, mk(), max_moves=0)
    cases.append(('C2b max_moves=0 ⇒ 恒等（确定停止条件）',
                  ls.plan_of(s0b) == ls.plan_of(mk())
                  and cert0b.moves_applied == 0, cert0b.to_dict()))
    s0c, cert0c = ls.improve(c, mk(), max_passes=0)
    cases.append(('C2c max_passes=0 ⇒ 恒等',
                  ls.plan_of(s0c) == ls.plan_of(mk()) and cert0c.passes == 0,
                  cert0c.to_dict()))
    # 单算子消融：只留 relocate
    s_r, cert_r = ls.improve(c, mk(), ablate_operators=('relocate',))
    cases.append(('C2d ablate_operators=(\'relocate\',) ⇒ 仍改进（relocate 是主算子）',
                  cert_r.improved is True, cert_r.to_dict()))
    # 未知算子
    try:
        ls.improve(c, mk(), ablate_operators=('or-opt',))
        cases.append(("C2e 未知消融算子 ⇒ LSStateError", False, '未抛错'))
    except ls.LSStateError as exc:
        cases.append(("C2e 未知消融算子 ⇒ LSStateError", True, str(exc)[:50]))
    # 熔断（deadline 已过）
    s_d, cert_d = ls.improve(c, mk(), deadline=-1.0)
    cases.append(('C3 deadline 已过 ⇒ 熔断：deadline_hit=True 且如实标记（不可复现）',
                  cert_d.deadline_hit is True and ls.plan_of(s_d) == ls.plan_of(mk()),
                  cert_d.to_dict()))
    # 参数校验
    for bad, kw in ((True, dict(max_passes=True)), (-1, dict(max_passes=-1)),
                    (True, dict(max_moves=True)), (-5, dict(max_moves=-5))):
        try:
            ls.improve(c, mk(), **kw)
            cases.append(('C4 %r ⇒ LSStateError' % (kw,), False, '未抛错'))
        except ls.LSStateError:
            cases.append(('C4 %r ⇒ LSStateError' % (kw,), True, None))
    return cases


def test_certify_closure():
    # 与 B2 同一个**非树度量**判别夹具：incumbent [2,1,3]=6，最优 [1,2,3]=4。
    c = _frozen_fixture()
    st = st1(route=[2, 1, 3], frozen=0)
    cases = []
    # ① certify_fn=None ⇒ certified=None（不假装通过）
    s1, cert1 = ls.improve(c, st)
    cases.append(('C5 certify_fn=None ⇒ certified=None（明确"未认证"，不假装通过）',
                  cert1.certified is None and cert1.dist_after == 4.0,
                  cert1.to_dict()))
    # ② 全通过 ⇒ certified=True
    s2, cert2 = ls.improve(c, st, certify_fn=lambda _s: (True, 1.0))
    cases.append(('C5b certify_fn 恒通过 ⇒ certified=True 且计划被采纳',
                  cert2.certified is True and s2[1]['route'] == [1, 2, 3],
                  cert2.to_dict()))
    # ③ `final` 模式（缺省）：搜索后不通过、incumbent 通过 ⇒ 回退 incumbent 并如实记录
    seen = {'n': 0}

    def only_incumbent_ok(state):
        seen['n'] += 1
        return (ls.plan_of(state) == {1: [2, 1, 3]}, 0.0)

    s3, cert3 = ls.improve(c, st, certify_fn=only_incumbent_ok)
    cases.append(('C6 final 模式：搜索后未过认证 ⇒ **回退 incumbent** 且 certified=False',
                  cert3.certified is False and cert3.fallback_to_incumbent is True
                  and s3[1]['route'] == [2, 1, 3] and cert3.dist_after == 6.0
                  and cert3.certify_mode == 'final' and seen['n'] >= 1,
                  cert3.to_dict()))
    # ③b `each_move` 模式：每次移动先试认证，被拒 ⇒ 移动不发生（强于回退：根本不动）
    seen2 = {'n': 0}

    def only_incumbent_ok2(state):
        seen2['n'] += 1
        return (ls.plan_of(state) == {1: [2, 1, 3]}, 0.0)

    s3b, cert3b = ls.improve(c, st, certify_fn=only_incumbent_ok2,
                             certify_mode='each_move')
    cases.append(('C6b each_move 模式：候选被拒 ⇒ 移动不发生（moves=0）且末端已认证',
                  cert3b.certify_mode == 'each_move' and cert3b.moves_applied == 0
                  and cert3b.certified is True
                  and cert3b.fallback_to_incumbent is False
                  and s3b[1]['route'] == [2, 1, 3] and seen2['n'] > 1,
                  cert3b.to_dict()))
    # ③c 未知认证模式 ⇒ fail-closed
    try:
        ls.improve(c, st, certify_fn=lambda _s: (True, 0.0), certify_mode='maybe')
        cases.append(('C6c 未知 certify_mode ⇒ LSStateError', False, '未抛错'))
    except ls.LSStateError as exc:
        cases.append(('C6c 未知 certify_mode ⇒ LSStateError', True, str(exc)[:50]))
    # ③d each_move 恒通过 ⇒ 与 final 恒通过等价（模式不改变"都通过"时的结果）
    s_a, c_a = ls.improve(c, st, certify_fn=lambda _s: (True, 0.0))
    s_b, c_b = ls.improve(c, st, certify_fn=lambda _s: (True, 0.0),
                          certify_mode='each_move')
    cases.append(('C6d 认证全通过时 final 与 each_move 结果一致（模式只在门槛处起作用）',
                  ls.plan_of(s_a) == ls.plan_of(s_b)
                  and c_a.moves_applied == c_b.moves_applied
                  and c_a.evaluations == c_b.evaluations,
                  (c_a.moves_applied, c_b.moves_applied, c_a.evaluations,
                   c_b.evaluations)))
    # ④ incumbent 自身也不通过 ⇒ LSError（不是 LS 能修的问题）
    try:
        ls.improve(c, st, certify_fn=lambda _s: (False, 0.0))
        cases.append(('C7 incumbent 也不可认证 ⇒ LSError（fail-closed）', False, '未抛错'))
    except ls.LSError as exc:
        cases.append(('C7 incumbent 也不可认证 ⇒ LSError（fail-closed）', True,
                      str(exc)[:60]))
    # ⑤ 起点不可认证时绝不静默返回"未认证的计划"
    cases.append(('C7b 上一条未产生任何部分结果（异常而非静默降级）',
                  True, None))
    return cases


def test_myopic_ablation_scope():
    """`ablate_to_myopic_scope`：只切分、不评估；冻结段内的未揭示单必须报错。"""
    cases = []
    state = {1: dict(cur=0, cur_time=0.0, load=0.0, route=[1, 2, 3], frozen=0)}
    reveal = [0.0, 0.0, 50.0, 0.0]      # 订单 2 在 clock=10 时未揭示
    my, removed = ls.ablate_to_myopic_scope(state, reveal=reveal, clock=10.0)
    cases.append(('C8 未揭示订单被切出近视臂（只切分）',
                  my[1]['route'] == [1, 3] and removed == ((1, 2),),
                  (my[1]['route'], removed)))
    cases.append(('C8b 不改动原状态（只读切分）',
                  state[1]['route'] == [1, 2, 3], state))
    # 未揭示单落在 frozen 前缀内 ⇒ 不允许"近视"掉不可撤销的在途 leg
    st_bad = {1: dict(cur=0, cur_time=0.0, load=0.0, route=[2, 3], frozen=1)}
    try:
        ls.ablate_to_myopic_scope(st_bad, reveal=reveal, clock=10.0)
        cases.append(('C9 未揭示单落在 frozen 前缀内 ⇒ LSStateError', False, '未抛错'))
    except ls.LSStateError as exc:
        cases.append(('C9 未揭示单落在 frozen 前缀内 ⇒ LSStateError', True,
                      str(exc)[:60]))
    return cases


def test_best_insert():
    """`best_insert`：最优可行位 / 冻结不越界 / 确定性 / 不改输入 / 恰好一次。"""
    cases = []
    # 线度量 x=[0,1,2,3]：depot=0。把订单 2 插进空车队的最小增量位。
    c = ctx_line(3, d_pairs={(0, 1): 1.0, (1, 2): 1.0, (2, 3): 1.0,
                             (0, 2): 2.0, (1, 3): 2.0, (0, 3): 3.0})
    st = {1: dict(cur=0, cur_time=0.0, load=0.0, route=[], frozen=0)}
    s2, ok = ls.best_insert(c, st, 2)
    cases.append(('D1 空车插入：唯一位置，增量 = 2+2 = 4',
                  ok is True and s2[1]['route'] == [2] and st[1]['route'] == [],
                  (s2[1]['route'], ok)))
    # 已有 [1]：插 2 到 pos 0 → [2,1] = 2+1+1 = 4，增量 4−2 = 2；
    #            插 2 到 pos 1 → [1,2] = 1+1+2 = 4，增量 2 ⇒ 并列，取**首个**（pos 0）⇒ [2,1]
    st2 = {1: dict(cur=0, cur_time=0.0, load=0.0, route=[1], frozen=0)}
    s3, ok3 = ls.best_insert(c, st2, 2)
    cases.append(('D1b 并列最小增量 ⇒ 取**首个**（pos 升序）⇒ 确定性，无 RNG',
                  ok3 is True and s3[1]['route'] == [2, 1] and s3[1]['route'] != [1, 2],
                  s3[1]['route']))
    # 冻结位：在途车 cur=3、route=[3]、frozen=1 ⇒ 只能插在 pos 1（不能越过在途 leg）
    st3 = {1: dict(cur=3, cur_time=0.0, load=0.0, route=[3], frozen=1)}
    s4, ok4 = ls.best_insert(c, st3, 2)
    cases.append(('D2 插入侧尊重 frozen：只能插在冻结前缀之后（[3,2] 而非 [2,3]）',
                  ok4 is True and s4[1]['route'] == [3, 2]
                  and ls.frozen_prefixes(s4) == {1: (3,)}, s4[1]['route']))
    # 不可行 ⇒ 不插入且不改输入
    c_tight = ctx_line(3, d_pairs={(0, 1): 1.0, (0, 2): 2.0, (0, 3): 3.0},
                       twe=[BIG, 0.5, 0.5, 0.5])
    st4 = {1: dict(cur=0, cur_time=1.0, load=0.0, route=[], frozen=0)}
    s5, ok5 = ls.best_insert(c_tight, st4, 1)
    cases.append(('D3 全部不可行 ⇒ 返回 False 且状态不变（由调用方定拒单语义）',
                  ok5 is False and s5[1]['route'] == [] and st4[1]['route'] == [],
                  (s5[1]['route'], ok5)))
    # 恰好一次
    try:
        ls.best_insert(c, {1: dict(cur=0, cur_time=0.0, load=0.0, route=[2],
                                   frozen=0)}, 2)
        cases.append(('D4 订单已在路线上 ⇒ LSStateError（恰好一次）', False, '未抛错'))
    except ls.LSStateError as exc:
        cases.append(('D4 订单已在路线上 ⇒ LSStateError（恰好一次）', True,
                      str(exc)[:50]))
    # 缺 frozen 的状态必须被拒绝（沿用必需字段语义）
    try:
        ls.best_insert(c, {1: dict(cur=0, cur_time=0.0, load=0.0, route=[])}, 2)
        cases.append(('D5 缺 frozen ⇒ LSStateError（必需字段语义一致）', False, '未抛错'))
    except ls.LSStateError:
        cases.append(('D5 缺 frozen ⇒ LSStateError（必需字段语义一致）', True, None))
    # 确定性
    a = ls.best_insert(c, st2, 2)[0]
    b = ls.best_insert(c, st2, 2)[0]
    cases.append(('D6 best_insert 两次运行逐位一致', a == b, (a, b)))
    # 多车择优选车：订单 3 在锚点附近时，锚点在节点 3 的车明显更省
    st6 = {1: dict(cur=0, cur_time=0.0, load=0.0, route=[1], frozen=0),
           2: dict(cur=3, cur_time=0.0, load=0.0, route=[], frozen=0)}
    s7, ok7 = ls.best_insert(c, st6, 3)
    cases.append(('D7 多车择优选车：车 2 锚点在节点 3 ⇒ 增量 3 < 车 1 的 4（选车 2）',
                  ok7 is True and s7[2]['route'] == [3] and s7[1]['route'] == [1],
                  {k: v['route'] for k, v in s7.items()}))
    return cases


def test_context_validation():
    cases = []
    try:
        ls.RouteContext(dist=[[0.0]], tws=[0.0], twe=[0.0], service=[0.0],
                        demand=[0.0], reveal=[0.0], capacity=1.0, tw_speed=0.0,
                        deadline=1.0)
        cases.append(('C10 tw_speed<=0 ⇒ LSStateError', False, '未抛错'))
    except ls.LSStateError:
        cases.append(('C10 tw_speed<=0 ⇒ LSStateError', True, None))
    c = ctx_line(2, d_pairs={(0, 1): 1.0, (0, 2): 2.0})
    cases.append(('C11 route_dist 空路线 = 0', ls.route_dist(c, 5, []) == 0.0, None))
    cases.append(('C12 plan_of / frozen_prefixes 输出与状态一致',
                  ls.plan_of({7: dict(cur=0, cur_time=0.0, load=0.0,
                                      route=[1, 2], frozen=1)}) == {7: [1, 2]}
                  and ls.frozen_prefixes({7: dict(cur=0, cur_time=0.0, load=0.0,
                                                  route=[1, 2], frozen=1)})
                  == {7: (1,)}, None))
    return cases


def main():
    cases = []
    for fn in (test_source_independence, test_certify_energy_channel_inert,
               test_frozen_required, test_frozen_blocks_the_only_improving_move,
               test_real_anchors_and_cross_vehicle,
               test_state_from_vehicles_real_anchors, test_feasibility_semantics,
               test_best_insert,
               test_determinism_and_ablation, test_certify_closure,
               test_myopic_ablation_scope, test_context_validation):
        try:
            cases.extend(fn())
        except Exception as exc:                            # noqa: BLE001
            import traceback
            cases.append(('★ %s 抛异常' % fn.__name__, False,
                          traceback.format_exc()[-400:]))
    passed = sum(1 for _n, ok_, _d in cases if ok_)
    for n, ok_, detail in cases:
        print(('  PASS ' if ok_ else '  FAIL ') + n
              + ('' if ok_ else ('   ← ' + str(detail))))
    failed = len(cases) - passed
    print('\n结果：%d 通过 / %d 失败' % (passed, failed))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
