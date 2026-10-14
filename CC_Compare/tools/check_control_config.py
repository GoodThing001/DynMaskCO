#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P1-A 对照配置核对器：证明某次运行的配置与**冻结对照规格**逐字段一致。

用途（P1-A §0.2.2 与 T 系列 T8）：
  * 跑批**前**：确认即将运行的 `cond_hist + SAA` 与 Role C 用的是同一套冻结旋钮；
  * 跑批**后**：从两臂 gate 各自的实际 `config` + `_identity` 出发，核对
    「两臂可配对的共享字段逐字段相同」——不配对就**不许**进入效果比较。

事实来源：`CC_Compare/P1A_CONTROL_FROZEN.json`（值取自 gate 落盘实据）。

用法：
    python CC_Compare/tools/check_control_config.py --config <run_config.json>
    python CC_Compare/tools/check_control_config.py --against-a <gateA.json> --against-b <gateB.json>
    # 自检（不需要任何实验产物）：
    python CC_Compare/tools/check_control_config.py --self-test
"""
import argparse
import hashlib
import io
import json
import os
import sys

# ⚠️ **2026-10-14 修复（缺陷已由用户裁定）**：原实现是
#     `sys.stdout = io.TextIOWrapper(sys.stdout.buffer, ...)`
# —— 它把 `sys.stdout` **换成新对象**，旧 `TextIOWrapper` 失去引用后被 GC，而
# `TextIOWrapper.__del__` 会按所有权**关闭底层 buffer** ⇒ **任何 import 本模块的进程**
# 此后所有 `print` 都抛 `ValueError: I/O operation on closed file`（Role C 的 T 系列
# 第一版就是这样崩的；`role_c_t_series` 因此改为 AST 抽取常量、不敢 import 本模块）。
# 修法：用 `reconfigure` **就地**改造编码，不替换对象 ⇒ 不产生"旧对象被 GC"的副作用。
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:                                         # noqa: BLE001
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SPEC = os.path.join(ROOT, 'CC_Compare', 'P1A_CONTROL_FROZEN.json')
# 必须在两臂之间**逐字段相同**的共享字段（来自 spec 的 role_c_alignment_requirements
# + CC_Compare/COMPARISON_PROTOCOL_2026-10-14.md §4.1 的协议字段）
PAIR_FIELDS = ('n_orders', 'capacity', 'num_vehicles', 'rho', 'gate_instances', 'gate_seed',
               'K', 'time_limit', 'future_policy', 'shadow_mode', 'k_neighbors',
               # ⚠️ **2026-10-14 加入（用户裁定）**：`arm_seed` 不只是日志属性 —— 它决定
               # `CondHistoricalSampler` 的随机**场景流**（`scenario_saa.py:1002`）。
               # P1-A 要求信息匹配与共同随机数流 ⇒ 场景种子属于**正式配对身份**的一部分。
               # `cond_hist` 用已核实的 7002（**不是** `SaaReplanner` 默认的 7001）；
               # 任一侧缺失或两侧不一致 ⇒ 新协议一律判配对身份失败。
               # 历史 gate **不因新增字段被追溯修改**，仍保留原资格标签。
               'arm_seed',
               'budget_B', 'cooling_share',
               'contract_sha256', 'data_sha256', 'data_meta_sha256')
# 两臂都必须为 **True**（不是"相等"）—— 各自"启动==结束"
REQUIRE_TRUE = ('source_stable',)
# **跨 gate 同源**字段：必须逐字段相同。⚠️ 这是 `source_stable=True` **不能**替代的那一层
# （A3 被判 DIAGNOSTIC/CROSS-SOURCE 正是因为它自身 source_stable=True 但跨源不符）。
CROSS_FIELDS = ('shared_source_sha256',)
# 新协议**附加**要求：正式比较必须记录并相等。旧 gate 无这些字段 ⇒ 只能作**历史比较**，
# 不能作为新协议的正式比较（除非显式 `--legacy-pair`，此时降级为告警并明确标注）。
#
# ⚠️ `booking_horizon` 是**因果性开关**：`StrictOnlineEnv(booking_horizon=None)` 会让
# `has_future_reveal` **读真实未来**（源码自述"旧行为…非因果主张"）⇒ 协议失效。
# 既有 gate 的 `config` **没有**记录它（只能从 runner 源码看到 `BOOKING_HORIZON=16.0`）
# ⇒ 因果性无法从落盘产物审计。新协议必须显式记录且两臂相等。
NEW_PROTOCOL_FIELDS = ('protocol_id', 'booking_horizon')


def load_spec(path=SPEC):
    with io.open(path, encoding='utf-8') as f:
        return json.load(f)


def _norm(v):
    """比较用归一：浮点按 1e-9 容差（跑批会经过 JSON 往返），其余按值。"""
    if isinstance(v, bool):
        return ('bool', v)
    if isinstance(v, (int, float)):
        return ('num', round(float(v), 9))
    return ('other', v)


def compare_config(cfg, spec):
    """把一份运行配置与冻结 config 逐字段比。返回 `(ok, rows)`。

    `rows` 每项 `(字段, 冻结值, 实际值, 状态)`；缺失/多出/不等都记下来 ——
    **多出的字段不算失败**（如 provider/arm 等 runner 自有字段），但要列出来供人看。
    """
    frozen = spec['frozen_config']
    rows, bad = [], []
    for k, want in frozen.items():
        got = (cfg or {}).get(k, '<缺>')
        if _norm(got) != _norm(want):
            bad.append(k)
            rows.append((k, want, got, '✗ 不一致'))
        else:
            rows.append((k, want, got, 'OK'))
    extra = sorted(set((cfg or {}).keys()) - set(frozen.keys()))
    return (not bad), rows, extra, bad


def _digest(v):
    """把 dict/list 归一成一个短摘要，便于比对与打印（避免刷屏）。"""
    if isinstance(v, dict):
        items = sorted((str(k), str(x)) for k, x in v.items())
        h = hashlib.sha256(('\n'.join('%s %s' % kv for kv in items)).encode('utf-8'))
        return 'dict(n=%d, sha=%s)' % (len(v), h.hexdigest()[:16])
    if isinstance(v, list):
        return 'list(n=%d)' % len(v)
    return v


def compare_pair(a, b, spec=None, legacy_pair=False):
    """两臂 gate 的配置/身份/协议**配对性**核对。返回 `(ok, rows, bad)`。

    判据分三层（`CC_Compare/COMPARISON_PROTOCOL_2026-10-14.md` §4）：

      ① `PAIR_FIELDS` —— 必须**逐字段相同**（可见信息协议/合同/主档/旋钮/预算）；
      ② `REQUIRE_TRUE` —— 两臂各自必须为 **True**（`source_stable`，只证"启动==结束"）；
      ③ `CROSS_FIELDS` —— **跨 gate 同源**：必须逐字段相同。
         ⚠️ ③ **不能**被 ② 替代 —— A3 正是 `source_stable=True` 却跨源不符（CROSS-SOURCE）。
      ④ `NEW_PROTOCOL_FIELDS` —— 新协议附加：必须记录且相等；缺失 ⇒ 只能作**历史比较**
         （除非 `legacy_pair=True`，此时降级为告警并在输出中明确标注）。

    ⚠️ **fail-closed**（2026-10-14 实测踩到并修正）：第一版 `pick()` 只看 gate 顶层与
    `results[*]._identity`，**没看 `config` / `identity` 子块** ⇒ 真实 baseline gate 的
    `K` / `time_limit` / `budget_B` / 合同哈希**全部取到 `'<缺>'`**，而"缺失 == 缺失"
    被 `_norm` 判为相等 ⇒ 输出 `PAIRABLE` —— 一个**假通过**。
    现在：① 依次在 `顶层 → config → identity → shared_config → results[*]._identity` 里找；
    ② **任一侧缺失即判失败**（拿不到值就**不能**证明"相同"）。
    """
    def pick(d, key):
        d = d or {}
        if key in d:
            return d[key]
        for sub in ('config', 'identity', 'shared_config'):
            blk = d.get(sub)
            if isinstance(blk, dict) and key in blk:
                return blk[key]
        for _yr, yb in (d.get('results') or {}).items():
            if not isinstance(yb, dict):
                continue
            idt = yb.get('_identity')
            if isinstance(idt, dict) and key in idt:
                return idt[key]
            if key in yb:
                return yb[key]
        return None                       # None 表示**缺失**（与"值为 None"不同）

    def missing(v):
        return v is None

    rows, bad = [], []

    for k in PAIR_FIELDS:
        va, vb = pick(a, k), pick(b, k)
        if missing(va) or missing(vb):
            bad.append(k)
            rows.append((k, '<缺>' if missing(va) else _digest(va),
                         '<缺>' if missing(vb) else _digest(vb), '✗ 缺失（无法证明相同）'))
        elif _norm(va) != _norm(vb):
            bad.append(k)
            rows.append((k, _digest(va), _digest(vb), '✗ 两臂不一致'))
        else:
            rows.append((k, _digest(va), _digest(vb), 'OK'))

    for k in REQUIRE_TRUE:
        va, vb = pick(a, k), pick(b, k)
        if va is not True or vb is not True:
            bad.append(k)
            rows.append((k, va, vb, '✗ 必须两臂都为 True'))
        else:
            rows.append((k, True, True, 'OK（启动==结束；⚠️ 不等于跨源相符）'))

    for k in CROSS_FIELDS:
        va, vb = pick(a, k), pick(b, k)
        if missing(va) or missing(vb):
            bad.append(k)
            rows.append((k, _digest(va) if not missing(va) else '<缺>',
                         _digest(vb) if not missing(vb) else '<缺>',
                         '✗ 缺失（无法证明跨源相符）'))
        elif _norm(va) != _norm(vb):
            bad.append(k)
            rows.append((k, _digest(va), _digest(vb), '✗ **跨源不符**（各自 source_stable 不能替代）'))
        else:
            rows.append((k, _digest(va), _digest(vb), 'OK（跨 gate 共享源码逐文件相符）'))

    for k in NEW_PROTOCOL_FIELDS:
        va, vb = pick(a, k), pick(b, k)
        if missing(va) or missing(vb):
            tag = ('⚠️ 缺失（`--legacy-pair` 故仅告警：该对只能作**历史比较**）'
                   if legacy_pair else '✗ 缺失（新协议要求记录且相等）')
            rows.append((k, va if not missing(va) else '<缺>',
                         vb if not missing(vb) else '<缺>', tag))
            if not legacy_pair:
                bad.append(k)
        elif _norm(va) != _norm(vb):
            bad.append(k)
            rows.append((k, va, vb, '✗ 两臂不一致'))
        else:
            rows.append((k, va, vb, 'OK'))
    return (not bad), rows, bad


def _self_check_spec(spec):
    """规格自洽性：必须存在的键 + 明确的未授权项。"""
    need = ('frozen_config', 'shared_identity_anchor', 'evidence',
            'must_rerun_same_batch', 'forbidden')
    missing = [k for k in need if k not in spec]
    problems = []
    if missing:
        problems.append('规格缺键: %s' % missing)
    if spec.get('must_rerun_same_batch', {}).get('required') is not True:
        problems.append('必须声明 must_rerun_same_batch.required=true（历史基线不可复用）')
    if spec.get('frozen_config', {}).get('future_policy') not in ('reveal', 'density'):
        problems.append('future_policy 未冻结为已知值')
    # 保留最终测试种子不得进入本次可用种子
    res = spec.get('shared_identity_anchor', {}).get('seeds', {}).get('final_test_reserved')
    if not res:
        problems.append('规格未声明 final_test_reserved（禁止触碰的种子）')
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', help='单个运行配置 JSON（与冻结 config 逐字段比）')
    ap.add_argument('--against-a', help='gate A json（配对核对）')
    ap.add_argument('--against-b', help='gate B json（配对核对）')
    ap.add_argument('--spec', default=SPEC)
    ap.add_argument('--legacy-pair', action='store_true', dest='legacy_pair',
                    help='旧 gate 无 protocol_id：降级为告警（该对只能作历史比较）')
    ap.add_argument('--self-test', action='store_true', dest='self_test')
    a = ap.parse_args()
    spec = load_spec(a.spec)
    rc = 0

    print('=== 规格自洽性 ===')
    probs = _self_check_spec(spec)
    for p in probs:
        print('   ✗ %s' % p)
    if probs:
        rc = 1
    else:
        print('   OK 规格齐备（含 must_rerun_same_batch / forbidden / 保留种子）')
    print('   冻结 config 字段数 = %d；共享身份锚点 = %d 项'
          % (len(spec['frozen_config']), len(spec['shared_identity_anchor'])))
    print('   future_policy = %r（对照强度由此固定）' % spec['frozen_config']['future_policy'])

    if a.self_test:
        # 合成自检：用冻结值本身当"一次运行"⇒ 必须全 OK
        ok, rows, extra, bad = compare_config(dict(spec['frozen_config']), spec)
        print('\n=== 自检：用冻结值自身当运行配置 ===')
        print('   逐字段 = %d；不一致 = %d；多出字段 = %s'
              % (len(rows), len(bad), extra or '无'))
        if not ok:
            print('   ✗ 自检失败：冻结值与自己都不一致（规格内部矛盾）')
            rc = 1
        else:
            print('   OK 全部一致（规格自洽）')
        # 反面：改一个字段必须被抓到
        broken = dict(spec['frozen_config'])
        broken['K'] = 20
        ok2, _r2, _e2, bad2 = compare_config(broken, spec)
        print('   反例 K=20 ⇒ 被抓到 = %s（%s）' % (bad2 == ['K'], bad2))
        if bad2 != ['K']:
            rc = 1
        # 配对：同一配置两臂 ⇒ OK；身份不一致 / 缺字段 ⇒ 必须失败
        g = {'config': dict(spec['frozen_config']), **spec['shared_identity_anchor']}
        g['source_stable'] = True
        g['shared_source_sha256'] = {'a.py': 'a' * 64}
        g['protocol_id'] = 'p1a-v1'
        g['booking_horizon'] = spec['must_record_causal_flag']['booking_horizon']
        ok3, _r3, bad3 = compare_pair(g, g, spec)
        print('   配对（同一 gate，含 protocol_id/跨源）⇒ OK=%s' % ok3)
        g2 = json.loads(json.dumps(g))
        g2['contract_sha256'] = 'f' * 64
        ok4, _r4, bad4 = compare_pair(g, g2, spec)
        print('   配对（合同哈希不同）⇒ OK=%s，检出 %s' % (ok4, bad4))
        # 回归本日踩到的**假通过**：两个空 gate 不得被判 PAIRABLE
        ok5, _r5, bad5 = compare_pair({}, {}, spec)
        print('   配对（两个空 gate）⇒ OK=%s，检出缺失 %d 项（必须 >0）'
              % (ok5, len(bad5)))
        # 跨源不符：source_stable 都为 True 也不得通过（A3 的情形）
        g3 = json.loads(json.dumps(g))
        g3['shared_source_sha256'] = {'a.py': 'b' * 64}
        ok6, _r6, bad6 = compare_pair(g, g3, spec)
        print('   配对（跨源不符但双方 source_stable=True）⇒ OK=%s，检出 %s'
              % (ok6, bad6))
        # 新协议缺字段：默认 FAIL；--legacy-pair 降级为告警
        g4 = json.loads(json.dumps(g))
        g4.pop('protocol_id')
        g4.pop('booking_horizon')
        ok7, _r7, bad7 = compare_pair(g4, g4, spec)
        ok8, _r8, bad8 = compare_pair(g4, g4, spec, legacy_pair=True)
        print('   缺 protocol_id+booking_horizon ⇒ 默认 OK=%s（检出 %s）；--legacy-pair ⇒ OK=%s'
              % (ok7, sorted(bad7), ok8))
        # 因果开关必须两臂相等（不等 ⇒ FAIL）
        g5 = json.loads(json.dumps(g))
        g5['booking_horizon'] = None
        ok9, _r9, bad9 = compare_pair(g, g5, spec)
        print('   两臂 booking_horizon 不等（16.0 vs None）⇒ OK=%s，检出 %s' % (ok9, bad9))
        if (ok3 is not True or ok4 or bad4 != ['contract_sha256'] or ok5 or not bad5
                or ok6 or bad6 != ['shared_source_sha256']
                or ok7 or set(bad7) != {'protocol_id', 'booking_horizon'} or not ok8
                or ok9 or bad9 != ['booking_horizon'] or rc):
            rc = 1

    if a.config:
        cfg = json.load(io.open(a.config, encoding='utf-8'))
        cfg = cfg.get('config', cfg)
        print('\n=== 与冻结 config 逐字段比对：%s ===' % a.config)
        ok, rows, extra, bad = compare_config(cfg, spec)
        for k, want, got, st in rows:
            print('   %-20s 冻结=%-12r 实际=%-12r %s' % (k, want, got, st))
        if extra:
            print('   （额外字段，不计失败）: %s' % extra)
        print('   => %s%s' % ('CONFIG_MATCH' if ok else 'CONFIG_MISMATCH: ',
                             '' if ok else ','.join(bad)))
        if not ok:
            rc = 1

    if a.against_a and a.against_b:
        A = json.load(io.open(a.against_a, encoding='utf-8'))
        B = json.load(io.open(a.against_b, encoding='utf-8'))
        print('\n=== 两臂配对性核对（共享字段 / 必须为真 / 跨源 / 新协议字段）===')
        ok, rows, bad = compare_pair(A, B, spec, legacy_pair=a.legacy_pair)
        for k, va, vb, st in rows:
            print('   %-20s A=%-30s B=%-30s %s' % (k, va, vb, st))
        print('   => %s%s' % ('PAIRABLE' if ok else 'NOT_PAIRABLE: ', '' if ok else ','.join(bad)))
        if not ok:
            rc = 1

    print('\n%s' % ('CHECK_CONTROL_CONFIG_PASS' if rc == 0 else 'CHECK_CONTROL_CONFIG_FAIL'))
    return rc


if __name__ == '__main__':
    sys.exit(main())
