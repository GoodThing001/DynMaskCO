"""v1↔v2 逐臂对账：同方法、同代码、同协议应**逐位复现**；明显偏离即为「可疑/空臂」。

为什么需要（2026-10-13 血的教训）：PyVRP 的 v2 臂 rc=0、`mean_timeouts=0`、
`hard_feasible_rate=1.0`、身份与配对全合格 —— 一切「健康指标」都正常，**但它是空臂**：
pyvrp 0.14.0 的 `add_depot(location, ...)` 签名变了，每个实例都抛 TypeError，
driver 静默降级为「全部拒单」，于是 p_c=0 util 恰好 = **0.00**、主档 = **−1943.12**，
而 v1 同臂是 **909.2 / −569.8**。
⇒ 「rc=0 + 无超时 + 身份合格」**不足以**判定一条臂有效；必须与 v1 同臂对账。

已知会有**真实**差异、不应误报的项（已在 ALLOW 中注明原因）：
  - LIH / DeepACO：跨次不可复现（清缓存两次调用不一致）⇒ 容差放宽为「同量级」。
  - 新方法（coenrich_* / mapt_* 等 v1 不存在的臂）：无 v1 对照，报 NEW 而不判失败。

用法（在 1003 仓库根，任何有 numpy 的解释器都行）：
    python CC_Compare/tools/check_v1_v2_agreement.py
退出码：0 = 无异常偏离；1 = 存在偏离（需人工解释）。
"""
import glob
import json
import os
import sys

EXT = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..',
    'C-VRP_Cold-chainVehicleRoutingProblem'))
EXT_RES = os.path.join(EXT, 'results')
BASELINE_ARMS = {'cond_hist', 'uncond_hist', 'explicit_feat'}

# 方法 -> 允许的相对/绝对偏离；缺省为「几乎逐位一致」
ALLOW = {
    'lih': ('nondet', '跨次不可复现（清缓存两次调用不一致）'),
    'deepaco': ('nondet', '跨次不可复现（清缓存两次调用不一致）'),
}
NONDET_TOL_REL = 0.25   # 非确定性方法：25% 相对差以内算同量级


def own_util(path):
    """该 gate 里「本臂」（非 3 条基线臂）在各 p_c 档的 utility 均值。"""
    g = json.load(open(path, encoding='utf-8'))
    out = {}
    cont = g.get('results') or g.get('gate') or {}
    for _seed, per in cont.items():
        if not isinstance(per, dict):
            continue
        for p_c, blk in per.items():
            if not isinstance(blk, dict) or 'arms' not in blk:
                continue
            for an, ab in blk['arms'].items():
                if an in BASELINE_ARMS:
                    continue
                u = (ab or {}).get('utility_all_days') or {}
                if u.get('mean') is not None:
                    out[p_c] = float(u['mean'])
    return out


def method_of(name):
    n = name
    for pre in ('rule_',):
        if n.startswith(pre):
            n = n[len(pre):]
    for k in ('valuee20', 'valuee30'):
        if n.startswith(k):
            return k
    for k in ('coenrich_nopchgs', 'coenrich_pchgs', 'rrnco', 'routefinder', 'cada',
              'mvmoe', 'pomo', 'symnco', 'sgbs', 'deepaco', 'attention', 'omnivrp',
              'lih', 'ortools', 'pyvrp', 'mapt', 'myopic'):
        if n == k or n.startswith(k + '_'):
            return k
    return n


def find_v1(name):
    """v2 臂名 -> 可能的 v1 gate 路径。"""
    cands = [
        os.path.join(EXT_RES, 'a1_ext_extra', name + '_f', 'gate.json'),
        os.path.join(EXT_RES, 'a1_ext_extra', name, 'gate.json'),
        os.path.join(EXT_RES, 'a1_ext_a05e_%s' % method_of(name), 'gate.json'),
    ]
    for c in cands:
        if os.path.exists(c):
            return c
    return None


def degenerate(utils):
    """「全拒单」空臂的签名——**只用于没有 v1 对照的臂**（coenrich_*/mapt_*）。

    机制：driver 对 SOLVER_EXC / provider 异常是**静默降级为全部拒单**，而 `p_c=0` 档的
    拒单损失为 0 ⇒ utility **恰好 0.00**（合法臂几乎不可能恰好为 0：要 revenue 与 fuel+惩罚
    精确抵消，是零测集）。

    ⚠️ **为什么不能只看「恰好 0」就判缺陷**：`rule_valuee30`（λ=3 的 value/energy 规则）
    **本来就全拒单**——p_c=0 util 恰好 0.00、主档 −1943.1——而且 **v1 同臂逐位相同**。
    这是该规则的**真实、可复现**行为（λ 越大越倾向拒单，本身是个有意义的发现），不是缺陷。
    所以判定顺序必须是：**先做 v1↔v2 对账**（v1 复现 ⇒ 接受），
    只有**无 v1 对照**时才用这个签名兜底。
    （同理不要用「util < −1000」这类粗阈值：`ortools` 高罚档 −1258.8 是真实的差结果，
    而全拒单在高罚档应约为 −3886，两者并不相同。）
    """
    out = []
    for p_c, v in sorted(utils.items()):
        if p_c == 'p_c=0' and abs(v) < 1e-9:
            out.append(('DEGENERATE', '%s util=%.6f 恰好为 0 ⇒ 高度疑似「全部拒单」空臂' % (p_c, v)))
    return out


def main():
    v2 = sorted(glob.glob(os.path.join(EXT_RES, 'a1_ext_formal2', '*', 'gate.json')))
    if not v2:
        print('MISSING: results/a1_ext_formal2/*/gate.json')
        return 1
    bad, new, ok = [], [], 0
    for p in v2:
        v2name = os.path.basename(os.path.dirname(p))
        meth = method_of(v2name)
        try:
            a = own_util(p)
        except Exception as exc:  # noqa: BLE001
            bad.append((v2name, 'LOAD_FAIL', str(exc)))
            continue
        v1p = find_v1(v2name)
        if v1p is None:
            # 无 v1 对照（coenrich_* / mapt_*）：只能用「全拒单」签名兜底
            deg = degenerate(a)
            if deg:
                bad.append((v2name, 'degenerate-check', deg))
            else:
                new.append((v2name, sorted(a.items())))
            continue
        if v1p is None:
            new.append((v2name, sorted(a.items())))
            continue
        b = own_util(v1p)
        diffs = []
        for p_c, va in sorted(a.items()):
            if p_c not in b:
                continue
            vb = b[p_c]
            denom = max(abs(vb), 1e-9)
            rel = abs(va - vb) / denom
            if meth in ALLOW:
                if rel > NONDET_TOL_REL:
                    diffs.append((p_c, va, vb, rel))
            else:
                # 确定性方法：允许 ~2 个单位的绝对差（基线平移/浮点），其余视为异常
                if abs(va - vb) > 2.0:
                    diffs.append((p_c, va, vb, rel))
        if diffs:
            bad.append((v2name, os.path.relpath(v1p, EXT), diffs))
        else:
            ok += 1
            print('OK      %-22s v2 与 v1 一致（%d 档）' % (v2name, len(a)))

    print()
    for name, tiers in new:
        print('NEW     %-22s v1 无对照（%s）' % (name, ', '.join('%s=%.2f' % t for t in tiers)))
    if bad:
        print()
        print('FAIL: %d 个臂与 v1 明显偏离（需人工解释，可能是空臂/部分臂）：' % len(bad))
        for name, v1p, diffs in bad:
            print('  - %s' % name)
            if v1p == 'degenerate-check':
                print('      空臂体检（无需 v1 对照）：')
                for lvl, msg in diffs:
                    print('        [%s] %s' % (lvl, msg))
            elif isinstance(diffs, list) and diffs and len(diffs[0]) == 4:
                print('      vs %s' % v1p)
                for p_c, va, vb, rel in diffs:
                    print('        %-16s v2=%-10.2f v1=%-10.2f rel=%.1f%%' % (p_c, va, vb, rel * 100))
            else:
                print('      %s' % (diffs,))
        return 1
    print()
    print('PASS: %d 个臂与 v1 逐位/同量级一致；%d 个为 v2 新增（无 v1 对照）' % (ok, len(new)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
