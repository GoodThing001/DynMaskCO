"""版本漂移 parity 校验（控制层，只比较、不重跑 oracle）。

对比「旧批次（冻结 runner 38461263 产出）」与「新 runner（b74ea7cb）重跑」的同一批实例，
判定 runner 改动是否影响计算数字。**比较字段 / 容差 / 失败退出规则全部冻结在本文件内**，
不通过 CLI 调整，避免看到结果后临时改口径。

冻结比较（每实例，old vs new）：
  0. 严格检查优先：两边先各自通过 validate_instance_record（protocol / service / repair /
     inst_idx / local 逐项对账 / 非退化 / code_sha256 缺失不查），任一不通过即 FAIL，不比较数值；
  1. baseline / oracle 的 D/Q/E/J：distance_cost, distance_km, quality_loss, energy_kwh,
     coldchain_cost（绝对容差 1e-9）；
  2. 完整 hard vector（hard_vector_from_outcome，布尔精确相等）；
  3. repair 字段：ownership_violations, terminal_unresolved（原始值必须为合法整数，不截断、
     不默认补零）；
  4. local event_deltas（长度 + 逐元素绝对容差 1e-9；local 必须存在）；
  5. 完整决策上下文：log 按 seq 排序后的 (seq, event_id, customer, state_hash, selected_action)
     序列，含 KEEP/DEFER，精确相等（不只是 selected_action 子串）。
排除字段（不参与比较）：runtime、文件 mtime、manifest 自报 code_sha256。
NaN/Inf 一律不一致（即便两边相同）。

失败退出规则（冻结）：任一实例任一字段不一致 → 打印逐字段 diff 并**非零退出**；全部一致 →
退出 0。实例缺失/无法解析/重复请求 ID 也算 FAIL。

用法（只比较已落盘的实例，不跑 oracle）：
    python scripts/evaluation/check_drift_parity.py \
        --old-dir <归档后的旧 r1_02 cell 目录> --new-dir <parity 重跑 cell 目录> \
        --instance-ids 0,1,2,3,4 --objective coldchain --out <parity_summary.json>
"""
import argparse, hashlib, json, math, os, sys

_BASE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_BASE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from hard_gate import hard_vector_from_outcome
from instance_validation import validate_instance_record

# ---- 冻结口径（勿改）----
TOL = 1e-9  # D/Q/E/J 与 event_deltas 的绝对容差
NUMERIC_FIELDS = ['distance_cost', 'distance_km', 'quality_loss', 'energy_kwh', 'coldchain_cost']
REPAIR_FIELDS = ['ownership_violations', 'terminal_unresolved']
EXCLUDED = ('runtime', 'mtime', 'code_sha256')  # 明确不参与比较的字段


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


def _is_close(a, b):
    a, b = float(a), float(b)
    if not math.isfinite(a) or not math.isfinite(b):
        return False  # NaN/Inf 一律不一致，即便两边相同
    return abs(a - b) <= TOL


def _is_int(v):
    return isinstance(v, int) and not isinstance(v, bool)


def _compare_outcome(side, old, new, diffs):
    hv_old = hard_vector_from_outcome(old)
    hv_new = hard_vector_from_outcome(new)
    if hv_old != hv_new:
        diff_keys = [k for k in hv_old if hv_old[k] != hv_new.get(k)]
        diffs.append(f'{side}.hard_vector 不一致: keys={diff_keys}')
    for f in NUMERIC_FIELDS:
        if f not in old or f not in new:
            diffs.append(f'{side}.{f} 缺失')
            continue
        if not _is_close(old[f], new[f]):
            diffs.append(f'{side}.{f}: {old[f]!r} vs {new[f]!r} (Δ={float(old[f])-float(new[f]):.3e})')
    for f in REPAIR_FIELDS:
        ov, nv = old.get(f), new.get(f)
        if not _is_int(ov) or not _is_int(nv):
            diffs.append(f'{side}.{f} 非严格整数: {ov!r} vs {nv!r}')
            continue
        if ov != nv:
            diffs.append(f'{side}.{f}: {ov} vs {nv}')


def _decision_context(rec):
    """完整决策上下文（按 seq 排序）。log 缺失/非 list/元素非 dict/缺字段/seq 重复 → None。"""
    log = rec.get('log')
    if not isinstance(log, list):
        return None
    ctx = []
    seen_seq = set()
    for r in log:
        if not isinstance(r, dict):
            return None
        for k in ('seq', 'event_id', 'customer', 'state_hash', 'selected_action', 'inst_idx'):
            if k not in r:
                return None
        seq = r.get('seq')
        if not isinstance(seq, int) or isinstance(seq, bool) or seq in seen_seq:
            return None
        seen_seq.add(seq)
        ctx.append((seq, r['event_id'], r['customer'], r['state_hash'], r['selected_action']))
    ctx.sort(key=lambda x: x[0])
    return [t[1:] for t in ctx]


def compare_instance(old_rec, new_rec, idx, objective='coldchain'):
    diffs = []
    # 0. 严格检查优先：两边都必须是合法实例，否则不比较数值
    for side, rec in (('old', old_rec), ('new', new_rec)):
        state, err = validate_instance_record(rec, idx, objective, require_local=True)
        if state != 'valid':
            diffs.append(f'{side} 未通过严格检查: {state}: {err}')
    if diffs:
        return diffs
    for side in ('baseline', 'oracle'):
        _compare_outcome(side, old_rec[side], new_rec[side], diffs)
    # local event_deltas（local 必须存在，已由严格检查保证）
    ed_old = old_rec['local']['event_deltas']
    ed_new = new_rec['local']['event_deltas']
    if len(ed_old) != len(ed_new):
        diffs.append(f'local.event_deltas 长度: {len(ed_old)} vs {len(ed_new)}')
    else:
        for i, (a, b) in enumerate(zip(ed_old, ed_new)):
            if not _is_close(a, b):
                diffs.append(f'local.event_deltas[{i}]: {a!r} vs {b!r}')
    # 完整决策上下文（含 KEEP/DEFER + event/customer/state_hash）
    ctx_old = _decision_context(old_rec)
    ctx_new = _decision_context(new_rec)
    if ctx_old is None or ctx_new is None:
        diffs.append(f'log 无效：old_valid={ctx_old is not None} new_valid={ctx_new is not None}')
    elif ctx_old != ctx_new:
        n_common = sum(1 for a, b in zip(ctx_old, ctx_new) if a == b)
        diffs.append(f'decision 上下文: 长度 {len(ctx_old)} vs {len(ctx_new)}，'
                     f'前 {n_common} 项一致；'
                     f'old[..3]={ctx_old[:3]} new[..3]={ctx_new[:3]}')
    return diffs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--old-dir', required=True)
    ap.add_argument('--new-dir', required=True)
    ap.add_argument('--instance-ids', default='0,1,2,3,4')
    ap.add_argument('--objective', choices=['coldchain'], default='coldchain')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    ids = [int(x) for x in args.instance_ids.split(',') if x.strip() != '']
    if not ids:
        raise SystemExit('--instance-ids 为空')
    if len(ids) != len(set(ids)):
        raise SystemExit('--instance-ids 含重复请求 ID')

    comparison_script_sha = _sha256_file(os.path.abspath(__file__))

    per_instance = {}
    all_ok = True
    for i in ids:
        op = os.path.join(args.old_dir, 'instances', f'inst_{i}.json')
        np_ = os.path.join(args.new_dir, 'instances', f'inst_{i}.json')
        if not os.path.exists(op):
            per_instance[i] = {'status': 'FAIL', 'diff': [f'旧实例缺失: {op}']}
            all_ok = False
            continue
        if not os.path.exists(np_):
            per_instance[i] = {'status': 'FAIL', 'diff': [f'新实例缺失: {np_}']}
            all_ok = False
            continue
        try:
            old_rec = json.load(open(op))
            new_rec = json.load(open(np_))
        except Exception as e:
            per_instance[i] = {'status': 'FAIL', 'diff': [f'解析失败: {e}']}
            all_ok = False
            continue
        # 实际实例 ID 必须等于请求 ID
        if old_rec.get('inst_idx') != i or new_rec.get('inst_idx') != i:
            per_instance[i] = {'status': 'FAIL', 'diff': [
                f'inst_idx 不符: old={old_rec.get("inst_idx")} new={new_rec.get("inst_idx")} 期望 {i}']}
            all_ok = False
            continue
        diffs = compare_instance(old_rec, new_rec, i, args.objective)
        per_instance[i] = {'status': 'PASS' if not diffs else 'FAIL', 'diff': diffs,
                           'old_sha256': _sha256_file(op), 'new_sha256': _sha256_file(np_)}
        if diffs:
            all_ok = False

    summary = {
        'objective': args.objective,
        'instance_ids': ids,
        'all_match': all_ok,
        'tolerance': TOL,
        'numeric_fields': NUMERIC_FIELDS,
        'excluded_fields': list(EXCLUDED),
        'comparison_script_sha256': comparison_script_sha,
        'per_instance': per_instance,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'=== 版本漂移 parity（容差 {TOL}，实例 {ids}）===')
    for i in ids:
        st = per_instance[i]['status']
        print(f'  inst_{i}: {st}')
        for d in per_instance[i].get('diff', []):
            print(f'      {d}')
    verdict = 'ALL_MATCH' if all_ok else 'MISMATCH'
    print(f'  verdict = {verdict}')
    print(f'  saved: {args.out}')
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
