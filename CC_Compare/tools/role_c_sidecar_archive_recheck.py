#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""**只读**独立复核器：仅凭 sidecar 归档 JSONL 的**原始字节**重算输入身份（缺口 B）。

用法：
    python role_c_sidecar_archive_recheck.py <archive.jsonl> [--json]

性质：
  * **不** import 记录器的运行状态；**不**写任何文件；**不**触碰冻结模块；
  * 逐行重算 `input_snapshot_sha256` 与 `input_decision_id` 并与归档字段比对；
  * 缺口 B 之前的旧记录（无 `input_snapshot`）判
    `NOT_VERIFIABLE_FROM_ARCHIVED_BYTES`——**不推定**、**不补正**。

`--self-test` 会用**临时目录**内的合成归档做一次正负例自检（不触碰仓库任何文件）。
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import role_c_sidecar_recorder as sr      # noqa: E402

NOT_VERIFIABLE = 'NOT_VERIFIABLE_FROM_ARCHIVED_BYTES'


def recheck_lines(lines):
    """逐记录复核；返回逐条结论 + 汇总（**不做任何裁决之外的外推**）。"""
    rows, n_ok, n_bad, n_nv = [], 0, 0, 0
    for i, raw in enumerate(lines, 1):
        if not raw.strip():
            continue
        try:
            rec = json.loads(raw)
        except Exception as exc:                          # noqa: BLE001
            rows.append({'line': i, 'verdict': 'UNPARSABLE',
                         'why': '%s: %s' % (type(exc).__name__, exc)})
            n_bad += 1
            continue
        rc = sr.recompute_archived_input_identity(rec)
        if rc.get('ok'):
            verdict = 'VERIFIED'
            n_ok += 1
        elif rec.get('input_snapshot') is None:
            verdict = NOT_VERIFIABLE
            n_nv += 1
        else:
            verdict = 'MISMATCH'
            n_bad += 1
        rows.append({'line': i, 'decision_point': rec.get('decision_point'),
                     'arm': rec.get('arm'), 'verdict': verdict,
                     'detail': rc})
    return {
        'label': 'P1A-C1C4-independent-archive-recheck',
        'n_records': len(rows), 'n_verified': n_ok, 'n_mismatch': n_bad,
        'n_not_verifiable': n_nv,
        'rows': rows,
        'scope': ('本复核只重算**输入身份**；它**不**核验决策正确性、'
                  '**不**核验是否发生过真实提交、**不**核验方法优劣。'),
        'vocabulary_note': ('`%s` **不等于**"没有发生"——它只表示'
                            '归档字节不足以独立复建输入身份。' % NOT_VERIFIABLE),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description='只读复核 sidecar 归档的输入身份')
    ap.add_argument('archive', nargs='?')
    ap.add_argument('--json', action='store_true')
    ap.add_argument('--self-test', action='store_true',
                    help='在临时目录内做正负例自检（不触碰仓库文件）')
    a = ap.parse_args(argv)

    if a.self_test:
        import shutil
        import tempfile
        d = tempfile.mkdtemp(prefix='sr_recheck_')
        try:
            p = os.path.join(d, 'a.jsonl')
            rec = sr.SidecarRecorder(run_id='r', arm='c2', sidecar_path=p,
                                     protocol_identity='p1', output_root=d)
            snap = {'inst_idx': 0, 'clock': 1.0, 'order': 5,
                    'vehicles_fingerprint': [[1, 'idle', 0, 0.0, None]],
                    'plan': {'1': [2]}, 'frozen': {'1': 0}, 'visible_orders': [2, 3]}
            rec.record(decision_point=0, input_snapshot=snap, action='reject',
                       plan_before={'1': [2]}, candidate_plan={'1': [2, 5]},
                       accepted_plan=None, plan_after={'1': [2]}, commit_observed=False)
            rec.close()
            lines = io.open(p, encoding='utf-8').read().splitlines()
            good = recheck_lines(lines)
            # 负例：删掉 input_snapshot（旧版记录）
            old = json.loads(lines[0])
            old.pop('input_snapshot')
            old.pop('input_snapshot_sha256')
            neg = recheck_lines([json.dumps(old, ensure_ascii=False)])
            print(json.dumps({'positive': {'n_verified': good['n_verified']},
                              'negative_verdict': neg['rows'][0]['verdict']},
                             ensure_ascii=False, indent=2))
            ok = (good['n_verified'] == 1 and neg['rows'][0]['verdict'] == NOT_VERIFIABLE)
            print('ARCHIVE_RECHECK_SELFTEST_%s' % ('PASS' if ok else 'FAIL'))
            return 0 if ok else 1
        finally:
            shutil.rmtree(d, ignore_errors=True)

    if not a.archive:
        ap.error('需要 <archive.jsonl> 或 --self-test')
    with io.open(a.archive, encoding='utf-8') as f:
        rep = recheck_lines(f.read().splitlines())
    rep['archive'] = os.path.abspath(a.archive)
    if a.json:
        print(json.dumps(rep, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print('records=%d verified=%d mismatch=%d not_verifiable=%d'
              % (rep['n_records'], rep['n_verified'], rep['n_mismatch'],
                 rep['n_not_verifiable']))
    return 0


if __name__ == '__main__':
    sys.exit(main())
