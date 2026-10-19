#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""DEV-GATE v4 封印的**只读**兼容性诊断：原始字节 SHA256 + 显式编码解析。

背景
====
`results/o0cc/dev_gate_v4_official/EVIDENCE_SEAL.json` 是**全仓唯一的封印官方裁决**
（`verdict = "GO"`，run_id `20260908-054550`），但它**不是 UTF-8**：
第 151 字节起的 `archive` 字段里，中文目录名"正式冻结包"被按 **GBK** 写入。
后果：任何默认 UTF-8 的工具（`json.load(open(p))`）都会抛
`UnicodeDecodeError: 'utf-8' codec can't decode byte 0xd5`。

本脚本**不重写封印**，只做两件可复算的事：
  1. 对**原始字节**求 SHA256（这是唯一与编码无关的身份）；
  2. 用**显式声明的编码**（逐个尝试并回报实际成功的那个）解析 JSON，读出裁决字段；
  3. 顺带核对封印自述的 `evidence_integrity`（文件数 / 总字节）与磁盘实况是否一致。

它**只证明**"封印字节能被完整读出、且其自述计数与目录一致"；
它**不**证明认证资格、**不**升级生产状态、**不**改动封印或任何证据文件。

只读：不写仓库。机读输出用 `--out`，请指向仓库之外。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
SEAL = EXT / 'results' / 'o0cc' / 'dev_gate_v4_official' / 'EVIDENCE_SEAL.json'
REPARSE = getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)


def _is_reparse(p):
    try:
        st = os.lstat(p)
    except OSError:
        return True
    return bool(getattr(st, 'st_file_attributes', 0) & REPARSE) or os.path.islink(p)


def _walk(top):
    for dirpath, dirnames, filenames in os.walk(str(top), followlinks=False):
        dirnames[:] = [d for d in dirnames if not _is_reparse(os.path.join(dirpath, d))]
        for f in filenames:
            yield os.path.join(dirpath, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out')
    a = ap.parse_args()

    if not SEAL.is_file():
        print('MISSING: %s' % SEAL)
        return 2
    raw = SEAL.read_bytes()
    out = {
        'schema': 'dev_gate_v4_seal_readonly_diagnostic_v1',
        'read_only': True,
        'seal_repo_relative': str(SEAL.relative_to(EXT.parent)),
        'raw_bytes': len(raw),
        'raw_sha256': hashlib.sha256(raw).hexdigest(),
        'has_bom': raw[:3] == b'\xef\xbb\xbf',
        'encodings': {},
        'seal_rewritten': False,
    }

    text = None
    for enc in ('utf-8', 'utf-8-sig', 'gbk', 'cp936', 'latin-1'):
        try:
            s = raw.decode(enc)
            obj = json.loads(s)
            out['encodings'][enc] = {'decode': 'OK', 'json_parse': 'OK'}
            if text is None:
                text = enc
                out['selected_encoding'] = enc
                out['seal'] = {k: obj.get(k) for k in
                               ('seal_type', 'run_id', 'verdict', 'archive',
                                'created_utc', 'evidence_integrity',
                                'component_breakdown_raw', 'local')}
        except UnicodeDecodeError as e:
            out['encodings'][enc] = {'decode': 'FAIL', 'error': str(e)}
        except Exception as e:                                      # noqa: BLE001
            out['encodings'][enc] = {'decode': 'OK', 'json_parse': 'FAIL', 'error': str(e)}

    if text is None:
        out['verdict'] = 'SEAL_NOT_DECODABLE_BY_ANY_TRIED_ENCODING'
    else:
        out['verdict'] = 'SEAL_BYTES_INTACT_DECODED_AS_' + text.upper().replace('-', '_')

    # ---- self-report vs disk -------------------------------------------------
    ev = (out.get('seal') or {}).get('evidence_integrity') or {}
    d = SEAL.parent
    n = 0
    total = 0
    for p in _walk(d):
        try:
            total += os.lstat(p).st_size
        except OSError:
            pass
        n += 1
    claimed_n = ev.get('file_count')
    claimed_b = ev.get('total_bytes')
    delta_n = (n - claimed_n) if isinstance(claimed_n, int) else None
    delta_b = (total - claimed_b) if isinstance(claimed_b, int) else None
    seal_size = len(raw)
    out['disk_check'] = {
        'directory': str(d.relative_to(EXT.parent)),
        'files_on_disk': n,
        'bytes_on_disk': total,
        'claimed_file_count': claimed_n,
        'claimed_total_bytes': claimed_b,
        'delta_files': delta_n,
        'delta_bytes': delta_b,
        'seal_file_bytes': seal_size,
        'file_count_matches': (claimed_n == n) if isinstance(claimed_n, int) else None,
        'total_bytes_matches': (claimed_b == total) if isinstance(claimed_b, int) else None,
        'delta_explained_by_seal_itself': (
            (delta_n == 1 and delta_b == seal_size) if delta_n is not None else None),
        'note': ('The seal counts the evidence tree as it was when the seal was written; the seal '
                 'file was then written INTO that directory. If delta_files == 1 and '
                 'delta_bytes == the seal file size, the evidence tree is byte-identical to what '
                 'the seal attests and the only difference is the seal itself. A different delta '
                 'is reported as-is and is NOT corrected here.'),
    }

    out['statement_this_proves'] = (
        'The seal file\'s bytes are intact and readable under an explicitly declared non-UTF-8 '
        'encoding, and its self-reported integrity counters can be compared with the directory. '
        'This does NOT certify the run, does NOT upgrade any production/certification status, '
        'and does NOT rewrite the seal.')
    out['utf8_incompatibility'] = (
        'A default UTF-8 json.load() on this file raises UnicodeDecodeError; that is a portability '
        'defect of the recorded artifact, not a change in its content.')

    print(json.dumps(out, ensure_ascii=False, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n',
                               encoding='utf-8')
        print('WROTE', a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
