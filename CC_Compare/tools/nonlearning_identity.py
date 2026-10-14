#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""`nonlearning_solver` 身份闭包的**产出 + 独立重算**（P1-A Role C 用）。

为什么需要：按 `P1A_强动态对照_预声明_2026-10-14.md` §0.2.1，**非学习**求解器/规划器
（如 Role C）**没有也不得伪造** `checkpoint_sha256`，但必须闭合

  * `provider_dependency_sha256` —— 源码 + 第三方依赖闭包
  * `solver_config_sha256`      —— 求解器配置（预算/时限/种子/各旋钮）

而 `qualification_gate.q_dependency_closure` 是 **fail-closed 三重校验**：
① 字段齐备 ② 64 位小写十六进制 ③ 调用方提供**从磁盘独立重算**的 `actual_hashes`
且与身份块**逐字段相符**（身份块自述不能自证）。

本模块就是那个"独立重算"的规范实现，刻意做成**纯函数 + 显式文件清单**，
不硬编码任何路径、不读任何 gate、**不涉及任何效用**，因此可完全离线单测。

用法（库）：

    from nonlearning_identity import build_identity, recompute_actual_hashes
    ident = build_identity(dependency_files=[...], config=cfg,
                           contract_sha256=..., data_sha256=..., data_meta_sha256=...)
    actual = recompute_actual_hashes(dependency_files=[...], config=cfg)
    # 再把 (ident, actual) 交给 qualification_gate.evaluate(..., method_class='nonlearning_solver')

用法（CLI，只读）：

    python CC_Compare/tools/nonlearning_identity.py --verify --config <config.json> \\
        --dependency-file <f1> --dependency-file <f2> ...
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys

if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

REQUIRED_NONLEARNING_IDENTITY = ('provider_dependency_sha256', 'solver_config_sha256')


def file_sha256(path):
    """单文件 SHA-256（分块读，避免大文件占内存）。"""
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def _rel(path, base=None):
    """相对路径（用 `/` 分隔），使闭包**与工作目录无关**。"""
    p = os.path.abspath(path)
    if base:
        try:
            p = os.path.relpath(p, os.path.abspath(base))
        except ValueError:                                    # 不同盘符（Windows）
            pass
    return p.replace('\\', '/')


def dependency_closure_sha256(dependency_files, base=None):
    """源码 + 依赖闭包哈希：对每个文件取 `(相对路径, 文件 SHA-256)`，**按相对路径排序**后聚合。

    * **顺序无关**：调用方传入的文件顺序不影响结果（先排序再聚合）。
    * **路径敏感**：改名/移动会改变结果（这是有意的 —— 闭包要能发现文件被替换）。
    * 空清单 ⇒ 返回 None（**不得**用空清单冒充"已闭合"）。
    """
    files = [f for f in (dependency_files or [])]
    if not files:
        return None
    rows = sorted((_rel(f, base), file_sha256(f)) for f in files)
    h = hashlib.sha256()
    for rel, sha in rows:
        h.update(('%s %s\n' % (rel, sha)).encode('utf-8'))
    return h.hexdigest()


def _canonical(obj):
    """把配置规范化成**稳定文本**：键排序 + 浮点规范化。

    ⚠️ 为什么不能直接 `json.dumps`：`10 == 10.0` 但 `repr` 不同，
    而"配置"里 `K=10` 与 `time_limit=10.0` 必须**可区分**；同时
    `10.0` 与 `10.00` 必须**不可区分**（同一配置的两种写法）。故 float 一律
    用 `repr(float(x))` 归一。
    """
    if isinstance(obj, dict):
        return '{' + ','.join('%s:%s' % (json.dumps(str(k), ensure_ascii=False),
                                        _canonical(obj[k]))
                              for k in sorted(obj, key=lambda x: str(x))) + '}'
    if isinstance(obj, (list, tuple)):
        return '[' + ','.join(_canonical(v) for v in obj) + ']'
    if isinstance(obj, bool):
        return 'true' if obj else 'false'
    if isinstance(obj, int):
        return 'int:%d' % obj
    if isinstance(obj, float):
        return 'float:%s' % repr(float(obj))
    if obj is None:
        return 'null'
    return 'str:%s' % json.dumps(str(obj), ensure_ascii=False)


def solver_config_sha256(config):
    """求解器配置哈希。`config` 必须是 dict；空/None ⇒ None（**不得**冒充已冻结）。"""
    if not config:
        return None
    return hashlib.sha256(_canonical(config).encode('utf-8')).hexdigest()


def build_identity(dependency_files, config, contract_sha256, data_sha256,
                   data_meta_sha256, source_stable=True, base=None, extra=None):
    """产出 `nonlearning_solver` 的**身份块**（不含也不要求 `checkpoint_sha256`）。

    ⚠️ 共享身份（合同/数据/数据元）对非学习求解器**同样强制** —— 它是被比较的**对照**，
    不是豁免对象。缺任一 ⇒ 该字段为 None，`qualification_gate` 会因此否决。
    """
    ident = {
        'source_stable': bool(source_stable),
        'contract_sha256': contract_sha256,
        'data_sha256': data_sha256,
        'data_meta_sha256': data_meta_sha256,
        'provider_dependency_sha256': dependency_closure_sha256(dependency_files, base),
        'solver_config_sha256': solver_config_sha256(config),
        'n_dependency_files': len(list(dependency_files or [])),
    }
    if extra:
        ident.update(extra)
    return ident


def recompute_actual_hashes(dependency_files, config, base=None):
    """**从磁盘独立重算**的 `actual_hashes`（交给 gate 做第 ③ 重校验）。

    刻意与 `build_identity` 走同两条纯函数，但**语义不同**：这里有"独立"责任 ——
    调用方应在**跑批后**再调用一次，用当前磁盘状态与身份块比对。
    """
    return {
        'provider_dependency_sha256': dependency_closure_sha256(dependency_files, base),
        'solver_config_sha256': solver_config_sha256(config),
    }


def verify_closure(identity, dependency_files, config, base=None):
    """三重校验的**本地**复刻（与 `qualification_gate` 同判据），返回 `(ok, reasons)`。

    用于**跑批前**自检：身份块能不能通过独立重算。不改任何 gate 逻辑，
    真正的裁决仍由 `qualification_gate.evaluate` 给出。
    """
    import re
    reasons = []
    sha_re = re.compile(r'^[0-9a-f]{64}$')
    actual = recompute_actual_hashes(dependency_files, config, base)
    for f in REQUIRED_NONLEARNING_IDENTITY:
        got = (identity or {}).get(f)
        if not got:
            reasons.append('缺字段 %s' % f)
            continue
        if not sha_re.match(str(got)):
            reasons.append('%s 格式非法（须 64 位小写十六进制）' % f)
            continue
        if actual.get(f) is None:
            reasons.append('%s 无法独立重算（清单/配置为空）' % f)
            continue
        if str(actual[f]).lower() != str(got).lower():
            reasons.append('%s 与磁盘实际重算不符（文件或配置被改动？）' % f)
    if (identity or {}).get('checkpoint_sha256'):
        reasons.append('非学习求解器**不得**声明 checkpoint_sha256（疑误标或伪造）')
    for f in ('contract_sha256', 'data_sha256', 'data_meta_sha256'):
        if not (identity or {}).get(f):
            reasons.append('缺共享身份 %s（对非学习求解器**同样强制**）' % f)
    return (not reasons), reasons


def _main():
    ap = argparse.ArgumentParser(description='nonlearning_solver 身份闭包产出/重算（只读）')
    ap.add_argument('--config', required=True, help='求解器配置 JSON（§3 全部旋钮）')
    ap.add_argument('--dependency-file', action='append', default=[],
                    help='依赖/源码文件（可重复）；建议含自己的 provider 与求解器源码')
    ap.add_argument('--base', default=None, help='相对路径基准（默认当前目录）')
    ap.add_argument('--contract-sha256', default='0' * 64)
    ap.add_argument('--data-sha256', default='0' * 64)
    ap.add_argument('--data-meta-sha256', default='0' * 64)
    a = ap.parse_args()

    cfg = json.load(open(a.config, encoding='utf-8'))
    missing = [f for f in a.dependency_file if not os.path.isfile(f)]
    if missing:
        print('FAIL: 依赖文件不存在: %s' % missing[:5])
        return 2
    ident = build_identity(a.dependency_file, cfg, a.contract_sha256, a.data_sha256,
                           a.data_meta_sha256, base=a.base)
    ok, reasons = verify_closure(ident, a.dependency_file, cfg, a.base)
    out = {'identity': ident,
           'actual_hashes': recompute_actual_hashes(a.dependency_file, cfg, a.base),
           'closure_ok': ok, 'reasons': reasons,
           'note': ('只读输出；**不**构成实验、**不**涉及效用。身份块与独立重算结果'
                    '一并交给 qualification_gate.evaluate(method_class="nonlearning_solver")。')}
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print('CLOSURE_%s' % ('PASS' if ok else 'FAIL'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(_main())
