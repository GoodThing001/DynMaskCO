#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Q3 A-05 独立复核：用 AST / 调用关系判断"生产路径是否真的引用了实验模块"。

背景（只读、不改任何旧工件）
============================
已提交的 `scripts/tests/q3_package_acceptance.py` 的第 A-05 行用**整文件子串匹配**
（`if EXPERIMENTAL_MODULE in p.read_text()`）判断"生产路径是否引用实验模块"。
它在 2026-10-17 16:36（commit cfa1fa8）新增的
`scripts/coldchain/q3_strict_float_experimental.py` 的 **docstring 散文**里命中了一次
（那句散文提到旧模块的文件名），于是 A-05 由 PASS 变 FAIL，整个验收器报 `ACCEPTANCE_FAIL`。

**本脚本不改旧验收器、不改旧 FAIL、不改那个 docstring。**它只做一件事：
用一个**语法级**判据回答同一个问题——生产路径里到底有没有**真的 import/调用**该模块。

判据分三档，逐文件 AST 解析后汇总：
  L1 静态 import         `import X` / `from X import ...`  —— 真正的依赖
  L2 动态取模块          `sys.modules[...]` / `importlib.import_module("X")` / `__import__("X")`
  L3 仅文本出现          出现在 docstring / 注释 / 普通字符串里 —— **不是依赖**

结论当且仅当 L1 与 L2 都为空时为 `NO_PRODUCTION_REFERENCE`（即 A-05 想表达的那句话成立）。

只读：不写仓库、不写结果目录。需要机读输出时用 `--out`，请指向**仓库之外**的路径。

用法
----
    python scripts/tests/q3_a05_reference_check.py
    python scripts/tests/q3_a05_reference_check.py --out %TEMP%/q3_a05.json
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent

# 与旧验收器一致的"生产面"清单（刻意逐条抄写，便于对照；不 import 旧脚本，保持独立）。
PRODUCTION_DIRS = ('scripts/m1b_adapter', 'scripts/simulation', 'scripts/evaluation',
                   'scripts/coldchain', 'scripts/data', 'scripts/training', 'scripts/models',
                   'scripts/expert', 'scripts/baseline', 'scripts/lib')

# 目标模块名用拼接构造，**避免本文件自身在旧验收器的子串扫描里多出一条**，
# 从而让旧验收器的输出保持可比（旧 FAIL 原样保留）。
TARGET = 'q3_thermal_' + 'interval_experimental'


def _module_names(tree):
    """所有 import 的模块名（L1）。"""
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                names.add(a.name)
                names.add(a.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.add(node.module)
                names.add(node.module.split('.')[0])
            for a in node.names:
                names.add(a.name)
    return names


def _dynamic_refs(tree, target):
    """动态导入/取模块（L2）：importlib.import_module("X") / __import__("X") / sys.modules["X"]。"""
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            fname = None
            if isinstance(fn, ast.Attribute):
                fname = fn.attr
            elif isinstance(fn, ast.Name):
                fname = fn.id
            if fname in ('import_module', '__import__'):
                for a in node.args:
                    if isinstance(a, ast.Constant) and isinstance(a.value, str):
                        if target in a.value:
                            hits.append('%s(%r)' % (fname, a.value))
        if isinstance(node, ast.Subscript):
            v = node.value
            if (isinstance(v, ast.Attribute) and v.attr == 'modules'
                    and isinstance(v.value, ast.Name) and v.value.id == 'sys'):
                s = node.slice
                if isinstance(s, ast.Constant) and isinstance(s.value, str) \
                        and target in s.value:
                    hits.append('sys.modules[%r]' % s.value)
    return hits


def _docstring_text(tree):
    """模块/类/函数 docstring 的拼接（用于区分 L3）。"""
    parts = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            d = ast.get_docstring(node, clean=False)
            if d:
                parts.append(d)
    return '\n'.join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', help='optional JSON output path (please keep it OUTSIDE the repo)')
    a = ap.parse_args()

    rows = []
    l1_hits, l2_hits, l3_hits = [], [], []
    n_py = 0
    for d in PRODUCTION_DIRS:
        base = EXT / d
        if not base.is_dir():
            continue
        for p in sorted(base.rglob('*.py')):
            n_py += 1
            try:
                src = p.read_text(encoding='utf-8', errors='replace')
            except OSError:
                continue
            if TARGET not in src:
                continue                       # 文本都不出现 -> 三种引用都不可能
            rel = str(p.relative_to(EXT))
            try:
                tree = ast.parse(src)
            except SyntaxError as e:
                rows.append({'file': rel, 'parse': 'SyntaxError: %s' % e})
                continue
            mods = _module_names(tree)
            imported = sorted(m for m in mods if TARGET in m)
            dyn = _dynamic_refs(tree, TARGET)
            doc = _docstring_text(tree)
            text_mentions = src.count(TARGET)
            # 去掉 docstring 后还剩多少处（注释无法从 AST 拿，故再粗判一次）
            residual = text_mentions - doc.count(TARGET)
            row = {'file': rel, 'imports': imported, 'dynamic': dyn,
                   'in_docstring': doc.count(TARGET), 'text_mentions': text_mentions,
                   'mentions_outside_docstring': residual}
            rows.append(row)
            if imported:
                l1_hits.append(row)
            if dyn:
                l2_hits.append(row)
            if not imported and not dyn:
                l3_hits.append(row)

    verdict = 'NO_PRODUCTION_REFERENCE' if (not l1_hits and not l2_hits) else 'PRODUCTION_REFERENCE_FOUND'
    payload = {
        'schema': 'q3_a05_ast_reference_check_v1',
        'read_only': True,
        'target_module': TARGET,
        'production_dirs': list(PRODUCTION_DIRS),
        'n_py_scanned': n_py,
        'n_files_mentioning_target': len(rows),
        'L1_static_import_hits': l1_hits,
        'L2_dynamic_module_access_hits': l2_hits,
        'L3_text_only_hits': l3_hits,
        'verdict': verdict,
        'statement_this_proves': (
            'No production path imports or dynamically loads the experimental interval module. '
            'This does NOT establish any production/certification qualification, and it does not '
            'change the old acceptance harness or its recorded FAIL.'),
        'note_on_old_harness': (
            'q3_package_acceptance.py checks a raw substring over the whole file text, so a '
            'docstring that merely NAMES the module trips it. This checker is syntax-level and is '
            'kept separate on purpose; the old harness and its FAIL are left untouched.'),
        'rows': rows,
    }
    print(json.dumps({k: payload[k] for k in
                      ('verdict', 'n_py_scanned', 'n_files_mentioning_target',
                       'L1_static_import_hits', 'L2_dynamic_module_access_hits', 'L3_text_only_hits')},
                     ensure_ascii=False, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(payload, ensure_ascii=False, indent=1) + '\n',
                               encoding='utf-8')
        print('WROTE', a.out)
    return 0 if verdict == 'NO_PRODUCTION_REFERENCE' else 1


if __name__ == '__main__':
    sys.exit(main())
