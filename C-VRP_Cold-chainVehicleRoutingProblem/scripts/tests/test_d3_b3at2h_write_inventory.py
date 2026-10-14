#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D3-G0 Evidence B3-AT2H: READ-ONLY write-point inventory of the REAL environment.

The user ruling authorises a read-only ownership-handover design plus a write-point audit, and sets
the stop condition: "形成完整、可机械核验的真实 env 写入清单、权威状态所有权方案、权限分类和预声明
回归规格。不得因为设计完成而自动启动 AT2B。"

This script produces the mechanically checkable part: every write point in `strict_online_env.py`
(and the D3 modules), classified by the PRE-REGISTERED rules in `B3AT2H_COUNTEREXAMPLE_SPEC.json`:

  * the classification key is the ENCLOSING FUNCTION, not the target field alone, because `status`
    is written both by `_commit` (a decision write) and by `_advance_fleet` (an execution advance);
  * DECISION writes need certification; EXECUTION advances are performed by the designated env
    primitives and need invariants instead;
  * anything else is recorded as UNCLASSIFIED -- a finding that must be adjudicated explicitly, never
    silently defaulted;
  * every row carries a drift anchor (file:line + the source text) that is re-read on every run, so a
    code change invalidates the recorded rationale instead of ageing quietly.

Nothing here modifies any file: it reads, classifies, and writes one JSON inventory.

Run: python scripts/tests/test_d3_b3at2h_write_inventory.py
"""
import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent
EXT = SCRIPTS.parent
for _p in (SCRIPTS / 'simulation', SCRIPTS, SCRIPTS / 'coldchain', HERE):
    sys.path.insert(0, str(_p))

TARGET_FILE = 'scripts/simulation/strict_online_env.py'
D3_FILES = ['scripts/simulation/d3_state_container.py',
            'scripts/simulation/d3_commit_coordinator.py',
            'scripts/simulation/d3_commit_transaction.py',
            'scripts/simulation/d3_geo_adapter.py',
            'scripts/simulation/d3_composed_certifier.py']

# PRE-REGISTERED classification rules (see B3AT2H_COUNTEREXAMPLE_SPEC.json R2/R3)
DECISION_FUNCS = ('plan', 'on_reveal', '_plan_and_commit', '_commit', 'prepare_decision_point',
                  '_force_return_all')
EXECUTION_FUNCS = ('_advance_fleet', '_advance_coldchain_to', '_dispatch_coldchain', '_run_loop',
                   '_apply_door_event', 'run', 'run_resumed')
# `plan`/`on_reveal` exist on several classes; each is a decision entry point (R2).
METHOD_MUTATORS = ('append', 'extend', 'pop', 'clear', 'insert', 'remove', 'sort', 'reverse',
                   'update', 'add', 'discard')
VEHICLE_FIELDS = ('mutable_suffix', 'committed_next', 'committed_arrive', 'committed_finish',
                  'committed_coldchain_before', 'needs_replan', 'replan_reason', 'status',
                  'current_node', 'current_load', 'ready_time', 'dispatch_time', 'return_finish',
                  'coldchain_state', 'coldchain_updated_time', 'coldchain_accounting_anchor',
                  'served_route', 'served_route')
REQUIRED_FUNCTIONS = ['Replanner.plan', 'Replanner.on_reveal', 'GreedyReplanner.plan',
                      '_plan_and_commit', '_commit', '_advance_fleet', '_advance_coldchain_to',
                      '_dispatch_coldchain', '_run_loop', '_force_return_all',
                      'prepare_decision_point']
REQUIRED_TARGETS = ['mutable_suffix', 'committed_next', 'committed_arrive', 'committed_finish',
                    'committed_coldchain_before', 'needs_replan', 'replan_reason', 'status',
                    'current_node', 'current_load', 'ready_time', 'dispatch_time', 'return_finish',
                    'coldchain_state', 'coldchain_updated_time', 'coldchain_accounting_anchor',
                    'served_route', 'services', 'depot_unload_records', 'served_mask']
# failure semantics declared per enclosing function (R6); any function not listed -> UNKNOWN finding
FAILURE_SEMANTICS = {
    '_commit': 'partial_possible',
    '_plan_and_commit': 'partial_possible',
    '_force_return_all': 'partial_possible',
    'prepare_decision_point': 'refuse_before_write',
    'plan': 'refuse_before_write',
    'on_reveal': 'refuse_before_write',
    '_advance_fleet': 'partial_possible',
    '_advance_coldchain_to': 'partial_possible',
    '_dispatch_coldchain': 'partial_possible',
    '_run_loop': 'partial_possible',
    'run': 'partial_possible',
    'run_resumed': 'partial_possible',
}


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def target_name(node):
    """Human-readable target of an assignment / mutation."""
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Subscript):
        return target_name(node.value)
    return None


class Collector(ast.NodeVisitor):
    def __init__(self, path, lines):
        self.path, self.lines = path, lines
        self.writes = []
        self.classes = {}
        self._cls, self._fn = None, None

    def visit_ClassDef(self, node):
        prev = self._cls
        self._cls = node.name
        self.classes.setdefault(node.name, node)
        self.generic_visit(node)
        self._cls = prev

    def _fn(self, node):
        prev = self._fn
        self._fn = node.name
        self.generic_visit(node)
        self._fn = prev

    visit_FunctionDef = _fn
    visit_AsyncFunctionDef = _fn

    def _record(self, lineno, target, kind, value_repr):
        self.writes.append({'file': self.path, 'line': lineno, 'class': self._cls,
                            'function': self._fn, 'target': target, 'kind': kind,
                            'source': self.lines[lineno - 1].strip(), 'value': value_repr})

    def visit_Assign(self, node):
        for t in node.targets:
            name = target_name(t)
            if isinstance(t, (ast.Attribute, ast.Subscript)) and name:
                self._record(t.lineno, name, 'assign',
                             ast.unparse(node.value)[:60] if hasattr(ast, 'unparse') else '')
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        name = target_name(node.target)
        if isinstance(node.target, (ast.Attribute, ast.Subscript)) and name:
            self._record(node.target.lineno, name, 'ann_assign', '')
        self.generic_visit(node)

    def visit_AugAssign(self, node):
        name = target_name(node.target)
        if isinstance(node.target, (ast.Attribute, ast.Subscript)) and name:
            self._record(node.target.lineno, name, 'aug_assign',
                         ast.unparse(node.value)[:40] if hasattr(ast, 'unparse') else '')
        self.generic_visit(node)

    def visit_Call(self, node):
        f = node.func
        if isinstance(f, ast.Attribute) and f.attr in METHOD_MUTATORS:
            name = target_name(f.value)
            if name:
                self._record(node.lineno, name, 'mutator_call:' + f.attr, '')
        self.generic_visit(node)


def classify(row):
    """Pre-registered rules R1-R6."""
    fn = row['function']
    cls = row['class']
    if fn in DECISION_FUNCS:
        perm = 'DECISION_WRITE'
    elif fn in EXECUTION_FUNCS:
        perm = 'EXECUTION_ADVANCE'
    else:
        perm = 'UNCLASSIFIED'
    return {
        'permission_class': perm,
        'requires_recertification': (perm == 'DECISION_WRITE'),
        'failure_semantics': FAILURE_SEMANTICS.get(fn, 'UNKNOWN'),
        'classification_key': 'enclosing function=%s%s' % (cls or '-', '.' + str(fn)),
        'note': ('decision writes change the un-executed suffix or the dispatch head and must go '
                 'through the certified transaction; execution advances move the frozen prefix and '
                 'time forward and are performed by the designated env primitives under invariants'
                 if perm != 'UNCLASSIFIED' else
                 'UNCLASSIFIED: this write point is not covered by the pre-registered function sets '
                 'and must be adjudicated explicitly'),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    path = EXT / TARGET_FILE
    lines = path.read_text(encoding='utf-8').splitlines()
    tree = ast.parse('\n'.join(lines), filename=str(path))
    c = Collector(TARGET_FILE, lines)
    c.visit(tree)

    rows = []
    for w in c.writes:
        r = classify(w)
        r.update(w)
        rows.append(r)
    rows.sort(key=lambda r: (r['line'], r['target']))

    # ---- coverage: required function x target combinations --------------------------------
    def cls_fn(r):
        return ('%s.%s' % (r['class'], r['function'])) if r['class'] else r['function']

    present = {(cls_fn(r), r['target']) for r in rows}
    coverage, missing = [], []
    for req_fn in REQUIRED_FUNCTIONS:
        for req_t in REQUIRED_TARGETS:
            hit = (req_fn, req_t) in present
            # a function may legitimately not write a given target; that must be adjudicated, so we
            # record the pair either way and only *flag* the ones the ruling named explicitly
            if hit:
                coverage.append({'function': req_fn, 'target': req_t, 'found': True})
            else:
                missing.append({'function': req_fn, 'target': req_t,
                                'adjudication_required': True})
    # the ruling named the fields that MUST appear somewhere on the real path
    must_appear = ['mutable_suffix', 'committed_next', 'status', 'ready_time', 'current_node',
                   'current_load', 'served_route', 'coldchain_state', 'coldchain_updated_time']
    absent_targets = sorted(t for t in must_appear if not any(r['target'] == t for r in rows))
    unclassified = [r for r in rows if r['permission_class'] == 'UNCLASSIFIED']
    unknown_failure = [r for r in rows if r['failure_semantics'] == 'UNKNOWN']

    d3_rows = {}
    for rel in D3_FILES:
        p = EXT / rel
        if not p.is_file():
            d3_rows[rel] = {'present': False}
            continue
        ls = p.read_text(encoding='utf-8').splitlines()
        cc = Collector(rel, ls)
        cc.visit(ast.parse('\n'.join(ls), filename=str(p)))
        # only vehicle-state writes are of interest for the D3 modules
        d3_rows[rel] = {'present': True,
                        'vehicle_state_writes': [w for w in cc.writes
                                                 if w['target'] in VEHICLE_FIELDS],
                        'n_all_writes': len(cc.writes)}

    out = {
        'stage': 'D3_G0_Evidence_B3AT2H_real_env_write_inventory',
        'read_only': True, 'source_modified': False,
        'target_file': TARGET_FILE,
        'n_write_points': len(rows),
        'permission_class_counts': {k: sum(1 for r in rows if r['permission_class'] == k)
                                    for k in ('DECISION_WRITE', 'EXECUTION_ADVANCE',
                                              'UNCLASSIFIED')},
        'findings': {
            'unclassified_write_points': [{'line': r['line'], 'function': r['function'],
                                           'target': r['target']} for r in unclassified],
            'n_unclassified': len(unclassified),
            'failure_semantics_unknown': [{'line': r['line'], 'function': r['function'],
                                           'target': r['target']} for r in unknown_failure],
            'n_failure_semantics_unknown': len(unknown_failure),
            'required_targets_absent_from_the_real_path': absent_targets,
            'uncovered_function_target_pairs_requiring_adjudication': len(missing),
        },
        'rows': rows,
        'coverage': {'required_functions': REQUIRED_FUNCTIONS, 'required_targets': REQUIRED_TARGETS,
                     'covered_pairs': len(coverage), 'pairs_requiring_adjudication': missing},
        'd3_modules': d3_rows,
        'threat_model': {
            'covered': ['项目正常执行路径', '意外持有的可变引用', '历史 replanner 直接写入',
                        '已登记的动态调用（属性赋值 / 列表原地修改 / 元素替换）'],
            'explicitly_not_claimed': ['任意反射代码', '恶意内存修改'],
            'why': ('`_vehicles` / `_authoritative()` 这类单下划线名称在 Python 中只是访问约定，'
                    '不是语言级私有权限边界；因此不得仅凭私有属性宣称任意代码都无法越权修改'),
        },
        'identities': {
            'this_file_sha256': sha256_file(Path(__file__)),
            'target_file_sha256': sha256_file(path),
            'prereg_spec_sha256': sha256_file(EXT.parent / 'research' / 'review_3080_20261010' /
                                              'd3_g0_evidence_root_v1' /
                                              'B3AT2H_COUNTEREXAMPLE_SPEC.json'),
            'code': {rel: sha256_file(EXT / rel) for rel in [TARGET_FILE] + D3_FILES
                     if (EXT / rel).is_file()},
        },
        'algorithmic_gain_claim': 'none',
        'does_not_authorise_AT2B': True,
    }
    dest = (EXT.parent / 'research' / 'review_3080_20261010' / 'd3_g0_evidence_root_v1' /
            'B3AT2H_REAL_ENV_WRITE_INVENTORY.json')
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str) + '\n',
                    encoding='utf-8', newline='\n')

    print('  write points in %s: %d' % (TARGET_FILE, len(rows)))
    print('  permission classes:', json.dumps(out['permission_class_counts'], ensure_ascii=False))
    print('  unclassified write points: %d  (must be adjudicated explicitly)'
          % len(unclassified))
    print('  failure semantics UNKNOWN: %d' % len(unknown_failure))
    print('  required targets absent from the real path:', absent_targets or 'none')
    print('  function x target pairs requiring adjudication: %d' % len(missing))
    print('  D3 modules scanned: %d' % len(d3_rows))
    print('wrote', dest)
    return 0


if __name__ == '__main__':
    sys.exit(main())
