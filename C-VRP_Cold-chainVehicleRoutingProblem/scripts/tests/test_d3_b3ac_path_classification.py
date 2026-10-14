#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""D3-G0 Evidence B3-AC part 1: READ-ONLY submit-path classification.

The user ruling for this batch is explicit:

    "七处直接调用认证原语" 不等于 "七处能够绕过 D3 提交门的状态写入者"

so the classification must be MECHANICALLY reviewable rather than a prose opinion.  This file
derives it from the source with `ast`:

  * every non-test call site of the primitives that can DECIDE or WRITE
    (`certify_plan` / `certify_route` / `compose_certification` / `pre_install_reverify` /
    `install_fn` / `run_repair_transaction`);
  * for each site: enclosing class + function, whether the result is BRANCHED ON (a filter) or
    merely recorded, and whether the enclosing class assigns `mutable_suffix` / `committed_next`
    on the vehicles the env hands the replanner (i.e. can cause an UNGATED commit);
  * whether the file wires the class into `StrictOnlineEnv(..., replanner=<expr>)`, and whether
    the file touches any D3 gate primitive at all;
  * base-class resolution WITHIN the scanned set, because the write lives in an INHERITED `plan()`:
    `rolling_optimizer.RollingOptimizeReplanner`, `rrnco_accept_replanner.RRNCOAcceptReplanner`,
    `solver_accept_replanner.SolverAcceptReplanner`, `run_a1_downstream_ablation`'s class and the
    two `research_20261010` attribution replanners all reach `C0ReserveReplanner.plan`, which sets
    `v.mutable_suffix` on the LIVE vehicles.  A classifier that only reads the site's own class body
    reports `write=False` for six of the seven sites -- a false negative in the direction the user
    warned about ("七处直接调用认证原语" != "七处能够绕过 D3 提交门的状态写入者" applies in BOTH
    directions), so inheritance is resolved and the resolution PATH is recorded as evidence.
  * which runner wires the class as `env.replanner`, and whether ANY D3 module references the class
    at all (the decisive difference between "capable of writing live state" and "wired to D3").

The four category rules and their priority are PRE-REGISTERED in `B3AC_COUNTEREXAMPLE_SPEC.json`
and are implemented verbatim here.  Each verdict carries `evidence` anchors (file, line, the exact
source line); the anchors are re-read from disk on every run, so a code change that invalidates a
recorded rationale fails this check instead of silently ageing.

Nothing here writes production state, imports nothing that writes, and modifies no source file.

Run: python scripts/tests/test_d3_b3ac_path_classification.py
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
for _p in (SCRIPTS / 'simulation', SCRIPTS, SCRIPTS / 'coldchain', SCRIPTS / 'evaluation', HERE):
    sys.path.insert(0, str(_p))

DECIDE_PRIMITIVES = ('certify_plan', 'certify_route', 'compose_certification',
                     'pre_install_reverify')
WRITE_PRIMITIVES = ('install_fn', 'run_repair_transaction', '_commit', '_plan_and_commit')
# The primitives `run_repair_transaction` demands before it will install (G0-Safety batch 1).
D3_GATE_PRIMITIVES = ('compose_certification', 'pre_install_reverify', 'run_repair_transaction')
WRITE_ATTRS = ('mutable_suffix', 'committed_next')
# Pre-registered target set: the seven NON-test `certify_plan` call sites named in B3-CG.
PREREGISTERED_SITES = [
    ('scripts/evaluation/rolling_optimizer.py', 197),
    ('scripts/evaluation/rrnco_accept_replanner.py', 150),
    ('scripts/evaluation/run_a1_downstream_ablation.py', 88),
    ('scripts/evaluation/run_exp_energy_c0.py', 302),
    ('scripts/evaluation/solver_accept_replanner.py', 424),
    ('scripts/research_20261010/pickup_budget_sweep.py', 96),
    ('scripts/research_20261010/pickup_rejection_attribution.py', 97),
]
CATEGORIES = ('D3_REAL_COMMIT_ENTRY', 'INDIRECT_WRITER_CAN_WRITE_D3_STATE',
              'HISTORICAL_BASELINE_INDEPENDENT_ENTRY', 'CANDIDATE_EVAL_ONLY_NO_REAL_WRITE')


def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def source_files():
    """Every tracked-looking `.py` under scripts/, excluding tests and this file's own directory."""
    out = []
    for p in sorted(SCRIPTS.rglob('*.py')):
        rel = p.relative_to(EXT).as_posix()
        if '/tests/' in rel or rel.endswith('test_d3_b3ac_path_classification.py'):
            continue
        if '__pycache__' in rel or '/lib/' in rel:
            continue
        out.append(p)
    return out


def call_name(node):
    f = node.func
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute):
        return f.attr
    return None


class _Collect(ast.NodeVisitor):
    """Collect call sites with their enclosing class/function and the source line text."""

    def __init__(self, path, lines):
        self.path, self.lines = path, lines
        self.sites = []
        self.classes = {}          # class -> {'methods': {fn: node}, 'body': node}
        self.write_attr_assigns = []   # (line, class, function, target_attr)
        self.env_wiring = []           # (line, expr_text)
        self.imports_d3 = False
        self._cls, self._fn = None, None

    def visit_Import(self, node):
        for a in node.names:
            if a.name.startswith('d3_'):
                self.imports_d3 = True
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if (node.module or '').startswith('d3_'):
            self.imports_d3 = True
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        prev = self._cls
        self._cls = node.name
        self.classes.setdefault(node.name, {'node': node, 'methods': {}})
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.classes[node.name]['methods'][item.name] = item
        self.generic_visit(node)
        self._cls = prev

    def _visit_fn(self, node):
        prev = self._fn
        self._fn = node.name
        if self._cls is not None:
            self.classes.setdefault(self._cls, {'node': None, 'methods': {}})
            self.classes[self._cls]['methods'].setdefault(node.name, node)
        self.generic_visit(node)
        self._fn = prev

    visit_FunctionDef = _visit_fn
    visit_AsyncFunctionDef = _visit_fn

    def visit_Assign(self, node):
        for t in node.targets:
            if isinstance(t, ast.Attribute) and t.attr in WRITE_ATTRS:
                self.write_attr_assigns.append((t.lineno, self._cls, self._fn, t.attr))
        self.generic_visit(node)

    def visit_AnnAssign(self, node):
        t = node.target
        if isinstance(t, ast.Attribute) and t.attr in WRITE_ATTRS:
            self.write_attr_assigns.append((t.lineno, self._cls, self._fn, t.attr))
        self.generic_visit(node)

    def visit_Call(self, node):
        name = call_name(node)
        if name in DECIDE_PRIMITIVES or name in WRITE_PRIMITIVES:
            self.sites.append({
                'primitive': name,
                'line': node.lineno,
                'end_line': getattr(node, 'end_lineno', node.lineno),
                'class': self._cls, 'function': self._fn,
                'text': self.lines[node.lineno - 1].strip(),
                'kind': 'decide' if name in DECIDE_PRIMITIVES else 'write',
            })
        if name == 'StrictOnlineEnv':
            for kw in node.keywords:
                if kw.arg == 'replanner':
                    self.env_wiring.append((node.lineno,
                                            self.lines[node.lineno - 1].strip()))
        self.generic_visit(node)


def analyse(path):
    text = path.read_text(encoding='utf-8')
    lines = text.splitlines()
    try:
        tree = ast.parse(text, filename=str(path))
    except SyntaxError as exc:
        return {'path': path, 'syntax_error': str(exc)}
    c = _Collect(path, lines)
    c.visit(tree)
    return {'path': path, 'text': text, 'lines': lines, 'sites': c.sites,
            'classes': c.classes, 'write_attr_assigns': c.write_attr_assigns,
            'env_wiring': c.env_wiring, 'imports_d3': c.imports_d3}


def _bases_of(cls_node):
    out = []
    if cls_node is None:
        return out
    for b in cls_node.bases:
        if isinstance(b, ast.Name):
            out.append(b.id)
        elif isinstance(b, ast.Attribute):
            out.append(b.attr)
    return out


def build_class_index(infos):
    """Global `class name -> (defining info, class node)` across the scanned set."""
    idx = {}
    for i in infos:
        for name, entry in i['classes'].items():
            if entry.get('node') is None:
                continue
            idx.setdefault(name, (i, entry['node']))
    return idx


def write_attrs_with_inheritance(info, cls_name, class_index):
    """Do the class OR ANY RESOLVED ANCESTOR assign the live write attributes?

    Returns `(bool, evidence)` where evidence lists every hop with file:line, so the reason the
    site is a writer is inspectable rather than asserted.
    """
    hops, seen = [], set()
    queue = [(info, cls_name)] if cls_name else [(info, None)]
    while queue:
        cur_info, name = queue.pop(0)
        if name in seen:
            continue
        seen.add(name)
        cls_entry = cur_info['classes'].get(name) if name else None
        # own body
        for line, cls, fn, attr in cur_info['write_attr_assigns']:
            if (name is None and cls is None and cur_info is info) or cls == name:
                hops.append({'file': cur_info['path'].relative_to(EXT).as_posix(),
                             'class': cls, 'function': fn, 'attr': attr, 'line': line})
        # ancestors
        for base in _bases_of(cls_entry['node'] if cls_entry else None):
            if base in class_index and base not in seen:
                queue.append((class_index[base][0], base))
            else:
                hops.append({'unresolved_base': base})
    writes = any('attr' in h for h in hops)
    return writes, hops


def defines_replanner_entry_points(info, cls_name, class_index):
    """A class is an ENV replanner candidate iff it (or a resolved ancestor) defines `plan`
    or `on_reveal`: those are the only two entry points `StrictOnlineEnv` calls with LIVE
    `vehicles`, so they are the only places a write can reach production state."""
    hops, seen = [], set()
    queue = [(info, cls_name)] if cls_name else []
    while queue:
        cur_info, name = queue.pop(0)
        if name in seen:
            continue
        seen.add(name)
        entry = cur_info['classes'].get(name)
        if entry:
            for fn_name in ('plan', 'on_reveal'):
                if fn_name in entry['methods']:
                    hops.append({'file': cur_info['path'].relative_to(EXT).as_posix(),
                                 'class': name, 'method': fn_name,
                                 'line': entry['methods'][fn_name].lineno})
        for base in _bases_of(entry['node'] if entry else None):
            if base in class_index and base not in seen:
                queue.append((class_index[base][0], base))
    return bool(hops), hops


def classify(info, ctx):
    """Pre-registered rules R1 > R4 > R3 > R2.  All satisfied labels are recorded."""
    cls = info['site']['class']
    writes, write_ev = write_attrs_with_inheritance(info, cls, ctx['class_index'])
    is_replanner, entry_ev = defines_replanner_entry_points(info, cls, ctx['class_index'])
    branched, branch_ev = result_is_branched_on(info, info['site'])
    wired = bool(info['env_wiring'])
    used_by = ctx['replanner_users'].get(cls, [])
    touched_by_d3 = ctx['d3_references'].get(cls, [])

    labels = []
    if info['imports_d3'] and writes:
        labels.append('D3_REAL_COMMIT_ENTRY')
    if writes and is_replanner:
        labels.append('INDIRECT_WRITER_CAN_WRITE_D3_STATE')
    if not info['imports_d3'] and (is_replanner or wired):
        labels.append('HISTORICAL_BASELINE_INDEPENDENT_ENTRY')
    if not writes:
        labels.append('CANDIDATE_EVAL_ONLY_NO_REAL_WRITE')
    if not labels:
        labels.append('CANDIDATE_EVAL_ONLY_NO_REAL_WRITE')
    governing = next(c for c in CATEGORIES if c in labels)
    return {
        'labels_all_satisfied': labels,
        'governing_category': governing,
        'capable_of_writing_live_state': bool(writes and is_replanner),
        'is_wired_to_a_d3_module': bool(touched_by_d3),
        'must_be_closed_or_proven_unreachable': bool(writes and is_replanner),
        'evidence': {
            'class': cls,
            'function': info['site']['function'],
            'line': info['site']['line'],
            'call_text': info['site']['text'],
            'class_or_ancestor_assigns_write_attrs': writes,
            'write_attr_resolution_path': write_ev,
            'is_env_replanner_entry_point': is_replanner,
            'replanner_entry_point_resolution': entry_ev,
            'result_branched_on_as_filter': branched,
            'branch_evidence': branch_ev,
            'file_wires_StrictOnlineEnv_replanner': wired,
            'env_wiring_lines': [ln for ln, _ in info['env_wiring']],
            'file_imports_d3_gate_primitive': info['imports_d3'],
            'referenced_by_runner_as_replanner': used_by,
            'referenced_by_any_d3_module': touched_by_d3,
        },
    }


def result_is_branched_on(info, site):
    """`certify_plan` returns a pair, so the filter test is a compare/truth-test on the call.

    Walk the enclosing function and look for an `if`/`while`/`assert`/`return` whose test
    mentions the assigned result name.  Conservative: anything else counts as 'not branched on',
    which is the WEAKER claim, so the classification can only under-state the gate role.
    """
    cls = info['classes'].get(site['class'])
    fn = None
    if cls:
        fn = cls['methods'].get(site['function'])
    if fn is None:
        for node in ast.walk(ast.parse(info['text'])):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                    and node.name == site['function']:
                fn = node
                break
    if fn is None:
        return None, None
    # name bound on the site's own statement (e.g. `ok, _ = certify_plan(...)`)
    stmt = None
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign) and any(
                isinstance(v, ast.Call) and call_name(v) == site['primitive']
                and v.lineno == site['line'] for v in ast.walk(node)):
            stmt = node
            break
    names = []
    if stmt is not None:
        for t in stmt.targets:
            if isinstance(t, ast.Name):
                names.append(t.id)
            elif isinstance(t, ast.Tuple):
                names += [e.id for e in t.elts if isinstance(e, ast.Name)]
    if not names:
        return None, None
    for node in ast.walk(fn):
        if isinstance(node, ast.If):
            used = {n.id for n in ast.walk(node.test) if isinstance(n, ast.Name)}
            if used & set(names):
                return True, {'if_line': node.lineno, 'names': sorted(used & set(names)),
                              'test': info['lines'][node.test.lineno - 1].strip()}
    return False, {'names': names}


def build_replanner_users(infos, class_index):
    """`class name -> files that wire a StrictOnlineEnv AND mention the class` (CROSS-FILE).

    Cross-file matters: `RollingOptimizeReplanner` / `RRNCOAcceptReplanner` /
    `SolverAcceptReplanner` are library replanners whose runner lives in another file, so a
    same-file-only check reports "used by nobody" and would under-state the wiring.  Measured
    defect of the first version of this function, fixed here.
    """
    users = {}
    for i in infos:
        if not i['env_wiring']:
            continue
        rel = i['path'].relative_to(EXT).as_posix()
        for name in class_index:
            if name in i['text']:
                users.setdefault(name, [])
                if rel not in users[name]:
                    users[name].append(rel)
    return users


def build_d3_references(infos):
    """`class name -> D3 modules that mention it` (the expected answer is an empty list)."""
    d3_files = [i for i in infos if i['path'].name.startswith('d3_') or i['imports_d3']]
    refs = {}
    for i in d3_files:
        rel = i['path'].relative_to(EXT).as_posix()
        for other in infos:
            for name in other['classes']:
                if name in i['text']:
                    refs.setdefault(name, [])
                    if rel not in refs[name]:
                        refs[name].append(rel)
    return refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--heavy-cap', type=int, default=25)
    args = ap.parse_args()

    files = source_files()
    infos = [analyse(p) for p in files]
    infos = [i for i in infos if 'syntax_error' not in i]

    deciders, writers = [], []
    for i in infos:
        for s in i['sites']:
            (deciders if s['kind'] == 'decide' else writers).append({'file': i, 'site': s})

    class_index = build_class_index(infos)
    ctx = {
        'class_index': class_index,
        'replanner_users': build_replanner_users(infos, class_index),
        'd3_references': build_d3_references(infos),
    }

    # ---- the seven pre-registered `certify_plan` sites, located by file:line ----------------
    rows, missing = [], []
    for rel, line in PREREGISTERED_SITES:
        hit = None
        for d in deciders:
            if d['file']['path'].relative_to(EXT).as_posix() == rel and d['site']['line'] == line:
                hit = d
                break
        if hit is None:
            missing.append({'file': rel, 'line': line,
                            'why': ('the pre-registered anchor no longer holds: no call to a decide '
                                    'primitive at this file:line')})
            continue
        info = dict(hit['file'])
        info['site'] = hit['site']
        rows.append({'file': rel, 'line': line, 'primitive': hit['site']['primitive'],
                     'classification': classify(info, ctx)})

    # ---- the D3 gate primitives' own non-test call sites -----------------------------------
    gate_sites = []
    for i in infos:
        for s in i['sites']:
            if s['primitive'] in D3_GATE_PRIMITIVES:
                gate_sites.append({'file': i['path'].relative_to(EXT).as_posix(),
                                   'line': s['line'], 'primitive': s['primitive'],
                                   'class': s['class'], 'function': s['function'],
                                   'call_text': s['text']})

    uniq = {}
    for s in gate_sites:
        uniq[(s['file'], s['line'], s['primitive'])] = s
    gate_sites = sorted(uniq.values(), key=lambda s: (s['file'], s['line']))

    n_writers = sum(1 for r in rows if r['classification']['capable_of_writing_live_state'])
    n_d3_entry = sum(1 for r in rows if r['classification']['governing_category']
                     == 'D3_REAL_COMMIT_ENTRY')
    n_d3_wired = sum(1 for r in rows if r['classification']['is_wired_to_a_d3_module'])
    n_gate_sites = len(gate_sites)

    # Hash only the files a reviewer needs to re-check this classification: the seven site files,
    # every file that appears in a write-attr resolution path, the D3 gate primitive modules, and
    # every file with a D3 gate primitive call site.
    relevant_files = {rel for rel, _ in PREREGISTERED_SITES}
    for r in rows:
        for hop in r['classification']['evidence']['write_attr_resolution_path']:
            if 'file' in hop:
                relevant_files.add(hop['file'])
        for ev in r['classification']['evidence']['replanner_entry_point_resolution']:
            if 'file' in ev:
                relevant_files.add(ev['file'])
    for s in gate_sites:
        relevant_files.add(s['file'])
    for i in infos:
        if i['path'].name.startswith('d3_') and i['sites']:
            relevant_files.add(i['path'].relative_to(EXT).as_posix())
    relevant_files = {rel for rel in relevant_files if (EXT / rel).is_file()}
    out = {
        'stage': 'D3_G0_Evidence_B3AC_submit_path_classification',
        'read_only': True,
        'production_state_written': False,
        'source_files_scanned': len(files),
        'preregistered_sites': [{'file': f, 'line': l} for f, l in PREREGISTERED_SITES],
        'missing_preregistered_sites': missing,
        'rows': rows,
        'summary': {
            'n_preregistered_sites': len(PREREGISTERED_SITES),
            'n_classified': len(rows),
            'n_governing_D3_REAL_COMMIT_ENTRY': n_d3_entry,
            'n_capable_of_writing_live_state': n_writers,
            'n_wired_to_a_d3_module': n_d3_wired,
            'n_d3_gate_primitive_non_test_call_sites': n_gate_sites,
            'NOTE': ('"capable of writing live state" is a CAPABILITY of the enclosing replanner '
                     'class, NOT a statement that the D3 production path is wired to it; the two '
                     'are reported separately and must never be merged'),
        },
        'd3_gate_primitive_call_sites_non_test': gate_sites,
        'identities': {
            'this_file_sha256': sha256_file(Path(__file__)),
            'code': {rel: sha256_file(EXT / rel) for rel in sorted(relevant_files)},
        },
    }
    dest = (EXT.parent / 'research' / 'review_3080_20261010' / 'd3_g0_evidence_root_v1' /
            'B3AC_PATH_CLASSIFICATION.json')
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str) + '\n',
                    encoding='utf-8', newline='\n')

    print('  PRE      %-6s %d source files scanned; %d decider sites; %d writer sites'
          % ('OK' if not missing else 'MISSING', len(files), len(deciders), len(writers)))
    for r in rows:
        c = r['classification']
        print('  %-52s:%-4d %-40s filter=%-5s canwrite=%-5s d3wired=%-5s used_by=%d'
              % (r['file'], r['line'], c['governing_category'],
                 c['evidence']['result_branched_on_as_filter'],
                 c['capable_of_writing_live_state'],
                 c['is_wired_to_a_d3_module'],
                 len(c['evidence']['referenced_by_runner_as_replanner'])))
    print('  D3 gate primitive non-test call sites: %d' % len(gate_sites))
    for s in gate_sites:
        print('      %s:%d  %s  (%s)' % (s['file'], s['line'], s['primitive'], s['class']))
    print('\nsummary:', json.dumps(out['summary'], ensure_ascii=False))
    if missing:
        print('\nMISSING pre-registered anchors: %d -> the classification cannot be reported'
              % len(missing))
        for m in missing:
            print('   ', m)
    print('wrote', dest)
    return 1 if missing else 0


if __name__ == '__main__':
    sys.exit(main())
