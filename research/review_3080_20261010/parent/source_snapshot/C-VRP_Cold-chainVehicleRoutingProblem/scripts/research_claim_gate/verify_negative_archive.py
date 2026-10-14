"""Read-only R1/C1 archive -> current paper check. No model evaluation or input writes.

Pins are a present-day inventory, not evidence of historical preregistration.
Numerical reconstruction below is independent of the historical aggregation code.
"""
import argparse
import json
from pathlib import Path
import re
import numpy as np
from evidence_gate import EvidenceError, require, finite, integer, load_json, file_hash

PROGRAM_SHA = '8edbc3cd5cbe4563992112bf55acadef0be4242b12d2e8139684c3f146cdbbd8'
PAPER = '03_负结果线/文档/论文与投稿/论文工作稿_收口修订_2026-10-09.md'
ARMS = ('R', 'Mpre', 'Mtrained')
PAIRS = ('Mtrained_minus_Mpre', 'Mtrained_minus_R', 'Mpre_minus_R')
BASE = 'negative_result_paper/results/'
QUALIFICATION = 'results/three_line_corrections_20261009/FINAL_PROTECTED_ACCEPTANCE_20261010_v2.json'
RECOVERY = '03_负结果线/恢复证据/B_REPRO_CLEAN_20261009/RECOVERY_MANIFEST.json'
FIXED = {k: BASE + '09_unified_v1/' + k + '.analysis.json'
         for k in ('s42_train', 's42_cal', 's43_train', 's43_cal', 's44_train', 's44_cal', 'heldout')}
FIXED.update({'c1_s' + s: BASE + '10_confirm_c1/c1_s' + s + '.analysis.json' for s in ('42', '43', '44')})
LOOP = {'r1_s' + s: BASE + '09_unified_v1/step5_heldout_s' + s + '.summary.json' for s in ('42', '43')}
LOOP.update({'c1_s' + s: BASE + '10_confirm_c1/c1_step5_s' + s + '.summary.json' for s in ('42', '43')})
RAW = {k: 'results/m0_scale/_bverify_pinned/fsq_' + k + '_v1/per_state.json' for k in FIXED}
RAW.update({'s42_train': 'results/m0_scale/fsq_clean_train/per_state.json',
            's42_cal': 'results/m0_scale/fsq_clean_cal/per_state.json',
            'heldout': 'results/m0_scale/_bverify_pinned/fsq_heldout_v1/per_state.json'})
RAW.update({'c1_s' + s: 'results/m0_scale/_bverify_pinned/confirm_c1/fsq_s' + s + '_c1_v1/per_state.json' for s in ('42', '43', '44')})


def inside(root, relative):
    require(isinstance(relative, str), 'path must be a string')
    p = Path(relative)
    require(not p.is_absolute() and '..' not in p.parts, 'unsafe input path')
    result = (root / p).resolve()
    require(result.is_relative_to(root.resolve()), 'input path escaped project')
    return result


def close(actual, expected, name, tol=1e-10):
    require(abs(finite(actual, name) - finite(expected, name)) <= tol, name + ': numeric mismatch')


def interval(value, name):
    require(isinstance(value, list) and len(value) == 2, name + ': interval requires two endpoints')
    lo, hi = (finite(x, name) for x in value)
    require(lo <= hi, name + ': reversed interval')
    return lo, hi


def bootstrap(values, reps, seed):
    values = np.asarray(values, dtype=float)
    require(values.size > 0 and np.isfinite(values).all(), 'invalid bootstrap values')
    rng = np.random.default_rng(seed)
    # integers and choice replace=True have the same default generator draw law.
    indices = rng.integers(0, values.size, size=(reps, values.size))
    return [float(v) for v in np.percentile(values[indices].mean(axis=1), [2.5, 97.5])]


def check_fixed(summary, rows=None):
    require(type(summary['n_states']) is int and summary['n_states'] == 32, 'fixed state count is not 32')
    require(type(summary['n_instances']) is int and summary['n_instances'] == 16, 'fixed instance count is not 16')
    for arm in ARMS:
        close(summary['improvement_vs_J0'][arm], summary['J0_mean'] - summary['accepted_mean'][arm], 'improvement ' + arm)
        for group in ('acceptance_rate', 'fail_rate', 'cert_fail_rate'):
            count, rate = summary[group][arm]
            integer(count, group)
            require(count <= 32, 'count exceeds states')
            close(rate, count / 32, group + arm)
    require(sum(integer(summary['Mtrained_vs_R_accepted'][k], k) for k in ('win', 'tie', 'loss')) == 32,
            'win/tie/loss coverage')
    for stage in ('candidate', 'accepted'):
        for pair in PAIRS:
            a, b = pair.split('_minus_')
            evidence = summary[stage + '_paired'][pair]
            close(evidence['mean'], summary[stage + '_mean'][a] - summary[stage + '_mean'][b], stage + pair)
            interval([evidence['ci_lo'], evidence['ci_hi']], stage + pair)
    if rows is None:
        return {'raw_state_recomputation': 'unknown_missing_raw', 'trajectory_replay': 'not_established'}
    require(isinstance(rows, list) and len(rows) == 32, 'raw fixed rows coverage')
    keys = [(integer(r['inst'], 'inst'), integer(r['event'], 'event')) for r in rows]
    require(len(set(keys)) == 32 and set(k[0] for k in keys) == set(range(16)), 'duplicate/changed fixed split')
    require(all(sum(k[0] == i for k in keys) == 2 for i in range(16)), 'two events per fixed instance required')
    close(summary['J0_mean'], float(np.mean([finite(r['J0'], 'J0') for r in rows])), 'J0 mean')
    values = {'candidate': {}, 'accepted': {}}
    for arm in ARMS:
        fails = cert_fails = accepts = 0
        for r in rows:
            det = r[arm]['det']
            require(type(det['fail']) is bool and type(det['cert_ok']) is bool, 'literal fail/cert flags required')
            j, j0 = finite(det['J'], arm + ' J'), finite(r['J0'], 'J0')
            fails += det['fail']; cert_fails += not det['cert_ok']
            accepts += (not det['fail'] and det['cert_ok'] and j < j0 - 1e-9)
        for group, count in (('fail_rate', fails), ('cert_fail_rate', cert_fails), ('acceptance_rate', accepts)):
            require(summary[group][arm][0] == count, group + ': raw count mismatch')
        for stage in values:
            per = {}
            for r in rows:
                det, j0 = r[arm]['det'], r['J0']
                val = j0 if det['fail'] else det['J']
                if stage == 'accepted':
                    val = j0 if det['fail'] or not det['cert_ok'] else min(j0, det['J'])
                per.setdefault(r['inst'], []).append(val)
            # Original grouping uses first encountered instance order, retained here.
            values[stage][arm] = np.array([np.mean(v) for v in per.values()])
            close(summary[stage + '_mean'][arm], float(values[stage][arm].mean()), stage + arm + ' raw mean')
    for stage in values:
        for pair in PAIRS:
            a, b = pair.split('_minus_')
            diff = values[stage][a] - values[stage][b]
            evidence = summary[stage + '_paired'][pair]
            close(evidence['mean'], float(diff.mean()), stage + pair + ' raw paired mean')
            ci = bootstrap(diff, 2000, 0)
            close(evidence['ci_lo'], ci[0], stage + pair + ' CI lo')
            close(evidence['ci_hi'], ci[1], stage + pair + ' CI hi')
    return {'raw_state_recomputation': 'passed_32_states_16_clusters_2000_bootstrap', 'trajectory_replay': 'not_established'}


def check_loop(summary):
    require(type(summary['n']) is int and summary['n'] == 16, 'loop count is not 16')
    close(summary['budget'], 4.0, 'per replanning call budget')
    rows = summary['rows']
    require(isinstance(rows, list) and len(rows) == 16, 'loop rows count')
    require({integer(r['inst'], 'inst') for r in rows} == set(range(16)), 'duplicate/changed loop split')
    for arm in ('B',) + ARMS:
        require(summary[arm + '_svc'] is True, 'literal complete service required')
        for r in rows:
            require(r[arm + '_svc'] is True, 'literal per-instance complete service required')
            close(r[arm + '_delta'], finite(r[arm + '_J'], 'J') - finite(r['B_J'], 'B J'), arm + ' delta')
            for component in ('D', 'Q', 'E'):
                require(finite(r[arm + '_' + component], component) >= 0, 'negative objective component')
        deltas = [r[arm + '_delta'] for r in rows]
        close(summary[arm + '_mean_delta'], float(np.mean(deltas)), arm + ' mean delta')
        for actual, expected in zip(interval(summary[arm + '_ci'], arm), bootstrap(deltas, 10000, 42)):
            close(actual, expected, arm + ' CI')
    # Historical labels indicate right-minus-left, explicitly normalize the direction.
    for label, a, b in (('R_vs_Mpre', 'Mpre', 'R'), ('R_vs_Mtrained', 'Mtrained', 'R'), ('Mpre_vs_Mtrained', 'Mtrained', 'Mpre')):
        vals = []
        for r in rows:
            close(r[label], r[a + '_J'] - r[b + '_J'], label + ' row direction')
            vals.append(r[label])
        close(summary[label + '_mean'], float(np.mean(vals)), label + ' mean')
        for actual, expected in zip(interval(summary[label + '_ci'], label), bootstrap(vals, 10000, 42)):
            close(actual, expected, label + ' CI')
    for field in ('attempts', 'improve', 'elapsed'):
        vals = [r['Mtrained_' + field] for r in rows]
        for value in vals:
            require(finite(value, field) >= 0, 'negative runtime/count')
            if field != 'elapsed': integer(value, field)
        close(summary['Mtrained_' + field + '_total'], sum(vals), field + ' total')
    require(all(r['Mtrained_improve'] <= r['Mtrained_attempts'] for r in rows), 'improves exceed attempts')
    return {'summary_row_recomputation': 'passed_16_instances_10000_bootstrap',
            'hard_trajectory_replay': 'unknown_missing_original_event_json',
            'budget_execution_proof': 'unknown_summary_only'}


def c1_joint(fixed, loops):
    pairs = {s: fixed['c1_s' + s]['accepted_paired'] for s in ('42', '43', '44')}
    a = all(pairs[s]['Mtrained_minus_R']['ci_hi'] >= 0 for s in pairs)
    b = all(pairs[s]['Mtrained_minus_Mpre']['ci_hi'] < 0 for s in ('42', '43'))
    b = b and pairs['44']['Mtrained_minus_Mpre']['ci_lo'] <= 0 <= pairs['44']['Mtrained_minus_Mpre']['ci_hi']
    c = all(loops['c1_s' + s]['R_vs_Mtrained_ci'][0] <= 0 <= loops['c1_s' + s]['R_vs_Mtrained_ci'][1]
            and loops['c1_s' + s]['Mpre_vs_Mtrained_ci'][1] < 0
            and all(loops['c1_s' + s][arm + '_svc'] is True for arm in ('B',) + ARMS) for s in ('42', '43'))
    return {'a': bool(a), 'b': bool(b), 'c': bool(c), 'joint_pass': bool(a and b and c),
            'rule_scope': 'historical_registered_case_not_new_program_confirmation'}


def table_rows(paper, title):
    starts = [i for i, line in enumerate(paper.splitlines()) if line.startswith('**表 ' + title + ' ·')]
    require(len(starts) == 1, 'missing/duplicate paper table ' + title)
    found = []
    for line in paper.splitlines()[starts[0] + 1:]:
        if line.startswith('|'):
            found.append([cell.strip() for cell in line.strip().strip('|').split('|')])
        elif found:
            break
    require(len(found) >= 3, 'missing paper table rows')
    return found[2:]


def paper_cell(cell, expected, name):
    text = cell.replace('**', '').replace('−', '-').replace('–', '-')
    nums = re.findall(r'[+-]?\d+\.\d+', text)
    require(len(nums) in (1, 3), name + ': numeric cell format')
    require(re.fullmatch(r'\s*[+-]?\d+\.\d+(?:\s*\[\s*[+-]?\d+\.\d+\s*,\s*[+-]?\d+\.\d+\s*\])?\s*', text) is not None, name + ': extra cell text')
    require(len(nums) == len(expected), name + ': missing/extra interval')
    for actual, target in zip(nums, expected):
        # Decimal display is four-place reporting; mismatch outside rounding tolerance fails.
        require(len(actual.split('.')[1]) == 4, name + ': four decimals required')
        close(float(actual), target, name, 0.00005000001)


def check_paper(paper, fixed, loops, joint):
    required = ('C1 联合确认未通过', 'R1 协议案例', 'CI 跨零', '不能因此称完整复现',
                '原闭环逐事件 JSON', '原训练身份仍缺', '不是双侧等效性检验',
                '未将共享状态的多项探索性比较当作独立确认', '结论只针对声明的仿真协议')
    for phrase in required: require(phrase in paper, 'missing cautious claim: ' + phrase)
    for assertion in ('已经建立稳定正向增益', '已建立稳定正向增益', 'C1联合确认已通过', '从零完整复现已通过'):
        require(assertion not in paper.replace(' ', ''), 'unsupported affirmative claim: ' + assertion)
    require(joint['joint_pass'] is False, 'historical C1 outcome changed; manual scholarly review required')
    count = 0
    for table, names in (('1', list(FIXED)[:7]), ('4', ['c1_s42', 'c1_s43', 'c1_s44'])):
        rows = table_rows(paper, table)
        require(len(rows) == len(names), 'paper fixed table coverage')
        for row, name in zip(rows, names):
            expected_seed = '留出(42)' if name == 'heldout' else name[-2:] if name.startswith('c1') else name[1:3]
            require(row[0].replace(' ', '') == expected_seed, 'paper seed/split order changed')
            offset = 2 if table == '1' else 1
            if table == '1':
                split = '留出' if name == 'heldout' else name.split('_')[1]
                require(row[1] == split, 'paper split mismatch')
            require(len(row) == offset + 4, 'paper fixed columns')
            for pos, stage, pair in ((0, 'candidate', 'Mtrained_minus_R'), (1, 'accepted', 'Mtrained_minus_R'),
                                     (2, 'candidate', 'Mtrained_minus_Mpre'), (3, 'accepted', 'Mtrained_minus_Mpre')):
                ev = fixed[name][stage + '_paired'][pair]
                expected = [ev['mean']]
                if pos == 1 or (table == '4' and pos == 3 and name != 'c1_s44'): expected += [ev['ci_lo'], ev['ci_hi']]
                paper_cell(row[offset + pos], expected, table + name + pair); count += 1
    for table, prefix in (('2', 'r1'), ('5', 'c1')):
        rows = table_rows(paper, table)
        require(len(rows) == 2, 'paper loop table coverage')
        labels = ['Mtrained_mean_delta', 'R_mean_delta', 'R_vs_Mtrained_mean', 'Mpre_vs_Mtrained_mean'] if table == '2' else ['R_vs_Mtrained_mean', 'Mpre_vs_Mtrained_mean']
        for row, seed in zip(rows, ('42', '43')):
            require(row[0] == seed and len(row) == len(labels) + 1, 'paper loop seed/columns mismatch')
            summary = loops[prefix + '_s' + seed]
            for cell, label in zip(row[1:], labels):
                ci_key = label.replace('_mean_delta', '_ci').replace('_mean', '_ci')
                paper_cell(cell, [summary[label]] + summary[ci_key], table + seed + label); count += 1
    require(count == 52, 'paper numeric cell coverage')
    return {'numeric_cells_checked': count, 'cautious_claim_checks': len(required), 'publication_or_full_reproduction_qualification': False}


def audit(project, manifest_path, paper_path=None):
    project = Path(project).resolve()
    manifest = load_json(Path(manifest_path))
    require(manifest['schema'] == 'negative-archive-current-inventory-v1', 'manifest schema')
    require(manifest['program_sha256'] == PROGRAM_SHA, 'program anchor changed')
    require(file_hash(inside(project, 'results/research_20261010/PROGRAM.json')) == PROGRAM_SHA, 'actual PROGRAM mismatch')
    inventory = manifest['archive_inventory']
    require(len(inventory) == 183, 'original package inventory not 183')
    actual = {p.relative_to(project).as_posix(): file_hash(p) for p in (project / 'negative_result_paper').rglob('*') if p.is_file()}
    require(actual == inventory, 'immutable original package inventory mismatch')
    qualification = {'status': 'unknown_missing_qualification_artifact', 'full_reproduction': False}
    qpath = inside(project, QUALIFICATION)
    if qpath.exists():
        require(manifest.get('qualification_sha256') == file_hash(qpath), 'qualification artifact identity mismatch')
        proof = load_json(qpath)
        require(proof['negative_static_checks_pass'] is True and proof['static_qualification_pass'] is True,
                'literal static qualification required')
        require(proof['historical_full_reproduction_ready'] is False and
                proof['scientific_execution_authorized_by_this_check'] is False, 'static proof promoted to scientific qualification')
        require(proof['protected_problems'] == [], 'reported protected artifact problems')
        qualification = {'status': 'passed_current_static_artifact_only', 'full_reproduction': False,
                         'sha256': file_hash(qpath), 'unresolved': proof['unresolved']}
    recovery = {'status': 'unknown_missing_recovery_manifest', 'original_training_identity_restored': False}
    rpath = inside(project, RECOVERY)
    if rpath.exists():
        require(manifest.get('recovery_manifest_sha256') == file_hash(rpath), 'recovery manifest identity mismatch')
        entries = load_json(rpath)['files']
        require(isinstance(entries, list) and len(entries) == 315, 'recovery inventory coverage')
        seen = set()
        for entry in entries:
            relative = entry['relative']
            require(relative not in seen, 'duplicate recovery path'); seen.add(relative)
            recovered = inside(rpath.parent, relative)
            require(recovered.is_file() and file_hash(recovered) == entry['sha256'], 'recovered dependency SHA mismatch')
            require(type(entry['bytes']) is int and recovered.stat().st_size == entry['bytes'], 'recovered dependency size mismatch')
        recovery = {'status': 'passed_315_actual_recovered_file_identities', 'sha256': file_hash(rpath),
                    'original_training_identity_restored': False}
    fixed, loops, findings = {}, {}, {}
    for key, rel in FIXED.items():
        fixed[key] = load_json(inside(project, rel))
        mirror = rel.replace('/results/', '/release/results/', 1)
        require(file_hash(inside(project, rel)) == file_hash(inside(project, mirror)), 'formal release mirror mismatch')
        rawpath = inside(project, RAW[key])
        if rawpath.exists():
            require(manifest['raw_state_sha256'].get(RAW[key]) == file_hash(rawpath), 'raw state identity mismatch')
            raw = load_json(rawpath)
        else: raw = None
        findings[key] = check_fixed(fixed[key], raw)
    for key, rel in LOOP.items():
        loops[key] = load_json(inside(project, rel))
        require(file_hash(inside(project, rel)) == file_hash(inside(project, rel.replace('/results/', '/release/results/', 1))), 'loop release mirror mismatch')
        findings['loop_' + key] = check_loop(loops[key])
    joint = c1_joint(fixed, loops)
    paper = Path(paper_path) if paper_path else inside(project, PAPER)
    claims = check_paper(paper.read_text(encoding='utf-8'), fixed, loops, joint)
    return {'schema': 'negative-archive-paper-audit-v1', 'engineering_consistency_pass': True,
            'scientific_positive_gain': False, 'stable_maskco_core_online_gain': False,
            'full_from_zero_reproduction': False, 'equivalence_established': False,
            'program_sha256': PROGRAM_SHA, 'manifest_sha256': file_hash(Path(manifest_path)),
            'paper_sha256': file_hash(paper), 'original_package_files_unchanged': len(actual),
            'formal_json_count': len(FIXED) + len(LOOP), 'c1_registered_joint_adjudication': joint,
            'current_static_qualification': qualification, 'recovery_inventory': recovery,
            'paper_checks': claims, 'recomputation': findings,
            'limitations': ['Inventory was recorded now; it is not proof of historical source or preregistration time.',
                           'Original training identity, s43/s44 training lineage, and closed-loop event trajectories are unresolved.',
                           'Recomputed fixed summaries and per-instance CIs do not certify full online execution, hard trajectories, or equal budgets.',
                           'CI crossing zero is not equivalence. Historical R1/C1 outcomes are not new PROGRAM confirmation.',
                           'Current paper check is finite required-phrase/numeric-table checking, not complete semantic peer review.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True)
    parser.add_argument('--manifest', default=str(Path(__file__).with_name('NEGATIVE_ARCHIVE_CURRENT_INVENTORY.json')))
    parser.add_argument('--paper')
    parser.add_argument('--output')
    args = parser.parse_args()
    try: result = audit(args.project, args.manifest, args.paper)
    except (EvidenceError, KeyError, TypeError, ValueError, OSError) as exc:
        result = {'engineering_consistency_pass': False, 'full_from_zero_reproduction': False,
                  'stable_maskco_core_online_gain': False, 'error': str(exc)}
    rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        target = Path(args.output).resolve()
        require(not target.is_relative_to(Path(args.project).resolve()), 'output must be outside authoritative input project')
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('x', encoding='utf-8') as stream:
            stream.write(rendered + '\n')
    print(rendered)
    return 0 if result['engineering_consistency_pass'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
