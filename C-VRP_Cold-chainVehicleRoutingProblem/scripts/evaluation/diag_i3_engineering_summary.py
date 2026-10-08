"""Independently audit engineering probes; never promote to online efficacy."""
import argparse
import json
import math
from pathlib import Path


def exit_ok(path):
    return {'RUN_EXIT=0', 'SEAL_EXIT=0', 'DONE=1'} <= set(path.read_text().splitlines())


def audit_teacher(folder):
    report = json.loads((folder / 'probe.json').read_text(encoding='utf-8'))
    rows, config = report['records'], report['config']
    problems = []
    if not exit_ok(folder / 'run.exit'):
        problems.append('run_or_end_seal_failed')
    if config['train_seed'] != 20260925 or config['orders'] != 200 or config['probe_days'] != 1:
        problems.append('not_fixed_training_probe')
    expected = [(0, i) for i in (24, 99, 174)]
    ids = [(r['day_id'], r['ordinal']) for r in rows]
    if sorted(ids) != expected or len(set(ids)) != 3:
        problems.append('missing_duplicate_or_unexpected_records')
    actual = {'records': len(rows), 'valid_teachers': 0, 'eligible': 0,
              'positive_delta': 0, 'negative_delta': 0}
    shadow_agree = 0
    for row in rows:
        a, r = row['branches']
        valid = all(b.get('action_feasible') and b.get('hard_feasible') and
                    b.get('utility') is not None and math.isfinite(b['utility'])
                    for b in (a, r))
        if row['teacher_valid'] != valid:
            problems.append('validity_mismatch:' + str(row['ordinal']))
        if valid:
            for branch in (a, r):
                reconstructed = branch['revenue'] - branch['fuel_cost'] - branch['reject_loss']
                if not math.isclose(reconstructed, branch['utility'], abs_tol=1e-8, rel_tol=0):
                    problems.append('utility_decomposition_mismatch:' + str(row['ordinal']))
            delta = a['utility'] - r['utility']
            if not math.isclose(delta, row['teacher_delta'], abs_tol=1e-8, rel_tol=0):
                problems.append('delta_mismatch:' + str(row['ordinal']))
            eligible = delta != 0 and abs(delta) >= config['minimum_gap']
            if row['decision_eligible'] != eligible:
                problems.append('eligibility_mismatch:' + str(row['ordinal']))
            if not math.isclose(row['loss_teacher_margin'], delta if eligible else 0,
                                abs_tol=1e-8, rel_tol=0):
                problems.append('loss_margin_mismatch:' + str(row['ordinal']))
            shadow = row.get('hist_shadow_margin')
            shadow_agree += int(shadow is not None and math.isfinite(shadow) and delta * shadow > 0)
            actual['valid_teachers'] += 1
            actual['eligible'] += int(eligible)
            actual['positive_delta'] += int(delta > 0)
            actual['negative_delta'] += int(delta < 0)
        elif (row['teacher_delta'] is not None or row['decision_eligible'] or
              row['loss_teacher_margin'] != 0):
            problems.append('invalid_teacher_not_retained_with_zero_weight:' + str(row['ordinal']))
        if not row['same_action_full_trajectory_parity']:
            problems.append('runner_parity_failure:' + str(row['ordinal']))
    if any(report['summary'].get(k) != v for k, v in actual.items()):
        problems.append('summary_mismatch')
    verify = (folder / 'verify_end.txt').read_text(encoding='utf-8').splitlines()
    if len(verify) != 14 or any(not line.endswith(': OK') for line in verify):
        problems.append('incomplete_source_seal')
    return {'engineering_qualified': not problems, 'problems': problems,
            'independently_recomputed': actual, 'elapsed_s': report['elapsed_s'],
            'conditional_hist_teacher_sign_agreement': [shadow_agree, actual['valid_teachers']],
            'online_gain_established': False,
            'limits': ['Only one training day and three fixed states',
                       'K=2/infinite continuation, not deployed K=10/10s',
                       'Parity is runner-measured; private snapshot archive not exported',
                       'Shadow/teacher agreement is a probe diagnostic, not method gain',
                       'Teacher labels do not show MaskCO beating any baseline']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--teacher-dir', required=True)
    parser.add_argument('--optimizer-dir', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    teacher = audit_teacher(Path(args.teacher_dir))
    optimizer = Path(args.optimizer_dir)
    log = (optimizer / 'run.log').read_text(encoding='utf-8')
    verify = (optimizer / 'verify_end.txt').read_text(encoding='utf-8').splitlines()
    optimizer_ok = (exit_ok(optimizer / 'run.exit') and
        'ALL PASS: actual frozen CVRP checkpoint branch' in log and bool(verify) and
        all(line.endswith(': OK') for line in verify))
    result = {'teacher_probe': teacher,
              'actual_pretrained_optimizer_engineering_qualified': optimizer_ok,
              'formal_I3_training_completed': False, 'online_gain_established': False}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))
    if not teacher['engineering_qualified'] or not optimizer_ok:
        raise SystemExit(1)
