"""Independent read-only audit of the user's new handover; old bytes stay put."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / 'C-VRP_Cold-chainVehicleRoutingProblem'
P = ROOT / 'research/pickup_20round_audit'
NEW = EXT / 'results/research_20261010_continuation_v2'
OUT = ROOT / 'research/review_3080_20261010'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)


def main():
    audit = OUT / 'parent_v3'
    audit.mkdir(parents=True, exist_ok=False)
    selected = [NEW / 'HANDOVER_2026-10-10.md',
                NEW / 'HANDOVER_CONTINUATION_DECISIONS_2026-10-10.md',
                NEW / 'pickup/P_LIVE_DEV_PANEL_V1_REGISTRATION.json',
                NEW / 'pickup/P_LIVE_DEV_PANEL_V2_REGISTRATION.json',
                NEW / 'pickup/P_LIVE_DEV_PANEL_V2_RUN_REGISTRATION.json',
                NEW / 'pickup/P_LIVE_DEV_PANEL_V2_AMENDMENT.md',
                ROOT / 'tools/_srv_exec.py', ROOT / 'tools/_srv.py']
    selected += list(P.glob('*.py')) + list(P.glob('*.md'))
    selected += list((EXT / 'scripts/research_20261010').glob('delivery_*'))
    selected += list((EXT / 'scripts/research_claim_gate').glob('*.py'))
    frozen = []
    for p in selected:
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT)
        dest = audit / 'source_snapshot' / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        frozen.append({'path': rel.as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p)})
    inventory = []
    for base in [P, NEW, EXT / 'results/research_20261010']:
        for p in sorted(base.rglob('*')):
            if p.is_file() and p.suffix in ('.json', '.jsonl', '.npz', '.pkl', '.ckpt', '.md', '.py'):
                inventory.append({'path': p.relative_to(ROOT).as_posix(),
                                  'bytes': p.stat().st_size, 'sha256': sha(p)})
    save(audit / 'INPUT_MANIFEST.json', {'time_utc': datetime.now(timezone.utc).isoformat(),
                                       'snapshot': frozen, 'inventory': inventory})
    checks = []
    def check(name, ok, evidence=None):
        checks.append({'name': name, 'pass': bool(ok), 'evidence': evidence})
    registry = json.loads((NEW / 'pickup/P_LIVE_DEV_PANEL_V2_RUN_REGISTRATION.json').read_text())
    for name, item in registry['pins'].items():
        if 'sha256' not in item:
            continue
        pp = ROOT / item['path']
        if not pp.exists():
            pp = EXT / item['path']
        check('registered_pin:' + name, pp.exists() and sha(pp) == item['sha256'])
    check('original_I3_protocol', sha(EXT / 'configs/i3_20261008_v2/protocol.json') ==
          '4825392b56399719a9a0042d97986c4c683362af80d9925b35f74b65e4427f5c')
    check('original_I3_gate', sha(EXT / 'results/i3_online_recovery_1009_v1/online_20260926/gate.json').startswith('48306155'))
    local = json.loads((P / 'stage_a_iter_v1/result.json').read_text())
    remote = json.loads((P / 'stage_a_iter_3080_v1/result.json').read_text())
    # Actual archived clocks are not bit-identical across hosts. Match the
    # capture ordinal within each instance/mode, then explicitly check order
    # identity and clock tolerance. New runs must store event and vehicle IDs.
    key = lambda r: (r['inst'], r['clock'], r['mode'])
    for host, report in [('local', local), ('3080', remote)]:
        check(host + ':unique_record_keys', len(set(map(key, report['rows']))) == len(report['rows']))
        for row in report['rows']:
            best = row['baseline_utility']
            for rr in row['per_round']:
                if rr['utility'] is None:
                    check(host + ':retained_failure', False, {'key': key(row), 'round': rr['round']})
                    continue
                expected = bool(rr['hard'] and rr['utility'] > best + 1e-10)
                check(host + ':oracle_acceptance_arithmetic', rr['accepted'] == expected)
                if expected:
                    best = rr['utility']
                check(host + ':round_delta', abs(rr['delta_vs_baseline'] -
                    (rr['utility'] - row['baseline_utility'])) < 1e-9)
            check(host + ':best_oracle_utility', abs(best - row['best_utility']) < 1e-9)
    ordered = lambda d: sorted(d['rows'], key=lambda r: (r['inst'], r['mode'], r['clock']))
    check('cross_host_record_coverage', len(local['rows']) == len(remote['rows']))
    clock_differences = []
    for row, peer in zip(ordered(local), ordered(remote)):
        check('cross_host_capture_ordinal', (row['inst'], row['mode'], row['orders']) ==
              (peer['inst'], peer['mode'], peer['orders']))
        clock_differences.append(abs(row['clock'] - peer['clock']))
        check('cross_host_clock_bounded', clock_differences[-1] < 1e-5)
        check('cross_host_discrete', row['n_rounds_accepted'] == peer['n_rounds_accepted'] and
              row['decoder_calls_total'] == peer['decoder_calls_total'])
        check('cross_host_numeric', abs(row['best_delta_vs_baseline'] -
              peer['best_delta_vs_baseline']) < 1e-8)
    stats = {}
    for mode in ['maskco', 'recon_only', 'discard']:
        rows = [r for r in local['rows'] if r['mode'] == mode]
        vals = [r['best_utility'] - r['baseline_utility'] for r in rows]
        mean = sum(vals) / len(vals)
        check('reported_mean:' + mode, abs(mean - local['per_mode'][mode][
              'mean_best_delta_vs_doing_nothing']) < 1e-10)
        check('decode_count:' + mode, sum(r['decoder_calls_total'] for r in rows) == 21)
        stats[mode] = {'n': len(rows), 'mean_best_delta': mean,
                       'individual_nonzero_deltas': [x for x in vals if abs(x) > 1e-10],
                       'mean_state_relative_delta_pct': sum(100 * v / abs(r['baseline_utility'])
                           for r, v in zip(rows, vals)) / len(vals),
                       'failures': sum(rr['utility'] is None for r in rows for rr in r['per_round'])}
    summary = {'scope': 'Independent JSON arithmetic, source review and identity checks; not terminal trajectory replay',
               'checks': checks, 'n_checks': len(checks),
               'n_failed_checks': sum(not x['pass'] for x in checks),
               'stage_a_recomputed': stats,
               'cross_host_max_abs_clock_difference': max(clock_differences),
               'qualification': {'arithmetic_pass': all(x['pass'] for x in checks),
                    'causal_online_policy_qualified': False, 'complete_all_day_census_qualified': False,
                    'terminal_trace_independent_replay_available': False,
                    'stable_online_gain_established': False},
               'corrections': [
                   'Whole-day future utility is fed back into round acceptance. This is retrospective oracle selection, not an online actor.',
                   'min_clock=6 and early return skip early times and some vehicles. The reported size survey is not exhaustive.',
                   'Best includes do-nothing, so both arms worse than do-nothing cannot trigger on this summary.',
                   'The nonzero recon-only STATE delta is 0.02401654459549718; 0.003430934942213883 is the seven-state mean.',
                   'A state intervention effect is not a day-level deployable policy effect; compare no confirmatory day gate here.',
                   'cond_hist=uncond_hist says nothing logically about maskco-cond_hist identifiability.',
                   'Two measured baseline ordering points do not prove strong_rolling is uniformly stronger.',
                   'n>=12 observed positive oracle cases do not establish a necessary size threshold.',
                   'Immediate insertion cost minimum creates a tautological non-negative guide excess; it does not close long-run value.',
                   'Per-decision10s and per-day10s in the run registry conflict; a new prospective operating contract is necessary.'
               ]}
    save(audit / 'INDEPENDENT_USER_STAGE_A_AUDIT.json', summary)
    print(json.dumps({k: summary[k] for k in ['n_checks', 'n_failed_checks', 'stage_a_recomputed', 'qualification']}, indent=2))


if __name__ == '__main__':
    main()
