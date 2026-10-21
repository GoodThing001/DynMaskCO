"""Seal the verified dev04 validation reference; never replace an existing baseline."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import zipfile

EXT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXT/'scripts/static_profit'))
from profit_core import ProfitTask, verify_research_freeze


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    out = EXT/'results/static_delivery/algorithm_baselines/v0_dev04_validation'
    if args.verify:
        lock = read(out/'BASELINE_LOCK.json')
        require(sha(out/'BASELINE_LOCK.json') == (out/'BASELINE_LOCK.sha256').read_text().split()[0], 'Lock SHA')
        for name, expected in lock['files'].items():
            require(sha(out/name) == expected, name)
        print(json.dumps(dict(status='PASS_BASELINE_ARCHIVE_SHA', files=len(lock['files']),
                             scientific_freeze=verify_research_freeze())))
        return
    require(not out.exists(), 'Baseline exists; use --verify. Never overwrite baseline.')
    root = EXT/'results/static_delivery/algorithm_dev_20261021/remote/dev04'
    receipt = read(root/'download_verification.json')
    manifest = read(root/'output_manifest.json')
    require(receipt['exit_code'] == 0, 'Incomplete source run')
    for name in receipt['downloaded_verified']:
        require(sha(root/name) == manifest[name], 'Source SHA: '+name)
    methods = ('maskco_base', 'quality_only', 'directed_only', 'quality_directed', 'alns', 'vns')
    rows = []
    for case in ('c205', 'c206', 'c207', 'c208'):
        folder = root/'outputs/validation'/case
        summary = read(folder/'summary.json')
        task = ProfitTask(case, fold=0, role='validation')
        require(summary['task'] == task.identity(), 'Task identity')
        require(not summary['formal_experiment'] and summary['outer_test_cases_used'] == 0, 'Test contamination')
        initial_routes = []
        for method in methods:
            result = read(folder/(method+'_search.json'))
            value = task.evaluate(result['routes'], audit=True)
            for key in ('net_benefit_U', 'quality_loss_value_U', 'distance_km', 'refrigeration_cost_U', 'minimum_quality', 'used_vehicles'):
                require(abs(value[key]-result['final'][key]) < 1e-6, 'Re-score: '+key)
            initial_routes.append(result['initial']['net_benefit_U'])
            rows.append(dict(case=case, method=method, **value,
                             elapsed_s=result['elapsed_s'], iterations=result['iterations']))
        require(max(initial_routes)-min(initial_routes) < 1e-6, 'Unequal initialization')
    report = dict(baseline_id='v0_dev04_validation', status='SEALED_DEVELOPMENT_REFERENCE',
        scientific_freeze=verify_research_freeze(), source_run='dev04', fold=0, seed=1000,
        training_steps=240, validation_cases=['c205', 'c206', 'c207', 'c208'],
        formal_experiment=False, outer_test_cases_used=0, rows=rows,
        mean_net_benefit_U={m: statistics.mean(r['net_benefit_U'] for r in rows if r['method']==m) for m in methods},
        method_identity={'maskco_base':'Frozen-profit task adaptation of original backbone; newly trained',
                         'alns':'Pinned official ALNS framework with project profit operators',
                         'vns':'Project implementation, not a claimed exact paper reproduction'},
        limitations=['One development seed and one validation geometry; no significance/generalization claim.',
                     'Nominal 10s search; recorded elapsed includes closing-audit overshoot.',
                     'Neural eager inference and 4-CPU host allocation; formal single-CPU/60s protocol pending.',
                     'NPZ weights remain on the authorized server; SHA and checkpoint paths are archived.'],
        iteration_rule='Keep v0 unchanged. Compare future runs on identical cases/objective/conditions; report recipe, hardware and timing differences. Use new validation versions, no outer test tuning.')
    out.mkdir(parents=True)
    (out/'comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    # Preserve actual executed sources, routes, configuration, checkpoint identities and logs.
    with zipfile.ZipFile(out/'evidence.zip', 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob('*')):
            if path.is_file() and path.suffix in ('.json', '.py', '.sh', '.log'):
                archive.write(path, path.relative_to(root).as_posix())
        pins = EXT/'comparisons/static_coldchain/SOURCE_PINS.json'
        archive.write(pins, 'comparator_source_pins.json')
    lock = dict(baseline_id=report['baseline_id'], files={p.name:sha(p) for p in out.iterdir() if p.is_file()})
    (out/'BASELINE_LOCK.json').write_text(json.dumps(lock, indent=2), encoding='utf-8')
    (out/'BASELINE_LOCK.sha256').write_text(sha(out/'BASELINE_LOCK.json')+'  BASELINE_LOCK.json\n', encoding='ascii')
    print(json.dumps(dict(status=report['status'], rows=len(rows), mean_net_benefit_U=report['mean_net_benefit_U'], output=str(out))))


if __name__ == '__main__':
    main()
