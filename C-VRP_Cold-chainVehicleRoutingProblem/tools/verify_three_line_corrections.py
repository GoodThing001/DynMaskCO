"""Read-only qualification check. Static PASS never authorizes scientific execution."""
from pathlib import Path
import argparse
import hashlib
import json
import importlib.util
import datetime

EXT = Path(__file__).resolve().parents[1]
AUDIT = EXT / 'docs/当前规划/文献工件/三线协议审计_2026-10-09'

def protected_check(root, expected):
    problems = []
    for relative, digest in expected.items():
        path = (root / relative).resolve()
        # Original audit explicitly pins this upstream checkpoint read-only.
        pinned_upstream = (root.parent / 'MASKCO_code/ckpts/cvrp100.ckpt').resolve()
        if not path.is_relative_to(root.resolve()) and path != pinned_upstream:
            problems.append({'path': relative, 'reason': 'outside_root'})
        elif not path.is_file():
            problems.append({'path': relative, 'reason': 'missing'})
        elif hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            problems.append({'path': relative, 'reason': 'hash_mismatch'})
    return problems

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='New local JSON only; existing files cannot be overwritten')
    args = parser.parse_args()
    baseline = json.loads((AUDIT / 'PROTECTED_BEFORE.json').read_text(encoding='utf-8-sig'))
    problems = protected_check(EXT, baseline)
    candidate = EXT / '03_负结果线/发布候选/2026-10-09'
    spec = importlib.util.spec_from_file_location('static_candidate_verifier', candidate / 'verify_static_artifacts.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    static = module.validate(candidate)
    result = {'checked_at': datetime.datetime.now().isoformat(),
              'protected_count': len(baseline), 'protected_problems': problems,
              'negative_static_checks_pass': static['all_checks_pass'],
              'static_qualification_pass': not problems and static['all_checks_pass'],
              'scientific_execution_authorized_by_this_check': False,
              'historical_full_reproduction_ready': False,
              'unresolved': ['original training source identity and full historical lineage',
                             'original online event trajectories',
                             'new scientific protocol and matched strong dynamic controls'],
              'own_physical_calibration_required_for_transparent_simulation': False,
              'check_scope': 'read-only static/protected check only; not a history of all authorized server engineering work',
              'scientific_experiments_run': False, 'remote_writes': False}
    if args.output:
        target = args.output.resolve()
        allowed = (EXT / 'results/three_line_corrections_20261009').resolve()
        if not target.is_relative_to(allowed):
            parser.error('Output must be below results/three_line_corrections_20261009')
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result['static_qualification_pass']:
        raise SystemExit(1)

if __name__ == '__main__':
    main()
