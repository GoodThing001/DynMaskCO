"""Copy only new reviewed artifacts, retaining agent provenance and old files."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
WORKTREE = Path('C:/Users/mmc/.codex/worktrees/ba63/MASKCO-Main')


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('line', choices=['delivery', 'pickup', 'reproduction'])
    a = ap.parse_args()
    source = WORKTREE / '.arbor' / a.line / 'research/review_3080_20261010' / a.line
    target = ROOT / 'research/review_3080_20261010' / a.line
    target.mkdir(parents=True, exist_ok=False)
    rows = []
    for p in sorted(source.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts or p.suffix == '.pyc':
            continue
        if any(part.startswith('engineering_qa_registered') for part in p.parts):
            continue
        rel = p.relative_to(source)
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
        rows.append({'source': str(p), 'local_path': str(dest),
                     'sha256': hashlib.sha256(dest.read_bytes()).hexdigest()})
    with (target / 'INTEGRATION_PROVENANCE.json').open('x', encoding='utf-8') as f:
        json.dump({'kind': 'exact_copy_of_new_agent_review_only',
                   'not_an_independent_rerun': True, 'files': rows}, f, indent=2)
    print(json.dumps({'line': a.line, 'files': len(rows), 'old_files_overwritten': 0}))


if __name__ == '__main__':
    main()
