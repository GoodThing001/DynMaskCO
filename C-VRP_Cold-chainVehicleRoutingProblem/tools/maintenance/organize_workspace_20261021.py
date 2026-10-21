"""One-time, reversible workspace classification with provenance checks.

No experiment is run, no historical report is rewritten, and no linked or
private directory is traversed. Run with python -B from any directory.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import zipfile

SCRIPT_DIR = next(p / 'scripts' for p in Path(__file__).resolve().parents
                  if (p / 'scripts/project_paths.py').is_file())
sys.path.insert(0, str(SCRIPT_DIR))
from project_paths import EXTENSION_ROOT as EXT, WORKSPACE_ROOT as ROOT

PREFIX = EXT.relative_to(ROOT).as_posix() + '/'
REPORT = EXT / 'results/workspace_organization_20261021'
BACKUP = EXT / 'archive/workspace_organization_20261021/originals.zip'
PLAN = REPORT / 'migration_manifest.json'
LOCK = EXT / 'results/static_delivery/publication_protocol_v1_scientific_freeze_20261021/FREEZE_LOCK.json'
ACQUIRE = [
    'acquire_dataset_product_review_20261020.py',
    'acquire_direct_value_evidence_20261020.py',
    'acquire_parameter_origins_20261020.py',
    'acquire_publication_model_evidence_20261021.py',
    'acquire_static_evidence_20261020.py',
    'acquire_static_evidence_fallback_20261020.py',
    'acquire_static_supplement_20261020.py',
    'acquire_static_vehicle_20261020.py',
]
ANALYZE = ['build_dataset_product_review_20261020.py',
           'build_static_research_delivery_20261020.py',
           'summarize_net_benefit_scenario_20261020.py']
VERIFY = ['verify_direct_value_evidence_20261020.py',
          'verify_parameter_origins_20261020.py',
          'verify_publication_model_evidence_20261021.py',
          'verify_static_research_20261020.py']
MOVES = [(PREFIX + 'tools/' + name, PREFIX + 'tools/evidence/' + group + '/' + name)
         for group, names in [('acquire', ACQUIRE), ('analyze', ANALYZE), ('verify', VERIFY)]
         for name in names]
MOVES += [(PREFIX + 'results/static_delivery/' + name,
           PREFIX + 'results/static_delivery/history/quality_evidence_20261020/' + name)
          for name in ['quality_evidence_demo_20261020.json', 'quality_evidence_demo_v2_20261020.json']]
EDIT_DOCS = ['README.md', '工作区导航.md', 'CLAUDE.md'] + [PREFIX + name for name in [
    'README.md', '00_项目导航.md', 'CLAUDE.md', 'data/README.md',
    'results/static_delivery/README.md', 'docs/文档索引与版本状态.md',
    'docs/static_delivery_research_20261021/README.md',
    'data/published_cases_20261020/QDVRP_2021/README.md',
]]


def guarded(path):
    """Check the absolute target and every parent before any filesystem change."""
    path = Path(os.path.abspath(path))
    if not path.is_relative_to(ROOT) or path == ROOT:
        raise ValueError('Target outside workspace')
    rel = path.relative_to(ROOT)
    if rel.parts[0] in {'MASKCO_code', 'CC_Compare', 'research', 'results', 'tools'}:
        raise ValueError('Preserved workspace range')
    if any(p.lower() in {'private', '.git', '.vscode', '.codex', '.cursor', '.claude'} for p in rel.parts):
        raise ValueError('Private or configuration path')
    for item in [path, *path.parents]:
        if item.exists() or item.is_symlink():
            if item.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                raise ValueError('Reparse point is protected')
        if item == ROOT:
            break
    return path


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(4 * 1024 * 1024):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    guarded(path).parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def preservation_checks():
    expected = LOCK.with_suffix('.sha256').read_text(encoding='ascii').split()[0]
    assert sha(LOCK) == expected, 'Freeze lock SHA changed'
    freeze = json.loads(LOCK.read_text(encoding='utf-8'))
    for row in freeze['files']:
        assert sha(EXT / row['path']) == row['sha256'], row['path']
    archives = []
    for name in ['MANIFEST.json', 'LEGACY_CONSOLIDATION.json']:
        source = EXT / 'archive/transition_20261020' / name
        data = json.loads(source.read_text(encoding='utf-8'))
        path = source.parent / data['archive']
        got = sha(path)
        assert got == data['archive_sha256'], path.name
        archives.append(dict(path=path.relative_to(ROOT).as_posix(), sha256=got, bytes=path.stat().st_size))
    rrp = EXT / 'data/public_sources_20261020/refrigerated_routing_rrp'
    raw_rrp = json.loads((rrp / 'download_manifest.json').read_text(encoding='utf-8'))
    assert len(raw_rrp) == 61
    for row in raw_rrp:
        assert sha(rrp / row['file']) == row['sha256']
    solomon = EXT / 'data/classic_static_sources_20261020/solomon_100'
    raw_solomon = json.loads((solomon / 'FILES_SHA256.json').read_text(encoding='utf-8'))
    assert len(raw_solomon) == 56
    for row in raw_solomon:
        assert sha(solomon / 'instances' / row['path']) == row['sha256']
    return dict(scientific_freeze_files=len(freeze['files']), lock_sha256=expected,
                original_rrp_files=61, original_solomon_files=56,
                author_originals_in_scientific_freeze=56, archives=archives)


def backup_check(plan):
    assert sha(BACKUP) == plan['backup_sha256']
    with zipfile.ZipFile(BACKUP) as bundle:
        for row in plan['originals']:
            assert hashlib.sha256(bundle.read(row['path'])).hexdigest() == row['sha256']


def prepare():
    if PLAN.exists() or BACKUP.exists():
        raise ValueError('One-time snapshot already exists')
    preserved = preservation_checks()
    originals = []
    for rel in sorted(set(EDIT_DOCS + [a for a, b in MOVES])):
        path = guarded(ROOT / rel)
        originals.append(dict(path=rel, sha256=sha(path), bytes=path.stat().st_size))
    baseline = []
    for folder in ['scripts/static_profit', 'scripts/static_delivery']:
        for path in sorted((EXT / folder).glob('*')):
            if path.is_file():
                baseline.append(dict(path=path.relative_to(ROOT).as_posix(), sha256=sha(path)))
    protected = json.loads((EXT / 'archive/transition_20261020/LEGACY_CONSOLIDATION.json').read_text(encoding='utf-8'))['protected']
    protected_metadata = []
    for row in protected:
        path = ROOT / row['path']
        info = path.lstat()
        protected_metadata.append(dict(path=row['path'], bytes=info.st_size, mtime_ns=info.st_mtime_ns,
                                       attributes=info.st_file_attributes))
    guarded(BACKUP).parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BACKUP, 'x', zipfile.ZIP_DEFLATED) as bundle:
        for row in originals:
            bundle.write(ROOT / row['path'], row['path'])
    plan = dict(schema='workspace-organization-v1', date='2026-10-21', status='PREPARED',
                scope='Local directory classification only; no experiment, commit, push, or credential access',
                backup=BACKUP.relative_to(ROOT).as_posix(), backup_sha256=sha(BACKUP),
                originals=originals, moves=[dict(source=a, destination=b) for a, b in MOVES],
                untouched_algorithm_files=baseline, protected_metadata=protected_metadata,
                preservation_before=preserved,
                git_status_before=subprocess.check_output(['git', '-c', 'core.quotepath=false', 'status', '--short'], cwd=ROOT).decode('utf-8'),
                removed_empty_directories=[])
    backup_check(plan)
    save(PLAN, plan)
    print(json.dumps(dict(status='PREPARED', moves=len(MOVES), backed_up=len(originals))))


def apply():
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    assert plan['status'] == 'PREPARED'
    preservation_checks()
    backup_check(plan)
    for row in plan['originals']:
        assert sha(guarded(ROOT / row['path'])) == row['sha256'], row['path']
    for row in plan['moves']:
        assert not guarded(ROOT / row['destination']).exists()
    # Preflight all destinations before the first move. Rename stays on this volume.
    for row in plan['moves']:
        source = guarded(ROOT / row['source'])
        dest = guarded(ROOT / row['destination'])
        dest.parent.mkdir(parents=True, exist_ok=True)
        source.rename(dest)
        original = next(r for r in plan['originals'] if r['path'] == row['source'])
        assert sha(dest) == original['sha256']
        row['original_sha256'] = original['sha256']
        row['moved_sha256'] = sha(dest)
        save(PLAN, plan)
    for row in plan['moves']:
        path = ROOT / row['destination']
        if path.suffix != '.py':
            continue
        text = path.read_text(encoding='utf-8')
        text, count = re.subn(r"sys\.path\.insert\(0,\s*str\(Path\(__file__\)\.resolve\(\)\.parents\[1\]\s*/\s*'scripts'\)\)",
            "sys.path.insert(0, str(next(p / 'scripts' for p in Path(__file__).resolve().parents\n"
            "                           if (p / 'scripts/project_paths.py').is_file())))", text)
        assert count in (0, 1)
        for name in ACQUIRE:
            text = text.replace('tools/' + name, 'tools/evidence/acquire/' + name)
        if path.name == 'verify_static_research_20261020.py':
            text = text.replace("list((E/'tools').glob('*static*20261020.py'))",
                                "list((E/'tools/evidence').rglob('*static*20261020.py'))")
        if text != path.read_text(encoding='utf-8'):
            path.write_text(text, encoding='utf-8')
        row['final_sha256'] = sha(path)
        row['change'] = 'Path bootstrap and relocated acquisition references only' if row['final_sha256'] != row['original_sha256'] else 'Byte-identical relocation'
    empty = guarded(EXT / 'ckpts')
    if empty.exists() and not any(empty.iterdir()):
        empty.rmdir()  # Only this checked, empty directory; never recursive deletion.
        plan['removed_empty_directories'].append(empty.relative_to(ROOT).as_posix())
    plan['status'] = 'APPLIED_PENDING_DOCUMENTATION_CHECK'
    save(PLAN, plan)
    print(json.dumps(dict(status=plan['status'], moves=len(plan['moves']))))


def verify():
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    assert plan['status'] in {'APPLIED_PENDING_DOCUMENTATION_CHECK', 'VERIFIED'}
    backup_check(plan)
    preserved = preservation_checks()
    for row in plan['moves']:
        assert not (ROOT / row['source']).exists(), row['source']
        path = guarded(ROOT / row['destination'])
        assert sha(path) == row.get('final_sha256', row['original_sha256'])
        if path.suffix == '.py':
            ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    for row in plan['untouched_algorithm_files']:
        assert sha(ROOT / row['path']) == row['sha256'], row['path']
    for row in plan['protected_metadata']:
        info = (ROOT / row['path']).lstat()
        assert (info.st_size, info.st_mtime_ns, info.st_file_attributes) == (row['bytes'], row['mtime_ns'], row['attributes']), row['path']
    for row in plan.get('configuration_changes', []):
        assert sha(ROOT / row['path']) == row['sha256']
    docs = [ROOT / row['path'] for row in plan['documentation_changes']]
    checked = 0
    for path in docs:
        for target in re.findall(r'\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#'):
                continue
            assert (path.parent / target.split('#')[0]).exists(), (path, target)
            checked += 1
    plan['status'] = 'VERIFIED'
    plan['preservation_after'] = preserved
    plan['verification'] = dict(moved_files=len(plan['moves']), backed_up_originals=len(plan['originals']),
        protected_records=len(plan['protected_metadata']), untouched_algorithm_files=len(plan['untouched_algorithm_files']),
        syntax_checked=15, local_document_links_checked=checked,
        historical_result_bytes_unchanged=True, credentials_read=False, git_commit_or_push=False)
    save(PLAN, plan)
    print(json.dumps(plan['verification']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'apply', 'verify'])
    args = parser.parse_args()
    {'prepare': prepare, 'apply': apply, 'verify': verify}[args.action]()
