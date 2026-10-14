"""P0-I3 preservation: INDEPENDENT verification of the archived originals.

Independent of `preserve_originals.py`: it re-reads everything from disk, re-hashes it, and
additionally checks that GIT'S STAGED BLOB bytes hash to the same SHA-256 -- i.e. that no
EOL/filter transformation will change the recorded identity after a checkout.  It also
re-verifies that the ORIGINALS are still present and byte-identical to their pre-preservation
values, so "we copied" cannot silently become "we moved".

Exit codes: 0 = verified, 3 = verification failed.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _find_root(start):
    for cand in (start, *start.parents):
        if (cand / 'MASKCO_code').is_dir():
            return cand
    raise RuntimeError('repo root not found')


ROOT = _find_root(HERE)
ARCHIVE = HERE / 'archived_originals'
MANIFEST = HERE / 'ARCHIVE_MANIFEST.json'


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(p):
    return sha_bytes(Path(p).read_bytes())


def staged_blob_sha256(rel):
    """SHA-256 of the bytes git would store for `rel` (empty string if not staged)."""
    r = subprocess.run(['git', 'show', ':%s' % rel], cwd=str(ROOT), capture_output=True)
    if r.returncode != 0:
        return None
    return sha_bytes(r.stdout)


def main():
    ok = True
    report = {'stage': 'P0_I3_archive_independent_verification', 'items': [], 'problems': []}
    if not MANIFEST.is_file():
        print('MANIFEST MISSING'); return 3
    man = json.loads(MANIFEST.read_text(encoding='utf-8'))
    report['manifest_sha256'] = sha_file(MANIFEST)
    report['baseline_commit_at_preservation'] = man['baseline_commit_at_preservation']
    report['manifest_self_consistency'] = {
        'item_count_matches': man['item_count'] == len(man['items']),
        'all_sha256_match_flag': man['all_sha256_match'],
        'verdict': man['verdict']}

    for it in man['items']:
        arch = ROOT / it['archived_path']
        orig = ROOT / it['original_path']
        row = {'archived_path': it['archived_path'], 'original_path': it['original_path']}
        row['archived_present'] = arch.is_file()
        row['original_present'] = orig.is_file()
        if not arch.is_file() or not orig.is_file():
            report['problems'].append('missing file for %s' % it['archived_path'])
            ok = False
            report['items'].append(row)
            continue
        a_bytes = arch.read_bytes()
        row['archived_bytes'] = len(a_bytes)
        row['archived_sha256_recomputed'] = sha_bytes(a_bytes)
        row['matches_pre_preservation'] = (
            row['archived_sha256_recomputed'] == it['sha256_before_preservation'])
        row['archived_matches_manifest'] = (
            row['archived_sha256_recomputed'] == it['sha256_after_copy'])
        row['original_sha256_recomputed'] = sha_file(orig)
        row['original_unchanged'] = (
            row['original_sha256_recomputed'] == it['sha256_before_preservation'])
        row['original_bytes_unchanged'] = (orig.stat().st_size == it['bytes'])
        sb = staged_blob_sha256(it['archived_path'])
        row['staged_blob_sha256'] = sb
        row['staged_blob_equals_worktree'] = (sb == row['archived_sha256_recomputed'])
        for k in ('matches_pre_preservation', 'archived_matches_manifest',
                  'original_unchanged', 'original_bytes_unchanged', 'staged_blob_equals_worktree'):
            if not row[k]:
                report['problems'].append('%s: %s == False' % (it['archived_path'], k))
                ok = False
        report['items'].append(row)

    pc = man.get('existing_provenance_registration') or {}
    report['provenance_cross_check'] = {
        'registered_acceptance_sha256': pc.get('registered_acceptance_sha256'),
        'archived_acceptance_sha256': (report['items'][0]['archived_sha256_recomputed']
                                       if report['items'] else None)}
    report['provenance_cross_check']['consistent'] = (
        pc.get('registered_acceptance_sha256') ==
        report['provenance_cross_check']['archived_acceptance_sha256'])
    if not report['provenance_cross_check']['consistent']:
        report['problems'].append('archived acceptance sha differs from the provenance registry')
        ok = False

    report['verdict'] = 'verified' if ok else 'FAILED'
    (HERE / 'ARCHIVE_VERIFICATION.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')

    print('%-52s %-18s %-8s %-8s %-8s %-8s %s'
          % ('archived file', 'sha256[:16]', 'preP', 'manif', 'orig', 'blob', 'verdict'))
    for r in report['items']:
        if 'archived_sha256_recomputed' not in r:
            print('%-52s MISSING' % r['archived_path']); continue
        print('%-52s %-18s %-8s %-8s %-8s %-8s %s'
              % (Path(r['archived_path']).name, r['archived_sha256_recomputed'][:16],
                 r['matches_pre_preservation'], r['archived_matches_manifest'],
                 r['original_unchanged'], r['staged_blob_equals_worktree'],
                 'OK' if all([r['matches_pre_preservation'], r['archived_matches_manifest'],
                              r['original_unchanged'], r['staged_blob_equals_worktree']]) else 'FAIL'))
    print('\nprovenance cross-check:', json.dumps(report['provenance_cross_check'],
                                                  ensure_ascii=False))
    print('problems:', report['problems'])
    print('VERDICT:', report['verdict'])
    return 0 if ok else 3


if __name__ == '__main__':
    sys.exit(main())
