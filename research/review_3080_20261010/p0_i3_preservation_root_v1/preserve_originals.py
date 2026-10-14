"""P0-I3 preservation: copy the three historical originals byte-for-byte into a tracked
evidence directory, then build an independent ARCHIVE_MANIFEST.json.

Rules honoured:
  * the originals are COPIED, never moved/edited/deleted;
  * `sha256_before_preservation` is the value read BEFORE any copy, so the post-copy comparison
    is meaningful rather than self-certifying;
  * the manifest does NOT contain the commit that carries it (that would be circular); the
    commit identity is written afterwards into a separate ARCHIVE_COMMIT_IDENTITY.json;
  * the archived bytes are re-hashed from disk and compared item by item against the
    pre-preservation values AND against the existing provenance registration.
"""
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _find_root(start):
    for cand in (start, *start.parents):
        if (cand / 'MASKCO_code').is_dir():
            return cand
    raise RuntimeError('repo root not found above %s' % start)


ROOT = _find_root(HERE)
ARCHIVE = HERE / 'archived_originals'

ITEMS = [
    {
        'role': 'I3 acquisition qualification acceptance record',
        'original': ('C-VRP_Cold-chainVehicleRoutingProblem/results/i3_registered_1008_v5/'
                     'LOCAL_128_STAGE_ACCEPTANCE_20261009.json'),
        'sha256_before_preservation':
            '10e60da10a80d74ae59cc10333f012f1f09e9541497db2d69c84f48ed776564a',
        'bytes_before_preservation': 1338,
        'mtime_before_preservation': '2026-10-09 00:44:21',
        'server_copy': 'absent (whole-disk filename search, depth<=9, 0 hits)',
    },
    {
        'role': 'independent 128-state actual-identity check',
        'original': ('C-VRP_Cold-chainVehicleRoutingProblem/results/i3_registered_1008_v5/'
                     'INDEPENDENT_128_ACTUAL_IDENTITY_CHECK_20261009.json'),
        'sha256_before_preservation':
            '0825ad07afbf23926bad3b4a14b50f2a4930d988092afbbe86b8c9cbb3bfd7c6',
        'bytes_before_preservation': 24764,
        'mtime_before_preservation': '2026-10-09 00:43:33',
        'server_copy': 'absent',
    },
    {
        'role': 'independent online end-identity check',
        'original': ('C-VRP_Cold-chainVehicleRoutingProblem/results/i3_online_recovery_1009_v1/'
                     'INDEPENDENT_ACTUAL_ONLINE_END_IDENTITY_20261009.json'),
        'sha256_before_preservation':
            '1f860e7594948844305160e2120fd04d627bcc00416974cc34ab56dea494f1ef',
        'bytes_before_preservation': 153550,
        'mtime_before_preservation': '2026-10-09 15:33:02',
        'server_copy': 'absent',
    },
]


def sha_file(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    import datetime
    import subprocess
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    (HERE / '.gitattributes').write_text(
        '# I3 历史原件保全：冻结为 blob == 工作树字节，保证 ARCHIVE_MANIFEST 里登记的\n'
        '# sha256 在任意 checkout 上都能逐字节复现（仓库 core.autocrlf=true）。\n'
        '* -text\n', encoding='utf-8', newline='\n')
    (ARCHIVE / '.gitattributes').write_text('* -text\n', encoding='utf-8', newline='\n')

    rows, problems = [], []
    for it in ITEMS:
        src = ROOT / it['original']
        dst = ARCHIVE / Path(it['original']).name
        if not src.is_file():
            problems.append('ORIGINAL MISSING: %s' % it['original'])
            continue
        before = sha_file(src)
        before_bytes = src.stat().st_size
        if before != it['sha256_before_preservation']:
            problems.append('ORIGINAL CHANGED BEFORE COPY: %s (%s != %s)'
                            % (it['original'], before, it['sha256_before_preservation']))
            continue
        if before_bytes != it['bytes_before_preservation']:
            problems.append('ORIGINAL LENGTH MISMATCH: %s' % it['original'])
            continue
        shutil.copyfile(src, dst)                      # byte copy; original untouched
        after = sha_file(dst)
        same = after == before
        if not same:
            problems.append('ARCHIVED COPY MISMATCH: %s' % dst)
        rows.append({
            'role': it['role'],
            'original_path': it['original'],
            'archived_path': str(dst.relative_to(ROOT)).replace('\\', '/'),
            'sha256_before_preservation': before,
            'sha256_after_copy': after,
            'sha256_match': same,
            'bytes': before_bytes,
            'mtime_before_preservation': it['mtime_before_preservation'],
            'server_copy': it['server_copy'],
            'original_still_present': src.is_file(),
        })

    # cross-check item by item against the existing provenance registration
    prov = (ROOT / 'research/review_3080_20261010/p0_next_stage_audit_root_v1/'
                   'P0_I3_QUALIFICATION_PROVENANCE.json')
    prov_check = {'path': str(prov.relative_to(ROOT)).replace('\\', '/'),
                  'present': prov.is_file()}
    if prov.is_file():
        d = json.loads(prov.read_text(encoding='utf-8'))
        prov_check['sha256'] = sha_file(prov)
        prov_check['registered_acceptance_sha256'] = (d.get('local_copy') or {}).get('sha256')
        prov_check['matches_archived_acceptance'] = (
            prov_check['registered_acceptance_sha256'] == rows[0]['sha256_after_copy']
            if rows else None)

    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=str(ROOT), capture_output=True,
                          text=True).stdout.strip()
    manifest = {
        'stage': 'P0_I3_historical_original_preservation',
        'purpose': ('preserve byte-identical copies of three single-copy, git-ignored I3 '
                    'historical originals so their identity survives'),
        'not_a_claim': ('preservation success only means an additional hash-verified copy '
                        'exists; it does NOT re-establish the I3 acquisition qualification and '
                        'changes no scientific adjudication'),
        'baseline_commit_at_preservation': head,
        'preserved_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'environment': {'platform': platform.platform(), 'python': platform.python_version()},
        'method': 'binary copy (shutil.copyfile); originals never moved, edited or deleted',
        'originals_are_gitignored_by': '.gitignore:28  /C-VRP_.../results/*',
        'archive_commit_identity': ('NOT recorded here: the commit that carries this manifest '
                                    'cannot be named inside it without circularity/self-'
                                    'certification. See ARCHIVE_COMMIT_IDENTITY.json, written '
                                    'after the commit.'),
        'privacy_scan': {
            'script': 'scan_privacy.py',
            'script_sha256': sha_file(HERE / 'scan_privacy.py'),
            'result': 'clear_to_preserve',
            'blocking_findings': 0,
            'sftp_password_in_any_file': False,
            'sftp_host_in_any_file': False,
            'sftp_username_in_files': True,
            'report_only_note': ('the only report-only finding is the login username inside '
                                 '/home/hzeng/... absolute paths, which this public repository '
                                 'already publishes in CLAUDE.md and in many registered '
                                 'artifacts; no credential material is present'),
        },
        'items': rows,
        'item_count': len(rows),
        'all_sha256_match': bool(rows) and all(r['sha256_match'] for r in rows),
        'existing_provenance_registration': prov_check,
        'problems': problems,
        'verdict': 'preserved' if (rows and not problems) else 'FAILED',
    }
    (HERE / 'ARCHIVE_MANIFEST.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({k: manifest[k] for k in
                      ('item_count', 'all_sha256_match', 'problems', 'verdict')},
                     ensure_ascii=False, indent=2))
    for r in rows:
        print('%-8s %s  %s' % ('MATCH' if r['sha256_match'] else 'MISMATCH',
                               r['sha256_after_copy'][:16], r['archived_path']))
    print('provenance cross-check:', json.dumps(prov_check, ensure_ascii=False)[:300])
    return 0 if manifest['verdict'] == 'preserved' else 3


if __name__ == '__main__':
    sys.exit(main())
