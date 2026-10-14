"""Write ARCHIVE_COMMIT_IDENTITY.json AFTER the archive commit.

Why a separate file: the manifest cannot name the commit that carries it without circularity
(self-certification).  This file is written afterwards, reads the commit from git itself, and
records the identity of the commit that introduced the archived originals.  Its OWN commit is
deliberately NOT named here; it is resolvable with `git log -- <this file>` -- no self-claim.
"""
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _find_root(start):
    for cand in (start, *start.parents):
        if (cand / 'MASKCO_code').is_dir():
            return cand
    raise RuntimeError('repo root not found')


ROOT = _find_root(HERE)
ARCHIVE_REL = 'research/review_3080_20261010/p0_i3_preservation_root_v1/archived_originals'


def git(*args):
    return subprocess.run(['git'] + list(args), cwd=str(ROOT), capture_output=True,
                          text=True, encoding='utf-8', errors='replace').stdout.strip()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    commit = git('log', '-1', '--format=%H', '--', ARCHIVE_REL)
    commit_short = git('log', '-1', '--format=%h', '--', ARCHIVE_REL)
    subject = git('log', '-1', '--format=%s', '--', ARCHIVE_REL)
    committed_at = git('log', '-1', '--format=%cI', '--', ARCHIVE_REL)
    files = git('show', '--name-only', '--format=', commit).splitlines()
    files = [f for f in files if f.strip()]
    blob_check = {}
    for rel in files:
        blob = subprocess.run(['git', 'show', '%s:%s' % (commit, rel)], cwd=str(ROOT),
                              capture_output=True)
        work = ROOT / rel
        blob_check[rel] = {
            'blob_sha256': hashlib.sha256(blob.stdout).hexdigest(),
            'worktree_sha256': sha(work) if work.is_file() else None,
            'equal': (hashlib.sha256(blob.stdout).hexdigest() == sha(work)) if work.is_file()
                     else None,
        }
    out = {
        'stage': 'P0_I3_archive_commit_identity',
        'archive_commit': commit,
        'archive_commit_short': commit_short,
        'archive_commit_subject': subject,
        'archive_commit_committed_at': committed_at,
        'archive_commit_file_count': len(files),
        'archive_commit_files': files,
        'blob_vs_worktree_at_that_commit': blob_check,
        'all_blob_equal_worktree': all(v['equal'] for v in blob_check.values()
                                       if v['equal'] is not None),
        'baseline_commit_before_preservation': json.loads(
            (HERE / 'ARCHIVE_MANIFEST.json').read_text(encoding='utf-8'))[
                'baseline_commit_at_preservation'],
        'manifest_sha256': sha(HERE / 'ARCHIVE_MANIFEST.json'),
        'note': ('this file was written after the archive commit so the manifest never certifies '
                 'itself; the commit carrying THIS file is intentionally not named here and is '
                 'resolvable via `git log -- ARCHIVE_COMMIT_IDENTITY.json`'),
    }
    (HERE / 'ARCHIVE_COMMIT_IDENTITY.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({k: out[k] for k in
                      ('archive_commit', 'archive_commit_subject',
                       'archive_commit_file_count', 'all_blob_equal_worktree',
                       'baseline_commit_before_preservation')}, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
