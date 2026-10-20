"""Verified recoverable consolidation of legacy files, never follow reparse points.

prepare writes a content-deduplicated ZIP and per-path manifest. apply deletes
only unchanged file copies already verified inside that ZIP. Private files,
upstream, third-party repositories, credentials, and the new static line stay.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import zipfile

EXT = Path(__file__).resolve().parents[1]
ROOT = EXT.parent
OUT = EXT / 'archive/transition_20261020'
MANIFEST = OUT / 'LEGACY_CONSOLIDATION.json'
ARCHIVE = OUT / 'legacy_recovery_objects.zip'
REPARSE = stat.FILE_ATTRIBUTE_REPARSE_POINT
PROTECTED_NAMES = {'private', '.git', '.vscode', '.codex', '.claude', '.cursor'}
PROTECTED_SUFFIXES = {'.pem', '.key', '.pfx', '.p12', '.ppk'}


def digest(stream):
    h = hashlib.sha256()
    while block := stream.read(2 ** 20):
        h.update(block)
    return h.hexdigest()


def reparse(path):
    return bool(path.lstat().st_file_attributes & REPARSE)


def guarded(path):
    path = Path(os.path.abspath(path))
    if not path.is_relative_to(ROOT) or path == ROOT:
        raise ValueError('Path outside intended workspace')
    for parent in [path, *path.parents]:
        if parent == ROOT:
            break
        try:
            linked = reparse(parent)
        except FileNotFoundError:
            continue  # Missing prior journaled removals still need parent guards.
        if linked:
            raise ValueError('Reparse path is never a deletion target')
    return path


def deletion_target(path):
    path = guarded(path)
    # Independently enforce the preserved ranges even for an edited manifest.
    rel = path.relative_to(ROOT)
    if rel.parts[0] in {'MASKCO_code', 'CC_Compare', 'results'} or rel.parts[0].startswith('.'):
        raise ValueError('Path belongs to a preserved range')
    if rel.parts[0] in {'research', 'tools', 'docs'}:
        return path
    if path.parent == ROOT and (path.name.startswith('_') or path.name in {'EVIDENCE_MANIFEST.md', 'REPOSITORY_MAP.md'}):
        return path
    if path.is_relative_to(EXT):
        erel = path.relative_to(EXT)
        first = erel.parts[0]
        protected = (
            (first == 'scripts' and (erel.parts[1] in {'static_delivery', 'project_paths.py'} or
             (erel.parts[1] == 'tests' and erel.parts[-1].startswith('test_static_rrp')))),
            first == 'configs' and erel.parts[1] == 'static_delivery',
            first == 'docs' and erel.parts[1] in {'static_delivery', 'static_delivery_migration_20261020'},
            first == 'data' and erel.parts[1] == 'public_sources_20261020',
            first == 'results' and erel.parts[1] == 'static_delivery',
            first == 'tools' and path.name in {'static_rrp_remote.py', 'static_rrp_cleanup.py'},
        )
        allowed = {'01_取货线', '02_配送线', '03_负结果线', 'negative_result_paper', 'logs',
                   'tools_remote', 'research', 'scripts', 'configs', 'docs', 'data', 'results', 'tools'}
        if first in allowed and not any(protected) and not excluded(path):
            return path
    raise ValueError('Path is not in the allowed legacy deletion ranges')


def _remove_unchanged(path, row):
    """Hash and delete the SAME file under a Windows write/delete exclusion lock."""
    import ctypes
    import msvcrt
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.CreateFileW.argtypes = (wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                  wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE)
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.SetFileInformationByHandle.argtypes = (wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD)
    kernel.SetFileInformationByHandle.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    # READ|DELETE, share READ only: concurrent writers or renames cannot race SHA.
    handle = kernel.CreateFileW(str(path), 0x80010000, 1, None, 3, 128, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        kernel.CloseHandle(handle)
        raise
    with os.fdopen(fd, 'rb') as stream:
        if digest(stream) != row['sha256']:
            raise ValueError('Final source SHA mismatch; preserved new edit: ' + row['path'])
        disposition = wintypes.BOOL(True)
        if not kernel.SetFileInformationByHandle(handle, 4, ctypes.byref(disposition), ctypes.sizeof(disposition)):
            raise ctypes.WinError(ctypes.get_last_error())


def excluded(path):
    parts = set(part.lower() for part in path.relative_to(ROOT).parts)
    if parts & PROTECTED_NAMES or path.suffix.lower() in PROTECTED_SUFFIXES:
        return 'protected_private_or_credential_path'
    if path.name == 'synthetic_mapping.bin':
        return 'user_original_untracked_mapping'
    if any(term in path.name.lower() for term in ('password', 'credential', 'secret', '.env', 'sftp.')):
        return 'credential_name_not_read'
    # Scripts in the old deployment tool area can contain embedded credentials.
    # Check only small text, never print the matched content or copy it to ZIP.
    if path.is_file() and path.suffix.lower() in {'.py', '.sh', '.ps1', '.json', '.txt', '.md'} and path.stat().st_size < 2 ** 20:
        data = path.read_text(encoding='utf-8', errors='replace')
        if re.search(r'''(?im)(?:password|passwd|api_key|access_token)\s*[:=]\s*["'][^"'\r\n]{4,}["']''', data):
            return 'possible_embedded_credential_preserved_in_place'
    return None


def legacy_roots():
    roots = [ROOT / n for n in ('research', 'tools', 'docs')]
    roots += [EXT / n for n in ('01_取货线', '02_配送线', '03_负结果线',
                                'negative_result_paper', 'logs', 'tools_remote', 'research')]
    for parent, keep in (
        (EXT / 'scripts', {'static_delivery', 'project_paths.py', 'tests'}),
        (EXT / 'configs', {'static_delivery'}),
        (EXT / 'docs', {'static_delivery', 'static_delivery_migration_20261020'}),
        (EXT / 'data', {'public_sources_20261020'}),
        (EXT / 'results', {'static_delivery'}),
        (EXT / 'tools', {'static_rrp_remote.py', 'static_rrp_cleanup.py'}),
    ):
        roots += [p for p in parent.iterdir() if p.name not in keep]
    roots += [p for p in (EXT / 'scripts/tests').iterdir() if not p.name.startswith('test_static_rrp')]
    roots += [p for p in ROOT.iterdir() if p.is_file() and p.name.startswith('_')]
    roots += [ROOT / n for n in ('EVIDENCE_MANIFEST.md', 'REPOSITORY_MAP.md')]
    return roots


def collect():
    files, directories, protected = set(), set(), []
    for root in legacy_roots():
        if not root.exists():
            continue
        if reparse(root):
            protected.append({'path': root.relative_to(ROOT).as_posix(), 'reason': 'reparse_point'})
            continue
        if root.is_file():
            files.add(root)
            continue
        for base, dirs, names in os.walk(root, followlinks=False):
            base = Path(base)
            directories.add(base)
            kept = []
            for name in dirs:
                p = base / name
                if reparse(p) or name.lower() in PROTECTED_NAMES:
                    protected.append({'path': p.relative_to(ROOT).as_posix(), 'reason': 'reparse_or_private_directory'})
                else:
                    kept.append(name)
            dirs[:] = kept
            files.update(base / name for name in names)
    selected = []
    for p in sorted(files):
        reason = 'reparse_point' if reparse(p) else excluded(p)
        if reason:
            protected.append({'path': p.relative_to(ROOT).as_posix(), 'reason': reason})
        else:
            selected.append(guarded(p))
    return selected, directories, protected


def prepare():
    if MANIFEST.exists() or ARCHIVE.exists():
        raise ValueError('Refuse to overwrite a consolidation archive')
    files, dirs, protected = collect()
    rows, objects = [], set()
    with zipfile.ZipFile(ARCHIVE, 'x', zipfile.ZIP_DEFLATED, compresslevel=1, allowZip64=True) as bundle:
        for i, path in enumerate(files):
            before = path.stat()
            with path.open('rb') as f:
                sha = digest(f)
            if sha not in objects:
                bundle.write(path, 'objects/' + sha)
                objects.add(sha)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError('File changed while archiving')
            rows.append({'path': path.relative_to(ROOT).as_posix(), 'bytes': before.st_size,
                         'sha256': sha, 'action': 'remove_verified_active_copy',
                         'preserved_as': ARCHIVE.name + ':objects/' + sha})
            if i % 1000 == 0:
                print(json.dumps({'archived': i, 'total': len(files)}), flush=True)
    verify_objects(objects)
    with ARCHIVE.open('rb') as f:
        archive_sha = digest(f)
    data = {'schema': 'legacy-consolidation-v1', 'archive': ARCHIVE.name,
            'archive_sha256': archive_sha, 'files': rows, 'unique_objects': len(objects),
            'protected': protected, 'directories': [p.relative_to(ROOT).as_posix() for p in sorted(dirs)],
            'root': str(ROOT), 'scope': 'recoverable_legacy_snapshot_not_active_method_data',
            'validation': 'all_unique_objects_SHA256_verified', 'applied': False}
    MANIFEST.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'files': len(rows), 'unique_objects': len(objects),
                      'source_bytes': sum(r['bytes'] for r in rows), 'archive_bytes': ARCHIVE.stat().st_size,
                      'protected': len(protected)}, ensure_ascii=False), flush=True)


def verify_objects(objects):
    with zipfile.ZipFile(ARCHIVE) as bundle:
        for sha in objects:
            with bundle.open('objects/' + sha) as f:
                if digest(f) != sha:
                    raise RuntimeError('Archive member SHA mismatch')


def apply():
    data = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if data['applied']:
        raise ValueError('Consolidation already applied')
    with ARCHIVE.open('rb') as f:
        if digest(f) != data['archive_sha256']:
            raise ValueError('Archive SHA mismatch')
    journal_path = OUT / 'CLEANUP_JOURNAL.jsonl'
    intents = set()
    if journal_path.exists():
        for line in journal_path.read_text(encoding='utf-8').splitlines():
            item = json.loads(line)
            intents.add((item['path'], item['sha256']))
    # Verify every remaining current file before deleting any. A missing file
    # is resumable only if this exact archive object has a durable prior intent.
    for row in data['files']:
        path = deletion_target(ROOT / row['path'])
        if excluded(path):
            raise ValueError('Protected path entered deletion manifest')
        if not path.exists() and (row['path'], row['sha256']) in intents:
            continue
        with path.open('rb') as f:
            if digest(f) != row['sha256']:
                raise ValueError('Current source SHA mismatch: ' + row['path'])
    verify_objects({r['sha256'] for r in data['files']})
    # Append durable intent/result per item, so a Windows lock or interruption
    # preserves an honest record and can safely resume without restoring first.
    with journal_path.open('a', encoding='utf-8') as journal:
        def record(row, action):
            journal.write(json.dumps({'path': row['path'], 'sha256': row['sha256'], 'action': action}) + '\n')
            journal.flush()
            os.fsync(journal.fileno())
        for index, row in enumerate(data['files']):
            path = deletion_target(ROOT / row['path'])
            if not path.exists() and (row['path'], row['sha256']) in intents:
                row['result'] = 'prior_journaled_removal'
                continue
            record(row, 'intent_remove_verified_copy')
            _remove_unchanged(path, row)
            record(row, 'removed_verified_copy')
            row['result'] = 'removed_verified_copy'
            if index % 1000 == 0:
                print(json.dumps({'removed': index + 1, 'total': len(data['files'])}), flush=True)
    removed_dirs = []
    for name in sorted(data['directories'], key=lambda x: len(Path(x).parts), reverse=True):
        path = guarded(ROOT / name)
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()
            removed_dirs.append(name)
    data.update(applied=True, removed_empty_directories=removed_dirs)
    MANIFEST.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'removed_files': len(data['files']), 'removed_empty_directories': len(removed_dirs),
                      'archive_preserved': True, 'protected_paths_untouched': len(data['protected'])}), flush=True)


def restore(destination):
    """Recover into a NEW directory, never overwrite an active checkout."""
    target = Path(destination).resolve()
    target.mkdir(parents=True, exist_ok=False)
    data = json.loads(MANIFEST.read_text(encoding='utf-8'))
    with zipfile.ZipFile(ARCHIVE) as bundle:
        for row in data['files']:
            rel = Path(row['path'])
            if rel.is_absolute() or '..' in rel.parts:
                raise ValueError('Unsafe archived relative path')
            path = target / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256()
            with bundle.open('objects/' + row['sha256']) as source, path.open('xb') as output:
                while block := source.read(2 ** 20):
                    h.update(block)
                    output.write(block)
            if h.hexdigest() != row['sha256']:
                raise ValueError('Restoration SHA mismatch')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare', 'apply', 'restore'))
    p.add_argument('--destination', type=Path)
    args = p.parse_args()
    if args.action == 'restore':
        if args.destination is None:
            p.error('restore requires --destination pointing to a new folder')
        restore(args.destination)
    elif args.action == 'prepare':
        prepare()
    else:
        apply()
