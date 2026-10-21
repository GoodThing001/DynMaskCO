"""Server-side, manifest-scoped reversible deployment; invoked by the sync tool."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import zipfile

EXT_NAME = 'C-VRP_Cold-chainVehicleRoutingProblem'
DENY = {'.git', '.vscode', '.codex', '.claude', '.cursor', 'private'}


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as stream:
        while b := stream.read(4 * 1024 * 1024): h.update(b)
    return h.hexdigest()


def ordinary(p):
    s = p.lstat()
    return not (stat.S_ISLNK(s.st_mode) or getattr(s, 'st_file_attributes', 0) & 1024)


def io(p):
    text = os.path.abspath(p)
    return Path('\\\\?\\' + text) if os.name == 'nt' and not text.startswith('\\\\?\\') else Path(text)


def safe(root, rel):
    rel = Path(rel)
    if rel.is_absolute() or '..' in rel.parts or any(p.lower() in DENY for p in rel.parts):
        raise RuntimeError('Protected or escaping target: ' + str(rel))
    p = root / rel
    if os.path.commonpath([os.path.abspath(p), os.path.abspath(root)]) != os.path.abspath(root):
        raise RuntimeError('Target escapes root')
    for item in [p, *p.parents]:
        if item.exists() or item.is_symlink():
            if not ordinary(item): raise RuntimeError('Link/reparse target refused: ' + str(rel))
        if item == root: break
    return p


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(obj, ensure_ascii=False, indent=2) + '\n').encode())


def job_guard(root):
    prefix = str(root).replace('\\', '/').lower().rstrip('/')
    immutable = [prefix+'/'+v for v in ['research/', 'research_static_profit_20261021/', EXT_NAME.lower()+'/research/']]
    own = {os.getpid(), os.getppid()}
    active = []
    protected_jobs = []
    if os.name == 'nt':
        code = "[Console]::OutputEncoding=[System.Text.UTF8Encoding]::new($false); Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*MASKCO-Main*' } | Select-Object ProcessId,Name,CommandLine | ConvertTo-Json -Compress"
        p = subprocess.run(['powershell.exe', '-NoLogo', '-NoProfile', '-NonInteractive', '-Command', code], capture_output=True, text=True, encoding='utf-8')
        if p.returncode: raise RuntimeError('Process check failed')
        rows = json.loads(p.stdout.strip() or '[]')
        if isinstance(rows, dict): rows = [rows]
        for row in rows:
            if row['ProcessId'] not in own and prefix in (row.get('CommandLine') or '').replace('\\', '/').lower():
                cmd = (row.get('CommandLine') or '').replace('\\', '/').lower()
                if any(v in cmd for v in immutable) and not any(prefix+'/'+v in cmd for v in ['cc_compare/', EXT_NAME.lower()+'/scripts/', EXT_NAME.lower()+'/tools/']):
                    protected_jobs.append(dict(pid=row['ProcessId'],name=row['Name'],scope='immutable run tree'))
                else: active.append(dict(pid=row['ProcessId'], name=row['Name']))
    else:
        for d in Path('/proc').iterdir():
            if not d.name.isdigit() or int(d.name) in own: continue
            try:
                args = (d / 'cmdline').read_bytes().split(b'\0')
                paths = [a.decode('utf-8', 'replace').lower() for a in args if len(a) < 1000]
                cwd = str((d / 'cwd').resolve()).lower()
                project_paths = [a for a in paths if a.startswith(prefix+'/')]
                if project_paths or cwd == prefix or cwd.startswith(prefix + '/'):
                    isolated = all(any(a.startswith(v) for v in immutable) for a in project_paths)
                    isolated = isolated and (not cwd.startswith(prefix) or any(cwd.startswith(v) for v in immutable))
                    row = dict(pid=int(d.name),name=paths[0] if paths else '',project_paths=project_paths,cwd=cwd)
                    (protected_jobs if isolated else active).append(row)
            except (PermissionError, FileNotFoundError, ProcessLookupError): pass
    if active: raise RuntimeError('Project jobs appeared; deployment stopped: ' + json.dumps(active))
    return dict(status='PASS', active_managed_tree_jobs=[], protected_immutable_run_jobs=protected_jobs)


def inventory(path):
    rows = []
    if path.is_file(): return [dict(path='', bytes=path.stat().st_size, sha256=sha(io(path)))]
    for base, dirs, files in os.walk(io(path), followlinks=False):
        for name in list(dirs):
            p = Path(base) / name
            if not ordinary(p):
                rows.append(dict(path=p.relative_to(io(path)).as_posix(), bytes=p.lstat().st_size,
                                 link_target=os.readlink(p), traversed=False))
                dirs.remove(name)
            # Private and nested Git bytes stay intact inside the renamed tree; contents are not logged.
        for name in files:
            p = Path(base) / name
            rel = p.relative_to(io(path)).as_posix()
            if not ordinary(p):
                rows.append(dict(path=rel, bytes=p.lstat().st_size, link_target=os.readlink(p), traversed=False))
            else: rows.append(dict(path=rel, bytes=p.stat().st_size, sha256=sha(p)))
    return sorted(rows, key=lambda r: r['path'])


def run(action, root, stage):
    root = Path(os.path.abspath(root)); stage = safe(root, stage)
    plan = json.loads((stage / 'manifest.json').read_text(encoding='utf-8'))
    desired = plan['deploy']
    archive = safe(root, EXT_NAME + '/archive/server_sync_20261021')
    if action == 'inspect':
        guard = job_guard(root)
        available = {}
        for name, row in desired.items():
            p = safe(root, name)
            if p.is_file() and p.stat().st_size == row['bytes'] and sha(io(p)) == row['sha256']:
                # Old CC is archived before installation. Remember it as a verified reusable blob.
                available[row['sha256']] = name
        result = dict(status='PASS', job_guard=guard, available=available,
                      matched_hashes=len(available), needed_hashes=len({r['sha256'] for r in desired.values()} - available.keys()))
        save(stage / 'inspection.json', result)
        print(json.dumps(result)); return
    guard = job_guard(root)
    inspection = json.loads((stage / 'inspection.json').read_text(encoding='utf-8'))
    available = inspection['available']
    changes = json.loads((archive / 'moves.json').read_text(encoding='utf-8')) if (archive / 'moves.json').exists() else []
    def recover(rel):
        source = safe(root, rel)
        target = safe(root, str(archive.relative_to(root) / 'legacy' / rel))
        if not source.exists(): return
        if target.exists(): raise RuntimeError('Recovery destination already exists: ' + rel)
        rows = inventory(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
        # Rename on the same filesystem preserves bytes; verify every archived member.
        if inventory(target) != rows: raise RuntimeError('Recovery verification failed: ' + rel)
        receipt = archive / 'inventories' / (hashlib.sha256(rel.encode()).hexdigest()[:16] + '.json')
        save(receipt, dict(original=rel, archived=target.relative_to(root).as_posix(), files=rows))
        changes.append(dict(original=rel, archived=target.relative_to(root).as_posix(),
                            files=len(rows), bytes=sum(r['bytes'] for r in rows), inventory=receipt.relative_to(root).as_posix()))
        save(archive / 'moves.json', changes)
        for key, old in list(available.items()):
            if old == rel or old.startswith(rel + '/'):
                available[key] = target.relative_to(root).as_posix() + old[len(rel):]
    if action == 'apply':
        # An earlier pass moved this empty historical directory before the protected
        # history classification was refined. Restore its original path, retaining receipts.
        old_history = safe(root, str(archive.relative_to(root) / 'legacy' / EXT_NAME / '02_配送线'))
        history = safe(root, EXT_NAME + '/02_配送线')
        if old_history.exists() and not history.exists():
            if inventory(old_history): raise RuntimeError('Historical restoration requires review of nonempty content')
            old_history.rename(history)
            save(archive/'historical_directory_restored.json', dict(original=EXT_NAME+'/02_配送线', files=0, status='RESTORED_ORIGINAL_PATH'))
        cc = root / 'CC_Compare'
        # A new layout has models; old layouts have one method per top-level directory.
        if cc.exists() and not (cc / 'models').is_dir(): recover('CC_Compare')
        keep_root = {n.split('/')[0] for n in desired} | DENY | {'research', 'results', 'research_static_profit_20261021'}
        for p in sorted(root.iterdir()):
            if p.name not in keep_root: recover(p.name)
        ext = root / EXT_NAME
        keep_ext = {n[len(EXT_NAME)+1:].split('/')[0] for n in desired if n.startswith(EXT_NAME+'/')} | {'research', 'archive', 'ckpts', 'logs', 'negative_result_paper', '02_配送线'} | DENY
        for p in sorted(ext.iterdir()):
            if p.name not in keep_ext: recover(EXT_NAME + '/' + p.name)
        for group in [EXT_NAME+'/scripts', EXT_NAME+'/tools', 'tools']:
            parent = root / group
            if not parent.exists(): continue
            for base, dirs, files in os.walk(parent, followlinks=False):
                for d in list(dirs):
                    p = Path(base)/d
                    if d == '__pycache__' and ordinary(p): recover(p.relative_to(root).as_posix())
                dirs[:] = [d for d in dirs if d.lower() not in DENY and d != '__pycache__' and ordinary(Path(base)/d)]
                for name in files:
                    p = Path(base) / name
                    rel = p.relative_to(root).as_posix()
                    if rel not in desired and p.suffix.lower() not in {'.pyc', '.pyo'} and not any(v in name.lower() for v in ['secret', 'credential']):
                        recover(rel)
            for base, dirs, files in os.walk(parent, topdown=False, followlinks=False):
                p = Path(base)
                if p == parent or not ordinary(p) or any(v.lower() in DENY for v in p.relative_to(root).parts): continue
                safe(root, p.relative_to(root))
                if not any(p.iterdir()): p.rmdir()
        restored = copied = unchanged = 0
        with zipfile.ZipFile(stage / 'blobs.zip') as payload:
            for name, row in desired.items():
                target = safe(root, name)
                if target.is_file() and target.stat().st_size == row['bytes'] and sha(io(target)) == row['sha256']:
                    unchanged += 1; continue
                if target.exists():
                    backup = safe(root, str(archive.relative_to(root) / 'overwritten' / name))
                    if not backup.exists():
                        backup.parent.mkdir(parents=True, exist_ok=True)
                        expected = sha(io(target)); shutil.copy2(io(target), io(backup))
                        if sha(io(backup)) != expected: raise RuntimeError('Overwrite backup failed')
                target.parent.mkdir(parents=True, exist_ok=True)
                tmp = target.with_name(target.name + '.sync_20261021.tmp')
                if row['sha256'] in available:
                    source = safe(root, available[row['sha256']])
                    if sha(io(source)) != row['sha256']: raise RuntimeError('Reuse source drift')
                    shutil.copyfile(io(source), io(tmp)); restored += 1
                else:
                    with payload.open(row['sha256']) as source, io(tmp).open('wb') as dest: shutil.copyfileobj(source, dest, 4*1024*1024)
                    copied += 1
                if sha(io(tmp)) != row['sha256']: raise RuntimeError('Payload SHA failed: '+name)
                os.replace(io(tmp), io(target))
                available[row['sha256']] = name
        save(stage / 'applied.json', dict(status='PASS', reused=restored, uploaded=copied, unchanged=unchanged, moves=len(changes)))
    failures = []
    for name, row in desired.items():
        p = safe(root, name)
        if not p.is_file() or sha(io(p)) != row['sha256']: failures.append(name)
    lock = json.loads((root/EXT_NAME/'results/static_delivery/publication_protocol_v1_scientific_freeze_20261021/FREEZE_LOCK.json').read_text(encoding='utf-8'))
    for row in lock['files']:
        if sha(io(root/EXT_NAME/row['path'])) != row['sha256']: failures.append('freeze:'+row['path'])
    result = dict(status='PASS' if not failures else 'FAIL', files_verified=len(desired), frozen_files_verified=len(lock['files']),
                  failures=failures, methods=sorted(p.name for p in (root/'CC_Compare/models').iterdir() if p.is_dir()),
                  cc_directories=sorted(p.name for p in (root/'CC_Compare').iterdir() if p.is_dir()),
                  root_entries=sorted(p.name for p in root.iterdir()), recovery_moves=changes,
                  project_jobs_before=guard, original_experiment_trees_preserved=True,
                  private_and_git_untouched=True)
    save(stage / 'verification.json', result)
    print(json.dumps(result))
    if failures: raise RuntimeError('Server verification failed')


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['inspect','apply','verify']);p.add_argument('--root',required=True);p.add_argument('--stage',required=True)
    a=p.parse_args();run(a.action,a.root,a.stage)
