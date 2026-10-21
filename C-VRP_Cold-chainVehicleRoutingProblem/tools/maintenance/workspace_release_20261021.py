"""Explicit publication/deployment manifest for the approved workspace cleanup.

No whole-repository add, recursive deletion, credential upload or history rewrite.
Run with python -B. Remote operations require the existing SFTP profiles.
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
import time
from datetime import datetime, timezone

from workspace_sync_20261021 import ROOT, EXT, OUT, PROFILES, connect, execute, save, digest

PREFIX = EXT.relative_to(ROOT).as_posix() + '/'
LOCK = EXT / 'results/static_delivery/publication_protocol_v1_scientific_freeze_20261021/FREEZE_LOCK.json'
PLAN = OUT / 'release_manifest.json'
SNAPSHOT = OUT / 'snapshot_blobs'
DENY_DIRS = {'.git', '.vscode', '.codex', '.cursor', '.claude', 'private', '__pycache__', '.runtime', 'node_modules', 'snapshot_blobs', 'snapshot_workspace'}
BINARY = {'.npz', '.npy', '.pt', '.pth', '.ckpt', '.pkl', '.pickle', '.zip', '.gz', '.whl', '.so', '.dll', '.exe', '.a', '.o', '.obj', '.pdb', '.lib', '.bin'}
TEXT = {'.json', '.jsonl', '.csv', '.tsv', '.txt', '.md', '.log', '.sha256', '.bib', '.py', '.xml', '.yaml', '.yml', '.toml', '.sh', '.ps1'}
ROOT_FILES = {'.cursorignore', '.gitattributes', '.gitignore', 'AGENTS.md', 'CLAUDE.md', 'GITHUB_SYNC.md', 'README.md', '工作区导航.md'}


def git(*args, data=None):
    return subprocess.check_output(['git', '--no-optional-locks', '-c', 'core.longpaths=true', *args], cwd=ROOT, input=data)


def ordinary(path):
    s = path.lstat()
    return not (stat.S_ISLNK(s.st_mode) or getattr(s, 'st_file_attributes', 0) & 1024)


def walk(folder):
    for base, dirs, files in os.walk(folder, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d.lower() not in DENY_DIRS and ordinary(Path(base) / d))
        for name in sorted(files):
            p = Path(base) / name
            if ordinary(p):
                yield p


def protected(name):
    parts = Path(name).parts
    return (any(p.lower() in DENY_DIRS for p in parts)
            or bool(re.search(r'(?:execution_secret|execution_credential|exposed_secret|credentials?\.(?:json|yaml)|id_rsa)', name, re.I))
            or parts[0] in {'research', 'results'})


def prepare():
    tracked = {v.decode('utf-8') for v in git('ls-files', '-z').split(b'\0') if v}
    public = set()
    # Existing tracked evidence remains versioned, except missing paths and protected material.
    for name in tracked:
        p = ROOT / name
        if p.is_file() and ordinary(p) and not protected(name):
            public.add(name)
    for folder in ['MASKCO_code', 'CC_Compare', PREFIX + 'scripts', PREFIX + 'tools',
                   PREFIX + 'configs', PREFIX + 'data', PREFIX + 'docs',
                   PREFIX + 'comparisons/static_coldchain', PREFIX + 'results']:
        for p in walk(ROOT / folder):
            name = p.relative_to(ROOT).as_posix()
            rel = p.relative_to(EXT).as_posix() if p.is_relative_to(EXT) else ''
            if protected(name) or '/archives/' in name or '/sources/' in name:
                continue
            if folder == 'MASKCO_code':
                continue  # original first-party source is already tracked; do not publish local assets
            if p.suffix.lower() in BINARY or p.suffix.lower() == '.pdf':
                continue
            if name.startswith('CC_Compare/checkpoints/') and p.name not in {'README.md', 'ASSETS.json'}:
                continue
            if name.startswith('CC_Compare/results/') and p.name not in {'README.md', 'catalog.json'}:
                continue
            if rel.startswith('results/'):
                if rel.startswith('results/static_delivery/algorithm_dev_20261021/remote/dev11/') and p.name not in {'started.json','stage.json','resource_probe.json','job.sh'}:
                    continue  # live run logs/unsealed outputs remain in their original runtime tree
                if rel.startswith('results/workspace_sync_20261021/'):
                    if p.name not in {'README.md', 'verification.json', 'static_profit_tests.log', 'static_rrp_tests.log'}:
                        continue
                elif rel.startswith(('results/cc_compare_organization_20261021/', 'results/workspace_organization_20261021/')):
                    if p.name not in {'README.md', 'verification.json', 'migration_manifest.json', 'safe_tool_verification.json'}:
                        continue
                    if 'cc_compare_organization' in rel and p.name == 'migration_manifest.json':
                        continue
                elif rel.startswith('results/static_delivery/git_sync_20261020/'):
                    continue
                if p.suffix.lower() not in TEXT and p.name != '.gitignore':
                    continue
            if p.stat().st_size >= 90 * 2**20:
                continue
            public.add(name)
    for p in EXT.glob('*.md'):
        public.add(p.relative_to(ROOT).as_posix())
    for name in ['requirements.txt', '.gitignore']:
        if (EXT / name).is_file(): public.add(PREFIX + name)
    for p in (EXT / 'archive').glob('*/README.md'):
        public.add(p.relative_to(ROOT).as_posix())
    public.update(ROOT_FILES)
    # Deletions cover only the approved old CC tree and actual moved evidence utilities/results.
    deleted = []
    for name in tracked:
        if (ROOT / name).exists(): continue
        if name.startswith(('CC_Compare/', PREFIX + 'tools/', PREFIX + 'results/static_delivery/quality_evidence_demo')):
            deleted.append(name)
        else:
            raise RuntimeError('Unexplained tracked deletion: ' + name)
    public = sorted(public)
    deploy = set(public)
    freeze = json.loads(LOCK.read_text(encoding='utf-8'))
    for row in freeze['files']:
        p = EXT / row['path']
        if digest(p) != row['sha256']: raise RuntimeError('Frozen file changed: ' + row['path'])
        deploy.add(p.relative_to(ROOT).as_posix())
    # Only the nine selected candidate weights are deployed, never training payloads or archives.
    for p in walk(ROOT / 'CC_Compare/checkpoints'):
        deploy.add(p.relative_to(ROOT).as_posix())
    rows = {}
    SNAPSHOT.mkdir(parents=True, exist_ok=True)
    for name in sorted(deploy):
        p = ROOT / name
        data = p.read_bytes()
        sha256 = hashlib.sha256(data).hexdigest()
        blob = SNAPSHOT / sha256
        if not blob.exists(): blob.write_bytes(data)
        if digest(blob) != sha256: raise RuntimeError('Snapshot blob corruption')
        if name.startswith(PREFIX+'configs/') and name.endswith('.json'): json.loads(data)
        if name.startswith(PREFIX+'scripts/') and name.endswith('.py'): ast.parse(data.decode('utf-8'))
        rows[name] = dict(bytes=len(data), sha256=sha256, publish=name in public)
    for row in freeze['files']:
        if rows[PREFIX+row['path']]['sha256'] != row['sha256']: raise RuntimeError('Frozen source drift during snapshot')
    result = dict(schema=1, base_commit=git('rev-parse', 'HEAD').decode().strip(),
                  branch=git('branch', '--show-current').decode().strip(),
                  public_files=public, delete=sorted(deleted), deploy=rows,
                  frozen_files=len(freeze['files']), frozen_lock_sha256=digest(LOCK),
                  captured_at_utc=datetime.now(timezone.utc).isoformat(), immutable_content_snapshot=True,
                  binary_weights='nine local CC_Compare candidate checkpoints; not published',
                  exclusions='credentials/private/reparse/runtime/full recovery archives; source paper PDFs server only')
    save(PLAN, result)
    print(json.dumps(dict(public_files=len(public), delete=len(deleted), deploy_files=len(rows),
                         deploy_MiB=round(sum(r['bytes'] for r in rows.values())/2**20, 2))), flush=True)


def scan():
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    secrets = set()
    def collect(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if any(k in key.lower() for k in ['password', 'passphrase', 'token', 'api_key', 'secret']) and isinstance(item, str) and len(item) >= 8:
                    secrets.add(item.encode())
                elif isinstance(item, (dict, list)): collect(item)
        elif isinstance(value, list):
            for item in value: collect(item)
    for name in ['.vscode/sftp.json', '.vscode/sftp.3080.json']:
        collect(json.loads((ROOT / name).read_text(encoding='utf-8-sig')))
    patterns = {'github_token': rb'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})',
                'aws_access_key': rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
                'private_key': rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----',
                'credential_url': rb'https?://[^\s/:]{3,}:[^\s/@]{8,}@'}
    findings = []
    for name, row in plan['deploy'].items():
        data = (SNAPSHOT / row['sha256']).read_bytes()
        if hashlib.sha256(data).hexdigest() != row['sha256']: raise RuntimeError('Source drift: ' + name)
        if any(s in data for s in secrets): findings.append(dict(path=name, kind='local_connection_secret'))
        for kind, pattern in patterns.items():
            if re.search(pattern, data): findings.append(dict(path=name, kind=kind))
    result = dict(status='PASS' if not findings else 'FAIL', files=len(plan['deploy']),
                  credentials_checked_without_disclosure=True, findings=findings)
    save(OUT / 'secret_scan.json', result)
    print(json.dumps(result), flush=True)
    if findings: raise RuntimeError('Secret scan failed; matched values are not printed')


def deploy(label, verify_only=False):
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    client, remote = connect(label)
    stage_rel = PREFIX + 'archive/server_sync_20261021/staging'
    native = remote.lstrip('/') if PROFILES[label]['windows'] else remote
    sftp = client.open_sftp()
    def mkdirs(path):
        parts = path.rstrip('/').split('/'); current = ''
        for part in parts:
            if not part: continue
            current += '/' + part
            try:
                info = sftp.lstat(current)
                if not stat.S_ISDIR(info.st_mode): raise RuntimeError('Non-directory SFTP parent')
            except FileNotFoundError: sftp.mkdir(current)
    def invoke(action):
        runner = native + '/' + stage_rel + '/runner.py'
        code = 'import runpy,sys;sys.argv=' + repr([runner, action, '--root', native, '--stage', stage_rel]) + ';runpy.run_path(' + repr(runner) + ',run_name="__main__")'
        return execute(client, label, code, timeout=1800)
    try:
        mkdirs(remote + '/' + stage_rel)
        sftp.put(str(PLAN), remote + '/' + stage_rel + '/manifest.json')
        sftp.put(str(Path(__file__).with_name('workspace_remote_apply_20261021.py')), remote + '/' + stage_rel + '/runner.py')
        if verify_only:
            result = invoke('verify')
        else:
            inspection = invoke('inspect')
            save(OUT / ('server_' + label + '_inspection.json'), inspection)
            available = inspection['available']
            payload = OUT / ('payload_' + label + '.zip')
            seen = set(available)
            with zipfile.ZipFile(payload, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=2, allowZip64=True) as z:
                for name, row in plan['deploy'].items():
                    source = SNAPSHOT / row['sha256']
                    if digest(source) != row['sha256']: raise RuntimeError('Snapshot blob drift before transfer: ' + name)
                    if row['sha256'] not in seen:
                        z.write(source, row['sha256']); seen.add(row['sha256'])
            print(json.dumps(dict(server=label, phase='upload', needed_hashes=inspection['needed_hashes'], payload_MiB=round(payload.stat().st_size/2**20, 2))), flush=True)
            last = [time.monotonic()]
            def progress(done, total):
                if time.monotonic() - last[0] > 25:
                    print(json.dumps(dict(server=label, uploaded_MiB=round(done/2**20, 1), total_MiB=round(total/2**20, 1))), flush=True)
                    last[0] = time.monotonic()
            sftp.put(str(payload), remote + '/' + stage_rel + '/blobs.zip', callback=progress)
            print(json.dumps(dict(server=label, phase='archive_install_verify')), flush=True)
            result = invoke('apply')
        save(OUT / ('server_' + label + '_verification.json'), result)
        print(json.dumps(dict(server=label, status=result['status'], files=result['files_verified'], frozen=result['frozen_files_verified'], methods=result['methods'], recovery_moves=len(result['recovery_moves']))), flush=True)
    finally:
        sftp.close();client.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'scan', 'deploy', 'verify-server'])
    p.add_argument('--server', choices=list(PROFILES))
    args = p.parse_args()
    if args.action in {'deploy', 'verify-server'}:
        if not args.server: p.error('--server required')
        deploy(args.server, args.action == 'verify-server')
    else: {'prepare': prepare, 'scan': scan}[args.action]()
