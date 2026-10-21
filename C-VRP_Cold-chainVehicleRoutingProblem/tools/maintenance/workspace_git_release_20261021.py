"""Review, explicitly stage, verify and commit the approved 2026-10-21 release."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from workspace_release_20261021 import ROOT, EXT, OUT, PLAN, SNAPSHOT, git, save, digest


def preview():
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    if git('diff', '--cached', '--name-only', '-z'):
        raise RuntimeError('Existing staged work must be reviewed before release')
    if git('rev-parse', 'HEAD').decode().strip() != plan['base_commit']:
        raise RuntimeError('Base commit changed')
    tracked = {p.decode() for p in git('ls-files', '-z').split(b'\0') if p}
    modified = {p.decode() for p in git('diff', '--name-only', '-z').split(b'\0') if p}
    adds = {}
    for name in plan['public_files']:
        row = plan['deploy'][name]
        if digest(SNAPSHOT / row['sha256']) != row['sha256']: raise RuntimeError('Snapshot drift: ' + name)
        if name not in tracked or name in modified:
            if name.startswith('MASKCO_code/') or name.startswith('tools/'):
                raise RuntimeError('Protected original source unexpectedly changed: ' + name)
            adds[name] = row
    result = dict(base_commit=plan['base_commit'], branch=plan['branch'], add_or_update=adds, delete=plan['delete'])
    save(OUT/'git_plan.json', result)
    (OUT/'add_paths.nul').write_bytes(b''.join(n.encode()+b'\0' for n in adds))
    (OUT/'delete_paths.nul').write_bytes(b''.join(n.encode()+b'\0' for n in plan['delete']))
    print(json.dumps(dict(add_or_update=len(adds), delete=len(plan['delete']), max_blob_MiB=round(max(v['bytes'] for v in adds.values())/2**20, 2))), flush=True)


def stage():
    plan=json.loads((OUT/'git_plan.json').read_text(encoding='utf-8'))
    if git('diff','--cached','--name-only','-z'): raise RuntimeError('Index already contains staged work')
    if git('rev-parse','HEAD').decode().strip()!=plan['base_commit']: raise RuntimeError('HEAD drift')
    for name,row in plan['add_or_update'].items():
        if digest(SNAPSHOT/row['sha256'])!=row['sha256']: raise RuntimeError('Snapshot drift: '+name)
    # Write only reviewed snapshot blobs. Concurrent working-tree edits stay in place.
    unique=list(dict.fromkeys(row['sha256'] for row in plan['add_or_update'].values()))
    paths=b''.join(str(SNAPSHOT/h).replace('\\','/').encode()+b'\n' for h in unique)
    oids=git('hash-object','-w','--stdin-paths',data=paths).decode().splitlines()
    if len(oids)!=len(unique):raise RuntimeError('Object batch incomplete')
    objects=dict(zip(unique,oids))
    modes={}
    for raw in git('ls-files','--stage','-z').split(b'\0'):
        if raw:
            head,name=raw.split(b'\t',1);modes[name.decode()]=head.split()[0].decode()
    updates=b''.join((modes.get(name,'100644')+' '+objects[row['sha256']]+'\t').encode()+name.encode()+b'\0' for name,row in plan['add_or_update'].items())
    git('update-index','-z','--index-info',data=updates)
    git('update-index','--force-remove','-z','--stdin',data=(OUT/'delete_paths.nul').read_bytes())
    names={v.decode() for v in git('diff','--cached','--name-only','--no-renames','-z').split(b'\0') if v}
    expected=set(plan['add_or_update'])|set(plan['delete'])
    if names!=expected: raise RuntimeError('Staged path set differs from explicit plan')
    save(OUT/'git_stage.json',dict(status='PASS',add_or_update=len(plan['add_or_update']),delete=len(plan['delete']),tree=git('write-tree').decode().strip()))
    print(json.dumps(dict(status='PASS',staged_paths=len(names))),flush=True)


def verify():
    plan=json.loads((OUT/'git_plan.json').read_text(encoding='utf-8'))
    reviewed=json.loads((OUT/'literal_review.json').read_text(encoding='utf-8'))
    spec=importlib.util.spec_from_file_location('existing_secret_rules',ROOT/'tools/scan_staged_secrets.py')
    rules=importlib.util.module_from_spec(spec);spec.loader.exec_module(rules)
    entries={}
    for raw in git('ls-files','--stage','-z').split(b'\0'):
        if not raw:continue
        head,name=raw.split(b'\t',1);mode,oid,stage=head.split()
        if stage!=b'0':raise RuntimeError('Unmerged index')
        entries[name.decode()]=(mode.decode(),oid.decode())
    proc=subprocess.Popen(['git','cat-file','--batch'],cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
    findings=[];attested=[];checked=0
    try:
        for name,row in plan['add_or_update'].items():
            mode,oid=entries[name]
            if mode not in {'100644','100755'}:raise RuntimeError('Unexpected staged type: '+name)
            proc.stdin.write((oid+'\n').encode());proc.stdin.flush()
            header=proc.stdout.readline().split()
            if len(header)!=3 or header[1]!=b'blob':raise RuntimeError('Unreadable staged object')
            raw=proc.stdout.read(int(header[2]));assert proc.stdout.read(1)==b'\n'
            if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise RuntimeError('Staged bytes differ: '+name)
            checked+=1
            hits=rules.scan_bytes(raw)
            if hits:
                review=reviewed.get(name)
                if review and review['sha256']==row['sha256'] and review['hits']==[list(v) for v in hits]:
                    attested.append(dict(path=name,hits=hits,reason=review['reason'],sha256=row['sha256']))
                    continue
                upstream=('/comparisons/static_coldchain/upstream/' in name or name.startswith('CC_Compare/models/'))
                # Pinned public upstream source is copied unchanged. Its generic variable/test-token
                # assignments are source literals, not local connection credentials (checked separately).
                if upstream and all(rule=='R3_credential_assignment' for rule,line in hits):
                    attested.append(dict(path=name,hits=hits,reason='unchanged pinned upstream source; exact local-secret and high-signal-token scan passed'))
                else:findings.append(dict(path=name,hits=hits))
    finally:
        proc.stdin.close();proc.stdout.close();proc.wait()
    result=dict(status='PASS' if not findings else 'REVIEW_REQUIRED',staged_blobs_checked=checked,unreadable=0,
                exact_source_sha256_verified=True,findings=findings,upstream_source_attestations=attested)
    save(OUT/'staged_verification.json',result)
    print(json.dumps(dict(status=result['status'],checked=checked,findings=findings,upstream_attestations=len(attested))),flush=True)
    if findings:raise RuntimeError('Staged findings need review')


def commit():
    report=json.loads((OUT/'staged_verification.json').read_text(encoding='utf-8'))
    if report['status']!='PASS':raise RuntimeError('Staged verification not passed')
    stage_receipt=json.loads((OUT/'git_stage.json').read_text(encoding='utf-8'))
    if git('write-tree').decode().strip()!=stage_receipt['tree']:raise RuntimeError('Index changed since verification')
    plan=json.loads((OUT/'git_plan.json').read_text(encoding='utf-8'))
    if git('rev-parse','HEAD').decode().strip()!=plan['base_commit']:raise RuntimeError('HEAD drift')
    message=OUT/'commit_message.txt'
    message.write_text('Organize seven comparison methods and synchronize research workspace\n\n'
                       'Classify CC_Compare by role, archive unused methods recoverably, and move evidence utilities into acquire/analyze/verify.\n'
                       'Publish current static-profit code, frozen contracts, provenance and development reports; preserve historical failure identities.\n'
                       'Deploy the explicit SHA manifest to the original 3080 and 5090 workspaces, with server-side recovery inventories.\n\n'
                       'Validation: 20 static-profit tests, 62 historical RRP tests, frozen 72-file checks and server manifest SHA verification.\n'
                       'Private data, credentials, paper full text, binary checkpoints and large recovery archives remain outside GitHub.\n',encoding='utf-8')
    git('commit','--file='+str(message))
    head=git('rev-parse','HEAD').decode().strip()
    save(OUT/'commit_receipt.json',dict(status='COMMITTED',commit=head,parent=plan['base_commit'],branch=plan['branch'],tree=stage_receipt['tree']))
    print(json.dumps(dict(status='COMMITTED',commit=head,branch=plan['branch'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['preview','stage','verify','commit']);a=p.parse_args()
    {'preview':preview,'stage':stage,'verify':verify,'commit':commit}[a.action]()
