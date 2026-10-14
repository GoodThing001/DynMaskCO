"""Mirror this document reorganization without touching any experiment artifacts."""
from pathlib import Path
import hashlib
import json
import posixpath
import stat

import paramiko

from _sftp_put import _cfg


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    workspace = Path(__file__).resolve().parents[1]
    root = workspace / 'C-VRP_Cold-chainVehicleRoutingProblem'
    record_dir = root / 'docs/目录整理记录/2026-10-08'
    manifest = json.loads((record_dir / '迁移清单.json').read_text(encoding='utf-8'))
    files = {x['new']: None for x in manifest['moves']}
    for x in manifest['moves']:
        files[x['old']] = x['original_sha256']
    files['00_项目导航.md'] = None
    for line in ['01_取货线', '02_配送线', '03_负结果线']:
        for p in (root / line).glob('*.md'):
            files[p.relative_to(root).as_posix()] = None
    for p in record_dir.glob('*.json'):
        if p.name != '服务器文档同步.json':
            files[p.relative_to(root).as_posix()] = None
    remote_root = '/home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem'
    host, username, password = _cfg()
    transport = paramiko.Transport((host, 22))
    transport.connect(username=username, password=password)
    sftp = paramiko.SFTPClient.from_transport(transport)
    uploaded, identical, conflicts = [], [], []

    def ensure_directory(path):
        if path == remote_root:
            return
        if not path.startswith(remote_root + '/'):
            raise ValueError('Remote path outside the extension')
        try:
            if not stat.S_ISDIR(sftp.stat(path).st_mode):
                raise ValueError('Remote parent is not a directory: ' + path)
        except FileNotFoundError:
            ensure_directory(posixpath.dirname(path))
            sftp.mkdir(path)

    try:
        for relative, expected_original in files.items():
            local = (root / relative).resolve()
            if root.resolve() not in local.parents:
                raise ValueError('Local path outside the extension')
            remote = remote_root + '/' + relative
            data = local.read_bytes()
            try:
                with sftp.open(remote, 'rb') as stream:
                    existing_hash = digest(stream.read())
            except FileNotFoundError:
                existing_hash = None
            if existing_hash == digest(data):
                identical.append(relative)
                continue
            if existing_hash is not None and existing_hash != expected_original:
                conflicts.append({'path': relative, 'remote_sha256': existing_hash,
                                  'reason': 'Unexpected remote document; left unchanged'})
                continue
            ensure_directory(posixpath.dirname(remote))
            with sftp.open(remote, 'wb') as stream:
                stream.write(data)
            with sftp.open(remote, 'rb') as stream:
                assert digest(stream.read()) == digest(data)
            uploaded.append(relative)
        report = {'uploaded': uploaded, 'already_identical': identical, 'conflicts': conflicts,
                  'runtime_or_frozen_artifacts_modified': False,
                  'shared_status_documents_not_overwritten': True,
                  'windows_shortcuts_local_only': True}
        (record_dir / '服务器文档同步.json').write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps({'uploaded': len(uploaded), 'identical': len(identical),
                          'conflicts': conflicts}, ensure_ascii=False))
    finally:
        sftp.close()
        transport.close()


if __name__ == '__main__':
    main()
