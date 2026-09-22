"""本地 SFTP 下载助手：从服务器下载文件到本地对应路径（remote 相对 REMOTE_BASE，local 相对 workspace 根）。

用法：python _sftp_get.py <remote_rel> [<remote_rel> ...]
"""
import json
import os
import sys

import paramiko

REMOTE_BASE = '/home/hzeng/project/MASKCO-Main/'


def _cfg():
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.vscode', 'sftp.json')
    with open(p) as f:
        d = json.load(f)
    return d['host'], d['username'], d['password']


def main():
    rels = sys.argv[1:]
    assert rels, 'usage: _sftp_get.py <remote_rel> ...'
    host, user, pw = _cfg()
    base = os.path.dirname(os.path.abspath(__file__))
    t = paramiko.Transport((host, 22))
    t.connect(username=user, password=pw)
    sftp = paramiko.SFTPClient.from_transport(t)
    for rel in rels:
        remote = REMOTE_BASE + rel
        local = os.path.join(base, *rel.split('/'))
        os.makedirs(os.path.dirname(local), exist_ok=True)
        sftp.get(remote, local)
        print(f'  downloaded {rel}')
    sftp.close()
    t.close()


if __name__ == '__main__':
    main()
