"""本地 SFTP 上传助手：把本地文件上传到服务器对应路径。

远程根目录硬编码为 sftp.json 的 remotePath；只传「本地相对路径」（相对 workspace 根），
远程路径自动 = REMOTE_BASE + 本地相对路径。避免把 /home/... 绝对路径当命令行参数（Git Bash 会转义）。

用法：python _sftp_put.py <local_rel> [<local_rel> ...]
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
    assert rels, 'usage: _sftp_put.py <local_rel> ...'
    host, user, pw = _cfg()
    base = os.path.dirname(os.path.abspath(__file__))
    t = paramiko.Transport((host, 22))
    t.connect(username=user, password=pw)
    sftp = paramiko.SFTPClient.from_transport(t)
    for rel in rels:
        local = os.path.join(base, *rel.split('/'))
        remote = REMOTE_BASE + rel
        sftp.put(local, remote)
        print(f'  uploaded {rel}')
    sftp.close()
    t.close()


if __name__ == '__main__':
    main()
