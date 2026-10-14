"""本地 SFTP 上传助手：把本地文件上传到服务器对应路径。

远程根目录取自所用 sftp 配置的 `remotePath`（不再硬编码），因此同一套工具可以指向
任意一台已配置的服务器。只传「本地相对路径」（相对 workspace 根），
远程路径自动 = remotePath + 本地相对路径。

用法（从仓库根目录执行）：
    python tools/_sftp_put.py <local_rel> [<local_rel> ...] [--config .vscode/sftp.3080.json]

注意：Windows 服务器的 SFTP 根形如 /C:/Users/zh/project/MASKCO-Main/，
把绝对远端路径当命令行参数传给 shell 会被 Git Bash / PowerShell 转义，所以这里只接受相对路径。
"""
import json
import os
import sys

import paramiko


def config_path(argv):
    cfg = None
    rest = []
    i = 0
    while i < len(argv):
        if argv[i] == '--config':
            cfg = argv[i + 1]
            i += 2
            continue
        rest.append(argv[i])
        i += 1
    if cfg is None:
        cfg = os.path.join('.vscode', 'sftp.json')
    if not os.path.isabs(cfg):
        cfg = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', cfg)
    return cfg, rest


def _rel(rel):
    """把 Windows 风格的反斜杠统一成 posix 分隔符。

    2026-10-13 事故（缺陷 M）：`remote = remote_base + rel` 若 `rel` 是
    `CC_Compare\\tools\\x.sh`（PowerShell 里很自然），远端就会**新建一个名字里带反斜杠的
    单文件** `CC_Compare\\tools\\x.sh`，而真正的 `CC_Compare/tools/x.sh` 根本没被写。
    上传「看起来成功」，目标文件却是旧的 —— 静默失效。故一律先归一化。
    """
    return rel.replace('\\', '/').lstrip('/')


def _cfg(path):
    with open(path) as f:
        d = json.load(f)
    base = d.get('remotePath')
    if not base:
        raise SystemExit('config %s has no remotePath' % path)
    if not base.endswith('/'):
        base += '/'
    return d['host'], d['username'], d['password'], base


def main():
    cfg, rels = config_path(sys.argv[1:])
    assert rels, ('usage: _sftp_put.py <local_rel> ... '
                  '[--config .vscode/sftp.3080.json]')
    host, user, pw, remote_base = _cfg(cfg)
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    t = paramiko.Transport((host, 22))
    t.connect(username=user, password=pw)
    sftp = paramiko.SFTPClient.from_transport(t)
    for rel in rels:
        rel = _rel(rel)
        local = os.path.join(base, *rel.split('/'))
        remote = remote_base + rel
        d = os.path.dirname(remote)
        if d:
            try:
                sftp.stat(d)
            except IOError:
                sftp.mkdir(d)          # windows openssh does not create parents implicitly
        sftp.put(local, remote)
        print('  uploaded %s -> %s' % (rel, remote))
    sftp.close()
    t.close()


if __name__ == '__main__':
    main()
