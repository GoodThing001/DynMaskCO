"""本地 SFTP 下载助手：从服务器下载文件到本地对应路径。

远程根目录取自所用 sftp 配置的 `remotePath`（不再硬编码），因此同一套工具可以指向
任意一台已配置的服务器。remote 与 local 都用「相对路径」。

用法（从仓库根目录执行）：
    python tools/_sftp_get.py <remote_rel> [<remote_rel> ...] [--config .vscode/sftp.3080.json]
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
    """同 `_sftp_put.py::_rel`：反斜杠 → `/` 并去掉前导 `/`。

    缺陷 M（2026-10-13）：不归一化时 `remote_base + rel` 会去找一个名字里带反斜杠的文件
    （几乎必然不存在），于是下载恒失败；若两端恰好都存在则更糟——写错位置还自认成功。
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
    assert rels, ('usage: _sftp_get.py <remote_rel> ... '
                  '[--config .vscode/sftp.3080.json]')
    host, user, pw, remote_base = _cfg(cfg)
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
    t = paramiko.Transport((host, 22))
    t.connect(username=user, password=pw)
    sftp = paramiko.SFTPClient.from_transport(t)
    for rel in rels:
        rel = _rel(rel)
        remote = remote_base + rel
        local = os.path.join(base, *rel.split('/'))
        # 2026-10-13 加固（起因是一次真实事故，见 CC_Compare/README §13.3 缺陷 I 的后续）：
        # `sftp.get()` 在**远端不存在**时会**先创建/截断本地文件**再失败，留下一个 **0 字节**的
        # 本地文件。那个 0 字节文件随后被 Copy-Item + _sftp_put.py 一路带上去，
        # **覆盖掉目标端原本正确的内容**（本批曾把 1003 上 7 个 gate 打成 0 字节）。
        # 因此这里：①先 `stat` 远端，不存在就直接报错、**不碰本地**（连目录都不建）；
        # ②下载后再校验本地非空；③任何失败都**清掉残留的本地文件**，绝不留 0 字节假文件。
        try:
            rsize = sftp.stat(remote).st_size
        except IOError:
            print('  FAIL %s：远端不存在 <%s>（本地未改动）' % (rel, remote))
            sys.exit(2)
        # 只有远端确认存在，才创建本地目录（避免失败时留下空目录）
        os.makedirs(os.path.dirname(local), exist_ok=True)
        try:
            sftp.get(remote, local)
        except Exception as e:                                   # noqa: BLE001
            if os.path.exists(local) and os.path.getsize(local) == 0:
                os.remove(local)
            print('  FAIL %s：下载异常 %s（已清理残留）' % (rel, e))
            sys.exit(3)
        lsize = os.path.getsize(local) if os.path.exists(local) else -1
        if lsize != rsize:
            print('  FAIL %s：大小不符 local=%s remote=%s' % (rel, lsize, rsize))
            sys.exit(4)
        print('  downloaded %s <- %s (%d B)' % (rel, remote, lsize))
    sftp.close()
    t.close()


if __name__ == '__main__':
    main()
