"""本地 SSH 助手：从 .vscode/sftp.json 读服务器凭据，跑一条远端命令。

用法：python tools/_srv.py '<remote command>'（从仓库根目录执行）
避免把密码明文写进命令行（credential leakage）。
"""
import json
import os
import sys

import paramiko


def _cfg(config=None):
    if config:
        p = config if os.path.isabs(config) else os.path.join(
            os.path.dirname(os.path.abspath(__file__)), '..', config)
    else:
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.vscode', 'sftp.json')
    with open(p) as f:
        d = json.load(f)
    return d['host'], d['username'], d['password']


def run(cmd, timeout=60, config=None):
    host, user, pw = _cfg(config)
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host, port=22, username=user, password=pw, timeout=30)
    i, o, e = c.exec_command(cmd, timeout=timeout)
    out = o.read().decode('utf-8', 'replace') + e.read().decode('utf-8', 'replace')
    c.close()
    return out


if __name__ == '__main__':
    # usage: python tools/_srv.py '<remote command>' [timeout] [config-json]
    # config-json lets the same helper target either server, e.g. .vscode/sftp.3080.json
    print(run(sys.argv[1],
              timeout=int(sys.argv[2]) if len(sys.argv) > 2 else 120,
              config=sys.argv[3] if len(sys.argv) > 3 else None))
