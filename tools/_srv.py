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


def run_rc(cmd, timeout=60, config=None):
    """同 `run()`，但**同时返回远端退出码**：`(out, rc)`。

    2026-10-13 新增，起因是一次真实缺陷：`_srv.py` 原先**从不传递远端退出码**
    （CLI 里没有 `sys.exit`，永远退 0）。于是
    `CC_Compare/tools/_local_finalize.ps1` 里
    `$GATES['v1_v2_agreement'] = $LASTEXITCODE` 拿到的是 `_srv.py` 的 0，
    **即使 1003 上那道门禁 FAIL，汇总里也会显示 PASS** —— 正是「加了汇总反而给出假绿灯」。

    注意：`run()` 的返回类型**保持不变（字符串）**，因为它被 ~30 个工具
    `from _srv import run` 依赖；需要退出码的调用方请显式用本函数。
    """
    host, user, pw = _cfg(config)
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(host, port=22, username=user, password=pw, timeout=30)
    i, o, e = c.exec_command(cmd, timeout=timeout)
    out = o.read().decode('utf-8', 'replace') + e.read().decode('utf-8', 'replace')
    rc = o.channel.recv_exit_status()      # 输出已读完 ⇒ 命令已结束，不会阻塞
    c.close()
    return out, rc


if __name__ == '__main__':
    # usage: python tools/_srv.py '<remote command>' [timeout] [config-json]
    # config-json lets the same helper target either server, e.g. .vscode/sftp.3080.json
    # 退出码 = 远端命令的退出码（见 run_rc 的说明）。
    out, rc = run_rc(sys.argv[1],
                     timeout=int(sys.argv[2]) if len(sys.argv) > 2 else 120,
                     config=sys.argv[3] if len(sys.argv) > 3 else None)
    print(out)
    sys.exit(rc)
