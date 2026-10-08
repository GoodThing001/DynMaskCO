# -*- coding: utf-8 -*-
"""Sync docs/当前规划 (reorganized) to server: delete server tree, mirror local tree, verify.

Also syncs 项目当前状态.md + root CLAUDE.md/AGENTS.md.
"""
import json
import os
import stat

import paramiko

WS = r"D:\PyCharm_\MASKCO-Main"
CFG = os.path.join(WS, ".vscode", "sftp.json")
REMOTE_BASE = "/home/hzeng/project/MASKCO-Main/"
LOCAL_DOCS = os.path.join(WS, "C-VRP_Cold-chainVehicleRoutingProblem", "docs", "当前规划")
REMOTE_DOCS = REMOTE_BASE + "C-VRP_Cold-chainVehicleRoutingProblem/docs/当前规划"

d = json.load(open(CFG, encoding="utf-8"))
host, user, pw = d["host"], d["username"], d["password"]

t = paramiko.Transport((host, 22))
t.connect(username=user, password=pw)
sftp = paramiko.SFTPClient.from_transport(t)
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(host, port=22, username=user, password=pw, timeout=30)


def list_remote_files(base):
    out = []

    def walk(rd):
        for e in sftp.listdir_attr(rd):
            p = rd + "/" + e.filename
            if stat.S_ISDIR(e.st_mode):
                walk(p)
            else:
                out.append(p[len(base):].lstrip("/"))

    walk(base)
    return sorted(out)


def mkdir_p(remote_dir):
    parts = [p for p in remote_dir.split("/") if p]
    cur = ""
    for p in parts:
        cur += "/" + p
        try:
            sftp.stat(cur)
        except IOError:
            sftp.mkdir(cur)


# local file list (relative to LOCAL_DOCS)
local_files = []
local_dirs = set()
for root, dirs, files in os.walk(LOCAL_DOCS):
    for f in files:
        p = os.path.join(root, f)
        rel = os.path.relpath(p, LOCAL_DOCS).replace("\\", "/")
        local_files.append(rel)
        local_dirs.add(os.path.dirname(rel))
local_files.sort()

# 1. snapshot server tree
server_files = list_remote_files(REMOTE_DOCS)
extra = [f for f in server_files if f not in local_files]
print("server files before:", len(server_files), "| local files:", len(local_files))
print("server-only (to delete):", extra if extra else "none")

# 2. delete server tree
_stdin, stdout, stderr = c.exec_command("rm -rf " + REMOTE_DOCS)
stdout.read()
print("deleted server docs tree")

# 3. mirror local tree
mkdir_p(REMOTE_DOCS)
for d in sorted(local_dirs):
    if d:
        mkdir_p(REMOTE_DOCS + "/" + d)
for rel in local_files:
    sftp.put(os.path.join(LOCAL_DOCS, rel), REMOTE_DOCS + "/" + rel)
print("uploaded", len(local_files), "files")

# 4. extra root-level files
for rel in ["C-VRP_Cold-chainVehicleRoutingProblem/项目当前状态.md"]:
    sftp.put(os.path.join(WS, rel.replace("/", "\\")), REMOTE_BASE + rel)
for rel in ["CLAUDE.md", "AGENTS.md"]:
    sftp.put(os.path.join(WS, rel), REMOTE_BASE + rel)
print("synced 项目当前状态.md + root CLAUDE.md/AGENTS.md")

# 5. verify
server_files2 = list_remote_files(REMOTE_DOCS)
missing = [f for f in local_files if f not in server_files2]
extra2 = [f for f in server_files2 if f not in local_files]
print("after sync server files:", len(server_files2))
print("missing on server:", missing if missing else "none")
print("extra on server:", extra2 if extra2 else "none")
print("MATCH" if not missing and not extra2 else "MISMATCH")

sftp.close()
t.close()
c.close()
