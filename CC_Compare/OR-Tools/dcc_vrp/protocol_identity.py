"""OR6/OR7 三层身份（compute/control/analysis）与 protocol_id 的共享定义。

三层身份（OR7 启动项 1）：
  compute   共同计算链 + adapter/builder/mapper + OR native identity
            （= strict_online_runner.code_hash 口径）
  control   run_devproto / ortools_instance_validation / ortools_cell_summary
  analysis  aggregate_devproto / parity_compare / OR7 预算选择脚本

protocol_id 绑定三层 hash + data manifest + profile + config。两次独立运行的
protocol_id 必须相同；run_id 则每次唯一。
"""
import hashlib
import json
import os

CONTROL_FILES = ('run_devproto.py', 'ortools_instance_validation.py',
                 'ortools_cell_summary.py', 'protocol_identity.py')
ANALYSIS_FILES = ('aggregate_devproto.py', 'parity_compare.py',
                  'or7_selector.py')


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def layer_hash(prefix, files, base_dir):
    """按逻辑相对路径（prefix/name）排序组合「逻辑路径 + 文件 SHA-256」。"""
    h = hashlib.sha256()
    for name in sorted(files):
        p = os.path.join(base_dir, name)
        h.update(f'{prefix}/{name}'.encode())
        h.update(b'\x00')
        h.update(sha256_file(p).encode('ascii'))
        h.update(b'\x00')
    return h.hexdigest()


def build_protocol_id(compute_hash, control_hash, analysis_hash,
                      dev_manifest_sha256, profile_hash, effective_config):
    """确定性 protocol_id：绑定三层代码身份 + data + profile + config。"""
    payload = json.dumps({
        'compute_hash': compute_hash,
        'control_hash': control_hash,
        'analysis_hash': analysis_hash,
        'dev_manifest_sha256': dev_manifest_sha256,
        'profile_hash': profile_hash,
        'config': effective_config,
    }, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()
