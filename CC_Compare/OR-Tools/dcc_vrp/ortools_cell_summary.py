"""OR5/OR6：cell 汇总（从落盘实例记录重新计算，不信任旧 summary）。

发布顺序：BUILDING → instances 校验 → summary/manifest/log 原子写 →
canonical.COMPLETE（绑定全部正式产物 hash）。

seed/scene 期望值由 **DEV_MANIFEST**（expected_by_instance_id）提供，不从
实例记录自证（P0-3）。
"""
import hashlib
import json
import os
import sys
import time

_DCC_VRP = os.path.dirname(os.path.abspath(__file__))
_COMMON = os.path.normpath(os.path.join(_DCC_VRP, '..', '..', 'common'))
for p in (_DCC_VRP, _COMMON):
    if p not in sys.path:
        sys.path.insert(0, p)

import baseline_contract as bc


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def atomic_write_json(path, obj):
    """原子写 JSON；用与 hash 计算同口径的 _json_default（numpy→原生），
    保证 decision_hash/artifact_hash 落盘后重算一致。"""
    tmp = path + f'.tmp_{os.getpid()}'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=bc._json_default)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def rebuild_cell_summary(cell_dir, cell_name, expected_ids, expected_by_instance_id,
                         data_sha256, profile_hash, code_hash, checkpoint_hash):
    """从 cell_dir/instances/inst_*.json 重建 cell 汇总。

    expected_ids：该 cell 应包含的精确实例 id 集合（缺失/多出即失败）。
    expected_by_instance_id：{iid: {instance_seed, scene_instance_id}}，
        由 DEV_MANIFEST 提供（不自证）。
    checkpoint_hash：完整期望 checkpoint 身份（P0-2）。
    返回 summary dict；任何实例校验失败抛 RuntimeError。
    """
    from ortools_instance_validation import validate_instance_record

    inst_dir = os.path.join(cell_dir, 'instances')
    if not os.path.isdir(inst_dir):
        raise RuntimeError(f'{cell_dir} 无 instances 目录')
    records = {}
    for f in sorted(os.listdir(inst_dir)):
        if not f.startswith('inst_') or not f.endswith('.json'):
            continue
        iid = int(f[len('inst_'):-len('.json')])
        with open(os.path.join(inst_dir, f), encoding='utf-8') as fh:
            rec = json.load(fh)
        records[iid] = rec
    if set(records) != set(expected_ids):
        raise RuntimeError(f'实例集合不精确: 实际={sorted(records)} '
                           f'期望={sorted(expected_ids)}')

    problems_all = []
    for iid in sorted(expected_ids):
        rec = records[iid]
        exp = expected_by_instance_id[iid]
        problems = validate_instance_record(rec, {
            'instance_id': iid,
            'scene_instance_id': exp['scene_instance_id'],
            'instance_seed': exp['instance_seed'],
            'data_sha256': data_sha256,
            'profile_hash': profile_hash,
            'code_hash': code_hash,
            'checkpoint_hash': checkpoint_hash,
        })
        if problems:
            problems_all.append({'instance_id': iid, 'problems': problems})

    if problems_all:
        raise RuntimeError(f'{cell_name} 实例校验失败: {problems_all}')

    summary = {
        'cell': cell_name,
        'n_instances': len(expected_ids),
        'n_complete': sum(1 for r in records.values()
                          if r.get('outcome', {}).get('complete')),
        'protocol_errors': len(problems_all),
        'instance_problems': problems_all,
        'n_solver_calls': sum(r.get('stats', {}).get('n_solver_calls', 0)
                              for r in records.values()),
        'n_fast_path': sum(r.get('stats', {}).get('n_fast_path', 0)
                           for r in records.values()),
        'n_waits': sum(r.get('stats', {}).get('n_waits', 0)
                       for r in records.values()),
        'n_fallback_events': sum(r.get('stats', {}).get(
            'fallback_triggered_events', 0) for r in records.values()),
        'n_events_with_deferred': sum(
            (r.get('audit', {}) or {}).get('n_events_with_deferred', 0)
            for r in records.values()),
        'runtime_mean': (sum(r.get('runtime_s', 0.0) for r in records.values())
                         / len(expected_ids)),
        'instance_hashes': [{'instance_id': iid,
                             'decision_hash': records[iid].get('decision_hash'),
                             'artifact_hash': records[iid].get('artifact_hash')}
                            for iid in sorted(expected_ids)],
    }
    return summary


def publish_cell(cell_dir, cell_name, expected_ids, expected_by_instance_id,
                 data_sha256, profile_hash, code_hash, checkpoint_hash, run_id,
                 control_hash=None):
    """重建 + 原子发布 cell summary/manifest/canonical.COMPLETE。

    canonical 绑定全部正式产物：summary_sha256、manifest_sha256、
    instance_artifact_hashes（复现身份）+ instance_file_sha256（实例文件
    字节）+ run_id（P0-3：正式发布单元不可被部分替换）。manifest 绑定
    compute/control 身份（OR7 启动项 1）。

    P1-1：失败时保留 BUILDING 标记（status=FAILED + error），不暴露旧
    canonical.COMPLETE；只有新 canonical 成功原子发布后才清除 BUILDING。
    """
    marker = os.path.join(cell_dir, 'BUILDING')
    atomic_write_json(marker, {'status': 'BUILDING',
                               'started_at': time.strftime('%Y-%m-%d %H:%M:%S')})
    success = False
    try:
        summary = rebuild_cell_summary(cell_dir, cell_name, expected_ids,
                                       expected_by_instance_id, data_sha256,
                                       profile_hash, code_hash, checkpoint_hash)
        atomic_write_json(os.path.join(cell_dir, 'summary.json'), summary)
        atomic_write_json(os.path.join(cell_dir, 'manifest.json'), {
            'cell': cell_name,
            'run_id': run_id,
            'n_instances': len(expected_ids),
            'data_sha256': data_sha256,
            'profile_hash': profile_hash,
            'code_hash': code_hash,
            'control_hash': control_hash,
            'checkpoint_hash': checkpoint_hash,
            'published_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        })
        inst_dir = os.path.join(cell_dir, 'instances')
        instance_file_sha256 = {}
        for iid in sorted(expected_ids):
            instance_file_sha256[str(iid)] = sha256_file(
                os.path.join(inst_dir, f'inst_{iid}.json'))
        canonical = {
            'cell': cell_name,
            'run_id': run_id,
            'completed_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'n_instances': len(expected_ids),
            'summary_sha256': sha256_file(os.path.join(cell_dir, 'summary.json')),
            'manifest_sha256': sha256_file(os.path.join(cell_dir, 'manifest.json')),
            'instance_artifact_hashes': sorted(
                h['artifact_hash'] for h in summary['instance_hashes']),
            'instance_file_sha256': instance_file_sha256,
            'protocol_errors': summary['protocol_errors'],
        }
        atomic_write_json(os.path.join(cell_dir, 'canonical.COMPLETE'), canonical)
        success = True
        return summary
    except Exception as exc:
        # 失败：保留 BUILDING 标记为 FAILED（aggregator 会因 BUILDING 存在而拒绝）
        atomic_write_json(marker, {'status': 'FAILED', 'error': repr(exc),
                                   'failed_at': time.strftime('%Y-%m-%d %H:%M:%S')})
        raise
    finally:
        if success and os.path.exists(marker):
            os.remove(marker)
