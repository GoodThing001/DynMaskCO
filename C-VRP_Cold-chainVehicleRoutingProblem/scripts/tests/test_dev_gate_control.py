"""DEV-GATE 运行控制加固回归测试（纯 NumPy，不跑 oracle、不触服务器）。

覆盖上一轮审计列出的缺口：
  1. 严格实例验收（instance_validation）：缺 outcome / error / service fail / 缺 local /
     inst_idx 非严格整数 / 非退化违反 → 分类正确；
  2. 规范汇总重建（cell_summary）：部分补算后从全部原始实例重建，与一次跑完一致；
     实例写完、汇总未写完时能恢复；
  3. local 九 cell 等权：不对称 eligibility 的反例得 -0.05（非旧逻辑 -0.45）；
     单 eligible seed / 未采集 local → 报告不可估计，不静默删 cell / 填零；
  4. 聚合器身份：九份同时缺 code_sha256+profile_hash 拒绝；--profile 内容重算 hash 与
     manifest 不符拒绝；
  5. 版本分离：COMPUTE_FILES 含 authoritative_evaluator.py、不含 control/analysis 文件。

用法：python scripts/tests/test_dev_gate_control.py
产物：results/o0cc/dev_gate_control_tests.json
"""
import os, sys, json, subprocess, shutil, tempfile

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # scripts
_CVRPTW = os.path.dirname(_BASE)                                      # C-VRP root
_EVAL = os.path.join(_CVRPTW, 'scripts', 'evaluation')
for p in (_EVAL, _BASE):
    if p not in sys.path:
        sys.path.insert(0, p)

from instance_validation import validate_instance_record
from cell_summary import rebuild_cell_summary
from aggregate_cross_cell import _local_nine_cell_equal_weight, _recompute_profile_hash
import run_dev_gate

RESULTS = []


def record(name, ok, detail=''):
    RESULTS.append({'test': name, 'status': 'PASS' if ok else 'FAIL', 'detail': str(detail)})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


# --------------------------------------------------------------------------- #
# 合成 coldchain outcome / record / local
# --------------------------------------------------------------------------- #
def _outcome(cost=10.0, complete=True):
    return {
        'complete': complete, 'n_unserved': 0, 'n_duplicate': 0,
        'tw_feasible': True, 'capacity_feasible': True, 'depot_return_feasible': True,
        'distance_cost': cost, 'distance_km': cost, 'quality_loss': 1.0,
        'energy_kwh': 5.0, 'coldchain_cost': cost,
        'temperature_hard_feasible': True, 'all_orders_picked': True,
        'all_cargo_delivered_to_depot': True, 'terminal_manifests_empty': True,
        'trace_accounting_consistent': True, 'distance_accounting_consistent': True,
        'ownership_violations': 0, 'terminal_unresolved': 0,
        'num_unsalable': 0, 'thermal_violation_count': 0, 'thermal_violation_duration_h': 0.0,
    }


def _local(deltas):
    return {'event_deltas': list(deltas),
            'records': [{'event_id': i, 'n_customers': 2, 'n_customer_deltas': 2,
                         'mean_customer_delta': d}
                        for i, d in enumerate(deltas)],
            'base_cost': 11.0}


def _record(idx=0, o_cost=10.0, b_cost=11.0):
    return {'inst_idx': idx, 'baseline': _outcome(b_cost), 'oracle': _outcome(o_cost),
            'local': _local([-0.1, -0.2]), 'log': [], 'runtime': 1.0, 'error': None}


def test_validate_states():
    ok = True
    ok &= validate_instance_record(_record(), 0)[0] == 'valid'
    r = _record(); del r['oracle']
    ok &= validate_instance_record(r, 0)[0] == 'protocol_fail'
    r = _record(); r['error'] = 'boom'
    ok &= validate_instance_record(r, 0)[0] == 'error'
    r = _record(); r['baseline']['complete'] = False
    ok &= validate_instance_record(r, 0)[0] == 'service_fail'
    r = _record(); r['inst_idx'] = 0.0
    ok &= validate_instance_record(r, 0)[0] == 'protocol_fail'
    r = _record(); r['inst_idx'] = '0'
    ok &= validate_instance_record(r, 0)[0] == 'protocol_fail'
    r = _record(); r['inst_idx'] = 5
    ok &= validate_instance_record(r, 0)[0] == 'protocol_fail'
    r = _record(o_cost=20.0, b_cost=10.0)  # oracle 更差
    ok &= validate_instance_record(r, 0)[0] == 'protocol_fail'
    r = _record(); del r['local']
    ok &= validate_instance_record(r, 0, require_local=True)[0] == 'missing_local'
    ok &= validate_instance_record(r, 0, require_local=False)[0] == 'valid'
    r = _record(); r['local'] = {'event_deltas': [-0.1], 'records': []}  # 对账不一致
    ok &= validate_instance_record(r, 0)[0] == 'protocol_fail'
    r = _record(); r['code_sha256'] = 'abc'
    ok &= validate_instance_record(r, 0, expected_code_sha256='abc')[0] == 'valid'
    ok &= validate_instance_record(r, 0, expected_code_sha256='xyz')[0] == 'protocol_fail'
    r = _record()
    ok &= validate_instance_record(r, 0, expected_code_sha256='abc')[0] == 'protocol_fail'
    r = _record(); r['local']['event_deltas'][0] = -0.999  # 改 event_deltas 不改记录均值
    ok &= validate_instance_record(r, 0)[0] == 'protocol_fail'
    record('validate_instance_states', ok)
    return ok


def test_code_sha256_mixed_rejected():
    src = os.path.join(_CVRPTW, 'results', 'o0cc', 'r5_cross_r1_02')
    tmp = tempfile.mkdtemp(prefix='devgate_codesha_')
    try:
        shutil.copytree(src, os.path.join(tmp, 'cell'))
        cell = os.path.join(tmp, 'cell')
        insts = sorted(f for f in os.listdir(os.path.join(cell, 'instances'))
                       if f.startswith('inst_') and f.endswith('.json'))
        for i, f in enumerate(insts):
            p = os.path.join(cell, 'instances', f)
            j = json.load(open(p))
            j['code_sha256'] = 'deadbeef' if i == 0 else 'c0ffee'  # 混版本
            json.dump(j, open(p, 'w'))
        try:
            rebuild_cell_summary(cell, objective='coldchain', role='regression',
                                 profile_manifest={'profile_hash': 'x'}, data_sha256='d',
                                 expected_code_sha256='c0ffee', require_local=True)
            ok = False
        except RuntimeError:
            ok = True
        record('code_sha256_mixed_rejected', ok)
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_rebuild_canonical_and_recovery():
    src = os.path.join(_CVRPTW, 'results', 'o0cc', 'r5_cross_r1_02')
    tmp = tempfile.mkdtemp(prefix='devgate_rebuild_')
    try:
        shutil.copytree(src, os.path.join(tmp, 'cell'))
        cell = os.path.join(tmp, 'cell')
        # 手动期望值
        insts = sorted(f for f in os.listdir(os.path.join(cell, 'instances'))
                       if f.startswith('inst_') and f.endswith('.json'))
        deltas = []
        for f in insts:
            p = os.path.join(cell, 'instances', f)
            j = json.load(open(p))
            j['code_sha256'] = 'c0ffee'  # 模拟新 runner 在实例里写入启动 hash
            json.dump(j, open(p, 'w'))
            deltas.append(j['oracle']['coldchain_cost'] - j['baseline']['coldchain_cost'])
        expect_mean = sum(deltas) / len(deltas)

        rebuild_cell_summary(cell, objective='coldchain', role='regression',
                             profile_manifest={'profile_hash': 'x'}, data_sha256='d',
                             expected_code_sha256='c0ffee', require_local=True)
        s = json.load(open(os.path.join(cell, 'summary.json')))
        m = json.load(open(os.path.join(cell, 'manifest.json')))
        ok = (s['n'] == len(insts) and abs(s['mean_delta'] - expect_mean) < 1e-9
              and m['code_sha256'] == 'c0ffee' and m['n_instances'] == len(insts))

        # 模拟「实例写完、汇总未写完」：删汇总文件，重建应恢复
        for f in ('summary.json', 'manifest.json', 'per_instance.csv', 'oracle_log.jsonl'):
            p = os.path.join(cell, f)
            if os.path.exists(p):
                os.remove(p)
        rebuild_cell_summary(cell, objective='coldchain', role='regression',
                             profile_manifest={'profile_hash': 'x'}, data_sha256='d',
                             expected_code_sha256='c0ffee', require_local=True)
        s2 = json.load(open(os.path.join(cell, 'summary.json')))
        ok = ok and s2['n'] == len(insts) and abs(s2['mean_delta'] - expect_mean) < 1e-9
        record('rebuild_canonical_and_recovery', ok, f'n={len(insts)} mean={expect_mean:+.4f}')
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_local_equal_weight_counterexample():
    cells = ['r1_02', 'c1_02', 'rc1_02', 'r1_05', 'c1_05', 'rc1_05', 'r1_08', 'c1_08', 'rc1_08']
    # seed 1 只有 r1_02 eligible(-0.9)，其余 no_eligible；seed 2 九 cell 全 eligible(0)
    matrix = {
        1: {c: ('eligible', -0.9) if c == 'r1_02' else ('no_eligible', None) for c in cells},
        2: {c: ('eligible', 0.0) for c in cells},
    }
    stat = _local_nine_cell_equal_weight(matrix, [1, 2], cells, n_boot=2000, seed=42)
    ok = (stat['mean'] is not None and abs(stat['mean'] - (-0.05)) < 1e-9)
    record('local_equal_weight_counterexample', ok,
           f"mean={stat['mean']:.4f}（期望 -0.05，旧逻辑 -0.45）")
    return ok


def test_local_non_estimable_and_missing():
    cells = ['r1_02', 'c1_02', 'rc1_02', 'r1_05', 'c1_05', 'rc1_05', 'r1_08', 'c1_08', 'rc1_08']
    # 单 eligible seed（仅 seed1 在 r1_02 有值）→ 8 个 cell 无 eligible → 不可估计
    m1 = {1: {c: ('eligible', -0.9) if c == 'r1_02' else ('no_eligible', None) for c in cells},
          2: {c: ('no_eligible', None) for c in cells}}
    s1 = _local_nine_cell_equal_weight(m1, [1, 2], cells, n_boot=500, seed=42)
    ok = (s1['mean'] is None and len(s1['non_estimable_cells']) == 8)
    # 未采集 local（全部 missing）→ 计数 18、不可估计
    m2 = {s: {c: ('missing', None) for c in cells} for s in (1, 2)}
    s2 = _local_nine_cell_equal_weight(m2, [1, 2], cells, n_boot=500, seed=42)
    ok = ok and s2['mean'] is None and s2['n_missing'] == 18
    record('local_non_estimable_and_missing', ok,
           f"single-seed non-est={len(s1['non_estimable_cells'])} missing={s2['n_missing']}")
    return ok


def _agg_run(results, out, extra=None):
    args = [sys.executable, os.path.join(_EVAL, 'aggregate_cross_cell.py'),
            '--results'] + results + [
            '--manifest', os.path.join(_CVRPTW, 'data', 'baseline', '50_node', 'dev_proto',
                                       'DEV_MANIFEST.json'),
            '--mode', 'regression', '--out', out]
    if extra:
        args += extra
    return subprocess.run(args, capture_output=True, text=True)


def _cells():
    return [f'results/o0cc/r5_cross_{t}_{e}'
            for t in ('r1', 'c1', 'rc1') for e in ('02', '05', '08')]


def test_missing_identity_all_nine_rejected():
    tmp = tempfile.mkdtemp(prefix='devgate_identity_')
    try:
        dirs = []
        cells = _cells()
        for i, c in enumerate(cells):
            d = os.path.join(tmp, f'cell{i}')
            shutil.copytree(os.path.join(_CVRPTW, c), d)
            m = json.load(open(os.path.join(d, 'manifest.json')))
            del m['code_sha256']
            del m['profile_hash']
            json.dump(m, open(os.path.join(d, 'manifest.json'), 'w'))
            dirs.append(d)
        r = _agg_run(dirs, os.path.join(tmp, 'out.json'))
        ok = r.returncode != 0
        record('missing_identity_all_nine_rejected', ok, f"exit={r.returncode}")
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_profile_rehash_mismatch_rejected():
    profile = os.path.join(_CVRPTW, 'results', 'o0cc', 'scale_v2', 'objective_profile.json')
    tmp = tempfile.mkdtemp(prefix='devgate_profile_')
    try:
        dirs = []
        for i, c in enumerate(_cells()):
            d = os.path.join(tmp, f'cell{i}')
            shutil.copytree(os.path.join(_CVRPTW, c), d)
            m = json.load(open(os.path.join(d, 'manifest.json')))
            m['profile_hash'] = '0' * 64  # 全九份同一错误 hash，但 --profile 重算会戳穿
            json.dump(m, open(os.path.join(d, 'manifest.json'), 'w'))
            dirs.append(d)
        r = _agg_run(dirs, os.path.join(tmp, 'out.json'), extra=['--profile', profile])
        ok = r.returncode != 0
        record('profile_rehash_mismatch_rejected', ok, f"exit={r.returncode}")
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_local_missing_field_rejected():
    """删除整个 local 字段（当 --local 默认开）应报完整性错误，而非当作无 eligible。"""
    tmp = tempfile.mkdtemp(prefix='devgate_localmiss_')
    try:
        dirs = []
        for i, c in enumerate(_cells()):
            d = os.path.join(tmp, f'cell{i}')
            shutil.copytree(os.path.join(_CVRPTW, c), d)
            if i == 0:
                p = os.path.join(d, 'instances', 'inst_0.json')
                j = json.load(open(p)); del j['local']
                json.dump(j, open(p, 'w'))
            dirs.append(d)
        r = _agg_run(dirs, os.path.join(tmp, 'out.json'))
        combined = (r.stdout or '') + (r.stderr or '')
        ok = r.returncode != 0 and 'missing_local' in combined
        record('local_missing_field_rejected', ok, f"exit={r.returncode}")
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_version_separation():
    compute = set(run_dev_gate.COMPUTE_FILES)
    control = set(run_dev_gate.CONTROL_FILES)
    analysis = set(run_dev_gate.ANALYSIS_FILES)
    ok = ('scripts/evaluation/authoritative_evaluator.py' in compute
          and 'scripts/evaluation/aggregate_cross_cell.py' not in compute
          and 'scripts/evaluation/run_dev_gate.py' not in compute
          and 'scripts/evaluation/aggregate_cross_cell.py' in analysis
          and 'scripts/evaluation/run_dev_gate.py' in control
          and not (compute & control) and not (compute & analysis))
    record('version_separation', ok,
           f"compute={len(compute)} control={len(control)} analysis={len(analysis)}")
    return ok


def test_profile_rehash_matches_frozen():
    profile = os.path.join(_CVRPTW, 'results', 'o0cc', 'scale_v2', 'objective_profile.json')
    data = json.load(open(profile))
    recomputed = _recompute_profile_hash(profile)
    ok = recomputed == data['profile_hash']
    record('profile_rehash_matches_frozen', ok,
           f"recomputed={recomputed[:12]} file={data['profile_hash'][:12]}")
    return ok


def test_aggregator_frozen_plan_identity():
    """聚合器冻结计划身份：正确成功 / runner 不符拒绝 / 组合 hash 与文件清单不符拒绝。"""
    from aggregate_cross_cell import _validate_result_manifests, _combined_hash
    dirs = [os.path.join(_CVRPTW, 'results', 'o0cc', f'r5_cross_{t}_{e}')
            for t in ('r1', 'c1', 'rc1') for e in ('02', '05', '08')]
    m0 = json.load(open(os.path.join(dirs[0], 'manifest.json')))
    runner_hash = m0['code_sha256']
    profile_hash = m0['profile_hash']

    def make_fp():
        fp = {'compute_files': {'run_action_oracle.py': runner_hash},
              'profile_hash': profile_hash}
        fp['compute_sha256'] = _combined_hash(fp['compute_files'])
        return fp

    # 1. 正确冻结计划 → 成功
    ok = True
    try:
        ref = _validate_result_manifests(dirs, make_fp(), None)
        ok = ref[1] == runner_hash
    except SystemExit:
        ok = False
    # 2. runner hash 不符 → 拒绝
    fp = make_fp(); fp['compute_files'] = {'run_action_oracle.py': '0' * 64}
    fp['compute_sha256'] = _combined_hash(fp['compute_files'])
    try:
        _validate_result_manifests(dirs, fp, None); ok = ok and False
    except SystemExit:
        pass
    # 3. 组合 hash 与文件清单不符 → 拒绝
    fp = make_fp(); fp['compute_sha256'] = '0' * 64
    try:
        _validate_result_manifests(dirs, fp, None); ok = ok and False
    except SystemExit:
        pass
    record('aggregator_frozen_plan_identity', ok)
    return ok


def test_drift_parity_script():
    """parity 脚本：自比 ALL_MATCH；改一个 coldchain_cost 判 MISMATCH。"""
    src = os.path.join(_CVRPTW, 'results', 'o0cc', 'r5_cross_r1_02')
    tmp = tempfile.mkdtemp(prefix='devgate_parity_')
    try:
        old = os.path.join(tmp, 'old'); new = os.path.join(tmp, 'new')
        shutil.copytree(src, old); shutil.copytree(src, new)
        script = os.path.join(_EVAL, 'check_drift_parity.py')

        def run():
            return subprocess.run([sys.executable, script, '--old-dir', old,
                                   '--new-dir', new, '--instance-ids', '0,1',
                                   '--out', os.path.join(tmp, 's.json')],
                                  capture_output=True, text=True)

        r = run()
        ok = (r.returncode == 0)
        p = os.path.join(new, 'instances', 'inst_0.json')
        j = json.load(open(p)); j['oracle']['coldchain_cost'] += 0.5
        json.dump(j, open(p, 'w'))
        r2 = run()
        ok = ok and r2.returncode != 0
        record('drift_parity_script', ok, f"self={0 if r.returncode==0 else 'FAIL'} "
              f"mutated={r2.returncode}")
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ok = [test_validate_states(), test_rebuild_canonical_and_recovery(),
          test_code_sha256_mixed_rejected(), test_drift_parity_script(),
          test_local_equal_weight_counterexample(), test_local_non_estimable_and_missing(),
          test_missing_identity_all_nine_rejected(), test_profile_rehash_mismatch_rejected(),
          test_local_missing_field_rejected(), test_version_separation(),
          test_profile_rehash_matches_frozen(), test_aggregator_frozen_plan_identity()]
    out_dir = os.path.join(_CVRPTW, 'results', 'o0cc')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'dev_gate_control_tests.json'), 'w') as f:
        json.dump({'all_pass': all(ok), 'results': RESULTS}, f, indent=2)
    print(f"\n  ALL: {'PASS' if all(ok) else 'FAIL'}")
    return 0 if all(ok) else 1


if __name__ == '__main__':
    sys.exit(main())
