"""聚合器 aggregate_cross_cell 的防错测试（用已下载的 18 实例产物，不跑 oracle）。

验证：
  1. 正序/倒序输入 → 九 cell 等权均值一致（身份按 data hash 映射，不按输入顺序）；
  2. 重复目录 → 非零退出；
  3. 缺 cell（8 个）→ 非零退出；
  4. NaN 注入 → 非零退出（逐实例 Gate 复算拦截）；
  5. service 失败注入（complete=False）→ 非零退出。

用法：python scripts/tests/test_aggregate_cross_cell.py
产物：results/o0cc/aggregate_cross_cell_tests.json
"""
import os, sys, json, subprocess, shutil, tempfile

_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CVRPTW = os.path.dirname(_BASE)
_TMP = tempfile.mkdtemp(prefix="agg_test_")
_p = lambda n: os.path.join(_TMP, n)
RESULTS = []


def record(name, ok, detail=''):
    RESULTS.append({'test': name, 'status': 'PASS' if ok else 'FAIL', 'detail': str(detail)})
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")


def _run(results, out):
    args = ['python', os.path.join(_CVRPTW, 'scripts', 'evaluation', 'aggregate_cross_cell.py'),
            '--results'] + results + [
            '--manifest', os.path.join(_CVRPTW, 'data', 'baseline', '50_node', 'dev_proto', 'DEV_MANIFEST.json'),
            '--mode', 'regression', '--out', out]
    return subprocess.run(args, capture_output=True, text=True)


def _cells(prefix='r5_cross'):
    return [os.path.join(_CVRPTW, 'results', 'o0cc', f'{prefix}_{t}_{e}')
            for t in ('r1', 'c1', 'rc1') for e in ('02', '05', '08')]


def test_order_invariant():
    fwd = _cells()
    rev = list(reversed(fwd))
    r1 = _run(fwd, _p('agg_fwd.json'))
    r2 = _run(rev, _p('agg_rev.json'))
    ok = (r1.returncode == 0 and r2.returncode == 0)
    if ok:
        a = json.load(open(_p('agg_fwd.json'))); b = json.load(open(_p('agg_rev.json')))
        ok = (a['nine_cell_equal_weight_mean_delta'] == b['nine_cell_equal_weight_mean_delta']
              and a['grouped_bootstrap_ci'] == b['grouped_bootstrap_ci'])
    record('aggregate_order_invariant', ok,
           f"fwd==rev mean={json.load(open(_p('agg_fwd.json')))['nine_cell_equal_weight_mean_delta']:.4f}")
    return ok


def test_duplicate_rejected():
    cells = _cells()
    cells[1] = cells[0]  # 重复
    r = _run(cells, _p('agg_dup.json'))
    ok = (r.returncode != 0)
    record('aggregate_duplicate_rejected', ok, f"exit={r.returncode}")
    return ok


def test_missing_rejected():
    r = _run(_cells()[:8], _p('agg_miss.json'))
    ok = (r.returncode != 0)
    record('aggregate_missing_rejected', ok, f"exit={r.returncode}")
    return ok


def test_nan_rejected():
    cells = _cells()
    tmp = os.path.join(_CVRPTW, 'results', 'o0cc', '_tmp_nan')
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(cells[0], tmp)
    p = f'{tmp}/instances/inst_0.json'
    j = json.load(open(p)); j['oracle']['coldchain_cost'] = float('nan')
    json.dump(j, open(p, 'w'))
    cells[0] = tmp
    r = _run(cells, _p('agg_nan.json'))
    shutil.rmtree(tmp, ignore_errors=True)
    ok = (r.returncode != 0)
    record('aggregate_nan_rejected', ok, f"exit={r.returncode}")
    return ok


def test_service_fail_rejected():
    cells = _cells()
    tmp = os.path.join(_CVRPTW, 'results', 'o0cc', '_tmp_svc')
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(cells[0], tmp)
    p = f'{tmp}/instances/inst_0.json'
    j = json.load(open(p)); j['oracle']['complete'] = False
    json.dump(j, open(p, 'w'))
    cells[0] = tmp
    r = _run(cells, _p('agg_svc.json'))
    shutil.rmtree(tmp, ignore_errors=True)
    ok = (r.returncode != 0)
    record('aggregate_service_fail_rejected', ok, f"exit={r.returncode}")
    return ok


def test_ten_dirs_rejected():
    cells = _cells() + [_cells()[0]]  # 10 个（9 唯一 + 1 重复）
    r = _run(cells, _p('agg_ten.json'))
    ok = (r.returncode != 0)
    record('aggregate_ten_dirs_rejected', ok, f"exit={r.returncode}")
    return ok


def test_profile_mismatch_rejected():
    cells = _cells()
    tmp = os.path.join(_CVRPTW, 'results', 'o0cc', '_tmp_prof')
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(cells[0], tmp)
    p = f'{tmp}/manifest.json'
    j = json.load(open(p)); j['profile_hash'] = 'bad-profile-hash'
    json.dump(j, open(p, 'w'))
    cells[0] = tmp
    r = _run(cells, _p('agg_prof.json'))
    shutil.rmtree(tmp, ignore_errors=True)
    ok = (r.returncode != 0)
    record('aggregate_profile_mismatch_rejected', ok, f"exit={r.returncode}")
    return ok


def test_inst_idx_mismatch_rejected():
    cells = _cells()
    tmp = os.path.join(_CVRPTW, 'results', 'o0cc', '_tmp_idx')
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(cells[0], tmp)
    p = f'{tmp}/instances/inst_0.json'
    j = json.load(open(p)); j['inst_idx'] = 99
    json.dump(j, open(p, 'w'))
    cells[0] = tmp
    r = _run(cells, _p('agg_idx.json'))
    shutil.rmtree(tmp, ignore_errors=True)
    ok = (r.returncode != 0)
    record('aggregate_inst_idx_mismatch_rejected', ok, f"exit={r.returncode}")
    return ok


def test_local_nan_rejected():
    cells = _cells()
    tmp = os.path.join(_CVRPTW, 'results', 'o0cc', '_tmp_lnan')
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.copytree(cells[0], tmp)
    p = f'{tmp}/instances/inst_0.json'
    j = json.load(open(p))
    j.setdefault('local', {})['event_deltas'] = [float('nan')]
    json.dump(j, open(p, 'w'))
    cells[0] = tmp
    r = _run(cells, _p('agg_lnan.json'))
    shutil.rmtree(tmp, ignore_errors=True)
    ok = (r.returncode != 0)
    record('aggregate_local_nan_rejected', ok, f"exit={r.returncode}")
    return ok


def main():
    ok = [test_order_invariant(), test_duplicate_rejected(), test_missing_rejected(),
          test_nan_rejected(), test_service_fail_rejected(), test_ten_dirs_rejected(),
          test_profile_mismatch_rejected(), test_inst_idx_mismatch_rejected(),
          test_local_nan_rejected()]
    out_dir = os.path.join(_CVRPTW, 'results', 'o0cc')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'aggregate_cross_cell_tests.json'), 'w') as f:
        json.dump({'all_pass': all(ok), 'results': RESULTS}, f, indent=2)
    print(f"\n  ALL: {'PASS' if all(ok) else 'FAIL'}")
    return 0 if all(ok) else 1


if __name__ == '__main__':
    sys.exit(main())
