"""RRNCO **原生 solver 面** 的 C2 权重因果对照（2026-10-13）。

补 `CC_Compare/results/a1_compare_table/模型参与证据.md` 中登记的**已知缺口**：

    RRNCO 走原生 `run_a1_external_accept.py --solver rrnco`（**不在**
    `ordering_providers._REGISTRY`），通用 harness 的
    `--provider-dir/--provider-module/--provider-class` 装载路径假定
    `cls(ckpt, device=...)`，而真实构造是
    `RRNCOPreferenceProvider(BackendConfig(checkpoint_path=..., device=...))`。
    ⇒ 需另写 C2：加载 `epoch_199.ckpt` 的策略后按 std 量级扰动权重、
       重跑同一子问题、比较返回排序。

本脚本就是那个入口：**完全复用** `verify_provider_participation.py` 的四对照 +
两守恒实现（`probe_provider` / `build_instance` / `make_sp` / `timed_order` /
`clear_cache` / `find_torch_models` / `randomize_weights` / `md_fragment`），
只替换 provider 的构造方式，使其与
`rrnco_accept_replanner.RRNCOAcceptReplanner.__init__`（line 183-185）**同一条加载路径**。

产物：`<outdir>/rrnco.json` + `<outdir>/rrnco.md`，schema 与其它 11 个 provider
逐字段一致，可直接并入同一台账。

纪律（沿用 1003 现状）：
- **只读**：不修改任何 provider / `scripts/**` / 上游 `rrnco/**`；
- 单线程（`OMP/MKL/OPENBLAS/NUMEXPR=1` + `torch.set_num_threads(1)`）；
- `SIGALRM` 墙钟熔断；**超时/异常不写产物**（避免把半成品当证据）。

用法（1003，cc_compare env）：
    cd /home/hzeng/project/MASKCO-Main
    /home/hzeng/envs/cc_compare/bin/python CC_Compare/tools/verify_rrnco_c2.py \
        --ckpt CC_Compare/RRNCO/checkpoints/rcvrptw/epoch_199.ckpt \
        --device cpu --outdir CC_Compare/results/a1_compare_table/participation

（3080 同源复验：把 `--device cpu` 换成 `--device cuda` 即可，加载路径不变。）
"""
import argparse
import os
import sys
import time

# --------------------------------------------------------------------------- #
# 负载纪律：必须在 import numpy/torch 之前由被复用的 harness 之外再压一遍
# （harness 自己也会 setdefault，这里保持同样顺序以免 transport 抢先）
# --------------------------------------------------------------------------- #
for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_v, '1')

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import verify_provider_participation as vp  # noqa: E402

ROOT = vp._ROOT


def build_rrnco_provider(args):
    """按 `RRNCOAcceptReplanner` 的同一条路径构造真实 provider。

    与 `run_a1_external_accept.py` 的 `solver == 'rrnco'` 分支一致：
        BackendConfig(checkpoint_path, device, seed) → RRNCOPreferenceProvider(config)

    需要两个 sys.path 位置：
      - `<rrnco_root>/`            → 上游 `rrnco` 包（`from rrnco.models import RRNet`）
      - `<rrnco_root>/dcc_rh_v4/`  → `rrnco_backend` / `preference`
    """
    rrnco_root = os.path.normpath(os.path.join(ROOT, *args.rrnco_root.split('/')))
    dcc = os.path.join(rrnco_root, 'dcc_rh_v4')
    for p in (dcc, rrnco_root):
        if not os.path.isdir(p):
            raise SystemExit('缺少路径：%s' % p)
        if p not in sys.path:
            sys.path.insert(0, p)

    ckpt = os.path.normpath(os.path.join(ROOT, *args.ckpt.split('/')))
    if not os.path.isfile(ckpt):
        raise SystemExit('缺少 checkpoint：%s' % ckpt)

    from rrnco_backend import BackendConfig, RRNCOPreferenceProvider  # noqa: E402
    cfg = BackendConfig(checkpoint_path=ckpt, device=args.device, seed=args.seed)
    return RRNCOPreferenceProvider(cfg), ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='CC_Compare/RRNCO/checkpoints/rcvrptw/epoch_199.ckpt',
                    help='仓库根相对路径（与原生 solver 臂同一文件）')
    ap.add_argument('--rrnco-root', default='CC_Compare/RRNCO',
                    help='上游 RRNCO 仓库根（仓库根相对）')
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--seed', type=int, default=0,
                    help='BackendConfig.seed（与原生臂默认一致）')
    ap.add_argument('--provider', default='rrnco',
                    help='证据里的方法名（应与总表 method key 一致）')
    ap.add_argument('--pool', type=int, default=40)
    ap.add_argument('--hidden', type=int, default=60)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--rand-seed', type=int, default=0)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--max-seconds', type=int, default=1800,
                    help='墙钟熔断（RRNCO 40 起点解码比其它 provider 慢，默认放宽到 1800s）')
    ap.add_argument('--outdir', required=True)
    args = ap.parse_args()

    import signal

    def _on_alarm(signum, frame):                       # noqa: ARG001
        raise TimeoutError('wall-clock budget %ss exceeded' % args.max_seconds)

    # SIGALRM 只有 POSIX 有；Windows（3080）没有该属性。2026-10-13 实测：不加判断时
    # 在 3080 上抛 `AttributeError: module 'signal' has no attribute 'SIGALRM'`，
    # 探针**两次（cpu/cuda）都秒退、零产物**，只在 `.done` 里留下 `rc=1 ok=False`。
    _armed = hasattr(signal, 'SIGALRM')
    if _armed:
        signal.signal(signal.SIGALRM, _on_alarm)
        signal.alarm(int(args.max_seconds))
    else:
        print('NOTE: 本平台无 SIGALRM（Windows）⇒ 跳过墙钟熔断；'
              '由外部守候（_gpu3080_rrnco_c2.ps1）兜底。')
    try:
        provider, ckpt = build_rrnco_provider(args)
        res = vp.probe_provider(provider, args)
    except TimeoutError as e:
        print('ABORTED: %s —— 未写产物（避免把半成品当证据）' % e)
        return 3
    finally:
        if _armed:
            signal.alarm(0)

    # 补 RRNCO 面特有的身份字段（诚实标注：本行不是 provider 注册表路径）
    res['ckpt'] = args.ckpt
    res['ckpt_abs'] = ckpt
    res['ckpt_sha256'] = provider.ckpt_sha256
    res['access_face'] = 'native-solver'
    res['load_path'] = ('rrnco_backend.BackendConfig -> RRNCOPreferenceProvider'
                        ' (== rrnco_accept_replanner.RRNCOAcceptReplanner.__init__)')
    res['in_registry'] = False
    if isinstance(res.get('C2_random_weights'), dict) and \
            res['C2_random_weights'].get('status') == 'DONE':
        # 记录被扰动的模块与「扰动后是否仍覆盖全池」
        res['C2_random_weights']['seeded'] = True
        res['C2_random_weights']['rand_seed'] = args.rand_seed

    os.makedirs(args.outdir, exist_ok=True)
    jp = os.path.join(args.outdir, '%s.json' % args.provider)
    with open(jp, 'w', encoding='utf-8') as f:
        import json
        json.dump(res, f, ensure_ascii=False, indent=1)
    mp = os.path.join(args.outdir, '%s.md' % args.provider)
    frag = vp.md_fragment(res)
    frag += ('- 接入面: **native solver**（`--solver rrnco`，不在 `_REGISTRY`）；'
             'ckpt sha256 `%s`\n' % provider.ckpt_sha256[:16])
    with open(mp, 'w', encoding='utf-8') as f:
        f.write(frag)
    print(frag)
    print('wrote', jp)
    return 0


if __name__ == '__main__':
    sys.exit(main() or 0)
