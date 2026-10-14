"""生成 3080 GPU 腿的 4 个 per-method .cmd（避免 batch 嵌套引号），并给出 WMI 启动命令。

每个 .cmd：顺序跑该方法的 p_c=0 与主档两臂，输出到 results\\a1_ext_formal2_3080\\<tag>_{0,51015}，
日志到 C:\\Users\\zh\\gpu3080_logs\\<tag>.log，完成写 <tag>.done。

两条不同的驱动面（**不能混用**）：
  1. 外部 provider 面：`--solver ordering --provider <name> --provider-ckpt ... --provider-device ...`
     适用于 routefinder / cada / mvmoe（三者在
     `C-VRP.../scripts/evaluation/ordering_providers.py::_REGISTRY` 里）。
  2. RRNCO 原生面：`--solver rrnco --rrnco-ckpt ... --rrnco-device ...`
     RRNCO **不在** `_REGISTRY` 里，`CC_Compare/RRNCO/a1_accept_v1/` 只有一个
     `run_formal_rr.sh`（原生 driver）。写成 `--provider rrnco` 会直接
     `SystemExit: unknown --provider` —— 这是启动前拦截掉的第二个缺陷。

env 映射对齐 1003 的权威解释器 /home/hzeng/envs/cc_compare/bin/python
（见 CC_Compare/_gpu_leg_1003_fallback.sh:13 与 CC_Compare/setup_envs.sh）：
  - routefinder / cada / rrnco 需要 rl4co + torchrl -> 3080 上的 cc_compare
    （torch 2.11.0+cu128 + rl4co 0.6.0，由 _setup_cc_compare_3080.cmd 建）
  - mvmoe 只 import torch（自带的 models/envs）     -> 3080 上的 pytorch
注意：旧的 cc_gpu env 不可用——它先装 torch==2.4.1+cu121，随后 pip install rl4co==0.6.0
因版本要求更高而把它换成了 PyPI 的 CPU build（实测 2.14.1+cpu / cuda False）。
DEV：4 个 provider 全部走 cuda（v1 中 routefinder=cuda、mvmoe/cada=cpu、rrnco=cuda；
v1 的 cpu 设置让 cada 每天超时 30.4/130.95 次，故 v2 统一在空闲 GPU 上重跑，逐臂报告
mean_timeouts）。详见 README §13.2。
"""
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_gpu3080')
os.makedirs(OUT, exist_ok=True)

BASE = r"results\a1_c1fresh_reveal_40\gate.json"
RES = r"results\a1_ext_formal2_3080"
LOGS = r"C:\Users\zh\gpu3080_logs"

# (tag, conda env, provider name, checkpoint)
PROVIDER_SPECS = [
    ('mvmoe', 'pytorch', 'mvmoe',
     r'..\CC_Compare\MVMoE\pretrained\pomo_vrptw_n100\epoch-5000.pt'),
    ('routefinder', 'cc_compare', 'routefinder',
     r'..\CC_Compare\RouteFinder\checkpoints\100\rf-transformer.ckpt'),
    ('cada', 'cc_compare', 'cada',
     r'..\CC_Compare\CaDA\100\result\2024-1121-1355\checkpoint-300.pt'),
]

# RRNCO：原生 solver 面（不在 provider registry）
RRNCO = ('rrnco', 'cc_compare',
         r'..\CC_Compare\RRNCO\checkpoints\rcvrptw\epoch_199.ckpt')

HEAD = r"""@echo off
REM 3080 GPU leg: {tag} ({env} env), two penalty tiers sequentially (device=cuda)
set PYTHONIOENCODING=utf-8
if not exist {logs} mkdir {logs}
cd /d C:\Users\zh\project\MASKCO-Main\C-VRP_Cold-chainVehicleRoutingProblem
echo == {tag} START %DATE% %TIME% == > {logs}\{tag}.log
"""

TAIL = r"""echo DONE_{tag} %DATE% %TIME% >> {logs}\{tag}.done
echo == {tag} END %DATE% %TIME% == >> {logs}\{tag}.log
"""

# 面 1：外部 provider
RUN_PROVIDER = ('conda run -n {env} python scripts\\evaluation\\run_a1_external_accept.py '
                '--solver ordering --provider {prov} --provider-ckpt {ckpt} '
                '--provider-device cuda --baseline-from {base} --dev-instances 40 '
                '--workers 4 --penalty "{pen}" --out {res}\\{tag}_{sfx} >> {logs}\\{tag}.log 2>&1\n')

# 面 2：RRNCO 原生
RUN_RRNCO = ('conda run -n {env} python scripts\\evaluation\\run_a1_external_accept.py '
             '--solver rrnco --rrnco-ckpt {ckpt} --rrnco-device cuda '
             '--baseline-from {base} --dev-instances 40 --workers 4 '
             '--penalty "{pen}" --out {res}\\{tag}_{sfx} >> {logs}\\{tag}.log 2>&1\n')

TIERS = (('p_c=0', '0'), ('p_c=(5,10,15)', '51015'))


def emit(tag, env, ckpt, runner, extra):
    body = HEAD.format(tag=tag, env=env, logs=LOGS)
    for pen, sfx in TIERS:
        body += runner.format(tag=tag, env=env, ckpt=ckpt, base=BASE, res=RES,
                              logs=LOGS, pen=pen, sfx=sfx, **extra)
        body += 'echo -- {tag} {sfx} rc=%ERRORLEVEL% -- >> {logs}\\{tag}.log\n'.format(
            tag=tag, sfx=sfx, logs=LOGS)
    body += TAIL.format(tag=tag, logs=LOGS)
    p = os.path.join(OUT, '_gpu3080_%s.cmd' % tag)
    with open(p, 'w', encoding='ascii') as f:
        f.write(body)
    print('wrote', p)


ALL_TAGS = []
for tag, env, prov, ckpt in PROVIDER_SPECS:
    emit(tag, env, ckpt, RUN_PROVIDER, {'prov': prov})
    ALL_TAGS.append(tag)
emit(RRNCO[0], RRNCO[1], RRNCO[2], RUN_RRNCO, {})
ALL_TAGS.append(RRNCO[0])

print()
print('WMI launch (one detached process per method, they run in parallel):')
for tag in ALL_TAGS:
    rel = r'CC_Compare\tools\_gpu3080\_gpu3080_%s.cmd' % tag
    print('  Invoke-CimMethod -ClassName Win32_Process -MethodName Create '
          r'-Arguments @{CommandLine="cmd /c C:\Users\zh\project\MASKCO-Main\%s"}' % rel)
