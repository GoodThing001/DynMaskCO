"""把追加 provider（coenrich / mapt）幂等写入 ordering_providers._REGISTRY。

背景：`ordering_providers.py` 属于各 ordering 臂的**启动封存清单**
（`run_a1_external_accept._ordering_seal_files`），F1 正式批次运行期间修改它会让
在跑臂 `source_stable=False`。因此本脚本**只能在 F1 四条腿全部 ALLDONE 之后**执行。

它**不在** `run_identity.SHARED_SOURCE_FILES` 里，所以事后接线不影响与基线的配对
（配对只核对 7 个共享计算文件 + contract/data/budget/config/seed）。

用法（在 C-VRP_Cold-chainVehicleRoutingProblem 下）：
    python ../CC_Compare/tools/wire_extra_providers.py            # 应用
    python ../CC_Compare/tools/wire_extra_providers.py --check    # 只看是否需要
"""
import argparse
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXT = os.path.abspath(os.path.join(HERE, '..', '..',
                                   'C-VRP_Cold-chainVehicleRoutingProblem'))
TARGET = os.path.join(EXT, 'scripts', 'evaluation', 'ordering_providers.py')

ENTRIES = {
    'coenrich': ("    'coenrich': ('CC_Compare/CO-enriched-ML/a1_accept_v1',\n"
                 "                 'coenrich_provider', 'COEnrichedMLProvider'),\n"),
    'mapt': ("    'mapt': ('CC_Compare/MAPT_code/a1_accept_v1',\n"
             "             'mapt_provider', 'MAPTProvider'),\n"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--only', default=None,
                    help='只接线这一个（coenrich / mapt）')
    args = ap.parse_args()

    src = open(TARGET, encoding='utf-8').read()
    missing = [k for k in ENTRIES if ("'%s':" % k) not in src]
    if args.only:
        missing = [k for k in missing if k == args.only]
    if not missing:
        print('already wired; nothing to do')
        return 0
    print('need to wire:', missing)
    if args.check:
        return 0

    # 在 _REGISTRY 的最后一个 '条目,' 之后插入（保持 dict 字面量合法）
    m = re.search(r'_REGISTRY = \{\n(.*?)\n\}\n', src, re.S)
    if not m:
        raise SystemExit('cannot locate _REGISTRY literal')
    body = m.group(1)
    ins = ''.join(ENTRIES[k] for k in ('coenrich', 'mapt')
                  if k in missing)
    new_body = body.rstrip('\n') + '\n' + ins.rstrip('\n')
    new_src = src[:m.start(1)] + new_body + src[m.end(1):]

    bak = TARGET + '.bak_before_extra_providers'
    if not os.path.exists(bak):
        shutil.copy2(TARGET, bak)
    with open(TARGET, 'w', encoding='utf-8') as f:
        f.write(new_src)
    print('wired %s (backup at %s)' % (missing, os.path.basename(bak)))
    # 语法自检
    import py_compile
    py_compile.compile(TARGET, doraise=True)
    print('py_compile OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
