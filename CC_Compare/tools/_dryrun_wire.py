"""在**不修改真文件**的前提下，预演 wire_extra_providers.py 的插入逻辑：
把 scripts/evaluation/ordering_providers.py 读进来，套用同一正则与插入，
对结果做 ast.parse / 关键结构断言，并打印 diff 预览。

用法：python CC_Compare/tools/_dryrun_wire.py
"""
import ast
import difflib
import os
import re
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

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
    src = open(TARGET, encoding='utf-8').read()
    missing = [k for k in ENTRIES if ("'%s':" % k) not in src]
    print('需要插入:', missing)
    if not missing:
        print('无需改动'); return 0
    m = re.search(r'_REGISTRY = \{\n(.*?)\n\}\n', src, re.S)
    if not m:
        print('FAIL: 找不到 _REGISTRY 字面量'); return 1
    ins = ''.join(ENTRIES[k] for k in ('coenrich', 'mapt') if k in missing)
    new_body = m.group(1).rstrip('\n') + '\n' + ins.rstrip('\n')
    new_src = src[:m.start(1)] + new_body + src[m.end(1):]

    try:
        tree = ast.parse(new_src)
    except SyntaxError as e:
        print('FAIL: 插入后语法错误', e); return 1
    print('ast.parse OK')

    # 结构断言：_REGISTRY 仍是 dict 字面量，且新增键在里面
    reg = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                getattr(t, 'id', None) == '_REGISTRY' for t in node.targets):
            reg = node.value
    if not isinstance(reg, ast.Dict):
        print('FAIL: _REGISTRY 不再是 dict 字面量'); return 1
    keys = [k.value for k in reg.keys if isinstance(k, ast.Constant)]
    print('插入后注册表键:', sorted(keys))
    for k in missing:
        if k not in keys:
            print('FAIL: 缺少键', k); return 1
    # 每个值必须是 3 元组字面量
    for k, v in zip(reg.keys, reg.values):
        if isinstance(k, ast.Constant) and isinstance(v, ast.Tuple) and len(v.elts) != 3:
            print('FAIL: %s 不是 3 元组' % k.value); return 1
    print('结构断言 OK（全部为 3 元组字面量）')

    print('---- diff 预览 ----')
    d = difflib.unified_diff(src.splitlines(), new_src.splitlines(),
                             'before', 'after', lineterm='', n=2)
    print('\n'.join(list(d)[:24]))
    return 0


if __name__ == '__main__':
    sys.exit(main())
