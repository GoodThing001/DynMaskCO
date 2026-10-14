"""
重新生成 archive/代码快照/完整代码归档.txt —— 把 scripts/ 下所有源码文件
（.py / .sh / .cpp / .hpp / Makefile）快照进一个 txt，供导师/归档审阅。

用法:
    python scripts/regenerate_archive.py
"""

import os
import glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, 'scripts')
OUT = os.path.join(ROOT, 'archive', '代码快照', '完整代码归档.txt')

EXTS = ('*.py', '*.sh', '*.cpp', '*.hpp', 'Makefile')


def collect_files():
    files = []
    for ext in EXTS:
        files += glob.glob(os.path.join(SCRIPTS, '**', ext), recursive=True)
    files = sorted(set(files))
    files = [f for f in files if '__pycache__' not in f and '.pyc' not in f]
    return files


def main():
    files = collect_files()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rel = [os.path.relpath(f, ROOT).replace('\\', '/') for f in files]

    out = []
    out.append('=' * 80)
    out.append('  MaskCO 动态冷链物流项目 — 完整代码归档')
    out.append('  日期: 2026-09-01 (HFR-M0 实现完成 + Gate A FAIL/F3；JF2 exact-vehicle No-Go；新增全部 JF2 文件 + HFR 文件)')
    out.append(f'  文件数: {len(files)}')
    out.append('=' * 80)
    out.append('')
    out.append('## 目录')
    out.append('')
    for p in rel:
        out.append(f'  {p}')
    out.append('')
    out.append('=' * 80)
    out.append('')

    for p in rel:
        out.append('=' * 80)
        out.append(f'## 文件: {p}')
        out.append('=' * 80)
        out.append('')
        with open(os.path.join(ROOT, p), 'r', encoding='utf-8', errors='replace') as f:
            out.append(f.read().rstrip('\n'))
        out.append('')
        out.append('')

    with open(OUT, 'w', encoding='utf-8') as f:
        f.write('\n'.join(out) + '\n')

    print(f'已生成 {OUT}')
    print(f'  文件数: {len(files)}')


if __name__ == '__main__':
    main()
