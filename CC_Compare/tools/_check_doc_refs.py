"""一次性检查：三份交付文档里引用的**本地可解析路径**是否都存在（找出死引用）。

为什么值得查：本会话新增/改名了很多文件（`print_readme_section2.py`、`_scan_timeouts.py`、
`_probe_*.py`、`_gpu3080_copyback.ps1` …），而 README §6/§11、NEXT_STEPS、读表注意里
到处是反引号路径引用 —— 一次改名就可能留下**指向不存在文件的引用**，读者会白找。
本脚本只检查**在本地工作区应当存在**的那一类（`CC_Compare/` 下的文件），
对「服务器侧路径」「相对扩展根的结果目录」「带通配符的引用」明确跳过，避免误报。
"""
import io
import os
import re
import sys

# 自保：中文 Windows 控制台默认 GBK，而本脚本会打印 `✅` ⇒ 直接跑会
# `UnicodeEncodeError: 'gbk' codec can't encode character '\u2705'`，
# 把一个**通过**的检查报成 rc=1（2026-10-14 实测踩到）。
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = '.'
DOCS = [
    'CC_Compare/results/a1_compare_table/README.md',
    'CC_Compare/results/a1_compare_table/NEXT_STEPS.md',
    'CC_Compare/results/a1_compare_table/读表注意与已知限定.md',
    'CC_Compare/results/a1_compare_table/模型参与证据.md',
]
# 反引号里的路径样貌：以这些前缀开头、以已知扩展名结尾
PAT = re.compile(r'`([A-Za-z0-9_./\-]*CC_Compare/[A-Za-z0-9_./\-]+\.(?:py|md|ps1|sh|cmd|json))`')
# 跳过：带通配符、第三方权重/数据、以及**服务器侧**结果路径。
# 但 **交付目录 `CC_Compare/results/a1_compare_table/<文件>` 要检查** ——
# 那里才是收口后新引用的落点（`README_SECTION2_paste.md`、`TIMEOUT_SCAN.md` 等）。
DELIV = 'CC_Compare/results/a1_compare_table/'
SKIP_SUBSTR = ('*', 'RRNCO/rrnco/', 'checkpoints', 'pretrained', 'artifacts')


def _skippable(p):
    if any(s in p for s in SKIP_SUBSTR):
        return True
    if 'results/' in p and not p.startswith(DELIV):
        return True
    if p.startswith(DELIV) and p[len(DELIV):].count('/') > 0:
        return True          # 交付目录下的子目录 = 服务器侧产物，跳过
    return False

missing = {}
checked = 0
for d in DOCS:
    if not os.path.exists(d):
        print('DOC MISSING: %s' % d)
        continue
    txt = io.open(d, encoding='utf-8').read()
    for m in PAT.finditer(txt):
        p = m.group(1)
        if _skippable(p):
            continue
        checked += 1
        if not os.path.exists(p):
            missing.setdefault(p, []).append(os.path.basename(d))

print('检查了 %d 个本地应存在的 CC_Compare 路径引用' % checked)
if missing:
    print('**未找到（死引用）%d 个：**' % len(missing))
    for p, where in sorted(missing.items()):
        print('  - %-64s  ← 出现在 %s' % (p, ', '.join(sorted(set(where)))))
else:
    print('全部存在 ✅')
