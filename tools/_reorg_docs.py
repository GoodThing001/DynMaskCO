# -*- coding: utf-8 -*-
"""One-off 2026-09-24: 把 docs/当前规划 的已收束历史文档迁入 历史已收束/ 并批量改链。

用法：
  python tools/_reorg_docs.py            # dry-run，只打印
  python tools/_reorg_docs.py --apply    # 真正移动 + 改链
"""
import os
import re
import shutil
import sys

APPLY = "--apply" in sys.argv

WS = r"D:\PyCharm_\MASKCO-Main"
EXT = os.path.join(WS, "C-VRP_Cold-chainVehicleRoutingProblem")
BASE = os.path.join(EXT, "docs", "当前规划")
HIST = os.path.join(BASE, "历史已收束")

MOVED = [
    ("决策与审计", "O0CC源码字节基线与行尾处置.md"),
    ("决策与审计", "代码与文档核查报告.md"),
    ("决策与审计", "event_plan两轮收束与后续决策.md"),
    ("决策与审计", "event_plan_v2运行身份与缺失项.md"),
    ("决策与审计", "N0服务失败与预留资源对照.md"),
    ("决策与审计", "N0源码核查与N1启动条件.md"),
    ("决策与审计", "N1N2收束与增强修复验证决策.md"),
    ("决策与审计", "直接目标训练后续决策_状态身份与固定状态诊断.md"),
    ("决策与审计", "头腔定位与下一步决策_2026-09-22.md"),
    ("结果与进度", "event_plan_v2表示对照结果.md"),
    ("结果与进度", "cc_lns_n0_r1结果.md"),
    ("结果与进度", "cc_lns_n0结果.md"),
    ("结果与进度", "cc_lns_n0r结果.md"),
    ("结果与进度", "n1n2_maskco条件联合重构结果.md"),
    ("结果与进度", "cc_lns_swap增强修复验证结果.md"),
    ("结果与进度", "MaskCO直接目标训练Step5收束结果.md"),
    ("结果与进度", "MaskCO直接目标训练_搜索增强验证与收束.md"),
    ("结果与进度", "搜索路线互补性分析_2026-09-22.md"),
    ("结果与进度", "头腔普查D1D2D3结果_2026-09-22.md"),
    ("结果与进度", "因果头腔诊断结果_2026-09-22.md"),
    ("结果与进度", "价值可预测性探针结果_2026-09-22.md"),
    ("结果与进度", "易腐排序头腔结果_2026-09-22.md"),
    ("结果与进度", "品质重标定路径A最终结论_2026-09-22.md"),
    ("结果与进度", "昼夜时变能耗方向最终结论_2026-09-23.md"),
    ("结果与进度", "预冷方向头腔确认_2026-09-23.md"),
    ("结果与进度", "预冷中心站正向突破_2026-09-23.md"),
    ("工程实现", "表征探针接口与数据规范.md"),
    ("工程实现", "R2到M1执行计划_2026-09-14.md"),
    ("工程实现", "event_plan_v2表示对照工作包.md"),
    ("工程实现", "event_plan_v1执行协议.md"),
    ("工程实现", "MaskCO直接目标训练工作包.md"),
    ("工程实现", "前瞻价值函数A_工作包.md"),
    ("研究设计", "MaskCO职责重定与下一阶段工作包.md"),
    ("研究设计", "搜索路线收束后_几何先验与冷链残差研究建议.md"),
    ("研究设计", "对比方法可迁移机制审读_2026-09-22.md"),
    ("研究设计", "距离先验与冷链残差_一页设计.md"),
    ("研究设计", "合同改动候选与推荐_2026-09-22.md"),
    ("研究设计", "B_静态耦合MCVRP_协议设计与无训练筛选.md"),
    ("实验与评估", "MaskCO新主线基线核查_2026-09-19.md"),
]
SUB_MOVES = [("结果与进度", "MaskCO直接目标训练_干净协议冻结")]


def norm(p):
    return os.path.normpath(os.path.abspath(p))


BASE_N = norm(BASE)
HIST_N = norm(HIST)

moved_old_files = set()
moved_old_dirs = set()
for sub, name in MOVED:
    moved_old_files.add(norm(os.path.join(BASE, sub, name)))
for sub, folder in SUB_MOVES:
    moved_old_dirs.add(norm(os.path.join(BASE, sub, folder)))


def is_moved_target(t):
    if t in moved_old_files:
        return True
    for d in moved_old_dirs:
        if t == d or t.startswith(d + os.sep):
            return True
    return False


def is_moved_linker(path):
    p = norm(path)
    if p.startswith(HIST_N + os.sep):
        return True
    if p in moved_old_files:
        return True
    for d in moved_old_dirs:
        if p == d or p.startswith(d + os.sep):
            return True
    return False


def linker_original_dir(path):
    p = norm(path)
    if p.startswith(HIST_N + os.sep):
        rel = os.path.relpath(p, HIST_N)
        return os.path.join(BASE, os.path.dirname(rel))
    return os.path.dirname(p)


LINK = re.compile(r'\]\(([^()\s]+)\)')


def rewrite(path):
    with open(path, encoding="utf-8") as f:
        s = f.read()
    orig = s
    li_moved = is_moved_linker(path)
    orig_dir = linker_original_dir(path)
    changes = []

    # 第一遍（非迁移文件）：字符串级前缀替换（覆盖根 CLAUDE.md 等以扩展根为基的写法）
    if not li_moved:
        for sub, name in MOVED:
            s = s.replace(f"{sub}/{name}", f"历史已收束/{sub}/{name}")
        for sub, folder in SUB_MOVES:
            s = s.replace(f"{sub}/{folder}", f"历史已收束/{sub}/{folder}")

    # 第二遍：路径解析级修正（覆盖同目录裸文件名 / 迁移文件对未迁移文件的 ../ 补层）
    def repl(m):
        t = m.group(1)
        if t.startswith(("http://", "https://", "mailto:", "#")):
            return m.group(0)
        base, _, anchor = t.partition("#")
        if not base:
            return m.group(0)
        if re.match(r'^[A-Za-z]:[/\\]', base):
            return m.group(0)  # Windows 绝对路径，不动
        target = norm(os.path.join(orig_dir, base))
        moved_target = is_moved_target(target)
        if moved_target and not li_moved:
            rel = os.path.relpath(target, BASE_N)
            hist_target = norm(os.path.join(HIST, rel))
            newpath = os.path.relpath(hist_target, os.path.dirname(norm(path))).replace(os.sep, "/")
            newt = newpath + ("#" + anchor if anchor else "")
            changes.append((t, newt))
            return "](" + newt + ")"
        if (not moved_target) and li_moved and os.path.exists(target):
            newt = "../" + t
            changes.append((t, newt))
            return "](" + newt + ")"
        return m.group(0)

    s2 = LINK.sub(repl, s)
    if s2 != orig:
        print("REWRITE", os.path.relpath(path, WS), "| %d links" % len(changes))
        for a, b in changes[:10]:
            print("    %s -> %s" % (a, b))
        if len(changes) > 10:
            print("    ...")
        if APPLY:
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.write(s2)


def scan_dir(root, skip_prefixes=()):
    for dirpath, dirnames, filenames in os.walk(root):
        if any(dirpath.startswith(p) for p in skip_prefixes):
            continue
        for fn in filenames:
            if fn.endswith(".md"):
                yield os.path.join(dirpath, fn)


def main():
    for sub, name in MOVED:
        src = os.path.join(BASE, sub, name)
        dst = os.path.join(HIST, sub, name)
        if os.path.exists(src):
            print("MOVE", os.path.relpath(src, WS), "->", os.path.relpath(dst, WS))
            if APPLY:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.move(src, dst)
        else:
            print("MISSING", src)
    for sub, folder in SUB_MOVES:
        src = os.path.join(BASE, sub, folder)
        dst = os.path.join(HIST, sub, folder)
        if os.path.isdir(src):
            print("MOVE", os.path.relpath(src, WS), "->", os.path.relpath(dst, WS))
            if APPLY:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.move(src, dst)
        else:
            print("MISSING", src)

    skip = [norm(os.path.join(EXT, "archive")),
            norm(os.path.join(EXT, "negative_result_paper"))]
    files = list(scan_dir(EXT, skip_prefixes=skip))
    for root_md in ("CLAUDE.md", "AGENTS.md", "README.md"):
        p = os.path.join(WS, root_md)
        if os.path.exists(p):
            files.append(p)
    for p in files:
        rewrite(p)


if __name__ == "__main__":
    main()
