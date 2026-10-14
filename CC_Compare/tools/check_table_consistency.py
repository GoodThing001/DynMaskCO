"""跨格式一致性自检：Markdown 主表 vs Excel 性能总表，逐方法比对关键数值。

防止「MD 与 Excel 是两套渲染路径、数字悄悄分叉」。比对字段：
utility p_c=0 的 mean、主档 mean、Δ vs cond_hist（p_c=0）mean、p_c=(10,20,30) mean。

用法：
    python CC_Compare/tools/check_table_consistency.py \
        --md CC_Compare/results/a1_compare_table/A-v1对比方法总表.md \
        --xlsx CC_Compare/results/a1_compare_table/A-v1对比方法总表.xlsx
"""
import argparse
import re
import sys


MD_KEYS = ['Utility p_c=0', 'Utility 主档', 'Utility p_c=(10,20,30)', 'Δ vs cond_hist (p_c=0)']
XL_KEYS = ['Utility p_c=0', 'Utility 主档', 'Utility p_c=10/20/30', 'Δ vs cond p_c=0']


def _find_cols(header_cells, keys):
    """按**表头名**定位列（而不是写死下标）—— 这样以后加/挪列不会把校验器带偏。

    2026-10-13 实测教训：给主表插入「参数量(M)」列后，写死下标的旧版校验器把
    「原生约束」当成了 util0，报出 21 处假 MISMATCH。
    """
    idx = []
    for k in keys:
        found = None
        for j, c in enumerate(header_cells):
            if c and c.startswith(k):
                found = j
                break
        idx.append(found)
    return idx


def parse_md(path):
    """返回 {方法: (util0, utilm, util3, diff0)}（浮点或 None）。"""
    cols = None
    out = {}
    for line in open(path, encoding='utf-8'):
        if line.startswith('| # |'):
            cols = _find_cols([c.strip() for c in line.strip().strip('|').split('|')], MD_KEYS)
            continue
        if not line.startswith('| ') or line.startswith('|---') or line.startswith('| # '):
            continue
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if len(cells) < 12 or cols is None or any(c is None for c in cols):
            continue
        name = cells[1]
        if name.startswith('**') or name.startswith('方法'):
            continue

        def num(s):
            m = re.match(r'^(-?[\d.]+)', s.replace('—', ''))
            return float(m.group(1)) if m and s not in ('—', '') else None
        out[name] = tuple(num(cells[c]) for c in cols)
    return out


def parse_xlsx(path):
    import openpyxl
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb['性能总表']
    hdr = [ws.cell(1, c).value for c in range(1, ws.max_column + 1)]
    cols = _find_cols([str(h) if h is not None else '' for h in hdr], XL_KEYS)
    out = {}
    for r in range(2, ws.max_row + 1):
        name = ws.cell(r, 2).value
        if not name:
            continue
        out[name] = tuple(
            (ws.cell(r, c + 1).value if c is not None else None) for c in cols)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--md', required=True)
    ap.add_argument('--xlsx', required=True)
    args = ap.parse_args()

    md = parse_md(args.md)
    xl = parse_xlsx(args.xlsx)
    print('MD 行数=%d  Excel 行数=%d' % (len(md), len(xl)))
    bad = 0
    only_md = [k for k in md if k not in xl]
    only_xl = [k for k in xl if k not in md]
    if only_md:
        print('仅在 MD:', only_md[:8])
    if only_xl:
        print('仅在 Excel:', only_xl[:8])
    for k in sorted(set(md) & set(xl)):
        a, b = md[k], xl[k]
        for i, fld in enumerate(('util0', 'utilm', 'util3', 'diff0')):
            va, vb = a[i], b[i]
            if va is None and vb is None:
                continue
            # MD 只显示 1 位小数，Excel 存全精度 ⇒ 容差取半个显示单位
            if va is None or vb is None or abs(va - vb) > 0.06:
                bad += 1
                print('  MISMATCH %-46s %-6s md=%s xlsx=%s' % (k[:46], fld, va, vb))
    print('不一致字段数:', bad, '| 结论:', 'OK' if bad == 0 and not only_md and not only_xl else 'CHECK')
    return 1 if bad else 0


if __name__ == '__main__':
    sys.exit(main())
