#!/usr/bin/env python3
"""
结果汇总脚本 — 从 eval_edod_matrix.py 或手动记录生成统一汇总表。

用法:
    # 从 EDoD 矩阵评估结果读取
    python scripts/analysis/summarize.py --from-edod analysis/edod_results.txt

    # 手动添加单条结果
    python scripts/analysis/summarize.py --add "st_r2|R1|0.5|94.5|14.67|0.0|-8.0|5000"

    # 查看已保存的汇总表
    python scripts/analysis/summarize.py --show

    # 导出为 markdown 表格
    python scripts/analysis/summarize.py --export docs/实验记录/实验结果总表.md
"""

import sys, os, argparse, json, re
from datetime import datetime

RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'docs', 'results.json')

HEADER = ['model', 'type', 'edod', 'feas%', 'cost', 'viol', 'gap_ref%', 'steps']
SORT_KEY = lambda r: (r.get('type',''), r.get('edod',''), -float(r.get('feas%',0)))

def load_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            return json.load(f)
    return []

def save_results(data):
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    with open(RESULTS_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def print_table(rows, title="Results Summary"):
    print(f"\n{'='*85}")
    print(f"  {title}  ({len(rows)} entries, {datetime.now().strftime('%Y-%m-%d %H:%M')})")
    print(f"{'='*85}")
    print(f"{'Model':<18} {'Type':<5} {'EDoD':<6} {'Feas%':>7} {'Cost':>8} {'Viol':>5} {'GapRef%':>8} {'Steps':>8}")
    print(f"{'-'*85}")
    for r in sorted(rows, key=SORT_KEY):
        print(f"{r.get('model','?'):<18} {r.get('type','?'):<5} {r.get('edod','?'):<6} "
              f"{float(r.get('feas%',0)):>6.1f}% {float(r.get('cost',0)):>8.2f} "
              f"{float(r.get('viol',0)):>5.1f} {float(r.get('gap%',0)):>+7.1f}% "
              f"{str(r.get('steps','')):>8}")
    print(f"{'='*85}")

    # EDoD 矩阵
    edod_rows = [r for r in rows if 'edod' in r and r['type'] in ('R1','C1','RC1')]
    if len(edod_rows) >= 3:
        print(f"\n  EDoD Matrix:")
        print(f"  {'Type':<5} {'EDoD=0.2':>20} {'EDoD=0.5':>20} {'EDoD=0.8':>20}")
        print(f"  {'-'*68}")
        for dt in ['R1','C1','RC1']:
            row = f"  {dt:<5}"
            for ed in ['0.2','0.5','0.8']:
                match = [r for r in edod_rows if r['type']==dt and r['edod']==ed]
                if match:
                    r = match[0]
                    row += f" {float(r['feas%']):>5.1f}% g={float(r.get('gap_ref%', r.get('gap%', 0))):+6.1f}% "
                else:
                    row += f" {'?':>5}  {'?':>7}  "
            print(row)


def parse_cvrptw_output(text, label=''):
    """从 cvrptw.py 输出解析指标。"""
    feas = re.search(r'TW feas rate:\s+([\d.]+)%', text)
    cost = re.search(r'mean cost:\s+([\d.]+)', text)
    viol = re.search(r'Avg TW viol/inst:\s+([\d.]+)', text)
    gap  = re.search(r'Gap_ref[^:]*:\s+([\d.-]+)\s*%', text)
    if not gap:
        gap  = re.search(r'Gap:\s+([\d.-]+)\s*%', text)  # backward compat
    opt  = re.search(r'opt cost:\s+([\d.]+)', text)

    return {
        'feas': float(feas.group(1)) if feas else -1,
        'cost': float(cost.group(1)) if cost else -1,
        'viol': float(viol.group(1)) if viol else -1,
        'gap':  float(gap.group(1)) if gap else -999,
        'opt':  float(opt.group(1)) if opt else -1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--add', type=str, help='添加结果: "model|type|edod|feas|cost|viol|gap|steps"')
    parser.add_argument('--parse', type=str, help='从 cvrptw.py 输出文本解析')
    parser.add_argument('--label', type=str, default='', help='标签: model|type|edod|steps')
    parser.add_argument('--show', action='store_true', help='显示汇总表')
    parser.add_argument('--export', type=str, help='导出 markdown')
    parser.add_argument('--clear', action='store_true', help='清空所有结果')
    args = parser.parse_args()

    rows = load_results()

    if args.clear:
        save_results([])
        print("Cleared all results.")
        return

    if args.add:
        parts = args.add.split('|')
        if len(parts) >= 7:
            row = {k: v for k, v in zip(HEADER, parts)}
            # 去重: 同 model+type+edod 替换
            rows = [r for r in rows if not (r.get('model')==row['model'] and r.get('type')==row.get('type','') and r.get('edod')==row.get('edod',''))]
            rows.append(row)
            save_results(rows)
            print(f"Added: {row['model']} | {row.get('type','')} | EDoD={row.get('edod','')} | feas={row['feas%']}%")
        else:
            print("Format: model|type|edod|feas|cost|viol|gap|steps")

    if args.parse:
        text = args.parse
        if os.path.exists(args.parse):
            with open(args.parse, 'r') as f:
                text = f.read()
        r = parse_cvrptw_output(text)
        label_parts = args.label.split('|') if args.label else ['?','?','?','?']
        row = {
            'model': label_parts[0] if len(label_parts)>0 else '?',
            'type':  label_parts[1] if len(label_parts)>1 else '?',
            'edod':  label_parts[2] if len(label_parts)>2 else '?',
            'feas%': f"{r['feas']:.1f}",
            'cost':  f"{r['cost']:.2f}",
            'viol':  f"{r['viol']:.1f}",
            'gap%':  f"{r['gap']:.1f}",
            'steps': label_parts[3] if len(label_parts)>3 else '?',
        }
        rows = [x for x in rows if not (x.get('model')==row['model'] and x.get('type')==row.get('type','') and x.get('edod')==row.get('edod',''))]
        rows.append(row)
        save_results(rows)
        print(f"Parsed: feas={r['feas']:.1f}% cost={r['cost']:.2f} gap={r['gap']:+.1f}%")

    if args.show or (not args.add and not args.parse and not args.clear and not args.export):
        print_table(rows)

    if args.export:
        with open(args.export, 'w') as f:
            f.write(f"# Results Summary ({datetime.now().strftime('%Y-%m-%d %H:%M')})\n\n")
            f.write(f"| Model | Type | EDoD | Feas% | Cost | Viol | Gap% | Steps |\n")
            f.write(f"|-------|------|------|-------|------|------|------|-------|\n")
            for r in sorted(rows, key=SORT_KEY):
                f.write(f"| {r.get('model','?')} | {r.get('type','?')} | {r.get('edod','?')} | "
                        f"{r.get('feas%','?')}% | {r.get('cost','?')} | {r.get('viol','?')} | "
                        f"{r.get('gap%','?')}% | {r.get('steps','?')} |\n")
        print(f"Exported to {args.export}")


if __name__ == '__main__':
    main()
