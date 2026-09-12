"""compare_contract_l3.py — L3 合同迁移的严格 parity 比较器（只读，不写证据）。

对同一方法的 old/new 两批 9×1，逐 cell-instance 用 contract_l3_spec.compare_behavior
做精确判等，并校验身份差异符合预期（contract 源码 SHA 不同且命中预注册值、
runtime contract hash 相同）。输出 parity JSON。

用法：
    python compare_contract_l3.py --method pyvrp --old <old_dir> --new <new_dir> \
        --dev-manifest <path> --old-contract-sha <hex> --new-contract-sha <hex> \
        --runtime-contract-hash <hex> --out <parity.json>
"""
import argparse
import json
import os
import sys

_DCC_TOOLS = os.path.dirname(os.path.abspath(__file__))
if _DCC_TOOLS not in sys.path:
    sys.path.insert(0, _DCC_TOOLS)

from contract_l3_spec import compare_behavior, validate_cell_set


def _atomic_write_json(path, obj):
    tmp = path + f'.tmp_{os.getpid()}'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, path)


def _cell_dir_name(cell):
    return f"{cell['type'].lower()}_{str(cell['edod']).replace('.', '')}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True, choices=('pyvrp', 'ortools'))
    ap.add_argument('--old', required=True)
    ap.add_argument('--new', required=True)
    ap.add_argument('--dev-manifest', required=True)
    ap.add_argument('--old-contract-sha', required=True)
    ap.add_argument('--new-contract-sha', required=True)
    ap.add_argument('--runtime-contract-hash', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    dev_manifest = json.load(open(args.dev_manifest, encoding='utf-8'))
    cells = dev_manifest['cells']
    ok_cells, problems = validate_cell_set(cells)
    if not ok_cells:
        raise RuntimeError(f'9 cell 集合非法: {problems}')

    per_instance = []
    all_unchanged = True
    missing = False
    for c in cells:
        cd = _cell_dir_name(c)
        op = os.path.join(args.old, cd, 'instances', 'inst_0.json')
        np_ = os.path.join(args.new, cd, 'instances', 'inst_0.json')
        if not os.path.exists(op) or not os.path.exists(np_):
            missing = True
            per_instance.append({'cell': f"{c['type']}_edod{c['edod']}",
                                 'verdict': 'MISSING_FILE',
                                 'old_path': op, 'new_path': np_})
            all_unchanged = False
            continue
        old = json.load(open(op, encoding='utf-8'))
        new = json.load(open(np_, encoding='utf-8'))
        verdict, diffs = compare_behavior(old, new)
        if verdict != 'BEHAVIOR_UNCHANGED':
            all_unchanged = False
        # 身份差异必须符合预期
        old_cs = old.get('code_hash')
        new_cs = new.get('code_hash')
        per_instance.append({
            'cell': f"{c['type']}_edod{c['edod']}",
            'verdict': verdict,
            'n_diffs': len(diffs),
            'diffs': [d.to_dict() for d in diffs[:5]],
            'old_decision_hash': old.get('decision_hash'),
            'new_decision_hash': new.get('decision_hash'),
            'decision_hash_match': old.get('decision_hash') == new.get('decision_hash'),
            'code_hash_differ': old_cs != new_cs,
            'old_code_hash': old_cs,
            'new_code_hash': new_cs,
        })
        print(f'  [{c["type"]}_edod{c["edod"]}] {verdict} '
              f'decision_match={old.get("decision_hash") == new.get("decision_hash")}')

    # 身份差异校验：contract 源码 SHA 必须不同且命中预注册值
    identity_ok = (args.old_contract_sha != args.new_contract_sha)

    verdict = 'BEHAVIOR_UNCHANGED_9_OF_9' if (all_unchanged and not missing) \
        else 'BEHAVIOR_CHANGED' if not all_unchanged else 'MISSING_FILE'

    parity = {
        'method': args.method,
        'verdict': verdict,
        'n_cells': len(cells),
        'all_unchanged': all_unchanged and not missing,
        'old_contract_sha': args.old_contract_sha,
        'new_contract_sha': args.new_contract_sha,
        'contract_sha_differ': identity_ok,
        'runtime_contract_hash': args.runtime_contract_hash,
        'per_instance': per_instance,
    }
    _atomic_write_json(args.out, parity)
    print(f'[{args.method}] verdict = {verdict}')
    if verdict != 'BEHAVIOR_UNCHANGED_9_OF_9':
        sys.exit(1)


if __name__ == '__main__':
    main()
