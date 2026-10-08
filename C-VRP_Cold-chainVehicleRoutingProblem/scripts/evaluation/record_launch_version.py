# -*- coding: utf-8 -*-
"""记录两批运行中批次的启动版本身份（源码 hash + 合同 hash）到各自的 LAUNCH_VERSION.json。"""
import datetime
import hashlib
import io
import json
import os
import sys

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_SCRIPTS, os.path.join(_SCRIPTS, 'evaluation'),
           os.path.join(_SCRIPTS, 'coldchain'), os.path.join(_SCRIPTS, 'simulation')):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_identity import SOURCE_FILES_STEP2
from scenario_saa import make_c0_contract_v2


def main():
    now = {}
    for rel in SOURCE_FILES_STEP2:
        h = hashlib.sha256()
        with io.open(rel, 'rb') as f:
            h.update(f.read())
        now[rel] = h.hexdigest()
    contract = make_c0_contract_v2()
    rec = {
        'recorded_at': datetime.datetime.now().isoformat(),
        'note': ('批次启动版本记录（Q-04 严格版于本批启动后落地；此记录 = 本批实际加载源码。'
                 'gate.json 的 identity 为运行时读取，二者结合核验配对身份。'),
        'contract_hash': contract.contract_hash,
        'source_sha256': now,
    }
    targets = [
        'results/a1_c1r_reveal_40/LAUNCH_VERSION.json',
        'results/a1_s3_2r_density_40/LAUNCH_VERSION.json',
    ]
    for out in targets:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, 'w', encoding='utf-8') as f:
            json.dump(rec, f, indent=2)
        print(out, 'written')


if __name__ == '__main__':
    main()
