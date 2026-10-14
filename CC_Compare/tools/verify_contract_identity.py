"""核对：当前磁盘代码是否仍产生与正式臂记录一致的 contract_hash / 数据身份。

若一致 → 现在重跑求解器臂在物理/合同口径上与 09-30 正式臂相同，
差异只来自驱动记账（可显式声明）；若不一致 → 重跑即为新协议。

用法（在 C-VRP_Cold-chainVehicleRoutingProblem 下）：
    python ../CC_Compare/tools/verify_contract_identity.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXT_ROOT = os.path.abspath(os.path.join(HERE, '..', '..',
                                        'C-VRP_Cold-chainVehicleRoutingProblem'))
for p in (os.path.join(EXT_ROOT, 'scripts'),
          os.path.join(EXT_ROOT, 'scripts', 'coldchain'),
          os.path.join(EXT_ROOT, 'scripts', 'simulation'),
          os.path.join(EXT_ROOT, 'scripts', 'evaluation')):
    if p not in sys.path:
        sys.path.insert(0, p)

RECORDED_CONTRACT = 'e25a12104b20c3ea99cb4033f6d3bf32adecf2974de6b5168dbfa0d5a058a41b'
RECORDED_DATA = '8767a03d787c0e0fcad112b4888edf46db44c55e9b50b9705bfc5830f4fb378d'
RECORDED_META = '72a805fd5bc0b2b7a94b0daa712cb11112bcc3e4a5daf7817ca8ca53e28322ba'


def main():
    from run_identity import contract_identity, dataset_identity, dataset_meta_sha256
    from run_exp_reserve import generate_dataset
    from coldchain_evaluator_a1 import add_v2_initial_quality
    from scenario_saa import make_c0_contract_v2

    contract = make_c0_contract_v2()
    ch = contract_identity(contract)
    print('contract_hash  now =', ch)
    print('contract_hash  arm =', RECORDED_CONTRACT)
    print('  ->', 'SAME' if ch == RECORDED_CONTRACT else 'DIFF')

    # 与驱动一致：先 generate_dataset，再 add_v2_initial_quality
    ds = add_v2_initial_quality(generate_dataset(40, 200, 20260926), contract)
    dh = dataset_identity(ds)
    dm = dataset_meta_sha256(ds)
    print('data_sha256    now =', dh)
    print('data_sha256    arm =', RECORDED_DATA)
    print('  ->', 'SAME' if dh == RECORDED_DATA else 'DIFF')
    print('data_meta      now =', dm)
    print('data_meta      arm =', RECORDED_META)
    print('  ->', 'SAME' if dm == RECORDED_META else 'DIFF')


if __name__ == '__main__':
    main()
