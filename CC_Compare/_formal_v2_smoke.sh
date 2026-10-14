#!/bin/bash
# 快速验证：同版基线 + 一个臂 → pairing_identity_check 是否真的通过（2 天小样）。
# 目的：在 25h 正式批次跑完之前，先证明「新基线 + 当前代码」能给出
#       identity_verified=true / formal_adjudication=true，避免整批白跑。
set -u
cd /home/hzeng/project/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem || exit 1
PY=/home/hzeng/envs/cc_compare/bin/python
SB=results/a1_c1smoke2_reveal/gate.json
OUT=results/a1_ext_formal2_smoke

echo "== [1/3] 2 天基线 =="
$PY scripts/evaluation/run_a1_step2_gate.py \
    --train-instances 4 --gate-instances 2 --n-orders 200 \
    --capacity 50.0 --num-vehicles 15 --rho 0.60 --K 10 \
    --time-limit 10.0 --k-neighbors 10 \
    --gate-seed 20260926 --shadow-mode greedy --energy-pricing marginal \
    --future-policy reveal --workers 3 \
    --out results/a1_c1smoke2_reveal
echo "base rc=$?"

echo "== [2/3] attention 臂（2 天）=="
$PY scripts/evaluation/run_a1_external_accept.py --solver ordering \
    --provider attention \
    --provider-ckpt ../CC_Compare/AttentionModel/pretrained/cvrp_100/epoch-99.pt \
    --provider-device cpu --baseline-from "$SB" --dev-instances 2 --workers 2 \
    --penalty "p_c=0" --out "$OUT/attention_0"
echo "arm rc=$?"

echo "== [3/3] 身份与配对判定 =="
$PY - <<'PYEOF'
import json
b = json.load(open('results/a1_c1smoke2_reveal/gate.json', encoding='utf-8'))
bi = b.get('identity') or {}
print('baseline identity keys :', sorted(bi.keys()))
print('baseline source_stable :', bi.get('source_stable'))
print('baseline shared files  :', len(bi.get('shared_source_sha256') or {}))
a = json.load(open('results/a1_ext_formal2_smoke/attention_0/gate.json', encoding='utf-8'))
for yr, yb in a['results'].items():
    ai = yb.get('_identity') or {}
    pid = yb.get('_pairing_identity') or {}
    print('arm source_stable      :', ai.get('source_stable'))
    print('arm shared files       :', len(ai.get('shared_source_sha256') or {}))
    for k in ('shared_source_sha256', 'contract_sha256', 'data_sha256',
              'data_meta_sha256', 'budget_B', 'cooling_share', 'shared_config'):
        same = (ai.get(k) == bi.get(k))
        print('  %-22s match=%s' % (k, same))
    print('PAIRING identity_verified :', pid.get('identity_verified'))
    print('PAIRING problems          :', pid.get('problems'))
    for pname, blk in yb.items():
        if pname.startswith('_') or not isinstance(blk, dict):
            continue
        for k, v in blk.items():
            if k.endswith('_minus_cond_hist'):
                print('paired diff', pname, v.get('mean'), 'adjudicable=', v.get('adjudicable'),
                      'formal_adjudication=', v.get('formal_adjudication'),
                      'identity_verified=', v.get('identity_verified'))
PYEOF
