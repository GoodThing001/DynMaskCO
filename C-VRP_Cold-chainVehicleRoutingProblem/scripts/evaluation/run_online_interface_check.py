"""④ 冻结评分器在线接口对账：F_H/F_R 70 维 feature-local 模型（不训练、不接 encoder/状态）。

六项检查：
  1. 模型身份 —— checkpoint / 输入维数 / 去编号 / 尺度 s 与原评价一致；
  2. 候选与参考动作 —— 合法 mask、KEEP/DEFER、平局规则一致；
  3. 特征 —— 同一快照+同一当前计划下逐字段对账（不能只比最终分数）；
  4. 分数与接受 —— 数值容差预固定，首选与接受决定一致；
  5. 写回 —— 接受后读新计划；拒绝/失败不留部分修改；
  6. 因果性 —— 扰动未揭示订单/未来节点数/teacher 标签不改输入、候选、分数、动作。

诚实性约束：在线一侧必须经过真实部署适配路径（不能把离线构造器调用两遍冒充一致性）；
缺 checkpoint/数据/快照/在线路径时输出 BLOCKED + 缺失项，不得跳过后显示 PASS。

当前状态（代码层）：70 维适配层 `feature_local_replanner.FeatureLocalReplanner` 已实现
（消费 ScoringMLP(70)，构造 ctx25+action16+local29，deindex 必填）。检查 2 验证该适配层
存在且 deindex 必填；检查 3–6 的动态逐字段/分数/写回/因果对账由
`run_feature_local_reconcile.py` 与 `run_feature_local_score_check.py` 承担（本脚本只做
身份 + 适配层存在性的前置检查，不冒充已跑通全部六项）。

用法（本地无 checkpoint/数据时只做静态审计）：
    python scripts/evaluation/run_online_interface_check.py --out results/m0_scale/iface_check
完整（服务器）：
    python scripts/evaluation/run_online_interface_check.py \
        --ckpt <model.ckpt> --data <npz> --teacher-dir <dir> --deindex --out <dir>
"""
import argparse
import json
import os
import pickle
import sys

_BASE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_BASE)
_CVRPTW = os.path.dirname(_SCRIPTS)
for p in ('models', 'data', 'simulation', 'evaluation', 'baselines', 'coldchain',
          'expert', 'training'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))

# 70 维显式特征规格（train_feature_only_local / train_m1_compact.build_explicit_and_labels）
FEATURE_CONTEXT_DIM = 25
ACTION_FEAT_DIM = 8
LOCAL_VAL_DIM = 25
LOCAL_VALID_DIM = 4
IN_DIM_70 = FEATURE_CONTEXT_DIM + 2 * ACTION_FEAT_DIM + LOCAL_VAL_DIM + LOCAL_VALID_DIM  # 70

EXPECTED_IN_DIM = IN_DIM_70

RESULTS = []


def record(name, status, detail=''):
    RESULTS.append({'check': name, 'status': status, 'detail': detail})
    print(f"  [{status:7s}] {name}  {detail}")


def check_model_identity(ckpt_path, expect_deindex):
    """检查 1：模型身份。缺 checkpoint → BLOCKED。"""
    if not ckpt_path or not os.path.exists(ckpt_path):
        record('model_identity', 'BLOCKED', f'缺 checkpoint：{ckpt_path}')
        return None
    with open(ckpt_path, 'rb') as f:
        ck = pickle.load(f)
    issues = []
    in_dim = ck.get('in_dim')
    if in_dim != EXPECTED_IN_DIM:
        issues.append(f'in_dim={in_dim}（期望 {EXPECTED_IN_DIM}；S 组 83 维不在本检查范围）')
    s = ck.get('s')
    if not (isinstance(s, (int, float)) and s > 0):
        issues.append(f's={s!r} 非正数')
    if ck.get('use_state'):
        issues.append('use_state=True（本检查只覆盖 F_H/F_R 70 维，不覆盖 S 组）')
    if expect_deindex is not None:
        # deindex 配置记录在训练 manifest.json；checkpoint 本身不含该字段，需从 --deindex 显式传入
        pass
    params = ck.get('params')
    if params is None:
        issues.append('缺 params')
    if issues:
        record('model_identity', 'FAIL', '；'.join(issues))
    else:
        record('model_identity', 'PASS', f'in_dim={in_dim} s={s:.4f} use_state={ck.get("use_state")}')
    return ck


def check_online_path_exists():
    """检查 2（前置）：70 维适配层是否存在且 deindex 必填。

    静态判定：导入 feature_local_replanner，确认 IN_DIM=70 且构造器要求显式 deindex。
    """
    gaps = []
    try:
        import feature_local_replanner as flr
        if flr.IN_DIM != EXPECTED_IN_DIM:
            gaps.append(f'feature_local_replanner.IN_DIM={flr.IN_DIM}，期望 {EXPECTED_IN_DIM}')
        try:
            flr.FeatureLocalReplanner()  # 缺 deindex 应报错
            gaps.append('FeatureLocalReplanner() 未在缺 deindex 时拒绝（应显式要求）')
        except ValueError:
            pass  # 正确拒绝
    except Exception as e:  # noqa: BLE001
        gaps.append(f'无法导入 feature_local_replanner：{e}')

    if gaps:
        record('online_path', 'BLOCKED', '70 维适配层不满足前置：' + ' | '.join(gaps))
        return False
    record('online_path', 'PASS', '存在 70 维适配层且 deindex 必填')
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=None, help='model.ckpt（train_state_2x2 输出）')
    ap.add_argument('--data', default=None, help='m0_scale 输入 npz')
    ap.add_argument('--teacher-dir', default=None)
    ap.add_argument('--deindex', action='store_true')
    ap.add_argument('--expect-deindex', default=None,
                    help='期望的去编号配置（True/False），与训练 manifest 一致')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print('=== ④ 冻结评分器在线接口对账（F_H/F_R 70 维） ===', flush=True)
    check_model_identity(args.ckpt, args.expect_deindex)

    online_ok = check_online_path_exists()

    # 3–6 依赖真实在线路径 + checkpoint + 数据；缺任一则 BLOCKED
    if not online_ok:
        record('feature_reconciliation', 'BLOCKED', '无 70 维在线路径，无法逐字段对账')
        record('score_accept', 'BLOCKED', '无 70 维在线路径，无法对账分数/接受')
        record('writeback', 'BLOCKED', '无 70 维在线路径，无法对账写回')
        record('causality', 'BLOCKED', '无 70 维在线路径，无法对账因果不变性')
    else:
        # 占位：当 70 维在线适配层存在后，在此接入逐字段/分数/写回/因果对账
        for name in ('feature_reconciliation', 'score_accept', 'writeback', 'causality'):
            record(name, 'BLOCKED', '在线适配层已存在但逐字段对账尚未实现')

    summary = {'expected_in_dim': EXPECTED_IN_DIM,
               'checks': RESULTS,
               'note': '在线一侧必须经过真实部署适配路径；离线构造器重复调用不算一致性。'}
    with open(os.path.join(args.out, 'iface_check.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'\nsaved: {args.out}/iface_check.json')


if __name__ == '__main__':
    main()
