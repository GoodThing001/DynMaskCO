"""Assemble cycle-6 scope, executor evidence and arbor update."""
import datetime
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TREE = ROOT / 'research/review_3080_20261010'
STAGE = TREE / 'dual_track_optimization_20261013_root_v1'
CYCLE = STAGE / 'cycle6_20261013_root_v1'
D13 = TREE / 'delivery/action_matching_generalization_root_v1'
P1 = TREE / 'pickup/lazy_base_distance_full_scenario_root_v1'
P2 = TREE / 'pickup/lazy_base_distance_full_scenario_root_v2'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rel(p):
    return Path(p).relative_to(ROOT).as_posix()


def hashes(paths):
    return {rel(p): sha(p) for p in paths}


def main():
    CYCLE.mkdir(parents=True, exist_ok=True)
    scope = CYCLE / 'CYCLE_SCOPE.json'
    if not scope.exists():
        scope.write_text(json.dumps(dict(
            cycle=6, engineering_cycle_budget=12,
            scientific_family_ledger_unchanged=True,
            delivery_node='n13', pickup_node='n14',
            delivery_scope=('fixed pre-registered 20/9 train/held-out split of the 29 sealed supervised states; three '
                            'pre-declared arms (SGD matched, SGD conservative, Adam control); held-out generalisation '
                            'and independent full-trajectory reproduction; no backbone forward'),
            pickup_scope=('full reveal-sorted order list of the fixed archived scenario index 8; paired unmodified '
                          'reference versus loop-scoped lazy base-distance reuse; interleaved repeats; no backbone'),
            cycle_scope_written_after_stage_registrations=True,
            binding_registration=('each stage was registered in its own REGISTRATION.json before that stage executed; '
                                  'this cycle file is a retrospective summary and is not the binding registration'),
            actual_resource_schedule='CPU7 serial; pickup production/audit then delivery production/audit',
            old_workloads_repeated=False, science_family2_closed=True, science_family3_started=False,
            stable_online_gain=False, no_GPU_or_shared_MD_upload=True,
            authoritative_status_at_start='cycle5_20261013_root_v1',
        ), ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
        print('wrote', rel(scope))

    d13 = [D13 / n for n in ('REGISTRATION.json', 'RESULT.json', 'RUN_EXIT.json', 'INDEPENDENT_AUDIT.json',
                             'AUDIT_RUN_EXIT.json', 'SOURCE_START.json', 'SOURCE_END.json',
                             'AUDIT_SOURCE_START.json', 'AUDIT_SOURCE_END.json', 'STAGE_ADJUDICATION.json',
                             'HEAD_AFTER_sgd_matched.npz', 'HEAD_AFTER_sgd_conservative.npz',
                             'HEAD_AFTER_adam_control.npz', 'RUN_SUPERVISION_RECEIPT.json',
                             'AUDIT_SUPERVISION_RECEIPT.json', 'run.py', 'audit.py', 'head_action_matching.py')]
    p1 = [P1 / n for n in ('SOURCE_START.json', 'RUN_SUPERVISION_RECEIPT.json', 'run.stdout.txt', 'run.stderr.txt')]
    p2 = [P2 / n for n in ('REGISTRATION.json', 'RESULT.json', 'RUN_EXIT.json', 'INDEPENDENT_AUDIT.json',
                           'AUDIT_RUN_EXIT.json', 'SOURCE_START.json', 'SOURCE_END.json', 'AUDIT_SOURCE_START.json',
                           'AUDIT_SOURCE_END.json', 'STAGE_ADJUDICATION.json', 'RUN_SUPERVISION_RECEIPT.json',
                           'AUDIT_SUPERVISION_RECEIPT.json', 'run.py', 'audit.py', 'lazy_base_distance.py')]
    for p in d13 + p1 + p2:
        assert p.exists(), p

    R13 = json.loads((D13 / 'RESULT.json').read_text(encoding='utf-8'))
    A13 = json.loads((D13 / 'INDEPENDENT_AUDIT.json').read_text(encoding='utf-8'))
    R2 = json.loads((P2 / 'RESULT.json').read_text(encoding='utf-8'))
    A2 = json.loads((P2 / 'INDEPENDENT_AUDIT.json').read_text(encoding='utf-8'))
    pa = R13['primary_arm']
    arm = {k: dict(train=v['after']['train'], held_out=v['after']['held_out']) for k, v in R13['arms'].items()}

    evidence = {
        'delivery': {
            'directory': rel(D13), 'stable_online_gain': False, 'scientific_family3_started': False,
            'candidate_installed': False, 'formal_trained_control_qualified': False,
            'producer_checks': R13['checks_count'], 'producer_failures': R13['failed'],
            'audit_checks': A13['checks_count'], 'audit_failures': A13['failed'],
            'registration_sha256': sha(D13 / 'REGISTRATION.json'),
            'generalisation': {
                'split': dict(train=R13['split']['train_records'], held_out=R13['split']['held_out_records']),
                'before': R13['arms'][pa]['before'], 'after': arm, 'primary_arm': pa},
            'mechanism': dict(
                gradient_sign_agreement_float32_vs_float64=A13['mechanism']['gradient_sign_agreement_on_nonzero_union'],
                adam_t1_step_max_abs_diff=A13['mechanism']['adam_t1_step_max_abs_diff_f32_vs_f64'],
                sgd_step_max_abs_diff=A13['mechanism']['sgd_step_max_abs_diff_f32_vs_f64'],
                train_step0_zero_gradient_fraction=R13['mechanism']['train_step0_zero_gradient_fraction'],
                adam_t1_saturated_fraction=R13['mechanism']['adam_t1_saturated_fraction']),
            'summary_zh': (
                '配送 n13 把动作匹配评分头放到固定预登记的 20/9 训练-留出划分上，并预声明三个臂(SGD 匹配步长 0.0860739、'
                'SGD 保守 0.01、Adam 3e-3 控制)，全部报告不按结果挑臂。生产 62 项 0 失败、自然退出 0；独立审计 72 项 2 FAIL。'
                '泛化结论是负向且清楚：真正在优化的两个臂训练 CE 大幅下降(保守 SGD 1.2509→0.2172、Adam 1.2509→0.2344)，'
                '但留出 CE 反而升高(1.3735→1.8231、1.3735→1.5245)；留出 argmax 只从 2/9 变成 4–5/9，n=9 不构成迁移证据。'
                '主臂 SGD 匹配步长虽然两端都降(训练 1.1349、留出 1.2522)，但其训练轨迹震荡(0.989→1.135)从未收敛，'
                '因此留出变化不可归因。这说明在已封存 latent 上做头-only 动作匹配是记忆 20 个状态，不学到可迁移评分函数，'
                '"在 pretrained_initial 封存 latent 上训头"这条路线作为在线增益来源就此封闭。'
                '机制同时被量化：float32 与 float64 的梯度符号在非零并集坐标上一致率只有 0.4978(等于随机)，'
                'Adam 的逐坐标归一化把更新变成符号的函数并饱和到 ±lr (t=1 步长跨库最大差 2.569e-3)，'
                '而 SGD 步长正比于 |g|(3.894e-8)。训练第一步 55.24% 坐标梯度恰为零、23.17% 坐标处于饱和。'
                '两个 SGD 臂仍超出预登记 1e-4 轨迹/参数界(保守 4.27e-4、匹配 0.674)，原"SGD 构造上可复现"的说法过强、'
                '按原样保留为 FAIL；真正可复现的是功能结果(三臂训练/留出 CE 与 argmax 跨库一致到 2–5e-8)。'),
            'next_zh': (
                '不重跑 v1/v2/n13 或诊断追 PASS，不放宽 1e-4 或改用功能等价冒充轨迹可复现。下一配送步骤必须离开已封存 '
                'latent：正式同初始化已训 CE、同任务训练随机与真实主干训练属于科学第3族，只有在全部已登记工程与科学'
                '屏障(完整 8+2 请求、参数/依赖发布前核验)满足后才可在新数据前登记；主干门满足前不加载 backbone。'),
            'evidence_hashes': hashes(d13),
        },
        'pickup': {
            'directory': rel(P2), 'preserved_prior_attempt': rel(P1),
            'stable_online_gain': False, 'formal_strong_dynamic_qualified': False,
            'full_request_budget_qualified': False, 'scientific_family3_started': False,
            'candidate_installed': False,
            'producer_checks': R2['checks_count'], 'producer_failures': R2['failed'],
            'audit_checks': A2['checks_count'], 'audit_failures': A2['failed'],
            'registration_sha256': sha(P2 / 'REGISTRATION.json'),
            'sequence_length': R2['inserted'], 'feasibly_inserted': R2['accepted'],
            'reference_distance_calls': R2['reference_distance_calls'],
            'candidate_distance_calls': R2['candidate_distance_calls'],
            'removed_repeated_base_calls': R2['proved_removed_base_calls'],
            'removed_fraction': R2['proved_removed_base_calls'] / R2['reference_distance_calls'],
            'summary_zh': (
                '取货把上一轮只覆盖前四单的惰性 base 复用扩到固定已封存 scenario index8 的整条揭示序订单表(166 单，'
                '其中 71 单实际可插入)，并与未修改原实现做同输入配对对照，交错 REF,CAND 各重复两次。'
                '原实现的路线距离求值 1042 次，候选 733 次，去掉 309 次同循环重复 base，占 29.7%(上一轮 13/109=11.9%)；'
                '能跑的精确门全部通过：两臂插入选择、可行性调用与数值、80 段 C0 物理转移、终点能量/距离 hex 逐位相同，'
                '候选日志确实是原日志的按序删除，每条被删项都是同循环 base 调用，1042+733 个距离值全部由独立有序双求和逐位复算，'
                '两臂跨重复确定，距离矩阵与原公共参数摘要未变。结果同时给出两个负向/受限证据：其一，实测耗时没有改善，'
                '候选两次计时 0.2704/0.3976 秒反而高于原实现 0.2635/0.2418 秒(仅两次、区间小，不作反向加速结论)；'
                '其二，生产 1 项 FAIL：原实现根本不记录它的评分候选，因此"评分候选严格相同"这一门没有参考侧日志可比，'
                '属本轮夹具设计缺陷而不是科学结果；审计 4 项 FAIL 中 2 项是拒绝为失败生产背书、1 项是同一夹具缺陷、'
                '1 项是审计自身用对象身份跨两份日志判 base 的检查缺陷。以上全部按原样保留，不重跑追 PASS。'),
            'next_zh': (
                '不重跑 v1/v2 或其审计、不调容量/重分组追速度，也不把省调用数当提速。取货运行时间线到此为负向结论，'
                '不再投入微优化；下一实质步骤应转向能产生在线收益的方向：正式学习提议、同初始化已训 CE、同任务训练随机'
                '与真正独立强动态接拒，并按已注册规则在全部屏障满足后另登记新鲜科学日。'),
            'evidence_hashes': hashes(p1 + p2),
        },
        'scientific_family3_started': False, 'stable_online_gain': False,
        'shared_MD_uploaded': False, 'final_pickup_seeds_used': False, 'GPU_used': False, 'server_used': False,
    }
    out = CYCLE / 'EXECUTOR_EVIDENCE.json'
    assert not out.exists()
    out.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    print('wrote', rel(out))

    # ---- arbor tree ----
    tp = STAGE / '.arbor/tree.json'
    t = json.loads(tp.read_text(encoding='utf-8'))
    now = int(time.time())
    t['nodes']['n13'] = dict(
        id='n13', parent='n12', depth=4, status='executed',
        hypothesis=('Delivery: an action-matched head-only CE update over sealed latents transfers to held-out sealed '
                    'states, and its optimisation trajectory can be reproduced by an independent implementation of '
                    'the same registered update'),
        insight=('Head-only action matching on sealed latents memorises 20 training states and does not transfer: held-out '
                 'CE rises while training CE falls. Independently, no normalised update can reproduce a float32 '
                 'trajectory because the gradient sign at the rounding floor agrees with float64 only at chance '
                 '(0.4978); only functional outcomes reproduce. The sealed-latent head route is closed.'),
        metadata=dict(dev_score=float(A13['checks_count']), test_score=None,
                      result=('Producer62/0 exit0; audit72/2FAIL. Train CE1.2509->0.2172(SGD0.01)/0.2344(Adam) while held-out '
                              'CE1.3735->1.8231/1.5245; held-out argmax2/9->4-5/9. Mechanism: gradient sign agreement '
                              'float32 vs float64 =0.4978, Adam t=1 step difference2.569e-3 vs SGD3.894e-8; 55.24 percent '
                              'zero and 23.17 percent saturated coordinates. Functional outcomes reproduce to 2-5e-8; '
                              'parameter trajectories do not. No backbone, RAM below the registered gate.'),
                      branch_ref=rel(D13)),
        created_at=now, updated_at=now)
    t['nodes']['n14'] = dict(
        id='n14', parent='n11', depth=4, status='executed',
        hypothesis=('Pickup: extending the loop-scoped lazy base reuse to the whole fixed archived scenario raises the '
                    'removed redundant work while preserving every exactness gate, and reduces the measured runtime'),
        insight=('The removed-work fraction rises from 11.9 to 29.7 percent and every runnable exactness gate still holds, '
                 'but measured runtime does not improve and the candidate is slower in both interleaved pairs. Reduced '
                 'call counts are not a speedup, so the pickup runtime micro-optimisation line is closed.'),
        metadata=dict(dev_score=float(A2['checks_count']), test_score=None,
                      result=('Paired REF/CAND on the full archived scenario-8 list: 166 orders, 71 feasibly inserted; '
                              'distance evaluations1042->733,309 repeated base calls removed (29.7 percent). Identical '
                              'choices, feasibility calls,80 C0 segments and terminal hexes; every deleted entry is a base '
                              'call; all1775 distance values independently recomputed. Times0.2635/0.2418 vs0.2704/0.3976. '
                              'Producer1/30 FAIL and audit4 include a registered-fixture design flaw; preserved, no rerun.'),
                      branch_ref=rel(P2)),
        created_at=now, updated_at=now)
    add = ('\n[from n13] Head-only action matching over sealed latents does not transfer to held-out sealed states, and no '
           'normalised update reproduces a float32 optimisation trajectory because the gradient sign at the rounding floor '
           'is not reproducible; only functional outcomes are.\n[from n14] Cutting redundant route evaluations by 29.7 '
           'percent left measured runtime unchanged and slower in both interleaved pairs, so the pickup runtime '
           'micro-optimisation line is closed.')
    if '[from n13]' not in t['nodes']['n0']['insight']:
        t['nodes']['n0']['insight'] += add
    t['nodes']['n0']['updated_at'] = now
    tp.write_text(json.dumps(t, ensure_ascii=False, indent=1) + '\n', encoding='utf-8', newline='\n')
    print('arbor n13/n14 executed; nodes:', len(t['nodes']))


if __name__ == '__main__':
    main()
