"""Assemble cycle-5 executor evidence with computed artifact hashes."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TREE = ROOT / 'research/review_3080_20261010'
CYCLE = TREE / 'dual_track_optimization_20261013_root_v1/cycle5_20261013_root_v1'
PK = TREE / 'pickup/lazy_base_distance_reuse_root_v1'
D1 = TREE / 'delivery/action_matching_head_update_root_v1'
D2 = TREE / 'delivery/action_matching_head_update_root_v2'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rel(p):
    return Path(p).relative_to(ROOT).as_posix()


def hashes(paths):
    return {rel(p): sha(p) for p in paths}


pickup_paths = [PK / n for n in (
    'REGISTRATION.json', 'RESULT.json', 'RUN_EXIT.json', 'INDEPENDENT_AUDIT.json', 'AUDIT_RUN_EXIT.json',
    'SOURCE_START.json', 'SOURCE_END.json', 'RUN_SUPERVISION_RECEIPT.json', 'AUDIT_SUPERVISION_RECEIPT.json',
    'lazy_base_distance.py', 'run.py', 'audit.py', 'supervise.py')]
delivery_paths = [D1 / n for n in ('RESULT.json', 'RUN_EXIT.json', 'CLOSURE_DIAGNOSIS.json',
                                   'closure_diff_diagnosis.py')]
delivery_paths += [D2 / n for n in (
    'REGISTRATION.json', 'RESULT.json', 'RUN_EXIT.json', 'INDEPENDENT_AUDIT.json', 'AUDIT_RUN_EXIT.json',
    'SOURCE_START.json', 'SOURCE_END.json', 'AUDIT_SOURCE_START.json', 'AUDIT_SOURCE_END.json',
    'HEAD_STEP0_GRAD.npz', 'HEAD_STEP1.npz', 'HEAD_AFTER.npz',
    'OPTIMIZER_REPRODUCIBILITY_DIAGNOSIS.json', 'RUN_SUPERVISION_RECEIPT.json',
    'AUDIT_SUPERVISION_RECEIPT.json', 'run.py', 'audit.py', 'head_action_matching.py')]

for p in pickup_paths + delivery_paths:
    assert p.exists(), p

evidence = {
    'pickup': {
        'directory': rel(PK),
        'stable_online_gain': False,
        'formal_strong_dynamic_qualified': False,
        'full_request_budget_qualified': False,
        'scientific_family3_started': False,
        'candidate_installed': False,
        'producer_checks': 7, 'producer_failures': 0, 'producer_exit': 0,
        'independent_checks': 155, 'independent_failures': 0, 'independent_exit': 0,
        'registration_sha256': sha(PK / 'REGISTRATION.json'),
        'original_candidates': 95, 'scored_feasible_candidates': 25,
        'original_reference_distance_calls': 109, 'candidate_distance_calls': 96,
        'proved_removed_base_calls': 13, 'base_computations': 12, 'base_reuses': 13,
        'complete_original_C0_transitions': 9,
        'operation_seconds': 0.027001500013284385, 'reference_operation_seconds': 0.024322999990545213,
        'summary_zh': (
            '取货新组件只把同一次原插入 vehicle 循环内重复求值的原 base 路线距离改成惰性复用，其余保持原串行算法：'
            '固定已存公共记录顺序下原未执行 scenario index8 的前四个揭示序订单(1450/1551/1455/1547)，K10 与 seed20261013'
            '0901 不变，不按效用挑状态、不增预算。生产 7 项 0 失败、独立 155 项 0 失败，两监督自然退出 0；逐位全部通过：'
            '可行调用序列与数值、每一步候选位置、全部原插入选择与私有路线状态、全部已评分候选 score hex 与类型、删除同循环'
            '重复 base 后仍保持 first-trial-then-base 顺序、原 incr_eval=False、9 次完整 C0 物理转移、终点能量/距离 hex、不可变'
            '距离矩阵 SHA 与原公共参数摘要未变、475 实际加载文件/映射起止一致。原候选 95、已评分可行 25；路线距离求值 109→96，'
            '证明省掉 13 次重复原 base 求值(12 次实际计算/13 次复用)。但机制边界同样明确：省调用数不等于提速，实测操作段 '
            '0.0270015 秒反而略高于原参考 0.0243230 秒，未取得任何加速、未执行完整 K10 请求、未触及在线收益。'),
        'next_zh': (
            '不重跑 277 项缓存诊断、103 局部组件或任何已完成 QA，不调容量追速度。下一另事前登记完整 K10 有界请求与公平计时，'
            '严格保留首 trial 后 base 求值顺序、原串行累加、全部可行性与完整 C0，不引入 O(1) 重分组或 incr_eval=True；'
            '局部语义合格后才谈等耗时部署，正式模型提议、同初始化已训 CE、同任务训练随机与真正独立强动态接拒仍需另闭合。'),
        'evidence_hashes': hashes(pickup_paths),
    },
    'delivery': {
        'directory': rel(D2),
        'preserved_failed_attempt': rel(D1),
        'stable_online_gain': False,
        'formal_trained_control_qualified': False,
        'formal_controls_qualified': False,
        'full_request_budget_qualified': False,
        'scientific_family3_started': False,
        'candidate_installed': False,
        'v1_producer_checks': 57, 'v1_producer_failures': 2, 'v1_exit': 1,
        'v1_closure_delta': 'encodings.cp437 lazy codec load only; no changed or removed file',
        'v2_producer_checks': 60, 'v2_producer_failures': 0, 'v2_exit': 0,
        'v2_audit_checks': 97, 'v2_audit_failures': 2, 'v2_audit_exit': 1,
        'registration_sha256': sha(D2 / 'REGISTRATION.json'),
        'head_parameter_elements': 196929,
        'sealed_records': 30, 'supervised_groups': 29, 'skipped_records': 1,
        'update_steps': 232,
        'forward_vs_sealed_scores_max_abs': 5.066394805908203e-07,
        'initial_mean_ce': 1.2889301165038127, 'final_mean_ce': 0.4199951999359259,
        'initial_argmax_correct': 8, 'final_argmax_correct': 25,
        'reload_bit_exact': True, 'reloaded_forward_max_abs': 0.0,
        'audit_failures': ['audit_epoch_ce_trajectory_within_bound', 'audit_final_parameters_within_bound'],
        'audit_functional_agreement': {
            'step0_gradient_max_abs': 4.524014431961376e-07,
            'adam_step1_max_abs': 2.367797341529254e-08,
            'final_ce_at_same_params_max_abs': 1.6368501620789289e-07,
            'final_argmax_correct': 25,
        },
        'diagnosis_float32_vs_float64': {
            'jax_float32_vs_saved_artifact': 0.0,
            'jax_float64_vs_numpy_float64_param': 4.363431629905712e-10,
            'jax_float32_vs_numpy_float64_param': 0.13946666304737146,
            'jax_float32_vs_jax_float64_param': 0.13946666302281596,
            'localisation_to_near_zero_gradients_supported': False,
        },
        'available_ram_gib': 1.9375648498535156, 'full_backbone_threshold_gib': 8.0, 'backbone_loaded': False,
        'summary_zh': (
            '配送本轮把动作匹配评分头更新/固定顺序/保存/重载真正接通，且只使用已封存节点表示、不加载主干。v1 首次生产 57 项 '
            '2 FAIL(设备标签字符串判据 + 加载源闭包起止不等)、退出 1、无 SOURCE_END，原件保留；只读机制诊断 CLOSURE_DIAGNOSIS 证明'
            '该闭包差异仅是第一份封存 npz 懒加载 encodings.cp437 及其 pyc，没有任何文件被改或缺失。v2 事前另登记后：生产 60 项 0 '
            '失败、自然退出 0；独立审计 97 项 2 FAIL。接线身份已逐位确立：30 份封存 pretrained_initial npz 共用同一评分头，其四叶 '
            '196929 参数与 construct_head(seed=20261012) 初值逐位相同；record0 全 JAX 重建 head 输入与封存 eager_concat 逐位相同，'
            '重算前向与封存 scores 最大差 5.066e-7(门 2e-5)。按每个客户的可行动作组构造 29 个监督组(可行子集 2–6，目标须在子集内)，'
            'record23(key[52,2]/客户6)因封存目标动作不在保存动作网格内如实跳过。固定升序 8 epoch=232 步 optax Adam(lr=3e-3) 只更新'
            '四叶：平均组 CE 1.28893→0.419995、argmax 命中 8/29→25/29，28/29 组全候选 argmax 改变；这是 196929 参数对 29 组的记忆'
            '工程接线，不是已训对照、不是收益。保存 HEAD_STEP0_GRAD/HEAD_LAST_GRAD/HEAD_STEP1/HEAD_AFTER 各有 SHA；四叶重载逐位相同，'
            '重载后前向与更新后内存结果逐位相同(0.0)，封存叶往返 5.066e-7。审计两 FAIL 都是跨库复现整条 232 步轨迹：epoch CE 轨迹 '
            '0.02711>1e-4、最终参数 0.13947>1e-4；但功能输出可复现(step0 梯度 4.524e-7、Adam 第一步 2.368e-8、同参数最终 CE 1.637e-7、'
            '最终 argmax 25/29 相同)。机制诊断确认这是 dtype 效应：JAX float32 重跑逐位复现 HEAD_AFTER(0.0)，两个独立 float64 实现互差 '
            '4.36e-10，而 float32 与任一 float64 差 0.13947，同一库内 JAX f32 vs f64 也是 0.13947 ⇒ float32 算术被 Adam 逐坐标归一化放大，'
            '不是算法不一致；同时"差异集中在近零梯度坐标"这一具体定位未被支持(step0 梯度大处平均参数差 0.00303、小处 3.72e-6)。'
            '资源边界如实记录：本机可用 1.94GiB(v1 2.30GiB) 低于 8GiB 主干门，因此本步只更新小评分头，无主干/C0/求解器/模型优化器更新。'),
        'next_zh': (
            '不重跑 v1/v2 或其诊断追 PASS，不用 2e-5 替代严格门。下一另事前有界候选：让步长随梯度幅度退化(纯 SGD 或带噪声下限的'
            '归一化)以获得可跨库复现的头更新轨迹，并接入正式同初始化已训 CE、同任务训练随机与完整 8+2 请求的参数/依赖发布前核验；'
            '在主干门满足前不加载 backbone，也不把本工程更新当已训对照。'),
        'evidence_hashes': hashes(delivery_paths),
    },
    'scientific_family3_started': False,
    'stable_online_gain': False,
    'shared_MD_uploaded': False,
    'final_pickup_seeds_used': False,
}

out = CYCLE / 'EXECUTOR_EVIDENCE.json'
assert not out.exists(), out
out.write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
print(json.dumps({'written': rel(out), 'pickup_evidence': len(evidence['pickup']['evidence_hashes']),
                  'delivery_evidence': len(evidence['delivery']['evidence_hashes'])}))
