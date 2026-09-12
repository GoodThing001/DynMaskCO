"""
R1.5-M Model-Utility Harness — 一次跑 H0/H1/H2/H3，输出四张表 + run_config.json。

H0 = EDD 贪心（无模型）
H1 = NN 贪心（无模型，teacher 风格）
H2 = MaskCO（NN incumbent + 真实 logits，raw selection，同 state 算 shuffled counterfactual 作 audit）
H3 = MaskCO（NN incumbent + shuffled logits，candidate-set 内 permutation，raw）
H2-G = MaskCO（NN incumbent + 真实 logits，guarded service-first selection，R1.6）

H2 vs H3 用同一 seed / mask / candidate set，唯一差异是 edge↔score 映射（H3 shuffle）。

输出（results/r1_5_model_utility/）：
  summary.csv          — 每 method 一行聚合（含 local H2/H3 divergence）
  instance_level.csv   — 每 (method, instance) 一行
  event_audit.csv      — 每 (method, event, vehicle) 一行（仅 model/model_shuffle）
  paired_h2_h3.csv     — H2/H3 按 instance 配对（长期 policy utility）
  run_config.json      — 复现元数据（data/ckpt sha256 + 配置）

用法:
    python scripts/decoding/run_r1_5_model_utility.py \
        --data data/baseline/50_node/val/dcc_50_r1_edod05_val.npz \
        --ckpt ckpts/r1_5_baseline/typed_v1_edge/phase3c/seed42/step50000.ckpt \
        --num_instances 16 --beam_width 16 --capacity 50
"""

import sys, os, argparse, csv, json, hashlib
import numpy as np

_BASE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_BOOTSTRAP = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _SCRIPTS_BOOTSTRAP not in sys.path:
    sys.path.insert(0, _SCRIPTS_BOOTSTRAP)
from project_paths import EXTENSION_ROOT, MASKCO_ROOT
_CVRPTW = str(EXTENSION_ROOT)
_MASKCO = str(MASKCO_ROOT)
sys.path.insert(0, _MASKCO)
sys.path.insert(0, _CVRPTW)
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'models'))
sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', 'simulation'))

from online_decode import StrictOnlineDecoder, build_replanner


def load_model(ckpt_path):
    from flax import nnx
    from training import load_ckpt
    from DynamicColdChainModel import DynamicColdChainModel, DynamicColdChainModelConfig
    params, _, _, model_config, _, _ = load_ckpt(ckpt_path)
    if model_config is None:
        model_config = DynamicColdChainModelConfig.get_config('softcap_fn')
        model_config.encoder_input_dim = 7
    model_config.dtype = 'float32'
    model = model_config.construct_model()
    if params is None:
        raise RuntimeError(f"load_ckpt 返回空 params: {ckpt_path}")
    model = nnx.merge(nnx.graphdef(model), params)
    return model


def run_method(method, dataset, capacity, model, tw_max, tw_speed, beam_width,
               decode_seed, num_instances, shuffle_seed=None,
               save_proposal_trace=False, candidate_mode='legacy_jf1h'):
    replan_fn = build_replanner(method, dataset, capacity, model, tw_max, tw_speed,
                                beam_width, decode_seed=decode_seed,
                                shuffle_seed=shuffle_seed,
                                save_proposal_trace=save_proposal_trace,
                                candidate_mode=candidate_mode)
    decoder = StrictOnlineDecoder(
        dataset, capacity=capacity, model=model, tw_max=tw_max,
        tw_speed=tw_speed, beam_width=beam_width, replan_fn=replan_fn,
    )
    n = min(num_instances, decoder.num_instances)
    per_inst = []
    audit = []
    proposal_trace = []
    for inst_idx in range(n):
        if hasattr(replan_fn, 'reset_audit'):
            replan_fn.reset_audit()
        route, m = decoder.decode_instance(inst_idx)
        per_inst.append(m)
        if hasattr(replan_fn, 'audit'):
            audit.extend(replan_fn.audit)
        if hasattr(replan_fn, 'proposal_trace'):
            proposal_trace.extend(replan_fn.proposal_trace)
    return per_inst, audit, proposal_trace


def _sha256(path):
    if path is None or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _fmt(x):
    """float → 固定精度字符串；None/nan → 'nan'。"""
    if x is None:
        return 'nan'
    x = float(x)
    return 'nan' if np.isnan(x) else f"{x:.4f}"


# ── audit 列（§31 冻结语义）───────────────────────────────────────────────
AUDIT_COLUMNS = [
    'method', 'instance_id', 'event_id', 'clock', 'vehicle_id', 'replan_reason',
    'incumbent_builder', 'logit_mode', 'selection_mode', 'decode_seed', 'decision_seed',
    'visible_pending_count', 'reserved_count', 'initial_actionable_count',
    'candidate_exists', 'candidate_accepted', 'selection_reason',
    'beam_complete', 'fallback_to_incumbent',
    'inc_first', 'candidate_first', 'executed_first',
    'candidate_vs_incumbent_changed',
    'local_real_exists', 'local_shuffle_exists', 'local_real_first', 'local_shuffle_first',
    'local_first_diverged',
    'inc_n_planned', 'candidate_n_planned',
    'inc_cost', 'candidate_cost', 'executed_cost',
]

SUMMARY_COLUMNS = [
    'method', 'n_instances',
    'completion_rate', 'tw_feas_rate', 'cap_feas_rate',
    'mean_unserved', 'mean_duplicate', 'mean_late',
    'distance_all', 'distance_complete_only', 'vehicles',
    'candidate_exists_rate', 'candidate_acceptance_rate', 'beam_complete_rate', 'fallback_rate',
    'candidate_vs_incumbent_change_rate', 'candidate_vs_incumbent_change_rate_valid',
    'local_h2_h3_first_divergence_rate', 'local_h2_h3_proposal_divergence_rate',
    'mean_initial_actionable_count',
]


def _write_audit_rows(w, method, audit):
    for a in audit:
        w.writerow([
            method, a['inst_idx'], a['event_id'], f"{a['clock']:.3f}", a['vehicle_id'],
            a['replan_reason'], a['incumbent_builder'], a['logit_mode'], a['selection_mode'],
            a['decode_seed'], a['decision_seed'],
            a['visible_pending_count'], a['reserved_count'], a['initial_actionable_count'],
            int(a['candidate_exists']), int(a['candidate_accepted']), a['selection_reason'],
            int(a['beam_complete']), int(a['fallback_to_incumbent']),
            a['inc_first'], a['candidate_first'], a['executed_first'],
            int(a['candidate_vs_incumbent_changed']),
            int(a['local_real_exists']), int(a['local_shuffle_exists']),
            a['local_real_first'], a['local_shuffle_first'], int(a['local_first_diverged']),
            a['inc_n_planned'], a['candidate_n_planned'],
            _fmt(a['inc_cost']), _fmt(a['candidate_cost']), _fmt(a['executed_cost']),
        ])


def _audit_stats(audit):
    """从 audit 行计算机制指标（H2 才有 local divergence，edd/nn 无 audit）。"""
    if not audit:
        return {
            'candidate_exists_rate': float('nan'),
            'candidate_acceptance_rate': float('nan'),
            'beam_complete_rate': float('nan'),
            'fallback_rate': float('nan'),
            'candidate_vs_incumbent_change_rate': float('nan'),
            'candidate_vs_incumbent_change_rate_valid': float('nan'),
            'local_h2_h3_first_divergence_rate': float('nan'),
            'local_h2_h3_proposal_divergence_rate': float('nan'),
            'mean_initial_actionable_count': float('nan'),
        }
    cand_exists = np.mean([a['candidate_exists'] for a in audit])
    # AR（R1.6）：#accepted / #candidate_exists（guarded 才有意义，raw 恒 =1）
    exists_rows = [a for a in audit if a['candidate_exists']]
    cand_accept = (np.mean([a['candidate_accepted'] for a in exists_rows])
                   if exists_rows else float('nan'))
    beam_complete = np.mean([a['beam_complete'] for a in audit])
    fallback = np.mean([a['fallback_to_incumbent'] for a in audit])
    cand_vs_inc = np.mean([a['candidate_vs_incumbent_changed'] for a in audit])
    valid = [a['candidate_vs_incumbent_changed'] for a in audit if a['candidate_exists']]
    cand_vs_inc_valid = np.mean(valid) if valid else float('nan')

    # local H2-vs-H3 divergence：只在 H2（logit_mode='real'）且有真实选择空间
    # （initial_actionable_count >= 2）的事件上统计（§30/§39）。
    actionable = [a for a in audit
                  if a['logit_mode'] == 'real' and a['initial_actionable_count'] >= 2]
    n_proposal_div = sum(1 for a in actionable
                         if (a['local_real_exists'] != a['local_shuffle_exists'])
                         or (a['local_real_first'] != a['local_shuffle_first']))
    local_proposal_div = (n_proposal_div / len(actionable)) if actionable else float('nan')
    both_exist = [a for a in actionable
                  if a['local_real_exists'] and a['local_shuffle_exists']]
    local_first_div = (np.mean([a['local_first_diverged'] for a in both_exist])
                       if both_exist else float('nan'))
    mean_actionable = np.mean([a['initial_actionable_count'] for a in audit])
    return {
        'candidate_exists_rate': cand_exists,
        'candidate_acceptance_rate': cand_accept,
        'beam_complete_rate': beam_complete,
        'fallback_rate': fallback,
        'candidate_vs_incumbent_change_rate': cand_vs_inc,
        'candidate_vs_incumbent_change_rate_valid': cand_vs_inc_valid,
        'local_h2_h3_first_divergence_rate': local_first_div,
        'local_h2_h3_proposal_divergence_rate': local_proposal_div,
        'mean_initial_actionable_count': mean_actionable,
    }


def main():
    parser = argparse.ArgumentParser(description='R1.5 model-utility harness (H0-H3)')
    parser.add_argument('--data', type=str, required=True)
    parser.add_argument('--ckpt', type=str, default=None,
                        help='model/model_shuffle 需要；edd/nn 不需要')
    parser.add_argument('--capacity', type=int, default=50)
    parser.add_argument('--beam_width', type=int, default=16)
    parser.add_argument('--num_instances', type=int, default=16)
    parser.add_argument('--tw_speed', type=float, default=1.0)
    parser.add_argument('--decode_seed', type=int, default=42)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--methods', type=str, default='edd,nn,model,model_shuffle,model_guarded',
                        help='逗号分隔的 method 子集（含 model_guarded = H2-G、model_guarded_shuffle = H3-G）')
    parser.add_argument('--shuffle_seeds', type=str, default=None,
                        help='H3-G 的 shuffle seed 列表（逗号分隔，如 42,43,44,45,46）；'
                             '为空则 model_guarded_shuffle 只跑一次（shuffle_seed=None）')
    parser.add_argument('--save_proposal_trace', action='store_true',
                        help='保存完整 route trace 到 proposal_trace.jsonl（R1.7-2 结构审计用）')
    parser.add_argument('--candidate_mode', type=str, default='legacy_jf1h',
                        choices=['legacy_jf1h', 'safe_pair'],
                        help='joint_beam 的 candidate 语义：legacy_jf1h=忠实 JF1-H 退化（JF2 Gate-0），'
                             'safe_pair=JF1.5 assignment beam（已判空）')
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = os.path.join(_CVRPTW, 'results', 'r1_5_model_utility')
    os.makedirs(args.out_dir, exist_ok=True)

    dataset = dict(np.load(args.data))
    tw_max = float(dataset['tw_end'].max())

    shuffle_seeds = ([int(s) for s in args.shuffle_seeds.split(',')]
                     if args.shuffle_seeds else None)

    model = None
    methods = [m for m in args.methods.split(',') if m]
    need_model = any(m in ('model', 'model_shuffle', 'model_guarded',
                           'model_guarded_shuffle', 'joint_real', 'joint_shuffle') for m in methods)
    if need_model:
        if args.ckpt is None:
            parser.error("model 类 method 需要 --ckpt")
        model = load_model(args.ckpt)

    # R1.7：把 model_guarded_shuffle 展开成 per-shuffle-seed 的 (label, method, seed) 列表。
    run_list = []
    for m in methods:
        if m == 'model_guarded_shuffle':
            seeds = shuffle_seeds if shuffle_seeds else [None]
            for s in seeds:
                label = f'model_guarded_shuffle_s{s}' if s is not None else 'model_guarded_shuffle'
                run_list.append((label, m, s))
        else:
            run_list.append((m, m, None))

    summary_path = os.path.join(args.out_dir, 'summary.csv')
    inst_path = os.path.join(args.out_dir, 'instance_level.csv')
    audit_path = os.path.join(args.out_dir, 'event_audit.csv')
    paired_path = os.path.join(args.out_dir, 'paired_h2_h3.csv')
    config_path = os.path.join(args.out_dir, 'run_config.json')
    proposal_trace_path = os.path.join(args.out_dir, 'proposal_trace.jsonl')

    # run_config.json（§56 复现元数据）
    config = {
        'data': args.data,
        'data_sha256': _sha256(args.data),
        'ckpt': args.ckpt,
        'ckpt_sha256': _sha256(args.ckpt),
        'num_instances': args.num_instances,
        'capacity': args.capacity,
        'beam_width': args.beam_width,
        'decode_seed': args.decode_seed,
        'methods': methods,
        'shuffle_seeds': shuffle_seeds,
        'selection_mode': 'raw',  # model_guarded 覆盖为 'guarded'
        'h3_shuffle': 'per_feasible_candidate_set',
    }
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    with open(summary_path, 'w', newline='') as f:
        csv.writer(f).writerow(SUMMARY_COLUMNS)
    with open(inst_path, 'w', newline='') as f:
        csv.writer(f).writerow(['method', 'instance_id', 'complete', 'tw_feasible',
                                'cap_feasible', 'n_unserved', 'n_duplicate', 'n_late',
                                'distance', 'vehicle_count'])
    with open(audit_path, 'w', newline='') as f:
        csv.writer(f).writerow(AUDIT_COLUMNS)

    per_inst_by_method = {}
    for label, method, shuffle_seed in run_list:
        print(f"=== {label} ===", flush=True)
        per_inst, audit, proposal_trace = run_method(
            method, dataset, args.capacity, model, tw_max,
            args.tw_speed, args.beam_width, args.decode_seed,
            args.num_instances, shuffle_seed=shuffle_seed,
            save_proposal_trace=args.save_proposal_trace,
            candidate_mode=args.candidate_mode)
        per_inst_by_method[label] = per_inst
        n = len(per_inst)
        complete = [m['complete'] for m in per_inst]
        dist_all = np.mean([m['distance_cost'] for m in per_inst])
        complete_only = [m['distance_cost'] for m in per_inst if m['complete']]
        dist_complete = np.mean(complete_only) if complete_only else float('nan')

        # R1.7-2：proposal_trace.jsonl（完整 route trace）
        if args.save_proposal_trace and proposal_trace:
            with open(proposal_trace_path, 'a', newline='') as f:
                for t in proposal_trace:
                    t['method'] = label
                    f.write(json.dumps(t, ensure_ascii=False) + '\n')

        with open(inst_path, 'a', newline='') as f:
            w = csv.writer(f)
            for iid, m in enumerate(per_inst):
                w.writerow([label, iid, int(m['complete']), int(m['tw_feasible']),
                            int(m['capacity_feasible']), m['n_unserved'],
                            m['n_duplicate'], m['tw_late_count'],
                            f"{m['distance_cost']:.4f}", m['vehicle_count']])

        if audit:
            with open(audit_path, 'a', newline='') as f:
                _write_audit_rows(csv.writer(f), label, audit)

        stats = _audit_stats(audit)
        with open(summary_path, 'a', newline='') as f:
            csv.writer(f).writerow([
                label, n,
                f"{np.mean(complete):.4f}",
                f"{np.mean([m['tw_feasible'] for m in per_inst]):.4f}",
                f"{np.mean([m['capacity_feasible'] for m in per_inst]):.4f}",
                f"{np.mean([m['n_unserved'] for m in per_inst]):.4f}",
                f"{np.mean([m['n_duplicate'] for m in per_inst]):.4f}",
                f"{np.mean([m['tw_late_count'] for m in per_inst]):.4f}",
                f"{dist_all:.4f}", f"{dist_complete:.4f}",
                f"{np.mean([m['vehicle_count'] for m in per_inst]):.2f}",
                _fmt(stats['candidate_exists_rate']),
                _fmt(stats['candidate_acceptance_rate']),
                _fmt(stats['beam_complete_rate']),
                _fmt(stats['fallback_rate']),
                _fmt(stats['candidate_vs_incumbent_change_rate']),
                _fmt(stats['candidate_vs_incumbent_change_rate_valid']),
                _fmt(stats['local_h2_h3_first_divergence_rate']),
                _fmt(stats['local_h2_h3_proposal_divergence_rate']),
                _fmt(stats['mean_initial_actionable_count']),
            ])

        print(f"    complete={np.mean(complete):.1%} "
              f"unserved={np.mean([m['n_unserved'] for m in per_inst]):.2f} "
              f"dup={np.mean([m['n_duplicate'] for m in per_inst]):.2f} "
              f"dist={dist_all:.2f} veh={np.mean([m['vehicle_count'] for m in per_inst]):.1f} "
              f"cand_exists={stats['candidate_exists_rate']:.1%} "
              f"accept={stats['candidate_acceptance_rate']} "
              f"local_first_div={stats['local_h2_h3_first_divergence_rate']}")

    # paired_h2_h3.csv（§42：长期 policy utility，按 instance 配对）
    if 'model' in per_inst_by_method and 'model_shuffle' in per_inst_by_method:
        h2 = per_inst_by_method['model']
        h3 = per_inst_by_method['model_shuffle']
        with open(paired_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['instance_id', 'h2_complete', 'h3_complete',
                        'h2_unserved', 'h3_unserved', 'h2_distance', 'h3_distance',
                        'h2_vehicles', 'h3_vehicles', 'service_equivalent',
                        'distance_delta', 'distance_delta_pct'])
            for iid in range(min(len(h2), len(h3))):
                a, b = h2[iid], h3[iid]
                service_eq = bool(a['complete'] and b['complete']
                                  and a['n_unserved'] == 0 and b['n_unserved'] == 0)
                delta = a['distance_cost'] - b['distance_cost']
                delta_pct = (delta / b['distance_cost'] * 100.0
                             if b['distance_cost'] > 0 else float('nan'))
                w.writerow([iid, int(a['complete']), int(b['complete']),
                            a['n_unserved'], b['n_unserved'],
                            f"{a['distance_cost']:.4f}", f"{b['distance_cost']:.4f}",
                            a['vehicle_count'], b['vehicle_count'],
                            int(service_eq), _fmt(delta), _fmt(delta_pct)])

    print(f"\n=== DONE ===")
    print(f"  summary:  {summary_path}")
    print(f"  instance: {inst_path}")
    print(f"  audit:    {audit_path}")
    print(f"  paired:   {paired_path if 'model' in per_inst_by_method and 'model_shuffle' in per_inst_by_method else '(skipped)'}")
    print(f"  config:   {config_path}")


if __name__ == '__main__':
    main()
