"""REINFORCE 直接目标训练：复用 MpreTrainedModel，对完整修复采样做策略梯度。

奖励 r(P) = (J_vis(P0) − J_vis(P)) / s_TRAIN（可行但变差保留负奖励；失败用预先固定值）。
基线 b_ik = (1/(K−1)) Σ_{j≠k} r_ij（同状态其他样本均值）。
损失 L = −(1/(BK)) Σ stopgrad(r−b) Σ_t log π(a_t | s_t)。

采样与可微回放分离：stepwise_sample 用 NumPy 采样并记录每步状态+选中索引；更新前用相同参数的
JAX score_fn 回放 log-prob 再反传。只更新可训练分区。旧轨迹不跨更新反复使用（每步重新采样）。

用法：
  python scripts/training/train_mpre_reinforce.py --smoke ...   # 单批 smoke
  python scripts/training/train_mpre_reinforce.py --num-steps 1000 --batch-states 8 --K 4 --seed 42 ...
"""
import argparse
import json
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
import optax

_BASE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.dirname(_BASE)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
from project_paths import EXTENSION_ROOT
_CVRPTW = str(EXTENSION_ROOT)
for p in ('simulation', 'evaluation', 'baselines', 'coldchain', 'models', 'data', 'training'):
    sys.path.insert(0, os.path.join(_CVRPTW, 'scripts', p))

from strict_online_env import StrictOnlineEnv
from jf1h_repair import JF1HRepairReplanner
from action_contract import build_vehicle_plans
from cc_lns_replanner import _select_one_mask, _plan_suffixes
from dynmaskco_cc_context import mask_candidate_pool, validate_mask_scope
from dynmaskco_cc_graph import build_visible_index, build_plan_adjacency
from repair_state import extract_repair_state_v1, F_NODE, F_ACTION_EXPLICIT, freeze_vehicles
from mpre_policy import stepwise_sample, replay_log_prob, enumerate_legal_actions
from mpre_replanner import retained_edge_ratio
from mpre import load_cvrp_model
from mpre_trained import load_mpre_trained, partition
from visible_state import build_visible_state, evaluate_visible_plan
from coldchain_contract import (default_pilot_contract, apply_objective_profile,
                                ObjectiveProfile, load_coldchain_contract)


def _load_profile(path):
    with open(path) as f:
        d = json.load(f)
    if d.get('name') == 'o0cc-pilot-devmean-equal-v1':
        raise ValueError("拒绝加载 INVALIDATED v1 profile")
    return ObjectiveProfile(name=d['name'], distance_scale=float(d['distance_scale']),
                            quality_scale=float(d['quality_scale']),
                            energy_scale=float(d['energy_scale']),
                            lambda_quality=float(d['lambda_quality']),
                            lambda_energy=float(d['lambda_energy']),
                            scale_source=d.get('scale_source', 'pilot'),
                            dev_statistics=d.get('dev_statistics'))


FAIL_REWARD = -1.0   # 预先固定的失败奖励（无合法动作 / 不可行 / 非有限）


def make_extract_fn(env, inst_idx, clock, vehicles, served_mask, visible_ids, capacity,
                    tw_max, node_to_local, local_to_node, Nv, P0, mutable_ids):
    def extract_fn(plans, remaining, legal):
        st = extract_repair_state_v1(env, inst_idx, clock, vehicles, served_mask, visible_ids,
                                     plans, remaining, legal, capacity, tw_max,
                                     allowed_vehicle_ids=mutable_ids,
                                     node_to_local=node_to_local, local_to_node=local_to_node,
                                     Nv=Nv)
        A0 = build_plan_adjacency(P0, node_to_local, Nv)
        st['timestep'] = retained_edge_ratio(A0, st['adjmat'][0])
        return st
    return extract_fn


def make_jit_score(gd, frozen_state, Nv_max, M_max):
    """JIT 功能式前向（固定容量 padding，避免逐形状重编译）。"""
    @jax.jit
    def _score(ts_, raw_3d, node_valid, node_feats, timestep, adjmat, cust, pred, succ,
               action_feats):
        m = nnx.merge(gd, ts_, frozen_state)
        return m.score_actions(raw_3d, node_valid, node_feats, timestep, adjmat,
                               cust, pred, succ, action_feats)[0][0]

    def score_from_state(ts_, st):
        Nv = st['raw_3d'].shape[1]
        M = st['cust'].shape[1]
        pn = Nv_max - Nv
        pm = M_max - M
        raw = jnp.pad(jnp.array(st['raw_3d']), ((0, 0), (0, pn), (0, 0)))
        nv = jnp.pad(jnp.array(st['node_valid']), ((0, 0), (0, pn)), constant_values=False)
        nf = jnp.pad(jnp.array(st['node_feats']), ((0, 0), (0, pn), (0, 0)))
        adj = jnp.pad(jnp.array(st['adjmat'][0]), ((0, pn), (0, pn)))
        cust = jnp.pad(jnp.array(st['cust']), ((0, 0), (0, pm)))
        pred = jnp.pad(jnp.array(st['pred']), ((0, 0), (0, pm)))
        succ = jnp.pad(jnp.array(st['succ']), ((0, 0), (0, pm)))
        af = jnp.pad(jnp.array(st['action_feats']), ((0, 0), (0, pm), (0, 0)))
        s = _score(ts_, raw, nv, nf, jnp.array([st['timestep']], jnp.float32), adj,
                   cust, pred, succ, af)
        return s[:M]
    return score_from_state


def sample_state_repairs(extract_fn, score_fn, env, inst_idx, partial, mask_set, mutable_ids,
                         vis, P0, J0, contract, objective, s_TRAIN, K, temperature, rng):
    records_list = []
    rewards = []
    for _ in range(K):
        P, steps, records, fail = stepwise_sample(score_fn, extract_fn, env, inst_idx, partial,
                                                  mask_set, mutable_ids, temperature, rng)
        if fail is not None:
            r = FAIL_REWARD
        else:
            r_eval = evaluate_visible_plan(vis, _plan_suffixes(P), contract, objective)
            r = ((J0 - r_eval.J_vis) / s_TRAIN) if (r_eval.finite and r_eval.feasible) else FAIL_REWARD
        rewards.append(float(r))
        records_list.append(records)
    return records_list, rewards


class CollectProbe(JF1HRepairReplanner):
    """采集 eligible 决策点状态（P0 + partial + mask + visible state）。

    修复（2026-09-21）：① 决策时刻深拷贝 vehicles/served_mask/visible_ids/replan_ids，
    隔离模拟器后续修改；② 每实例最多 max_per_instance 个状态（固定事件顺序，非全局上限）。
    """

    def __init__(self, contract, capacity, max_per_instance=4):
        super().__init__(slack_vehicles=1)
        self.contract = contract
        self.capacity = capacity
        self.max_per_instance = int(max_per_instance)
        self.per_instance = {}
        self.states = []

    def plan(self, env, inst_idx, clock, vehicles, served_mask, visible_ids, replan_ids=None):
        reserved_ids = set()
        idle_ids = sorted((v.vehicle_id for v in vehicles if v.status == 'idle'), reverse=True)
        reserved_ids = set(idle_ids[:1])
        super().plan(env, inst_idx, clock, vehicles, served_mask, visible_ids,
                     replan_ids=replan_ids)
        if self.per_instance.get(int(inst_idx), 0) >= self.max_per_instance:
            return
        protected = set()
        P0 = build_vehicle_plans(env, inst_idx, vehicles)
        for vid in reserved_ids:
            v = vehicles[vid]
            p = P0.get(vid)
            if v.status == 'idle' and v.current_node == 0 and (p is None or not p.suffix):
                protected.add(vid)
        mutable_ids = set(int(v.vehicle_id) for v in vehicles
                          if v.status in ('idle', 'ready')
                          and (replan_ids is None or v.vehicle_id in replan_ids)) - protected
        pool = mask_candidate_pool(vehicles, served_mask, visible_ids,
                                   protected=protected, plans=P0)
        mask = _select_one_mask(env, inst_idx, P0, pool, 2,
                                seed=self.per_instance.get(int(inst_idx), 0))
        if not mask:
            return
        validate_mask_scope(vehicles, served_mask, visible_ids, P0, mask, mutable_ids,
                            protected=protected)
        mask_set = set(mask)
        partial = {vid: type(p)(p.vehicle_id, p.anchor_node, p.anchor_time, p.anchor_load,
                                tuple(x for x in p.suffix if x not in mask_set))
                   for vid, p in P0.items()}
        vis = build_visible_state(env, inst_idx, clock, getattr(env, 'event_id', -1), vehicles,
                                  served_mask, visible_ids, replan_ids, self.deferred_customers)
        r0 = evaluate_visible_plan(vis, _plan_suffixes(P0), self.contract, self.contract.objective)
        # 决策时刻独立快照（冻结可变输入）
        self.states.append({'inst': int(inst_idx), 'clock': float(clock), 'env': env,
                            'event': int(getattr(env, 'event_id', -1)),
                            'vehicles': freeze_vehicles(vehicles),
                            'served_mask': np.array(served_mask, copy=True),
                            'visible_ids': list(visible_ids),
                            'replan_ids': list(replan_ids) if replan_ids is not None else None,
                            'P0': P0, 'partial': partial, 'mask_set': mask_set,
                            'mutable_ids': mutable_ids, 'vis': vis, 'J0': float(r0.J_vis)})
        self.per_instance[int(inst_idx)] = self.per_instance.get(int(inst_idx), 0) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--cvrp-ckpt', required=True)
    ap.add_argument('--capacity', type=float, default=50.0)
    ap.add_argument('--num-vehicles', type=int, default=25)
    ap.add_argument('--objective', choices=['distance', 'coldchain'], default='coldchain')
    ap.add_argument('--objective-profile', default=None)
    ap.add_argument('--coldchain-contract', default=None)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--num-steps', type=int, default=1000)
    ap.add_argument('--batch-states', type=int, default=8)
    ap.add_argument('--K', type=int, default=4)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--max-instances', type=int, default=64)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    dataset = dict(np.load(args.data))
    profile = _load_profile(args.objective_profile) if args.objective == 'coldchain' else None
    contract = (load_coldchain_contract(args.coldchain_contract)
                if args.coldchain_contract else default_pilot_contract())
    eff = apply_objective_profile(contract, profile) if args.objective == 'coldchain' else None
    tw_max = float(dataset['tw_end'][:, 0].max())
    backbone, cfg, _step = load_cvrp_model(args.cvrp_ckpt)
    model, _, _ = load_mpre_trained(args.cvrp_ckpt, F_NODE, F_ACTION_EXPLICIT, seed=args.seed)
    gd, train_state, frozen_state, other_state = partition(model)
    assert sum(1 for _ in jax.tree_util.tree_leaves(other_state)) == 0, 'param leak'

    # 采集决策点池（每实例最多 4 个，固定事件顺序）
    n_instances = 2 if args.smoke else args.max_instances
    collector = CollectProbe(eff, args.capacity, max_per_instance=4)
    for inst in range(n_instances):
        env = StrictOnlineEnv(dataset, args.capacity, 1.0, args.num_vehicles,
                              replanner=collector, coldchain_contract=eff)
        env.run(inst)
    pool = collector.states
    if not pool:
        raise RuntimeError('无可用决策点')
    from collections import Counter
    inst_dist = Counter(st['inst'] for st in pool)
    s_TRAIN = float(np.asarray([st['J0'] for st in pool]).std())
    s_TRAIN = max(s_TRAIN, 1e-6)
    print(f"collected {len(pool)} decision points across {len(inst_dist)} instances; "
          f"s_TRAIN={s_TRAIN:.4f}", flush=True)
    with open(os.path.join(args.out, 'state_manifest.json'), 'w') as f:
        json.dump({'inst_event': [(st['inst'], st['event']) for st in pool],
                   'inst_dist': dict(sorted(inst_dist.items())), 's_TRAIN': s_TRAIN}, f, indent=2)

    # 预计算每个状态的 extract_fn（模型无关）+ 固定信息
    metas = []
    for st in pool:
        env = st['env']
        inst_idx = st['inst']
        node_to_local, local_to_node, Nv = build_visible_index(
            [int(i) for i in st['visible_ids'] if int(i) != 0])
        extract_fn = make_extract_fn(env, inst_idx, st['clock'], st['vehicles'], st['served_mask'],
                                     st['visible_ids'], args.capacity, tw_max,
                                     node_to_local, local_to_node, Nv, st['P0'], st['mutable_ids'])
        metas.append({'extract_fn': extract_fn, 'env': env, 'inst_idx': inst_idx,
                      'partial': st['partial'], 'mask_set': st['mask_set'],
                      'mutable_ids': st['mutable_ids'], 'vis': st['vis'], 'P0': st['P0'],
                      'J0': st['J0']})

    rng = np.random.default_rng(args.seed)
    rng_sample = np.random.default_rng(args.seed + 1)
    temperature = args.temperature
    B = min(args.batch_states, len(pool))
    K = args.K
    tx = optax.adam(args.lr)
    opt_state = tx.init(train_state)

    # 固定容量：预扫确定 Nv_max / M_max（避免逐形状 JIT 重编译）
    Nv_max = max(1 + len([i for i in st['visible_ids'] if int(i) != 0]) for st in pool)
    M_max = 0
    for meta in metas:
        legal = enumerate_legal_actions(meta['env'], meta['inst_idx'], meta['partial'],
                                        list(meta['mask_set']), meta['mutable_ids'])
        M_max = max(M_max, len(legal))
    M_max = max(M_max, 1)
    print(f'Nv_max={Nv_max} M_max={M_max}', flush=True)
    score_from_state = make_jit_score(gd, frozen_state, Nv_max, M_max)
    num_steps = 1 if args.smoke else args.num_steps
    loss_hist = []
    for step in range(num_steps):
        score_fn = lambda st: np.asarray(score_from_state(train_state, st))
        idxs = rng.integers(0, len(pool), size=B)
        all_records = []
        all_rewards = []
        for i in idxs:
            meta = metas[int(i)]
            records, rewards = sample_state_repairs(
                meta['extract_fn'], score_fn, meta['env'], meta['inst_idx'], meta['partial'],
                meta['mask_set'], meta['mutable_ids'], meta['vis'], meta['P0'], meta['J0'],
                eff, eff.objective, s_TRAIN, K, temperature, rng_sample)
            all_records.extend(records)
            all_rewards.extend(rewards)
        rewards = np.asarray(all_rewards).reshape(B, K)
        baseline = np.array([(rewards[i].sum() - rewards[i, k]) / (K - 1)
                             for i in range(B) for k in range(K)])
        advantages = rewards.reshape(-1) - baseline

        def loss_fn(ts_):
            replay_score = lambda st: score_from_state(ts_, st)
            total = jnp.array(0.0)
            for ik, records in enumerate(all_records):
                if not records:
                    continue
                lp = replay_log_prob(replay_score, records)
                total = total + jax.lax.stop_gradient(advantages[ik]) * lp
            return -total / (B * K)

        loss, grads = jax.value_and_grad(loss_fn)(train_state)
        updates, opt_state = tx.update(grads, opt_state, train_state)
        train_state = optax.apply_updates(train_state, updates)
        loss_hist.append(float(loss))
        if (step + 1) % 100 == 0 or step == 0:
            print(f"  step {step+1}/{num_steps} loss={float(loss):.4f} "
                  f"reward_mean={rewards.mean():+.3f}", flush=True)

    # 保存 checkpoint（train_state + 身份）
    with open(os.path.join(args.out, 'model.ckpt'), 'wb') as f:
        import pickle
        pickle.dump({'train_state': train_state, 's_TRAIN': s_TRAIN,
                     'F_NODE': F_NODE, 'F_ACTION_EXPLICIT': F_ACTION_EXPLICIT,
                     'num_steps': num_steps, 'batch_states': B, 'K': K, 'seed': args.seed}, f)
    summary = {'num_steps': num_steps, 'batch_states': B, 'K': K, 's_TRAIN': s_TRAIN,
               'n_states': len(pool), 'final_loss': float(loss_hist[-1]),
               'loss_hist': loss_hist, 'seed': args.seed}
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"done: {num_steps} updates, final_loss={loss_hist[-1]:.4f}, s_TRAIN={s_TRAIN:.4f}")
    print(f"saved: {args.out}/model.ckpt + summary.json")


if __name__ == '__main__':
    main()
