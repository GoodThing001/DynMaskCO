"""I3-specific online adapter: qualified cards, fixed history pool and controls.

Reuses the audited simulator/task runner. Runs fresh concurrent cond_hist,
warmstart, reconstruction-only control and I3; never imports old baseline rows.
Engineering config can only produce explicitly nonformal smoke evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT.parent
# Resolve transitive cold-chain/simulator imports before loading the model.
# A subprocess entry cannot depend on the parent pipeline's sys.path.
for folder in ('training', 'evaluation', 'models', 'coldchain', 'simulation'):
    sys.path.insert(0, str(ROOT / folder))
from i3_pool_lineage import verified_pool_days
from maskco_scenario import HistoryPools, check_model_generation
import run_a1_step3_compare as base


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class I3OnlineAdapter:
    source_files = ['scripts/evaluation/run_i3_online_compare.py', 'scripts/training/i3_pool_lineage.py']

    def __init__(self, path, digest):
        if sha(path) != digest:
            raise ValueError('Online configuration identity mismatch')
        self.path, self.digest = Path(path), digest
        self.config = json.loads(self.path.read_text(encoding='utf-8'))
        if (self.config['schema'] != 'i3_online_comparison_v1' or
                self.config['qualification'] not in ('engineering', 'formal_development')):
            raise ValueError('Unknown online comparison qualification')
        self.formal_eligible = self.config['qualification'] == 'formal_development'
        self.cards = {}

    def worker_init(self):
        import multiprocessing
        import os
        sets = self.config.get('worker_cpu_sets')
        if sets:
            index = (multiprocessing.current_process()._identity[-1] - 1) % len(sets)
            selected = set(sets[index])
            if not selected <= os.sched_getaffinity(0):
                raise ValueError('Registered worker CPUs are unavailable')
            os.sched_setaffinity(0, selected)

    def qualify(self, args):
        if args.only or args.baseline_from or args.allow_code_mismatch or args.final_test:
            raise ValueError('I3 adapter requires fresh concurrent arms and no final test')
        if args.gate_seed not in (None, 20260926, 20260930, 20261001):
            raise ValueError('Unregistered I3 development seed')
        for key, expected in self.config['online_parameters'].items():
            if getattr(args, key) != expected:
                raise ValueError('Online parameter differs from registration: ' + key)
        specs = self.config['models']
        if set(specs) != {'maskco', 'warmstart', 'reconstruction_control'}:
            raise ValueError('Missing I3 or matched training control')
        if Path(args.model_bin).resolve() != Path(specs['maskco']['path']).resolve():
            raise ValueError('CLI model differs from registered I3 model')
        for name, spec in specs.items():
            if sha(spec['path']) != spec['sha256'] or sha(spec['card']) != spec['card_sha256']:
                raise ValueError('Changed model/card: ' + name)
            card = json.loads(Path(spec['card']).read_text(encoding='utf-8'))
            if Path(spec['card']).resolve() != Path(spec['path']).resolve().parent / 'config.json':
                raise ValueError('Wrong model lineage card: ' + name)
            if card.get('training_complete', True) is not True:
                raise ValueError('Incomplete training: ' + name)
            if self.formal_eligible and name != 'warmstart' and card.get('qualification') != 'formal_development':
                raise ValueError('Engineering checkpoint cannot enter formal I3 gate')
            if name != 'warmstart':
                train = card['i3_training']
                if (train['model_bin_sha256'] != spec['sha256'] or not train['source_stable'] or
                        not train['checkpoint_reload_verified'] or
                        train['initial_trainable_hash'] == train['final_trainable_hash']):
                    raise ValueError('Unqualified I3 training result: ' + name)
                if train['source_start'] != train['source_end']:
                    raise ValueError('Changed training source: ' + name)
            if check_model_generation(spec['path']) != 'ok':
                raise ValueError('Unknown model source generation: ' + name)
            self.cards[name] = card
        actor, control = self.cards['maskco'], self.cards['reconstruction_control']
        for card in self.cards.values():
            if (card.get('train_seed') != 20260925 or
                    card['args']['arm'] != args.arm or card['args']['dim'] != args.dim or
                    card['args']['m_max'] != args.m_max or
                    card['args']['train_instances'] != args.train_instances):
                raise ValueError('CLI architecture/history count differs from model lineage')
        if args.arm in ('pretrained', 'random_cvrp'):
            if not args.cvrp_ckpt or sha(args.cvrp_ckpt) != self.config['cvrp_sha256']:
                raise ValueError('Frozen CVRP checkpoint differs from registration')
        if (actor['warmstart_sha256'] != specs['warmstart']['sha256'] or
                actor['warmstart_sha256'] != control['warmstart_sha256'] or
                actor['history_pool_lineage'] != control['history_pool_lineage'] or
                actor['i3_training']['initial_trainable_hash'] != control['i3_training']['initial_trainable_hash']):
            raise ValueError('Control does not share initialization or decoder pool')
        a, c = dict(actor['i3_training']['settings']), dict(control['i3_training']['settings'])
        if a.pop('objective') != 'decision_reconstruction' or c.pop('objective') != 'reconstruction_control' or a != c:
            raise ValueError('Control changed more than the training objective')
        if actor['args'] != control['args'] or actor['args']['arm'] != args.arm:
            raise ValueError('Model architecture differs from registered online arm')
        if actor['history_pool_lineage'] != self.config['history_pool_lineage']:
            raise ValueError('Online history pool differs from training card')
        if self.formal_eligible:
            sets = self.config.get('worker_cpu_sets', [])
            if (len(sets) != args.workers or any(len(s) != 2 for s in sets) or
                    len(set(cpu for s in sets for cpu in s)) != 2 * args.workers):
                raise ValueError('Formal comparison requires equally sized, disjoint CPU allocations')
            required = {'K': 10, 'time_limit': 10., 'dev_instances': 40, 'n_orders': 200,
                'capacity': 50., 'num_vehicles': 15, 'train_instances': 200,
                'shadow_mode': 'greedy', 'future_policy': 'density', 'energy_pricing': 'marginal',
                'standby_orders': 0, 'rho': .6, 'k_neighbors': 10, 'main_only': True,
                'iterative': False}
            if any(getattr(args, key) != value for key, value in required.items()):
                raise ValueError('Formal I3 main comparison must use the registered problem/10s budget')
            if actor['history_pool_lineage']['excluded_days']:
                raise ValueError('This first formal I3 protocol preserves the full 200-day B pool')
            p = self.config['protocol']
            if sha(p['path']) != p['sha256']:
                raise ValueError('Online protocol identity mismatch')
            protocol = json.loads(Path(p['path']).read_text(encoding='utf-8'))
            if (protocol.get('predeclared') is not True or protocol['main_comparison'] != 'I3_minus_cond_hist' or
                    protocol['delta_3'] != 20 or protocol['online_days'] != 40 or
                    protocol['settings'] != actor['i3_training']['settings'] or
                    protocol['warmstart_sha256'] != specs['warmstart']['sha256']):
                raise ValueError('Online training choices differ from protocol')

    def build_pools(self, history_data, history):
        days = verified_pool_days(history_data, self.config['history_pool_lineage'])
        return HistoryPools([history[i] for i in days])

    def make_specs(self, specs):
        result = {'maskco': specs['maskco'], 'cond_hist': specs['cond_hist']}
        for name in ('warmstart', 'reconstruction_control'):
            result[name] = {**specs['maskco'], 'model_bin': str(Path(self.config['models'][name]['path']).resolve())}
        # Randomized, registered arm order balances process/core assignment;
        # no ordering choice depends on measured utility.
        if 'arm_order_seed' in self.config:
            import numpy as np
            names = list(result)
            np.random.default_rng(self.config['arm_order_seed']).shuffle(names)
            result = {name: result[name] for name in names}
        return result

    def identity(self):
        if sha(self.path) != self.digest:
            raise ValueError('Online config changed during run')
        for spec in self.config['models'].values():
            if sha(spec['path']) != spec['sha256'] or sha(spec['card']) != spec['card_sha256']:
                raise ValueError('Model/card changed during online run')
        return {'qualification': self.config['qualification'], 'config_sha256': self.digest,
                'history_pool_lineage': self.config['history_pool_lineage'],
                'model_bin_sha256': {name: s['sha256'] for name, s in self.config['models'].items()},
                'fresh_concurrent_controls': True, 'matched_offline_wall_compute': False}


def main(argv=None):
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument('--online-config', required=True)
    ap.add_argument('--online-config-sha256', required=True)
    args, remaining = ap.parse_known_args(argv)
    adapter = I3OnlineAdapter(args.online_config, args.online_config_sha256)
    return base.main(remaining, online_adapter=adapter)


if __name__ == '__main__':
    main()
