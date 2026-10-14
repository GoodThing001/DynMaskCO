"""D2 follow-up: READ-ONLY behaviour equivalence, training-era vs current sources.

Authorised scope (user ruling 2026-10-14): the only next step is a read-only audit of
whether the training-era and current mask-pool / geometric-action sources behave
differently.  No effect comparison, no training, no writes to any frozen tree.

Two independent lines of evidence:
  A. CONSTRUCTIVE: per-function source-segment comparison.  If a function body is
     byte-identical after whitespace normalisation, its behaviour is identical for ALL
     inputs -- a proof, not a sample.
  B. DYNAMIC: import both module versions side by side and compare, on a state suite
     declared BEFORE looking at results, the ORDERED outputs of
     decision_pool_from_vehicles / mask_candidate_pool, the accept-reject behaviour of
     validate_mask_scope (including masks that violate each clause), and a
     seed-dependent selection over the pool (order matters for seeded sampling, so set
     equality alone is not sufficient).

verdict is one of the user's three labels:
    behavior_equivalent_on_registered_states | behavior_different | not_verifiable
with an explicit `basis` and `not_proven` so the label is never read as more than it is.
"""
import ast
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
EXT = ROOT / 'C-VRP_Cold-chainVehicleRoutingProblem'
OUT = Path(__file__).resolve().parent

PAIRS = {
    'dynmaskco_cc_context': {
        'train': EXT / 'negative_result_paper/release/v1_modules/dynmaskco_cc_context.py',
        'cur': EXT / 'scripts/simulation/dynmaskco_cc_context.py',
        'train_sha_expected': '9de8ceb54d088b8afa087948d6469f8cbd6f503678715bbf39e9971f5a91be54',
    },
    'cc_lns_replanner': {
        'train': EXT / 'negative_result_paper/release/v1_modules/cc_lns_replanner.py',
        'cur': EXT / 'scripts/simulation/cc_lns_replanner.py',
        'train_sha_expected': '7ea14c07e13943ea41bf9123f8c4f47d40546970b09d0e70777f8d4f8d9437c3',
    },
}
TRAIN_MPRE_REINFORCE_SHA = '33c617a3e24bfc4c5d68c2ca4e72f41561999906dcea4795493cc8a2aad1b5fd'

# ---- declared BEFORE any comparison: the synthetic state suite and its coverage -----
ARM = 'D2_EQ_V1'
SUITE_NOTE = ('synthetic, branch-covering, deterministic; declared before execution. '
              'Covers: committed vehicles, non-replan frozen tails, deferred customers, '
              'protected vehicle tails, depot 0 in suffix, empty pool, id ordering.')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def func_segments(path):
    src = Path(path).read_text(encoding='utf-8')
    tree = ast.parse(src)
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            seg = ast.get_source_segment(src, node)
            # normalise only trailing whitespace per line: a real behavioural change
            # always survives this normalisation
            norm = '\n'.join(l.rstrip() for l in (seg or '').splitlines())
            out[node.name] = hashlib.sha256(norm.encode('utf-8')).hexdigest()
    return out


class V:
    def __init__(self, vid, status='idle', needs_replan=True, committed_next=None,
                 suffix=()):
        self.vehicle_id = vid
        self.status = status
        self.needs_replan = needs_replan
        self.committed_next = committed_next
        self.mutable_suffix = list(suffix)


class P:
    def __init__(self, suffix):
        self.suffix = list(suffix)


class Plans(dict):
    pass


def build_state_suite():
    """(name, vehicles, served_ids, visible_ids, protected, plans) -- fully deterministic."""
    s = []
    # S1 plain: two idle replan vehicles, all customers free
    s.append(('S1_plain_idle',
              [V(1, 'idle', True, None, (3, 4)), V(2, 'ready', True, None, (5,))],
              [], [1, 2, 3, 4, 5], frozenset(), None))
    # S2 committed vehicle contributes committed_next -> must be excluded
    s.append(('S2_committed_next',
              [V(1, 'committed', True, 3, (4,)), V(2, 'idle', True, None, (5,))],
              [], [1, 2, 3, 4, 5], frozenset(), None))
    # S3 non-replan vehicle frozen tail -> whole tail locked
    s.append(('S3_nonreplan_tail',
              [V(1, 'committed', False, None, (3, 4, 0)), V(2, 'idle', True, None, (5,))],
              [], [1, 2, 3, 4, 5], frozenset(), None))
    # S4 served customers excluded
    s.append(('S4_served',
              [V(1, 'idle', True, None, (3, 4)), V(2, 'idle', True, None, (5,))],
              [2], [1, 2, 3, 4, 5], frozenset(), None))
    # S5 protected vehicle without plans -> falls back to mutable_suffix
    s.append(('S5_protected_no_plans',
              [V(1, 'idle', True, None, (3, 4)), V(2, 'idle', True, None, (5, 6))],
              [], [1, 2, 3, 4, 5, 6], frozenset({2}), None))
    # S6 protected vehicle WITH plans -> uses plan.suffix (may differ from mutable_suffix)
    s.append(('S6_protected_with_plans',
              [V(1, 'idle', True, None, (3, 4)), V(2, 'idle', True, None, (5, 6))],
              [], [1, 2, 3, 4, 5, 6], frozenset({2}),
              Plans({2: P((5, 6, 0))})))
    # S7 deferred customer: not in any suffix, still in pool
    s.append(('S7_deferred',
              [V(1, 'idle', True, None, (3,)), V(2, 'idle', True, None, ())],
              [], [1, 2, 3, 4, 9], frozenset(), None))
    # S8 empty pool
    s.append(('S8_empty',
              [V(1, 'committed', False, None, (3, 4))],
              [], [3, 4], frozenset(), None))
    # S9 id ordering: visible_ids deliberately NOT sorted
    s.append(('S9_order_unsorted_visible',
              [V(1, 'idle', True, None, (7, 2)), V(2, 'idle', True, None, (9,))],
              [], [9, 2, 7, 1, 5], frozenset(), None))
    # S10 depot 0 inside a suffix must never enter the pool
    s.append(('S10_depot_in_suffix',
              [V(1, 'idle', True, None, (0, 4)), V(2, 'idle', True, None, (0,))],
              [], [1, 2, 4], frozenset(), None))
    return s


def load(path, name):
    # the modules import sibling project modules (strict_online_env, ...) by bare name
    for sub in ('simulation', 'models', 'evaluation', 'coldchain', 'data', 'training'):
        p = str(EXT / 'scripts' / sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    res = {'stage': 'D2_followup_behaviour_equivalence', 'arm': ARM,
           'scope': 'read-only', 'effect_comparison': False, 'training_started': False,
           'state_suite': {'note': SUITE_NOTE, 'n_states': None},
           'constructive': {}, 'dynamic': {}, 'source_identity': {}}

    # ---- source identity ----
    for name, p in PAIRS.items():
        res['source_identity'][name] = {
            'train_path': str(p['train'].relative_to(ROOT)),
            'train_sha256': sha(p['train']) if p['train'].is_file() else None,
            'train_sha_expected': p['train_sha_expected'],
            'train_sha_matches': (sha(p['train']) == p['train_sha_expected']
                                  if p['train'].is_file() else False),
            'current_path': str(p['cur'].relative_to(ROOT)),
            'current_sha256': sha(p['cur']) if p['cur'].is_file() else None,
            'bytes_train': (p['train'].stat().st_size if p['train'].is_file() else None),
            'bytes_current': (p['cur'].stat().st_size if p['cur'].is_file() else None),
        }
    trainer = EXT / 'scripts/training/train_mpre_reinforce.py'
    cur_trainer = sha(trainer) if trainer.is_file() else None
    res['source_identity']['train_mpre_reinforce.py'] = {
        'path': 'scripts/training/train_mpre_reinforce.py',
        'current_sha256': cur_trainer,
        'sha_at_F_training': TRAIN_MPRE_REINFORCE_SHA,
        'matches': cur_trainer == TRAIN_MPRE_REINFORCE_SHA,
        'why_it_matters': ('the TRAINER is where the F/E2 mask protocol must be comparable; '
                           'this file is separate from the evaluation-time mask pool and the '
                           'two requirements are NOT substitutes')}

    # ---- A. constructive: per-function segment hashes ----
    for name, p in PAIRS.items():
        if not (p['train'].is_file() and p['cur'].is_file()):
            res['constructive'][name] = {'comparable': False}
            continue
        a, b = func_segments(p['train']), func_segments(p['cur'])
        keys = sorted(set(a) | set(b))
        rows = {k: {'train': a.get(k), 'current': b.get(k),
                    'identical': a.get(k) == b.get(k) and a.get(k) is not None}
                for k in keys}
        res['constructive'][name] = {
            'comparable': True,
            'functions': rows,
            'n_identical': sum(1 for r in rows.values() if r['identical']),
            'n_functions': len(rows),
            'all_identical': all(r['identical'] for r in rows.values()),
        }

    # ---- B. dynamic: ordered outputs on the declared suite ----
    try:
        mt = load(PAIRS['dynmaskco_cc_context']['train'], 'ctx_train')
        mc = load(PAIRS['dynmaskco_cc_context']['cur'], 'ctx_cur')
        suite = build_state_suite()
        res['state_suite']['n_states'] = len(suite)
        rows, diffs = [], []
        for (sname, veh, served, visible, prot, plans) in suite:
            served_mask = [False] * 64
            for c in served:
                served_mask[c] = True
            out = {}
            for tag, mod in (('train', mt), ('current', mc)):
                dp = [int(c) for c in mod.decision_pool_from_vehicles(veh, served_mask, visible)]
                mp = [int(c) for c in mod.mask_candidate_pool(veh, served_mask, visible,
                                                              protected=prot, plans=plans)]
                mp_np = [int(c) for c in mod.mask_candidate_pool(veh, served_mask, visible)]
                # seeded selection over the ORDERED pool: order must be compared too
                sel = []
                if mp:
                    rng = random.Random(20261014)
                    sel = [rng.choice(mp) for _ in range(8)]
                # validate_mask_scope: legal mask and three violating masks.
                # mutable_ids MUST be supplied fairly: the training-era version uses it to
                # build `mutable_plan_customers`, so passing None would make it raise for a
                # harness reason rather than a semantic one.
                mutable_ids = {v.vehicle_id for v in veh
                               if v.status in ('idle', 'ready') and v.needs_replan}
                # FAITHFUL plans: the real pipeline passes the incumbent FleetPlan (P0), which
                # is populated.  Passing {} would make the training-era clause reject every
                # mask for a harness reason, overstating the difference.
                plans_full = Plans({v.vehicle_id: P(v.mutable_suffix) for v in veh})
                if plans:
                    plans_full.update(plans)
                vres, vres_empty = {}, {}
                for label, mask in (('legal', mp[:3]),
                                    ('not_in_pool', [999]),
                                    ('committed', [3] if sname == 'S2_committed_next' else [999]),
                                    ('protected_tail', sorted(prot)[:1])):
                    for store, pl in ((vres, plans_full), (vres_empty, {})):
                        try:
                            mod.validate_mask_scope(veh, served_mask, visible, pl,
                                                    mask, mutable_ids, protected=prot)
                            store[label] = 'ok'
                        except AssertionError as exc:
                            store[label] = 'AssertionError: %s' % str(exc)[:60]
                        except Exception as exc:
                            store[label] = type(exc).__name__
                # which pool customers are absent from every mutable vehicle's plan suffix
                in_mutable_plans = set()
                for vid, p in plans_full.items():
                    if vid in mutable_ids:
                        in_mutable_plans.update(int(x) for x in p.suffix if int(x) > 0)
                deferred_in_pool = [c for c in mp if c not in in_mutable_plans]
                out[tag] = {'decision_pool_ordered': dp, 'mask_pool_ordered': mp,
                            'mask_pool_noprotected': mp_np, 'seeded_selection': sel,
                            'mutable_ids': sorted(mutable_ids),
                            'deferred_customers_in_pool': deferred_in_pool,
                            'validate_with_faithful_plans': vres,
                            'validate_with_empty_plans': vres_empty}
            same = out['train'] == out['current']
            rows.append({'state': sname, 'identical': same,
                         'train': out['train'], 'current': out['current']})
            if not same:
                diffs.append(sname)
        res['dynamic'] = {
            'module': 'dynmaskco_cc_context',
            'n_states': len(rows), 'n_identical': sum(1 for r in rows if r['identical']),
            'n_different': len(diffs), 'different_states': diffs, 'rows': rows,
            'compared': ['decision_pool order', 'mask_candidate_pool order (protected & not)',
                         'seeded selection over the ordered pool', 'validate_mask_scope '
                         'accept/reject on a legal mask and on three clause-violating masks'],
        }
    except BaseException as exc:                                   # pragma: no cover
        import traceback
        res['dynamic'] = {'module': 'dynmaskco_cc_context', 'executed': False,
                          'traceback': traceback.format_exc()[-1500:],
                          'error': repr(exc)}

    # ---- verdict ----
    ctx_c = res['constructive'].get('dynmaskco_cc_context', {})
    ctx_d = res['dynamic']
    fns = ctx_c.get('functions') or {}
    changed = sorted(k for k, v in fns.items() if not v['identical'])
    pool_fns = ('decision_pool_from_vehicles', 'mask_candidate_pool')
    pool_identical = all(fns.get(f, {}).get('identical') for f in pool_fns)
    dyn_executed = ctx_d.get('executed', True) and ctx_d.get('n_states')
    dyn_diff = ctx_d.get('n_different', 0) or 0

    if changed or dyn_diff:
        verdict = 'behavior_different'
        bits = []
        if pool_identical:
            bits.append('the POOL GENERATORS (decision_pool_from_vehicles, '
                        'mask_candidate_pool) are byte-identical, so the pool contents and '
                        'order are the same')
        else:
            bits.append('a pool generator itself differs')
        if changed:
            bits.append('but these functions differ: %s' % ', '.join(changed))
        if dyn_diff:
            bits.append('and the ordered/validated outputs differ on %d/%s suite states'
                        % (dyn_diff, ctx_d.get('n_states')))
        basis = '; '.join(bits)
    elif pool_identical and dyn_executed:
        verdict = 'behavior_equivalent_on_registered_states'
        basis = ('every function of the context module is byte-identical after whitespace '
                 'normalisation (constructive proof for all inputs), AND the ordered outputs '
                 'agree on every state of the declared suite')
    else:
        verdict = 'not_verifiable'
        basis = 'neither line of evidence is conclusive'

    lns_c = res['constructive'].get('cc_lns_replanner', {})
    res['verdict'] = {
        'mask_pool': verdict, 'basis': basis,
        'geometric_arm_cc_lns_replanner': (
            'functions_identical' if lns_c.get('all_identical')
            else ('functions_differ: %s' % ', '.join(
                k for k, v in (lns_c.get('functions') or {}).items() if not v['identical']))),
        'not_proven': [
            'the state suite is synthetic and branch-covering, NOT the historical public '
            'states; no historical state contents were reconstructed',
            'a byte-identical function body proves input-output equivalence of that function, '
            'not of the module as a whole (module-level constants and imports are compared '
            'separately)',
            'differing source SHA in files whose relevant functions ARE identical still means '
            'the files are not the same artefact: record the training-era identity explicitly '
            'in any later comparison',
            'comparable mask protocol at EVALUATION time (F vs D) and comparable mask/data '
            'protocol at TRAINING time (F vs E2) are separate requirements; this audit only '
            'addresses the former',
        ],
        'consequence': ('The mask POOL is provably identical (contents and order), so a '
                        'shared-mask comparison (F vs D) is not invalidated by the drift. What '
                        'changed is the ENFORCEMENT STRENGTH of mask legality: the current '
                        'validate_mask_scope dropped the clause "mask customer must come from a '
                        'mutable_ids vehicle plan", so it is strictly MORE PERMISSIVE — masks '
                        'containing customers that sit in decision_pool but in no mutable '
                        'vehicle plan suffix (deferred customers) now pass where the '
                        'training-era version raised. A new experiment must therefore either '
                        'adopt the current gate and declare the relaxation, or freeze the '
                        'training-era gate; and F\'s training-era identity must be recorded '
                        'either way. Separately, the D arm\'s cc_lns_search gained a '
                        'time-budget break, so the D trajectory is not reproducible from the '
                        'training-era file without that break.'),
        'practical_frequency_unknown': ('the declared suite is synthetic and sparsely planned, '
                                        'so it does not measure how often real states contain '
                                        'pool customers outside every mutable plan suffix; the '
                                        'clause removal is proven, its real-world reach is not'),
    }
    (OUT / 'D2B_EQUIVALENCE.json').write_text(
        json.dumps(res, ensure_ascii=False, indent=2, default=str) + '\n',
        encoding='utf-8', newline='\n')
    print(json.dumps({'verdict': res['verdict'],
                      'constructive': {k: (v.get('n_identical'), v.get('n_functions'))
                                       for k, v in res['constructive'].items()},
                      'dynamic': {k: v for k, v in res['dynamic'].items()
                                  if k in ('n_states', 'n_identical', 'n_different',
                                           'different_states')},
                      'identity': {k: v.get('matches') if 'matches' in v else
                                   (v.get('train_sha_matches'), v.get('bytes_train'),
                                    v.get('bytes_current'))
                                   for k, v in res['source_identity'].items()}},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
