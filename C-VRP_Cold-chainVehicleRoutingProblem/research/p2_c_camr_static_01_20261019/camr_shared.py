"""P2-C-CAMR-STATIC-01  isolated shared machinery.

One implementation, used by BOTH sides, so that a plan difference can only come from the
action score and never from the action set, the tie-break, the search skeleton or the
acceptance rule. Nothing here changes the engine: it only enumerates actions on a saved
public view and rewrites FleetPlan objects in memory.

Contract fixed at T1 (`registration.json`) and reproduced verbatim here:

  shared legal action set : cand.feasible  AND  route_check on every plan of the trial
  allowed                 : mutable_ids(view) | {committed_vid}   (same object for both sides)
  capacity convention     : route_check capacity_ok, i.e. load <= capacity - 1e-4
  tie-break               : (primary ascending, FleetAction.action_id() ascending)
  search skeleton         : greedy first, then layer-by-layer beam (beam=2, <=2 per parent,
                            <=8 completions, plan_hash dedup)
  anchor assertions       : anchor triple frozen, committed_next absent from every suffix,
                            every accepted-unserved order exactly once fleet-wide

`primary` is a comparable float per action: `cand.incremental_distance` for the distance side
and `-score` for the model side, so "smaller is better" holds for both.
"""
from __future__ import annotations

import os

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '1')

import hashlib                                                      # noqa: E402
import json                                                         # noqa: E402
import sys                                                          # noqa: E402
from itertools import combinations                                  # noqa: E402
from pathlib import Path                                            # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = ROOT.parent
OUT = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT / 'scripts'))
from project_paths import configure_imports                         # noqa: E402
configure_imports(('experimental', 'simulation', 'coldchain', 'evaluation', 'training', 'models'))

import action_contract as ac                                        # noqa: E402
import p2_pos01_reference as ref                                    # noqa: E402
from p2_pos01c_gap import view_from_json                            # noqa: E402

PHASES = {'SAA': ROOT / 'research/p2_c_opp_01_20261019' / 'phases.json',
          'MYOPIC': ROOT / 'research/p2_c_ctrl_01_20261019' / 'phases.json'}
SCOPE = ROOT / 'research/p2_c_anchor_01_20261019' / 'anchor_scope.csv'
CAPACITY_MARGIN = ref.CAPACITY_MARGIN          # 1e-4, the stricter convention


# --------------------------------------------------------------------------- #
# identity / IO
# --------------------------------------------------------------------------- #
def sha_file(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def sha_text(s):
    return hashlib.sha256(s.encode()).hexdigest()


def load_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def write_json(p, payload):
    p = Path(p)
    if p.exists():
        raise SystemExit('refuse overwrite: %s already exists' % p)
    p.open('x', encoding='utf-8', newline='\n').write(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str) + '\n')


def read_scope():
    import csv
    with SCOPE.open('r', encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def load_phase(arm, day, ordinal):
    for ph in load_json(PHASES[arm]):
        if int(ph['day']) == int(day) and int(ph['ordinal']) == int(ordinal):
            return ph
    raise KeyError('%s d%s ord%s' % (arm, day, ordinal))


def view_of(ph):
    return view_from_json({'public_view': ph['public_view'],
                           'public_identity': ph['public_identity']})


# --------------------------------------------------------------------------- #
# shared legal action set
# --------------------------------------------------------------------------- #
def shared_allowed(view):
    """mutable vehicles plus the single in-transit vehicle: one object for both sides."""
    committed = [v for v in view.vehicles
                 if v.status == 'committed' and v.committed_next not in (None, 0)]
    if len(committed) > 1:
        raise ValueError('MULTIPLE_COMMITTED_VEHICLES')
    allowed = set(ref.mutable_ids(view))
    if committed:
        allowed = allowed | {int(committed[0].vehicle_id)}
    return frozenset(int(x) for x in allowed)


def legal_actions_shared(view, env, plans, remaining, allowed):
    """(customer, action, candidate) for every action that survives the SHARED filter.

    Mirrors mpre_policy.enumerate_legal_actions but additionally requires that every plan of
    the trial passes ref.route_check (capacity - 1e-4, TW end, depot return, accepted-only).
    """
    allowed_t = tuple(sorted(int(x) for x in allowed))
    out = []
    for c in sorted(int(x) for x in remaining):
        if c not in view.accepted or c in view.served:
            raise ValueError('illegal repair customer %d' % c)
        cands, _ = ac.enumerate_actions_from_plans(env, 0, plans, c,
                                                   allowed_vehicle_ids=allowed_t)
        for cand in cands:
            if not cand.feasible:
                continue
            trial = ac.apply_action(plans, cand.action, allowed_vehicle_ids=allowed_t)
            if any(not ref.route_check(view, p) for p in trial.values()):
                continue
            out.append((c, cand.action, cand))
    return out


def rank(entries, primary):
    """(primary ascending, action_id ascending). Deterministic total order, both sides."""
    keyed = [((float(primary(cand, action)), action.action_id()), c, action)
             for c, action, cand in entries]
    keyed.sort(key=lambda t: t[0])
    return [(c, action) for _k, c, action in keyed]


def distance_primary(cand, action):
    return float(cand.incremental_distance)


# --------------------------------------------------------------------------- #
# plan utilities
# --------------------------------------------------------------------------- #
def plans_of(view):
    return ref.plans_from_view(view)


def remove_mask(plans, mask):
    return ref.remove_mask(plans, tuple(mask))


def apply(plans, action, allowed):
    return ac.apply_action(plans, action, allowed_vehicle_ids=tuple(sorted(int(x) for x in allowed)))


def signature(plans):
    return ac.plan_hash(plans)


def anchor_of(plans, cvid):
    p = plans[int(cvid)]
    return (int(p.anchor_node), round(float(p.anchor_time), 6), round(float(p.anchor_load), 6))


def suffix_first(plans, cvid):
    p = plans[int(cvid)]
    s = [int(x) for x in p.suffix]
    return s[0] if s else 0


def check_plan(view, plans, cvid, anchor, committed_next, pending):
    """Registered anchor assertions plus the shared reference check. (ok, reason)."""
    if anchor_of(plans, cvid) != tuple(anchor):
        return False, 'ANCHOR_MISMATCH'
    pending = set(int(x) for x in pending)
    for p in plans.values():
        if int(committed_next) in [int(x) for x in p.suffix]:
            return False, 'COMMITTED_LEG_IN_SUFFIX'
    counts = {}
    for p in plans.values():
        for c in p.suffix:
            counts[int(c)] = counts.get(int(c), 0) + 1
    for o in sorted(pending):
        want = 0 if o == int(committed_next) else 1
        if counts.get(o, 0) != want:
            return False, 'EXACT_ONCE_VIOLATED_%d' % o
    for c in counts:
        if c not in pending:
            return False, 'NON_PENDING_ORDER_IN_PLAN'
    if any(not ref.route_check(view, p) for p in plans.values()):
        return False, 'REFERENCE_REJECT'
    return True, 'OK'


# --------------------------------------------------------------------------- #
# search skeleton (identical for both sides)
# --------------------------------------------------------------------------- #
def _complete(proposer, plans, remaining, allowed):
    P, rem, calls = plans, list(remaining), 0
    while rem:
        acts, k = proposer(P, rem, 1)
        calls += k
        if not acts:
            return None, 'NO_LEGAL_ACTION', calls
        c, action = acts[0]
        P = apply(P, action, allowed)
        rem.remove(c)
    return P, 'OK', calls


def greedy(proposer, plans, mask, allowed):
    """One deterministic greedy completion. Never raises on a legal dead end."""
    partial = remove_mask(plans, mask)
    P, reason, calls = _complete(proposer, partial, sorted(mask), allowed)
    return {'plan': P, 'reason': reason, 'scoring_calls': calls,
            'n_completions': 0 if P is None else 1}


def completions(proposer, plans, mask, allowed, beam=2, max_per_parent=2, max_completions=8):
    """Layer-by-layer beam. completions[0] is the branch-free greedy completion."""
    partial = remove_mask(plans, mask)
    seen, out, n_total, n_fail, calls = set(), [], 0, 0, 0

    def record(P):
        nonlocal n_total, n_fail
        n_total += 1
        if P is None:
            n_fail += 1
            return
        h = signature(P)
        if h not in seen:
            seen.add(h)
            out.append(P)

    P0, first_reason, c0 = _complete(proposer, partial, sorted(mask), allowed)
    calls += c0
    record(P0)

    frontier = [(partial, sorted(mask))]
    while frontier and len(out) < max_completions:
        nxt = []
        for P, rem in frontier:
            if not rem:
                continue
            acts, k = proposer(P, rem, max_per_parent)
            calls += k
            for c, action in acts:
                P2 = apply(P, action, allowed)
                rem2 = [x for x in rem if x != c]
                Pg, _r, cg = _complete(proposer, P2, rem2, allowed)
                calls += cg
                record(Pg)
                nxt.append((P2, rem2))
                if len(out) >= max_completions:
                    break
            if len(out) >= max_completions:
                break
        frontier = nxt[:beam]

    return {'completions': out, 'first_reason': first_reason,
            'stats': {'n_unique': len(out), 'n_total': n_total, 'n_fail': n_fail,
                      'scoring_calls': calls},
            'params': {'beam': beam, 'max_per_parent': max_per_parent,
                       'max_completions': max_completions}}


# --------------------------------------------------------------------------- #
# proposers  (the ONLY difference between the two sides is the score)
# --------------------------------------------------------------------------- #
def _distance_scores(entries, _backbone, _view, _base, _partial):
    """Higher is better, so the score is MINUS the incremental distance."""
    return [-float(cand.incremental_distance) for _c, _a, cand in entries]


def _model_scores(entries, backbone, view, base, partial):
    scores = backbone.insertion_scores(view, base, partial, entries)
    if len(scores) != len(entries):
        raise ValueError('score length mismatch %d != %d' % (len(scores), len(entries)))
    out = []
    for s in scores:
        f = float(s)
        if not (f == f) or f in (float('inf'), float('-inf')):
            raise FloatingPointError('NONFINITE_SCORE')
        out.append(f)
    return out


def _model_path_distance(entries, backbone, view, base, partial):
    """Degeneration gate: run the REAL model path, then discard its scores for -distance.

    If the result is not bit-identical to distance-greedy, then the difference between the two
    sides comes from the plumbing (batching, index mapping, score sign, tie-break, skeleton) and
    not from the learned score, so the gate fails and Pass A must not run.
    """
    backbone.insertion_scores(view, base, partial, entries)
    return [-float(cand.incremental_distance) for _c, _a, cand in entries]


SCORERS = {'distance': _distance_scores, 'model': _model_scores,
           'model_path_distance': _model_path_distance}


def proposer(view, env, allowed, score='model', backbone=None, base=None):
    """Rank the SHARED legal set by -score with the unified tie-break.

    `score='distance'` is the non-learned side; `score='model'` is the frozen MaskCO side.
    The degeneration gate calls this with `score='distance'` through the model plumbing.
    """
    fn = SCORERS[score]

    def propose(P, remaining, K):
        entries = legal_actions_shared(view, env, P, remaining, allowed)
        if not entries:
            return [], 0
        scores = fn(entries, backbone, view, base, P)
        keyed = [((-float(s), e[1].action_id()), e[0], e[1])
                 for e, s in zip(entries, scores)]
        keyed.sort(key=lambda t: t[0])
        return [(c, a) for _k, c, a in keyed][:K], len(entries)
    return propose


# --------------------------------------------------------------------------- #
# unit helpers
# --------------------------------------------------------------------------- #
def unit_from_row(row):
    arm, day, ordinal = row['arm'], int(row['day']), int(row['ordinal'])
    ph = load_phase(arm, day, ordinal)
    view = view_of(ph)
    plans = plans_of(view)
    committed = [v for v in view.vehicles
                 if v.status == 'committed' and v.committed_next not in (None, 0)]
    if len(committed) != 1:
        raise ValueError('expected one committed vehicle')
    cvid = int(committed[0].vehicle_id)
    base = plans[cvid]
    pending = {int(x) for x in view.accepted} - {int(x) for x in view.served}
    queue = [int(c) for c in base.suffix if int(c) in pending]
    masks = [tuple(sorted(p)) for p in combinations(queue, 2)]
    return {'arm': arm, 'day': day, 'ordinal': ordinal,
            'public_identity': ph['public_identity'], 'view': view, 'plans': plans,
            'env': ref.public_env(view), 'allowed': shared_allowed(view),
            'cvid': cvid, 'anchor': anchor_of(plans, cvid),
            'committed_next': int(base.anchor_node), 'pending': pending,
            'queue': queue, 'masks': masks}


def unit_key(u, mask):
    return {'arm': u['arm'], 'day_id': u['day'], 'ordinal': u['ordinal'],
            'public_identity': u['public_identity'], 'mask': [int(x) for x in mask]}
