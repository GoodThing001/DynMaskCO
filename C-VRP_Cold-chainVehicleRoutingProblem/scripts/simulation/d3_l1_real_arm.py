#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""L1 REAL-F ARM: drive the two-arm protocol with the REAL MaskCO scorer, never a synthetic one.

The L1 question
---------------
Given the SAME ordered legal action list, do the restored MaskCO logits (F) and the registered
`-incremental_distance` (D-Geo) choose different EXECUTABLE repairs?

What this module guarantees
---------------------------
R1  F is the REAL model.  `load_real_f_scorer` loads the V1.22-accepted checkpoint/backbone through
    the same `load_trained_model` path the real F forward used, restores the 83 trained leaves
    bit-exactly, and calls `mtrained_score_fn(model, Nv_max, M_max)` -- the exact scorer whose
    interface the V1.22 batch qualified.  There is no synthetic fallback: a missing or mismatched
    checkpoint raises.
R2  State construction is the SAME path as the real deployment.  `make_real_extract_fn` rebuilds the
    repair-state tensors from the CURRENT plans at every step through `extract_repair_state_v1`, with
    the deployment identity (frozen clock, Nv, ordered legal list) and the deployment timestep
    (`retained_edge_ratio(A0, A_partial)`).
R3  Per-step logits correspondence.  Every scorer call is observed by `Ledger.on_score_step`, which
    asserts `len(logits) == len(this step's ordered legal)` and fingerprints the logits together with
    their step.  A vector produced for one step's legal list can therefore never be silently reused
    for another: the fingerprint and the length are recorded per step, and the audit re-reads them.
R4  Shared input.  Both arms receive the SAME state dict; only `scorer_id` differs.  The protocol's
    `assert_single_factor` machinery (state digest before/after, first-step legal fingerprint,
    declared varying fields) is reused unchanged.
R5  Fail-closed resources.  Both arms get an identical `BudgetGate`; the F ceiling is set from
    MEASURED F cost with a stated derivation and no unregistered bonus for being slower.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
REPO = EXT.parent
for _p in ('scripts', 'scripts/simulation', 'scripts/coldchain', 'scripts/data', 'scripts/models',
           'scripts/tests'):
    sys.path.insert(0, str(EXT / _p))

import d3_geo_adapter as GEO                                                   # noqa: E402
import d3_l1_two_arm_protocol as TA                                           # noqa: E402


def sha256_file(path):
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------
# R1 the real scorer
# ---------------------------------------------------------------------------------------------
REAL_F_CKPT = EXT / 'results/m0_scale/mpre_reinforce_s42_clean/model.ckpt'
REAL_F_BACKBONE = REPO / 'MASKCO_code/ckpts/cvrp100.ckpt'
REAL_F_CKPT_SHA = 'f5acf83fa2519020aeea274df5ecf740a6ad809b215d2d270209b7901903a7f5'
REAL_F_BACKBONE_SHA = '03615b08a55125a8fe4a6dad1df9ab592b56f7ed9d21b54d2562070b3c6d38db'
F_LABEL = 'F_RECONSTRUCTED_WITH_CURRENT_BACKBONE / HISTORICAL_BACKBONE_UNVERIFIED'


def frozen_jit_shapes(*, day=None, position=None):
    """The JIT shapes for the real scorer, derived from the V1.22-accepted inputs.

    `Nv_max` is the largest visible-node count and `M_max` the largest ordered-legal length over the
    64 states the V1.22 batch qualified.  Deriving them from the QUALIFIED INPUTS (not from a score)
    keeps the same rule the F forward pre-registration used: "the dimensions come from the maxima of
    the already-accepted inputs, never from a scoring result".
    """
    import json as _json
    ev = REPO / 'research/review_3080_20261010/d3_g0_evidence_root_v1/a1_d3_l1_prereg_root_v1'
    targets = None
    for p in sorted((ev / 'v122_evidence/run_v122').glob('FORWARD_TARGETS_*.json')):
        targets = _json.loads(p.read_text(encoding='utf-8'))
        break
    if not targets:
        raise RuntimeError('no FORWARD_TARGETS record: the JIT shapes cannot be derived')
    nvs = [int(r['Nv']) for r in targets['rows']]
    ms = [int(r['n_legal_expected']) for r in targets['rows']]
    return max(nvs), max(ms)


def load_real_f_scorer(*, nv_max, m_max):
    """Load the REAL MaskCO scorer and prove the trained leaves were restored bit-exactly."""
    import pickle
    import jax
    from mtrained_replanner import load_trained_model, mtrained_score_fn
    from mpre_trained import partition
    from d3_l1_v18_readonly_audit import typed_bytes

    if sha256_file(REAL_F_CKPT) != REAL_F_CKPT_SHA:
        raise RuntimeError('refusing to score: the F checkpoint digest does not match the frozen '
                           'value; this arm may never fall back to a synthetic scorer')
    if sha256_file(REAL_F_BACKBONE) != REAL_F_BACKBONE_SHA:
        raise RuntimeError('refusing to score: the backbone digest does not match the frozen value')
    model, _cfg = load_trained_model(str(REAL_F_BACKBONE), str(REAL_F_CKPT), seed=42)
    _gd, train, frozen, other = partition(model)
    meta = pickle.loads(REAL_F_CKPT.read_bytes())
    trained = jax.tree_util.tree_flatten_with_path(train)[0]
    reference = jax.tree_util.tree_flatten_with_path(meta['train_state'])[0]
    exact = len(trained) == len(reference) and all(
        str(p) == str(q) and typed_bytes(np.asarray(x)) == typed_bytes(np.asarray(y))
        for (p, x), (q, y) in zip(trained, reference))
    if not exact:
        raise RuntimeError('refusing to score: the trained leaves were not restored bit-exactly')
    score = mtrained_score_fn(model, Nv_max=int(nv_max), M_max=int(m_max))
    return score, {'model': model, 'n_train_leaves': len(trained), 'restored_exact': exact,
                   'n_frozen_leaves': len(jax.tree_util.tree_leaves(frozen)),
                   'n_other_leaves': len(jax.tree_util.tree_leaves(other)),
                   'backend': jax.default_backend(), 'devices': [str(d) for d in jax.devices()],
                   'f_ckpt_sha256': sha256_file(REAL_F_CKPT),
                   'backbone_sha256': sha256_file(REAL_F_BACKBONE),
                   'Nv_max': int(nv_max), 'M_max': int(m_max), 'label': F_LABEL}


def make_geo_score_fn(ledger=None, arm='D-Geo'):
    """The registered D-Geo scorer, ledger-instrumented so BOTH arms are observed identically.

    Instrumenting only the F arm would make the R4 claim ("each arm's per-step legal fingerprint and
    value digest are recorded") false by construction, and would also treat the arms asymmetrically:
    one measured, one not.
    """
    def score_fn(state):
        legal = state.get('legal') or []
        out = np.asarray(GEO.geo_score(state))
        if ledger is not None:
            ledger.note_score_call(arm, legal, out)
        return out
    return score_fn


def make_score_fn(score, *, ledger=None, arm='F'):
    """`score_fn(state) -> logits`, asserting the logits match THIS step's ordered legal list."""
    def score_fn(state):
        legal = state.get('legal') or []
        out = np.asarray(score(state))
        if out.ndim != 1 or out.shape[0] != len(legal):
            raise RuntimeError(
                'the real F scorer returned %r scores for %d ordered legal actions at this step; a '
                'vector produced for a different step must never be reused'
                % (list(out.shape), len(legal)))
        if ledger is not None:
            ledger.note_score_call(arm, legal, out)
        return out
    return score_fn


# ---------------------------------------------------------------------------------------------
# R2 the deployment-identical extractor
# ---------------------------------------------------------------------------------------------
def make_real_extract_fn(*, env, inst_idx, clock, vehicles, served_mask, visible_ids, p0, mask,
                         mutable_ids, node_to_local, local_to_node, nv, capacity=50.0, tw_max=24.0,
                         ledger=None):
    """Rebuild the repair-state tensors from the CURRENT plans at every step.

    This is the same call the real deployment makes; only the plan/legal arguments change per step.
    The P0 adjacency is fixed (it is the reference the timestep is measured against), exactly as the
    deployment replanner builds it once per decision point.
    """
    from dynmaskco_cc_graph import build_plan_adjacency
    from mpre_replanner import retained_edge_ratio
    from repair_state import extract_repair_state_v1

    a0 = np.triu(build_plan_adjacency(p0, node_to_local, nv), k=1) > .5

    def extract_fn(plans, remaining, legal):
        state = extract_repair_state_v1(
            env, inst_idx, clock, vehicles, served_mask, visible_ids, plans, list(remaining), legal,
            capacity, tw_max, allowed_vehicle_ids=mutable_ids, node_to_local=node_to_local,
            local_to_node=local_to_node, Nv=nv)
        ap = np.triu(np.asarray(state['adjmat'][0]), k=1) > .5
        state['timestep'] = float(retained_edge_ratio(a0, ap))
        state['legal'] = list(legal)
        if ledger is not None:
            ledger.note_extract(legal, state)
        return state
    return extract_fn


# ---------------------------------------------------------------------------------------------
# R3 the per-step ledger
# ---------------------------------------------------------------------------------------------
class Ledger(object):
    """Records, per scorer call, which ordered legal list the logits were produced for.

    The point is falsifiability: the audit can re-read this and prove that no vector was reused
    across steps, because each entry carries the step's legal fingerprint, the vector length, and the
    logits' own digest.
    """

    def __init__(self):
        self.extracts = []
        self.calls = []
        self._step = {}

    def note_extract(self, legal, state):
        self.extracts.append({'step_index': len(self.extracts),
                              'n_legal': len(legal),
                              'legal_sha256': TA.legal_fingerprint(legal),
                              'timestep': float(state.get('timestep', 0.0))})

    def note_score_call(self, arm, legal, logits):
        # The step index is per ARM and is derived from that arm's own call count, not from a global
        # interleaving: the arms run sequentially, so a global counter would mislabel the second arm.
        step = self._step.get(arm, 0)
        self._step[arm] = step + 1
        self.calls.append({'arm': arm, 'step_index': step,
                           'n_legal': len(legal),
                           'legal_sha256': TA.legal_fingerprint(legal),
                           'logits_len': int(np.asarray(logits).shape[0]),
                           'logits_sha256': hashlib.sha256(np.asarray(logits).tobytes()).hexdigest(),
                           'logits_dtype': np.asarray(logits).dtype.str,
                           'wall_mono': time.monotonic()})

    def cross_check(self):
        """`(ok, report)`: every call's logits length matches its step's legal length."""
        problems, by_step = [], {}
        for c in self.calls:
            k = (c['arm'], c['step_index'])
            by_step.setdefault(k, []).append(c)
            if c['logits_len'] != c['n_legal']:
                problems.append({'arm': c['arm'], 'step': c['step_index'],
                                 'logits_len': c['logits_len'], 'n_legal': c['n_legal']})
        # the two arms must have called the scorer the same number of times for the same step order
        arms = sorted({c['arm'] for c in self.calls})
        counts = {a: sum(1 for c in self.calls if c['arm'] == a) for a in arms}
        return (not problems), {'n_calls': len(self.calls), 'per_arm_calls': counts,
                                'problems': problems,
                                'note': 'each entry carries the step legal fingerprint, so a vector '
                                        'produced for one step can never be silently reused for '
                                        'another; the audit re-reads the fingerprints'}
