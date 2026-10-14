"""Unified fail-closed closure protocol for a delivery request.

Round 20's candidate could capture and compare snapshots; the earlier registration for
this line said "any adjacent capture difference rejects publication", while the local
test admitted publication on a stable P2->P3 even though P1->P2 had gained a lazily
imported file.  That contradiction is resolved here by declaring the phase boundaries
explicitly:

  P0  before the declared warm-up          (informational only, never a gate)
      declared warm-up: model and explicit files load, first JAX call, first native call
  P1  after the declared warm-up           (gate reference)
  P2  after the timed execution segment    (gate)
  P3  immediately before publication       (gate)

Publication is refused unless P1 == P2 AND P2 == P3.  The P0 -> P1 delta is recorded
for audit and is explicitly NOT a gate, because the warm-up phase is declared in
advance and its whole purpose is to provoke lazy loading.  Any file, module, native
path or explicit-path change inside the timed region therefore still rejects.

Nothing is written to the publication target on any refusal.
"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]


def _candidate():
    spec = importlib.util.spec_from_file_location(
        'closure_candidate', ROOT / 'tools/actual_loaded_closure_candidate_r20.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


candidate = _candidate()

POINTS = ('P0_before_declared_warmup', 'P1_after_declared_warmup',
          'P2_after_timed_execution', 'P3_before_publication')
GATED_ADJACENCIES = (('P1_after_declared_warmup', 'P2_after_timed_execution'),
                     ('P2_after_timed_execution', 'P3_before_publication'))
INFORMATIONAL_ADJACENCIES = (('P0_before_declared_warmup', 'P1_after_declared_warmup'),)
WARMUP = ('load model and explicit files', 'first JAX call', 'first native call')


def snapshot(explicit_files):
    return candidate.capture(explicit_files)


def evaluate(p0, p1, p2, p3):
    snaps = {'P0_before_declared_warmup': p0, 'P1_after_declared_warmup': p1,
             'P2_after_timed_execution': p2, 'P3_before_publication': p3}
    gated, info = {}, {}
    for a, b in GATED_ADJACENCIES:
        gated['%s->%s' % (a.split('_')[0], b.split('_')[0])] = candidate.compare(snaps[a], snaps[b])
    for a, b in INFORMATIONAL_ADJACENCIES:
        info['%s->%s' % (a.split('_')[0], b.split('_')[0])] = candidate.compare(snaps[a], snaps[b])
    publishable = all(v['qualified'] for v in gated.values())
    return dict(publishable=publishable, gated=gated, informational=info,
                refusal_reasons=[k for k, v in gated.items() if not v['qualified']])
