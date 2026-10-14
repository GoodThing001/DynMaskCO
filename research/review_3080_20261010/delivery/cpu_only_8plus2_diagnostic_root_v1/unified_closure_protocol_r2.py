"""Unified fail-closed closure protocol for a delivery request.  r2.

r1 -> r2 is forced by the r1 server diagnostic (run_r1/DIAG_FAILURE.traceback.txt): the
protocol's semantics were correct, but the candidate it resolved to (r20) crashed on the
real server module set.  r2 resolves candidate r21 and is otherwise semantically
identical to r1:

  P0  before the declared warm-up          (informational only, never a gate)
      declared warm-up: model and explicit files load, first JAX call, first native call
  P1  after the declared warm-up           (gate reference)
  P2  after the timed execution segment    (gate)
  P3  immediately before publication       (gate)

Publication is refused unless P1 == P2 AND P2 == P3.  The P0 -> P1 delta is recorded for
audit and is explicitly NOT a gate, because the warm-up phase is declared in advance and
its whole purpose is to provoke lazy loading.  Any file, module, native path or
explicit-path change inside the timed region therefore still rejects.

Nothing is written to the publication target on any refusal.
"""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
CANDIDATE = ROOT / 'tools/actual_loaded_closure_candidate_r21.py'


def _candidate():
    spec = importlib.util.spec_from_file_location('closure_candidate_r21', CANDIDATE)
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
