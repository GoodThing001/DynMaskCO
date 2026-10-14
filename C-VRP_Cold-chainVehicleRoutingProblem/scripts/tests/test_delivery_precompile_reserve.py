"""Compiled runtime, rather than excluded first JIT, controls the safety reserve."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'delivery'))
from delivery_maskco import MaskCORouteProvider, improve_masked
from delivery_contract import DeliveryContract
from test_delivery_contract import fixture


class TimedProvider:
    prewarm = MaskCORouteProvider.prewarm

    def __init__(self):
        self.max_decode_s = 0.
        self.identity = {}
        self.calls = 0

    def features(self, data, capacity):
        return len(data['coords'])

    def logits(self, features, adjacency):
        self.calls += 1
        elapsed = 14. if self.calls == 1 else .001
        self.max_decode_s = max(self.max_decode_s, elapsed)
        out = np.ones_like(adjacency)
        np.fill_diagonal(out, -np.inf)
        return out


def main():
    data = fixture()
    contract = DeliveryContract(capacity=6., vehicles=2)
    initial = [[1, 2, 3]]
    blocked = TimedProvider()
    blocked.max_decode_s = 14.
    old = improve_masked(data, initial, contract, blocked, time_limit_s=.2, mask_fraction=1.)
    assert old['decode_calls'] == 0 and old['cycles'] == 0
    fixed = TimedProvider()
    fixed.prewarm(4, 6.)
    assert fixed.identity['prewarm_decode_including_compilation_s'] == 14.
    assert fixed.max_decode_s == .001 and fixed.calls == 4
    new = improve_masked(data, initial, contract, fixed, time_limit_s=.2, mask_fraction=1.)
    assert new['decode_calls'] == 1 and new['completed_candidates'] == 1
    assert new['report']['hard'] and new['retained_edges_preserved']
    assert not new['timeout']
    print('PASS: first-JIT reserve reproducer, postcompile reserve, active reconstruction, hard fallback')


if __name__ == '__main__':
    main()
