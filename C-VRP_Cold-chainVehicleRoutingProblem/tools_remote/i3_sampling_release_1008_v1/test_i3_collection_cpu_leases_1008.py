"""Regression for overlapping day residues and immediate failure reporting."""
from threading import Lock
import time

from i3_collection_cpu_leases_1008 import collect_shards


def run_tests():
    live, seen, lock = set(), [], Lock()

    def collect(day, cpu):
        with lock:
            assert cpu not in live, 'Two live collectors share a reserved CPU'
            live.add(cpu)
            seen.append((day, cpu))
        # Day 0 deliberately outlives day 1 and day 2; static day % 2 fails.
        time.sleep(.035 if day == 0 else .002)
        with lock:
            live.remove(cpu)
        return day

    assert collect_shards(range(16), [8, 9], collect, lambda *_: None) == list(range(16))
    assert not live and sorted(day for day, _ in seen) == list(range(16))
    failures, called = [], []

    def failing(day, cpu):
        called.append(day)
        if day == 1:
            raise ValueError('deliberate failure')
        time.sleep(.02)
        return day

    try:
        collect_shards(range(16), [8, 9], failing, lambda day, exc: failures.append((day, str(exc))))
    except (ValueError, RuntimeError):
        pass
    else:
        raise AssertionError('Failed collection must not return complete shards')
    assert failures == [(1, 'deliberate failure')] and set(called) <= {0, 1}
    print('ALL PASS: out-of-order days keep unique CPUs, registered order, fail-fast record, queued-day cancellation')


if __name__ == '__main__':
    run_tests()
