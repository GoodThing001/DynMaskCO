"""Bounded CPU leases for out-of-order teacher days; no scientific choices."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from threading import Event, Lock


def collect_shards(days, cpus, collect, on_failure):
    """Return shards in registered order, giving each live day one distinct CPU."""
    days, cpus = list(days), list(cpus)
    if not days or len(set(days)) != len(days) or not cpus or len(set(cpus)) != len(cpus):
        raise ValueError('Explicit unique days and unique reserved CPUs required')
    leases = Queue()
    for cpu in cpus:
        leases.put(cpu)
    abort, lock = Event(), Lock()

    def one(day):
        cpu = leases.get()
        try:
            if abort.is_set():
                raise RuntimeError('Collection stopped after another day failed')
            try:
                return collect(day, cpu)
            except Exception as exc:
                with lock:
                    if not abort.is_set():
                        abort.set()
                        on_failure(day, exc)
                raise
        finally:
            leases.put(cpu)

    # The failure callback records the first child error immediately, rather
    # than waiting for day 0 in ordered executor.map. Already running days keep
    # their original finite child timeout and their evidence is preserved.
    with ThreadPoolExecutor(max_workers=len(cpus)) as pool:
        tasks = {pool.submit(one, day): day for day in days}
        values = {}
        try:
            for future in as_completed(tasks):
                values[tasks[future]] = future.result()
        except BaseException:
            abort.set()
            for future in tasks:
                future.cancel()
            raise
    return [values[day] for day in days]
