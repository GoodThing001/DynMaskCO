"""Durable synthetic/preparation ledger; never starts or resumes computation."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import time


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False,
                      separators=(',', ':')).encode('utf-8')


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    data = encoded(value)
    with temp.open('wb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


class Journal:
    def __init__(self, root, unit_keys, *, create=True):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'events.jsonl'
        if create:
            if self.path.exists() or (self.root/'slots.json').exists():
                raise ValueError('refuse existing run')
            atomic(self.root/'slots.json', [dict(unit_key=k, state='NOT_ATTEMPTED') for k in unit_keys])
            with self.path.open('xb') as f:
                f.flush()
                os.fsync(f.fileno())
        self.seq = len(self.read()[0])

    def append(self, unit_key, event, **payload):
        row = dict(seq=self.seq, unit_key=unit_key, event=event,
                   monotonic_ns=time.monotonic_ns(), payload=payload)
        envelope = dict(row=row, sha256=hashlib.sha256(encoded(row)).hexdigest())
        with self.path.open('ab') as f:
            f.write(encoded(envelope) + b'\n')
            f.flush()
            os.fsync(f.fileno())
        self.seq += 1
        return row

    def read(self):
        rows, bad = [], False
        if not self.path.exists():
            return rows, bad
        for line in self.path.read_bytes().splitlines(keepends=True):
            try:
                assert line.endswith(b'\n')
                env = json.loads(line)
                row = env['row']
                assert row['seq'] == len(rows)
                assert hashlib.sha256(encoded(row)).hexdigest() == env['sha256']
                rows.append(row)
            except (ValueError, AssertionError, KeyError, TypeError):
                bad = True
                break
        return rows, bad

    def postmortem(self, *, no_live_workers_confirmed, package_timeout=False):
        if not no_live_workers_confirmed:
            raise ValueError('cannot infer termination while workers may be alive')
        rows, bad = self.read()
        slots = json.loads((self.root/'slots.json').read_text(encoding='utf-8'))
        for slot in slots:
            events = [r for r in rows if r['unit_key'] == slot['unit_key']]
            terminal = [r for r in events if r['event'] == 'END']
            started = any(r['event'] in ('START', 'RESUME') for r in events)
            slot.update(state='END' if terminal else 'INTERRUPTED' if started else 'NOT_ATTEMPTED',
                        failure_code=(terminal[-1]['payload'].get('failure_code') if terminal
                                      else 'PROCESS_INTERRUPTED' if started
                                      else 'NOT_ATTEMPTED_PACKAGE_TIMEOUT' if package_timeout
                                      else 'NOT_ATTEMPTED_PACKAGE_ABORT'))
        return dict(slots=slots, damaged_tail=bad,
                    evidence_valid=False, computation_resumed=False)
