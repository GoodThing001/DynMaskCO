"""Immutable phase publications: readers never block replacement of an open destination."""
import json
import os
import time
import uuid
from pathlib import Path


class PhaseJournal:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        # Restart must not overwrite an earlier producer's immutable sequence.
        files = sorted(self.directory.glob('phase_*.json'))
        self.sequence = int(files[-1].name.split('_')[1]) + 1 if files else 0

    def publish(self, stage, **extra):
        record = dict(stage=stage, mono=time.monotonic(), sequence=self.sequence, **extra)
        dest = self.directory / ('phase_%09d_%s.json' % (self.sequence, uuid.uuid4().hex))
        temp = dest.with_suffix('.tmp')
        with open(temp, 'xb') as fh:
            fh.write(json.dumps(record, sort_keys=True).encode())
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(temp, dest)  # a new destination; existing/open records remain untouched
        self.sequence += 1
        return dest


READ_DIAGNOSTICS = {'transient_permission_errors': 0, 'exhausted_permission_errors': 0}


def _read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def latest(directory, max_wait_s=.05):
    # Windows can briefly deny access while the rename handle is being closed.
    # Wait only for sharing/permission errors, and account this wait in the caller's clock.
    # Malformed final JSON is never hidden; persistent I/O denial is a hard error.
    deadline = time.monotonic() + max_wait_s
    while True:
        files = sorted(Path(directory).glob('phase_*.json'))
        if not files:
            return {}
        try:
            return _read_json(files[-1])
        except PermissionError:
            READ_DIAGNOSTICS['transient_permission_errors'] += 1
            if time.monotonic() >= deadline:
                READ_DIAGNOSTICS['exhausted_permission_errors'] += 1
                raise
            time.sleep(.002)
