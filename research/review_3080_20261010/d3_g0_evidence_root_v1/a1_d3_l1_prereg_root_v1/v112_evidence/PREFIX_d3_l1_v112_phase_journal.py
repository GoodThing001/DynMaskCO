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


def latest(directory):
    files = sorted(Path(directory).glob('phase_*.json'))
    return json.loads(files[-1].read_text(encoding='utf-8')) if files else {}
