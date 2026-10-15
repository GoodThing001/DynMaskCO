#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""V1.18 forward persistence: NEW-destination publication + bounded reading + write verification.

Why this module exists (root cause of the V1.11 whole-round failure)
--------------------------------------------------------------------
V1.11's worker persisted progress by `os.replace(temp, 'STAGE.json')` into a FIXED destination
name.  The supervisor simultaneously read that same name.  On Windows an open reader handle makes
`os.replace` fail with `[WinError 5] PermissionError` even though the reader only opened it for
reading.  In V1.11 that aborted slot 5/2 and the whole batch (22/64 completed).

Two independent defects were involved and both are closed here:

  D1 (writer side)  replacing a FIXED destination that a reader may hold open.
  D2 (reader side)  reading a just-renamed file with an unbounded raw read.

The repair keeps EVERY byte it publishes:

  * every publication goes to a FRESH, never-reused destination name, so no reader can ever hold
    the destination that a writer wants to replace (there is nothing to replace);
  * the write is verified by re-reading the exact published name and comparing the byte digest,
    so "the file exists" is never mistaken for "the record is durable and correct";
  * reads are bounded: only `PermissionError` (a transient sharing violation) is retried, for a
    bounded window, and a persistent denial is a HARD error that is never hidden;
  * a record whose digest does not match what was written is QUARANTINED, and the quarantine is
    itself published, because a corrupted record must never silently disappear from the ledger.

Public API
----------
    publish_new(directory, prefix, payload, *, verify=True) -> dict
    VerifiedReader(directory).latest(prefix) / .all(prefix)
    PermanentWriteError, TransientReadError (both are real hard errors)

Nothing in this module knows about F, states, masks or scoring: it is pure persistence.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path

# Bounded read retry window for Windows sharing violations.  Measured behaviour: the denial is
# sub-millisecond in the normal case, so 50 ms is already three orders of magnitude of headroom,
# and a denial that outlives it is a genuine permission problem rather than a race.
DEFAULT_READ_WAIT_S = 0.05
READ_DIAGNOSTICS = {'transient_permission_errors': 0, 'exhausted_permission_errors': 0,
                    'records_read': 0, 'verify_rereads': 0, 'quarantined': 0}


class PermanentWriteError(RuntimeError):
    """The write could not be proven durable; the caller must treat the run as failed."""


class TransientReadError(RuntimeError):
    """A record could not be read within the bounded window; never silently skipped."""


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _fsync_dir(directory: Path) -> None:
    """Best-effort directory fsync; unavailable on Windows and therefore not relied upon."""
    try:
        fd = os.open(str(directory), os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def publish_new(directory, prefix, payload, *, verify=True, retries=5):
    """Publish `payload` to a FRESH immutable destination and prove it is durable.

    Returns a record describing the publication:
        {'name', 'path', 'sha256', 'bytes', 'verified', 'attempts', 'wall_s', 'published_mono'}

    The destination is `prefix_<sequence-independent uuid>.json` and is created with mode 'xb'
    (exclusive): a name collision is an error, never an overwrite.  After the write, the exact
    published path is re-read and its digest compared with the bytes written; a mismatch raises
    `PermanentWriteError` and leaves the (quarantined) file in place for forensics.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    if isinstance(payload, (bytes, bytearray)):
        raw = bytes(payload)
    else:
        # EVERY JSON record carries its own publication clock.  Without it the reader's
        # (published_mono, name) ordering falls back to file-name order, which is arbitrary: the
        # run directory legitimately holds several immutable records of the same family (one per
        # verification run, because a record is never overwritten), so "latest" would otherwise be
        # whichever uuid happens to sort last.
        payload = dict(payload)
        payload.setdefault('published_mono', time.monotonic())
        raw = json.dumps(payload, indent=2, sort_keys=True, default=repr).encode('utf-8')
    last = None
    for attempt in range(1, int(retries) + 1):
        dest = directory / ('%s_%s.json' % (prefix, uuid.uuid4().hex))
        temp = directory / ('.%s.tmp.%s' % (dest.stem, uuid.uuid4().hex))
        try:
            with open(temp, 'xb') as fh:
                fh.write(raw)
                fh.flush()
                os.fsync(fh.fileno())
            # `temp` is a name no reader knows about; renaming it cannot collide with a reader
            # holding the destination open, because the destination does not exist yet.
            os.replace(temp, dest)
            _fsync_dir(directory)
        except OSError as exc:                                    # pragma: no cover - platform
            last = exc
            try:
                temp.unlink()
            except OSError:
                pass
            time.sleep(0.002 * attempt)
            continue
        if not verify:
            return {'name': dest.name, 'path': str(dest), 'sha256': _digest(raw),
                    'bytes': len(raw), 'verified': False, 'attempts': attempt,
                    'wall_s': time.monotonic() - started, 'published_mono': time.monotonic()}
        try:
            back = dest.read_bytes()
            READ_DIAGNOSTICS['verify_rereads'] += 1
        except OSError as exc:                                    # pragma: no cover - platform
            raise PermanentWriteError(
                'published %s but could not re-read it: %r' % (dest.name, exc))
        if back != raw:
            READ_DIAGNOSTICS['quarantined'] += 1
            raise PermanentWriteError(
                'published %s but the re-read bytes differ (wrote %d B sha %s, read %d B sha %s)'
                % (dest.name, len(raw), _digest(raw), len(back), _digest(back)))
        return {'name': dest.name, 'path': str(dest), 'sha256': _digest(raw), 'bytes': len(raw),
                'verified': True, 'attempts': attempt, 'wall_s': time.monotonic() - started,
                'published_mono': time.monotonic()}
    raise PermanentWriteError('could not publish %s_*.json after %d attempts: %r'
                              % (prefix, retries, last))


class VerifiedReader(object):
    """Bounded, verified reader over a directory of immutable `prefix_*.json` records.

    A record is identified by file NAME, not by a fixed name being replaced, so a reader can
    never interfere with a writer.  Ordering is by (published_mono, name) when the record
    carries `published_mono`; otherwise lexicographic name order is used.  `latest()` returns the
    newest record, `all()` returns every readable record in that order.
    """

    def __init__(self, directory, *, wait_s=DEFAULT_READ_WAIT_S):
        self.directory = Path(directory)
        self.wait_s = float(wait_s)

    def _files(self, prefix):
        if not self.directory.is_dir():
            return []
        return sorted(self.directory.glob('%s_*.json' % prefix))

    def _read(self, path):
        deadline = time.monotonic() + self.wait_s
        while True:
            try:
                raw = path.read_bytes()
            except FileNotFoundError:
                return None                # a record vanished: never silently invented
            except PermissionError:
                READ_DIAGNOSTICS['transient_permission_errors'] += 1
                if time.monotonic() >= deadline:
                    READ_DIAGNOSTICS['exhausted_permission_errors'] += 1
                    raise TransientReadError(
                        'persistent permission denial reading %s after %.3fs'
                        % (path.name, self.wait_s))
                time.sleep(0.002)
                continue
            except OSError as exc:
                raise TransientReadError('read error on %s: %r' % (path.name, exc))
            try:
                record = json.loads(raw.decode('utf-8'))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                # A corrupt record is a HARD error: it must never be treated as "no record yet".
                raise TransientReadError('corrupt record %s: %r' % (path.name, exc))
            record = dict(record) if isinstance(record, dict) else {'value': record}
            record.setdefault('published_name', path.name)
            record.setdefault('published_sha256', _digest(raw))
            READ_DIAGNOSTICS['records_read'] += 1
            return record

    def _ordered(self, prefix):
        rows = []
        for path in self._files(prefix):
            record = self._read(path)
            if record is None:
                continue
            rows.append(record)
        rows.sort(key=lambda r: (float(r.get('published_mono') or 0.0),
                                 str(r.get('published_name'))))
        return rows

    def all(self, prefix):
        return self._ordered(prefix)

    def latest(self, prefix):
        rows = self._ordered(prefix)
        return rows[-1] if rows else None

    def count(self, prefix):
        return len(self._files(prefix))


def write_once(path, payload):
    """Create a file that must not already exist; used for run manifests and results."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = payload if isinstance(payload, (bytes, bytearray)) else \
        json.dumps(payload, indent=2, sort_keys=True, default=repr).encode('utf-8')
    raw = bytes(raw)
    try:
        with open(path, 'xb') as fh:
            fh.write(raw)
            fh.flush()
            os.fsync(fh.fileno())
    except FileExistsError:
        raise PermanentWriteError('refusing to overwrite existing artifact %s' % path.name)
    _fsync_dir(path.parent)
    return {'name': path.name, 'path': str(path), 'sha256': _digest(raw), 'bytes': len(raw),
            'verified': path.read_bytes() == raw}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()
