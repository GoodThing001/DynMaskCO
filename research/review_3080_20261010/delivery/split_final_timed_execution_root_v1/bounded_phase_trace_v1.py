"""Prepared bounded single-writer trace. No experiment or process is launched.

Caller supplies an already allocated writable memoryview. Request elapsed must
include all record writes, operation calls and final trace collection. Records
are engineering telemetry, never authorization or a result-publication ticket.
Concurrent readers and torn writes have no qualification in this preparation.
"""
import struct
import time

HEADER = struct.Struct('<8sIIII40x')
RECORD = struct.Struct('<QII3di20x')
MAGIC = b'MCPHT001'
MAX_RECORDS = 4096


class TraceCapacityExceeded(RuntimeError):
    pass


class PhaseTrace:
    def __init__(self, storage, sequence, capacity):
        if not isinstance(sequence, int) or not 0 < sequence < 2**64:
            raise ValueError('Positive request sequence')
        if not isinstance(capacity, int) or not 1 <= capacity <= MAX_RECORDS:
            raise ValueError('Bounded record count')
        self.storage = memoryview(storage).cast('B')
        if self.storage.readonly or len(self.storage) != HEADER.size+capacity*RECORD.size:
            raise ValueError('Exact writable fixed buffer')
        self.sequence, self.capacity, self.count = sequence, capacity, 0
        HEADER.pack_into(self.storage, 0, MAGIC, 1, capacity, 0, 0)

    def measure(self, phase, step, operation, *args, **kwargs):
        """One synchronous call; async operations require a registered sync wrapper."""
        if not isinstance(phase, int) or not 1 <= phase <= 255:
            raise ValueError('Registered phase identifier')
        if not isinstance(step, int) or not 0 <= step < 2**32:
            raise ValueError('Nonnegative step identifier')
        if self.count >= self.capacity:
            raise TraceCapacityExceeded('No silent telemetry truncation')
        index = self.count
        offset = HEADER.size+index*RECORD.size
        began = time.perf_counter()
        RECORD.pack_into(self.storage, offset, self.sequence, phase, step, began, began, began, 0)
        self.count += 1
        HEADER.pack_into(self.storage, 0, MAGIC, 1, self.capacity, self.count, 0)
        operation_start = time.perf_counter()
        status = 2
        try:
            value = operation(*args, **kwargs)
            status = 1
            return value
        finally:
            operation_end = time.perf_counter()
            RECORD.pack_into(self.storage, offset, self.sequence, phase, step,
                             began, operation_start, operation_end, status)

    def close(self):
        # This marker is neither source closure nor successful policy completion.
        HEADER.pack_into(self.storage, 0, MAGIC, 1, self.capacity, self.count, 1)


def read_after_writer_reaped(storage, sequence):
    """Only after actual writer exit/reap; no live-read atomicity claim."""
    raw = bytes(storage)
    if len(raw) < HEADER.size:
        raise ValueError('Short header')
    magic, version, capacity, count, closed = HEADER.unpack_from(raw)
    if magic != MAGIC or version != 1 or not 1 <= capacity <= MAX_RECORDS:
        raise ValueError('Schema/capacity')
    if len(raw) != HEADER.size+capacity*RECORD.size or not 0 <= count <= capacity or closed not in (0, 1):
        raise ValueError('Buffer/header bounds')
    result = []
    for i in range(count):
        seq, phase, step, began, start, end, status = RECORD.unpack_from(raw, HEADER.size+i*RECORD.size)
        if seq != sequence or not 1 <= phase <= 255 or status not in (0, 1, 2):
            raise ValueError('Record identity/status')
        if not 0 <= began <= start <= end < float('inf'):
            raise ValueError('Finite monotonic record clocks')
        result.append(dict(sequence=seq, phase=phase, step=step, began=began,
                           operation_start=start, operation_end=end, status=status,
                           operation_seconds=end-start if status else None))
    return {'closed': bool(closed), 'records': result,
            'qualification': False, 'open_records_are_completed': False}
