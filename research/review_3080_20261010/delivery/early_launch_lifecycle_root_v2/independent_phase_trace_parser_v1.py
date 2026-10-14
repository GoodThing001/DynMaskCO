"""Independent byte parser: imports neither writer nor model/engine modules.

Only use archived raw bytes after actual matching writer exit/reap. Engineering
trace validity does not certify completed candidate, scientific source or budget.
"""
import hashlib
import math
import struct
from fractions import Fraction


def parse(raw, sequence, start_clock, end_clock, writer_reaped):
    if type(raw) is not bytes or type(sequence) is not int or not 0<sequence<2**64:
        raise ValueError('Exact byte/sequence input')
    if writer_reaped is not True:raise ValueError('Actual writer must be reaped')
    if not all(math.isfinite(x) for x in (start_clock,end_clock)) or not 0<=start_clock<=end_clock:
        raise ValueError('Outer clock bounds')
    if len(raw)<64:raise ValueError('Short raw trace')
    magic,version,capacity,count,closed=struct.unpack_from('<8sIIII',raw)
    if magic!=b'MCPHT001' or version!=1 or not 1<=capacity<=4096 or not 0<=count<=capacity or closed not in (0,1):
        raise ValueError('Header schema/bounds')
    if len(raw)!=64+64*capacity or any(raw[24:64]):raise ValueError('Exact buffer/header padding')
    records=[];intervals=[];last_began=start_clock
    for index in range(count):
        offset=64+64*index
        seq,phase,step,began,start,end,status=struct.unpack_from('<QII3di',raw,offset)
        if any(raw[offset+44:offset+64]):raise ValueError('Record padding')
        if seq!=sequence or not 1<=phase<=19 or status not in (0,1,2):raise ValueError('Record identity/registered phase/status')
        if not all(math.isfinite(x) for x in (began,start,end)) or not start_clock<=began<=start<=end<=end_clock or began<last_began:
            raise ValueError('Actual outer bounds/monotonic record allocation')
        last_began=began
        if status==0 and (start!=began or end!=began):raise ValueError('Open record must retain initial clocks')
        duration=Fraction.from_float(end)-Fraction.from_float(start) if status else None
        records.append(dict(index=index,sequence=seq,phase=phase,step=step,status=status,
            began_hex=began.hex(),operation_start_hex=start.hex(),operation_end_hex=end.hex(),
            exact_operation_seconds=None if duration is None else [duration.numerator,duration.denominator]))
        if status:intervals.append((Fraction.from_float(start),Fraction.from_float(end)))
    if any(raw[64+64*count:]):raise ValueError('Torn header count or dirty unused tail')
    # Union, not sum: nested operation spans overlap. Untraced gaps and final
    # record writes remain in the separately measured outer request duration.
    union=Fraction(0);right=None
    for left,end in sorted(intervals):
        if right is None or left>right:union+=end-left
        elif end>right:union+=end-right
        right=end if right is None else max(right,end)
    outer=Fraction.from_float(end_clock)-Fraction.from_float(start_clock)
    return dict(raw_sha256=hashlib.sha256(raw).hexdigest(),records=records,closed=bool(closed),
        exact_operation_union_seconds=[union.numerator,union.denominator],
        exact_outer_seconds=[outer.numerator,outer.denominator],
        exact_untraced_outer_seconds=[(outer-union).numerator,(outer-union).denominator],
        operation_union_is_complete_request=False,trace_validity_is_scientific_qualification=False,
        open_records_completed=False)
