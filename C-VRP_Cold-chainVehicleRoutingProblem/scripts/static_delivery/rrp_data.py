"""Read immutable public RRP matrices and reference records; no generator."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from types import MappingProxyType

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_paths import EXTENSION_ROOT

COMMIT = 'f5c5317a5ecd8fe64df779e79bd0397904bc0f9e'
PRIMARY_PROFILE = 'rrp_mip_dimensional_v1'
TIME_DOMAINS = ('source_day_domain', 'operational_tour_9h')
SOURCE = EXTENSION_ROOT / 'data/public_sources_20261020/refrigerated_routing_rrp'
PROTOCOL_PATH = EXTENSION_ROOT / 'configs/static_delivery/rrp_source.json'
REQUIRED = ('A', 'B', 'C', 'Tint', 'U', 'surface', 'carb_weight', 'mass', 'days',
            'month_days', 'driver_wage', 'fuel_cost', 'fix_time', 'opening_time',
            'row_q', 'row_time', 'time_up', 'start_time', 'clients', 'demands',
            'distances', 'speed_ranges', 'temperature_ranges')


def source_protocol():
    data = json.loads(PROTOCOL_PATH.read_text(encoding='utf-8'))
    if data['author_commit'] != COMMIT:
        raise ValueError('Unsupported author commit')
    return data


@dataclass(frozen=True)
class RRPInstance:
    name: str
    raw: dict
    demands: np.ndarray
    distances: np.ndarray
    source_sha256: str = ''
    source_commit: str = COMMIT

    @property
    def nodes(self):
        return len(self.demands)


@dataclass(frozen=True)
class ReferenceRecord:
    route: tuple[int, ...]
    waits_s: tuple[float, ...]
    saved: dict
    reported_costs: dict
    objective: str
    source_sha256: str
    raw_waits_s: tuple[float, ...] = ()
    wait_convention: str = 'incomplete_or_unresolved'
    label_origin: str = 'incomplete_or_unresolved'
    saved_clock_convention: str = 'incomplete_or_unresolved'


def parse_instance(raw, name, sha256=''):
    for field in REQUIRED:
        if field not in raw:
            raise ValueError('Missing required source field: ' + field)
    raw = json.loads(json.dumps(raw, allow_nan=True))
    n = raw['clients']
    if type(n) is not int or n < 2:
        raise ValueError('Invalid clients count including depot')
    for field in REQUIRED:
        value = np.asarray(raw[field], dtype=float)
        if not np.all(np.isfinite(value)):
            raise ValueError('Nonfinite source field: ' + field)
    demands = np.asarray(raw['demands'], dtype=np.float64)
    distances = np.asarray(raw['distances'], dtype=np.float64)
    if demands.shape != (n,) or demands[0] != 0 or np.any(demands[1:] <= 0) or np.any(demands != np.floor(demands)):
        raise ValueError('Invalid pallet demands')
    # The two published n34 matrices contain a nonzero unused diagonal entry.
    # Preserve source bytes/values. Hamiltonian route validation forbids self arcs.
    if distances.shape != (n, n) or np.any(distances < 0):
        raise ValueError('Invalid distances')
    speed = np.asarray(raw['speed_ranges'], dtype=float)
    thermal = np.asarray(raw['temperature_ranges'], dtype=float)
    if speed.shape != (24, 5) or thermal.shape != (288, 9):
        raise ValueError('Invalid hourly/monthly table dimensions')
    intervals = np.column_stack((np.arange(24), np.arange(1, 25)))
    if not np.array_equal(speed[:, :2], intervals) or not np.array_equal(thermal[:, :2], np.tile(intervals, (12, 1))):
        raise ValueError('Source slots must be contiguous half-open hours')
    if np.any(speed[:, 4] <= 0) or np.any(thermal[:, 8] <= 0):
        raise ValueError('Nonpositive speed or COP')
    if np.any(thermal[:, 5:8] < 0):
        raise ValueError('Negative thermal energy/power')
    positive = ('U', 'surface', 'carb_weight', 'mass', 'days', 'fuel_cost', 'fix_time', 'row_q', 'time_up')
    if any(raw[k] <= 0 for k in positive):
        raise ValueError('Nonpositive physical source field')
    if raw['row_q'] != int(raw['row_q']) or any(raw[k] < 0 for k in ('A', 'B', 'C', 'driver_wage', 'row_time', 'opening_time')):
        raise ValueError('Invalid physical source field')
    if len(raw['month_days']) != 12 or sum(raw['month_days']) != raw['days'] or any(x < 0 for x in raw['month_days']):
        raise ValueError('Inconsistent month_days/yearly tours')
    protocol = source_protocol()
    if sum(demands) > protocol['pallet_capacity']:
        raise ValueError('Demand exceeds sourced pallet capacity')
    if raw['Tint'] != -20 or not 0 <= raw['start_time'] < 24:
        raise ValueError('Outside original RRP temperature/day domain')
    demands.setflags(write=False)
    distances.setflags(write=False)
    return RRPInstance(name, MappingProxyType(raw), demands, distances, sha256)


def _verified_bytes(path, manifest=None):
    path = Path(path).resolve()
    rows = json.loads(Path(manifest or SOURCE / 'download_manifest.json').read_text(encoding='utf-8'))
    candidates = [r for r in rows if Path(r['file']).name == path.name]
    if len(candidates) != 1 or candidates[0]['commit'] != COMMIT:
        raise ValueError('Source file absent or ambiguous in fixed-commit manifest: ' + path.name)
    data = path.read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if sha != candidates[0]['sha256'] or len(data) != candidates[0]['bytes']:
        raise ValueError('Source SHA256/size mismatch: ' + path.name)
    return data, sha


def load_instance(path, manifest=None):
    data, sha = _verified_bytes(path, manifest)
    return parse_instance(json.loads(data), Path(path).stem, sha)


def _vector(text, label):
    lines = text.splitlines()
    for idx, line in enumerate(lines):
        if line.strip() == label + ':':
            for candidate in lines[idx + 1:idx + 4]:
                try:
                    values = [float(v) for v in candidate.split()]
                    if values:
                        return values
                except ValueError:
                    continue
    raise ValueError('Missing reference vector: ' + label)


def load_reference(path):
    data, sha = _verified_bytes(path)
    text = data.decode('utf-8-sig')
    route = _vector(text, 'Route')
    if any(not math.isfinite(v) or v != int(v) for v in route):
        raise ValueError('Noninteger reference route')
    saved = {label: _vector(text, label) for label in (
        'Arrival times', 'Stop times', 'Travel times', 'Departure times', 'Speeds')}
    costs = {}
    for key, label in {'traction': 'Traction', 'transmission': 'Transmission',
                       'infiltration': 'Infiltration', 'refrigeration': 'Refrigeration',
                       'driver': 'Driver', 'total': 'Total'}.items():
        match = re.search(label + r'(?: Consumption| cost| cons)?\s*:\s*([\d.]+)', text)
        if match:
            costs[key] = float(match.group(1))
    raw_waits = tuple(_vector(text, 'Waiting times'))
    _, clock_convention = reference_wait_mapping(route, raw_waits, saved)
    waits, origin = standard_reference_wait_mapping(route, raw_waits, saved)
    return ReferenceRecord(tuple(int(v) for v in route), waits,
                           saved, costs, 'with_wage' if 'driver' in costs else 'without_wage', sha,
                           raw_waits_s=raw_waits, wait_convention=origin,
                           label_origin=origin, saved_clock_convention=clock_convention)


def standard_reference_wait_mapping(route, raw_waits, saved):
    """Equation 33 pre-service indexing; printed clocks never override it.

    Table 7(b) independently confirms [0,*raw[:-1]] in 13/15 D2 records.
    The other records are explicitly distinct repository/paper snapshots.
    A missing all-zero entry is recoverable ONLY if the complete clock fields
    independently imply zero incoming and departure slack at EVERY position.
    """
    n = len(route)
    if any(not math.isfinite(v) or v < 0 for v in raw_waits):
        return raw_waits, 'incomplete_or_unresolved'
    if len(raw_waits) == n and raw_waits[-1] == 0:
        return (0.0, *raw_waits[:-1]), 'paper_pre_service_from_outgoing_arc'
    a, d, tr, s = (saved[k] for k in ('Arrival times', 'Departure times', 'Travel times', 'Stop times'))
    if (len(raw_waits) == n - 1 and not any(raw_waits) and len(a) == n + 1
            and all(len(v) == n for v in (d, tr, s))
            and all(math.isfinite(v) for values in (a, d, tr, s) for v in values)
            and all(a[i + 1] - d[i] - tr[i] == 0 and d[i] - a[i] - s[i] == 0 for i in range(n))):
        return (0.0,) * n, 'derived_unique_zero_from_author_clock_equalities'
    return raw_waits, 'incomplete_or_unresolved'


def reference_wait_mapping(route, raw_waits, saved):
    """Identify saved-clock serialization for archive DIAGNOSTICS only.

    Some CP records print waits on incoming arcs, others print departure waits
    by route position. This never sets canonical labels: those are determined
    by standard_reference_wait_mapping and the independent paper evidence.
    Incomplete/contradictory records remain invalid instead of padding a value.
    """
    n = len(route)
    a, d, tr, s = (saved[k] for k in ('Arrival times', 'Departure times', 'Travel times', 'Stop times'))
    if len(raw_waits) != n or len(a) != n + 1 or any(len(v) != n for v in (d, tr, s)):
        return raw_waits, 'incomplete_or_unresolved'
    arc = tuple(a[i + 1] - d[i] - tr[i] for i in range(n))
    post = tuple(d[i] - a[i] - s[i] for i in range(n))
    if not any(raw_waits) and arc == raw_waits and post == raw_waits:
        return raw_waits, 'zero_wait_equivalent'
    if arc == raw_waits and raw_waits[-1] == 0 and post != raw_waits:
        return (0.0, *raw_waits[:-1]), 'incoming_arc_slack'
    if post == raw_waits and arc != raw_waits:
        return raw_waits, 'departure_position_wait'
    return raw_waits, 'incomplete_or_unresolved'


def public_instances():
    return sorted((SOURCE / 'Datasets').rglob('*.json'))
