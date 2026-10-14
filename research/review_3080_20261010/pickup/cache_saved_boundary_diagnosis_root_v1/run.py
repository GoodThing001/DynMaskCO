import os, sys
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'; sys.dont_write_bytecode = True
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / 'tools'))
import ast, collections, json, struct, time
from fractions import Fraction
from engineering_stage_closure_v1 import initialize, closure, same, sha, save

def decode(x):
    if isinstance(x, dict):
        if '__mapping__' in x:
            return {decode(k): decode(v) for k, v in x['__mapping__']}
        if '__sequence__' in x:
            return [decode(v) for v in x['values']]
        return {k: decode(v) for k, v in x.items()}
    if isinstance(x, list): return [decode(v) for v in x]
    return x

def rational(x): return {'numerator': x.numerator, 'denominator': x.denominator}

def main():
    initialize(); reg = json.loads((HERE/'REGISTRATION.json').read_text(encoding='utf-8'))
    for rel, h in reg['inputs'].items(): assert sha(ROOT/rel) == h
    start = closure([Path(__file__)]); save(HERE/'SOURCE_START.json', start)
    t0 = time.perf_counter()
    full = ROOT / reg['request_root']; local = ROOT / reg['local_root']
    raw = (full/'PHASE_TRACE.bin').read_bytes()
    magic, cap, count, pid, first, padding = struct.unpack_from('<8sIIQQ32s', raw)
    assert magic == b'PKPHv001' and cap == 512 and len(raw) == 32832 and padding == bytes(32)
    rows = []
    for i in range(count):
        seq, clock, label, index, decision, v0, v1, pad = struct.unpack_from('<QQ16siiQQ8s', raw, 64+64*i)
        assert seq == i+1 and pad == bytes(8)
        rows.append({'sequence': seq, 'clock_ns': clock, 'stage': label.rstrip(b'\0').decode('ascii'), 'index': index, 'decision': decision, 'value0': v0, 'value1': v1})
    assert all(a['clock_ns'] <= b['clock_ns'] for a, b in zip(rows, rows[1:]))
    segments = []; prior = {'calls': 0, 'hits': 0, 'misses': 0, 'evictions': 0}
    for j in range(3):
        w = decode(json.loads((full/('ROLLOUT_reject_%02d.json' % j)).read_text(encoding='utf-8')))
        begin = next(x for x in rows if x['stage'] == 'SAA_begin' and x['index'] == j and x['decision'] == 0)
        end = next(x for x in rows if x['stage'] == 'SAA_end' and x['index'] == j and x['decision'] == 0)
        a = Fraction(begin['clock_ns']); b = Fraction.from_float(w['started']) * 1000000000
        c = Fraction.from_float(w['completed']) * 1000000000; d = Fraction(end['clock_ns'])
        assert a <= b <= c <= d
        delta = {k: w['cache_stats'][k]-prior[k] for k in prior}
        assert delta['calls'] == delta['hits'] + delta['misses']
        segments.append({'index': j, 'before_core_ns': rational(b-a), 'core_and_D_guard_ns': rational(c-b), 'after_core_to_marker_ns': rational(d-c), 'whole_marker_span_ns': rational(d-a), 'cache_delta': delta, 'cumulative_cache': w['cache_stats']})
        prior = {k: w['cache_stats'][k] for k in prior}
    old = json.loads((local/'RESULT.json').read_text(encoding='utf-8'))
    log = old['distance_log']; freq = collections.Counter((x['cur'], tuple(x['route'])) for x in log)
    distribution = collections.Counter(freq.values())
    src = ast.parse((full/'exact_distance_cache.py').read_text(encoding='utf-8'))
    clock_calls = [ast.unparse(n.func) for n in ast.walk(src) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in ('perf_counter', 'perf_counter_ns', 'monotonic', 'monotonic_ns')]
    assert not clock_calls
    result = {'schema': 'saved-boundary-diagnosis-v1', 'rows': rows, 'segments': segments, 'local_key_distribution': {'calls': len(log), 'distinct_keys': len(freq), 'repeated_calls': len(log)-len(freq), 'frequency_histogram': dict(sorted(distribution.items()))}, 'completed_prefix_cache_totals': prior, 'inner_cache_clocks_present': False, 'attribution': 'Core interval includes original simulation, cache operations, physics and final D guard; outer interval includes publishing/serialization plus other caller work. No query/miss/eviction times or full-request key stream are present.', 'reuse_scope': 'Local103 log reveals repeated exact keys only; cannot infer full132000 key working set or capacity optimum.', 'original_begin_marker_only': rows[-1], 'model_route_C0_or_old_parser_executions': 0, 'elapsed_seconds': time.perf_counter()-t0, 'stable_online_gain': False}
    end = closure([Path(__file__)]); save(HERE/'SOURCE_END.json', end)
    result['loaded_source_closed'] = same(start, end)
    save(HERE/'RESULT.json', result)
    assert result['loaded_source_closed']
    print(json.dumps({'segments': len(segments), 'local_distinct': len(freq), 'completed_calls': prior['calls'], 'inner_time_attribution': False}))

if __name__ == '__main__': main()
