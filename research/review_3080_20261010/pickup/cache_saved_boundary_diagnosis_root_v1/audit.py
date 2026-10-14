import os, sys
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'; sys.dont_write_bytecode = True
from pathlib import Path
HERE = Path(__file__).resolve().parent; ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT/'tools'))
import ast, json, time
from fractions import Fraction as Q
from engineering_stage_closure_v1 import initialize, closure, same, sha, save

def unpack(x):
    if isinstance(x, dict) and '__mapping__' in x: return dict((unpack(p[0]), unpack(p[1])) for p in x['__mapping__'])
    if isinstance(x, dict) and '__sequence__' in x: return list(map(unpack, x['values']))
    if isinstance(x, dict): return {k: unpack(v) for k, v in x.items()}
    if isinstance(x, list): return list(map(unpack, x))
    return x

def main():
    initialize(); reg = json.loads((HERE/'REGISTRATION.json').read_text(encoding='utf-8'))
    start = closure([Path(__file__)]); save(HERE/'AUDIT_SOURCE_START.json', start)
    checks = []
    def ck(n, ok): checks.append({'name': n, 'pass': bool(ok)})
    result = json.loads((HERE/'RESULT.json').read_text(encoding='utf-8'))
    for rel,h in reg['inputs'].items(): ck('input:'+rel, sha(ROOT/rel)==h)
    full = ROOT/reg['request_root']; local = ROOT/reg['local_root']; raw = (full/'PHASE_TRACE.bin').read_bytes()
    get = lambda b, signed=False: int.from_bytes(b, 'little', signed=signed)
    count = get(raw[12:16]); ck('header', raw[:8]==b'PKPHv001' and get(raw[8:12])==512 and len(raw)==32832)
    for i in range(count):
        b = raw[64+64*i:128+64*i]
        r = result['rows'][i]
        ck('row:'+str(i), (get(b[:8]),get(b[8:16]),b[16:32].rstrip(bytes([0])).decode('ascii'),get(b[32:36],True),get(b[36:40],True),get(b[40:48]),get(b[48:56])) == (r['sequence'],r['clock_ns'],r['stage'],r['index'],r['decision'],r['value0'],r['value1']) and b[56:]==bytes(8))
    prior = {k:0 for k in ('calls','hits','misses','evictions')}
    for j in range(3):
        w = unpack(json.loads((full/('ROLLOUT_reject_%02d.json'%j)).read_text(encoding='utf-8')))
        bb = next(r for r in result['rows'] if (r['stage'],r['index'],r['decision'])==('SAA_begin',j,0))
        ee = next(r for r in result['rows'] if (r['stage'],r['index'],r['decision'])==('SAA_end',j,0))
        limits = [Q(bb['clock_ns']),Q.from_float(w['started'])*10**9,Q.from_float(w['completed'])*10**9,Q(ee['clock_ns'])]
        spans = [limits[1]-limits[0],limits[2]-limits[1],limits[3]-limits[2],limits[3]-limits[0]]
        for k,v in zip(('before_core_ns','core_and_D_guard_ns','after_core_to_marker_ns','whole_marker_span_ns'),spans):
            d=result['segments'][j][k]; ck('span:'+str(j)+':'+k,Q(d['numerator'],d['denominator'])==v)
        for k in prior: ck('counter_delta:'+str(j)+':'+k,result['segments'][j]['cache_delta'][k]==w['cache_stats'][k]-prior[k]); prior[k]=w['cache_stats'][k]
    old=json.loads((local/'RESULT.json').read_text(encoding='utf-8')); keys=[(r['cur'],tuple(r['route'])) for r in old['distance_log']]
    ck('new_key_analysis', result['local_key_distribution']['distinct_keys']==len(set(keys)) and result['local_key_distribution']['repeated_calls']==len(keys)-len(set(keys)))
    ck('no_inner_clock', not any(isinstance(n,ast.Attribute) and n.attr in ('perf_counter','perf_counter_ns','monotonic','monotonic_ns') for n in ast.walk(ast.parse((full/'exact_distance_cache.py').read_text(encoding='utf-8')))))
    ck('no_utility_claim', result['stable_online_gain'] is False and result['inner_cache_clocks_present'] is False)
    producer_start=json.loads((HERE/'SOURCE_START.json').read_text(encoding='utf-8')); producer_end=json.loads((HERE/'SOURCE_END.json').read_text(encoding='utf-8'))
    ck('producer_closed', same(producer_start,producer_end))
    for path,h in producer_start['files'].items(): ck('producer_file:'+path,sha(path)==h)
    end=closure([Path(__file__)]); save(HERE/'AUDIT_SOURCE_END.json',end); ck('audit_closed',same(start,end))
    save(HERE/'INDEPENDENT_AUDIT.json',{'checks_count':len(checks),'failed':sum(not c['pass'] for c in checks),'checks':checks,'physics_or_original_parser_calls':0})
    print(json.dumps({'checks':len(checks),'failed':sum(not c['pass'] for c in checks)}))
    return int(any(not c['pass'] for c in checks))

if __name__=='__main__': raise SystemExit(main())
