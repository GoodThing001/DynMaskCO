"""Append-only engineering recovery: qualify real mutable rather than total tails."""
import ast
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
src=ROOT/'tools/qualify_pickup_actual_common_candidate_root_v1.py'
dst=ROOT/'tools/qualify_pickup_actual_common_candidate_root_v2.py'
s=src.read_text(encoding='utf-8')
s=s.replace("OUT=P/'actual_common_candidate_root_v1'","OUT=P/'actual_common_candidate_root_v2'")
s=s.replace('def get_fixture():','def get_fixture(require_two_mutable=True):')
s=s.replace("return next(r['public_request'] for r in raw['admissions'] if any(len(v)>=2 for v in r['public_request']['plan'].values()))",
 "return next(r['public_request'] for r in raw['admissions'] if (len(pf.family(r['public_request'])[1])>=2 if require_two_mutable else any(len(v)>=2 for v in r['public_request']['plan'].values())))")
old="""                n=len(mf.topology(packet)[2])+1
                if n not in range(3,14):raise ValueError('unwarmed observed graph; cannot exclude new setup')
                _,proposal=mf.propose(provider,packet,SCRIPTS,mode=mode,seed=202610101903,mask_fraction=.3,guide_weight=1.,deadline=deadline)
                candidate=proposal.get('raw_candidate',packet['plan'])"""
new="""                if len(pf.family(packet)[1])<2:
                    candidate=copy.deepcopy(packet['plan']);proposal={'kind':'ineligible_incumbent_fallback','reason':'fewer than two mutable orders; committed destination excluded','decoder_calls':0}
                else:
                    n=len(mf.topology(packet)[2])+1
                    if n not in range(3,14):raise ValueError('unwarmed observed graph; cannot exclude new setup')
                    _,proposal=mf.propose(provider,packet,SCRIPTS,mode=mode,seed=202610101903,mask_fraction=.3,guide_weight=1.,deadline=deadline)
                    candidate=proposal.get('raw_candidate',packet['plan'])"""
assert s.count(old)==1;s=s.replace(old,new)
s=s.replace("'fixture_sha256':sha(fixture_path),","'fixture_sha256':sha(fixture_path),'fixture_selection':'first recorded public state with >=2 mutable orders excluding committed destination; no score-based selection','old_failed_fixture_preserved':True,")
s=s.replace("write(OUT/'REGISTRATION.json',reg);packet=get_fixture();records=[]","write(OUT/'REGISTRATION.json',reg);packet=get_fixture();ineligible=get_fixture(False);records=[]")
anchor='            # A distinct10ms delayed request tests the hard response barrier only;'
addition=r'''            if row['parent_on_time'] and mode in ('pretrained','random','discard'):
                b0=time.perf_counter();bd=b0+10.
                p.stdin.write(json.dumps({'request':wire.wire({'packet':ineligible,'deadline':bd})})+'\n');p.stdin.flush()
                try:boundary=q.get(timeout=max(0.,bd-time.perf_counter()))
                except queue.Empty:boundary={'kind':'timeout'}
                row['ineligible_domain_boundary']={'started':b0,'deadline':bd,'received':time.perf_counter(),'response':wire.unwire(boundary['result']) if boundary.get('kind')=='response' else None,'reply_kind':boundary['kind']}
'''
assert s.count(anchor)==1;s=s.replace(anchor,addition+anchor)
ast.parse(s)
with dst.open('x',encoding='utf-8') as f:f.write(s)
print('Isolated v2 prepared; v1/source/results unchanged; no experiment started.')
