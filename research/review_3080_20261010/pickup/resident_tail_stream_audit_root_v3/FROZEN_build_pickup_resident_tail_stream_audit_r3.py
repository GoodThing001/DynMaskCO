"""Recover an unqualified audit footer in a new exclusive directory; no producer rerun."""
from pathlib import Path
import ast, json, hashlib
ROOT=Path(__file__).resolve().parents[1]
P=ROOT/'research/review_3080_20261010/pickup'
def main():
    source=ROOT/'tools/audit_pickup_resident_tail_stream_r2.py'
    frozen=P/'resident_day_tail_root_v1/FROZEN_audit_pickup_resident_tail_stream_r2.py'
    with frozen.open('xb') as f:f.write(source.read_bytes())
    receipt=P/'resident_day_tail_root_v1/STREAM_AUDIT_FOOTER_FAILURE_R2.json'
    with receipt.open('x',encoding='utf-8') as f:json.dump(dict(exit_code=1,exception='FileNotFoundError',missing='RESULT.json',phase='result footer after SOURCE_END was written',checks_output_absent=True,checks_count_unknown=True,qualification=False,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),scope='Actual independent replay executed, but in-memory checks were not archived. Do not invent counts or mark PASS. Original producer RUN_EXIT1 remains. A separately registered auditor recovery may replay the same unqualified audit inputs; never rerun the fifty production requests.'),f,indent=2)
    a=source.read_text(encoding='utf-8')
    a=a.replace("def main():\n", "A=P/'resident_tail_stream_audit_root_v3'\ndef main():\n    A.mkdir(exist_ok=False)\n")
    a=a.replace("def ck(n,v):checks.append(dict(name=n,pass_=bool(v)))", "def ck(n,v):\n        row=dict(name=n,pass_=bool(v));checks.append(row)\n        with (A/'CHECKS.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(row)+'\\n')")
    a=a.replace("+[SOURCE,Path(__file__)]", "+[SOURCE,Path(__file__),ROOT/'tools/build_pickup_resident_tail_stream_audit_r3.py']")
    for name in ['STREAM_AUDIT_REGISTRATION_R2.json','STREAM_AUDIT_SOURCE_START_R2.json','STREAM_AUDIT_SOURCE_END_R2.json','INDEPENDENT_RESIDENT_TAIL_STREAM_R2.json']:
        a=a.replace("(OUT/'"+name+"')", "(A/'"+name.replace('R2','R3')+"')")
    a=a.replace("producer_sha256=old.sha(OUT/'RESULT.json')", "producer_exit_sha256=old.sha(OUT/'RUN_EXIT.json'),producer_failure_sha256=old.sha(OUT/'FAILURE.json'),prior_unqualified_audit_source_sha256=old.sha(OUT/'FROZEN_audit_pickup_resident_tail_stream_r2.py')")
    a=a.replace("First independent stream replay of all50", "Audit-only footer recovery R3 of unqualified R2, same fixed all50")
    ast.parse(a)
    with (ROOT/'tools/audit_pickup_resident_tail_stream_r3.py').open('x',encoding='utf-8',newline='\n') as f:f.write(a)
    print('R2 failure preserved; R3 auditor prepared, no production execution')
if __name__=='__main__':main()
