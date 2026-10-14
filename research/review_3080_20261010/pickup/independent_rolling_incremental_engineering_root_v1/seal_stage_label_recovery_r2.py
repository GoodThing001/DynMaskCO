"""Saved JSON-only correction of a case-sensitive phase-label expectation."""
import json,hashlib,datetime
from pathlib import Path
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def put(p,x):
    with p.open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False)
def main():
    names=['seal_stage.py','SAVED_ONLY_STAGE_IDENTITY_AUDIT.json','STAGE_ADJUDICATION.json','controller.py','PHASE_TRACE.bin','FINAL_ARTIFACT_IDENTITY.json']
    inputs={n:sha(HERE/n) for n in names}
    put(HERE/'LABEL_RECOVERY_REGISTRATION_R2.json',dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=sha(__file__),inputs=inputs,goal='Only saved JSON actual uppercase SAA_begin/end phase labels; no raw parser or scenario audit replay',production_repeated=False))
    x=read(HERE/'SAVED_ONLY_STAGE_IDENTITY_AUDIT.json');rows=x['trace_records'];ended=[r for r in rows if r['stage']=='SAA_end'];begun=[r for r in rows if r['stage']=='SAA_begin'];source=(HERE/'controller.py').read_text(encoding='utf-8');checks=[]
    def ck(n,v):checks.append(dict(name=n,pass_=bool(v)))
    ck('only_old_label_expectation_failure',x['failure_names']==['saved_seven_finished_reject_stages'] and x['failed']==1)
    ck('actual_uppercase_labels_bound_source',"'SAA_begin'" in source and "'SAA_end'" in source)
    ck('seven_reject_ended_exact_indices',len(ended)==7 and [r['index'] for r in ended]==list(range(7)) and all(r['decision']==0 for r in ended))
    ck('next_eighth_started_not_ended',begun[-1]['index']==7 and rows[-1]==begun[-1] and rows[-1]['stage']=='SAA_begin')
    ck('prior_119_successes_retained',sum(c['pass_'] for c in x['checks'])==119)
    for n,h in inputs.items():ck('original_unchanged:'+n,sha(HERE/n)==h)
    out=dict(checks_count=len(checks),failed=sum(not c['pass_'] for c in checks),checks=checks,old120_one_FAIL_preserved=True,prior_successful_items=119,last_actual_stage=rows[-1],meaning='Eighth reject branch entry telemetry was written before internal deadline guard; _sim_scenario call for index7 did not occur. No accept score branch or result serialization/source-end reached.')
    put(HERE/'SAVED_ONLY_LABEL_RECOVERY_R2.json',out)
    stage=read(HERE/'STAGE_ADJUDICATION.json');stage.update(saved_only_identity_original=dict(checks_count=120,failed=1,old_file_preserved=True,failure='case-sensitive uppercase SAA label misread by lowercase expectation'),saved_only_identity_recovery=dict(checks_count=len(checks),failed=out['failed'],successful_original_items_carried=119,no_original_parser_or_scenario_replay=True),last_actual_phase=out['last_actual_stage'],failure_phase_interpretation=out['meaning'])
    put(HERE/'STAGE_ADJUDICATION_R2.json',stage)
    final={str(p):sha(p) for p in HERE.rglob('*') if p.is_file() and p.name!='FINAL_ARTIFACT_IDENTITY_R2.json'}
    put(HERE/'FINAL_ARTIFACT_IDENTITY_R2.json',dict(files=final,prior_original_identity_preserved=all(sha(HERE/n)==h for n,h in inputs.items())))
    print(json.dumps(dict(stage_path=str(HERE/'STAGE_ADJUDICATION_R2.json'),recovery_checks=len(checks),recovery_failed=out['failed'],files=len(final))))
if __name__=='__main__':main()
