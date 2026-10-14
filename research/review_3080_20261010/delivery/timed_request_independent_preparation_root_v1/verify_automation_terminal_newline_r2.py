"""Narrow read-only recovery: prove only the app's terminal LF was omitted."""
import hashlib,json,tomllib
from pathlib import Path
O=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def save(n,x):
    with (O/n).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,ensure_ascii=False)
def main():
    original=read(O/'SOURCE_PREPARATION_RECORDING_VERIFICATION.json')
    before=tomllib.loads((O/'AUTOMATION_BEFORE_TIMED_INDEPENDENT_PREPARATION.toml').read_text(encoding='utf-8-sig'))
    nowfile=Path(r'C:/Users/mmc/.codex/automations/maskco/automation.toml')
    actual=tomllib.loads(nowfile.read_text(encoding='utf-8-sig'))
    addition=read(O/'AUTOMATION_ADDED_EVIDENCE.json');expected=before['prompt']+addition
    checks=[]
    def ck(n,v):checks.append(dict(name=n,passed=bool(v)))
    ck('original79_oneFAIL_preserved',original['checks_count']==79 and original['failed']==1)
    ck('original_failure_only_exact_append',[c['name'] for c in original['checks'] if not c['passed']]==['automation_exact_append'])
    ck('original78_successes_retained',sum(c['passed'] for c in original['checks'])==78)
    ck('exact_original_prompt_prefix',actual['prompt'].startswith(before['prompt']))
    ck('requested_one_terminal_LF',expected.endswith('\n') and not expected.endswith('\n\n'))
    ck('only_one_terminal_LF_missing',actual['prompt']==expected[:-1] and len(actual['prompt'])==len(expected)-1)
    ck('all_addendum_content_exact',actual['prompt'][len(before['prompt']):]==addition[:-1])
    for k in ('id','kind','name','status','rrule','target_thread_id'):ck('preserved_field:'+k,actual[k]==before[k])
    result=dict(checks=checks,checks_count=len(checks),failed=sum(not c['passed'] for c in checks),
        original_verification_sha256=sha(O/'SOURCE_PREPARATION_RECORDING_VERIFICATION.json'),actual_automation_sha256=sha(nowfile),
        scope='Read-only original78 successes plus exact terminal-LF omission evidence. Original79/1FAIL unchanged; no model, QA, automation rewrite or scientific threshold change.',
        original_overall_PASS=False,model_executed=False,execution_authorized=False)
    save('AUTOMATION_TERMINAL_NEWLINE_RECOVERY_R2.json',result)
    print(json.dumps(dict(checks=result['checks_count'],failed=result['failed'],original_79_1FAIL_preserved=True)))
    return int(result['failed']>0)
if __name__=='__main__':raise SystemExit(main())
