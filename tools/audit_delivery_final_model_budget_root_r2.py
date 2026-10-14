"""Read actual executed bytes; preserve first audit and compare newline transport explicitly."""
import ast, hashlib, json, posixpath, sys
from datetime import datetime, timezone
from pathlib import Path
import remote_actual_fullstep_3080_v2 as remote
ROOT=Path(__file__).resolve().parents[1]
D=ROOT/'research/review_3080_20261010/delivery'

def sha(raw):return hashlib.sha256(raw).hexdigest()
def main(version):
    if version not in ('v2','v3'):raise ValueError('Isolated completed version required')
    folder=D/('remote_final_model_budget_bundle_root_'+version)
    actual=folder/'actual_executed_bytes_r2'
    actual.mkdir(exist_ok=False)
    cfg=json.loads(remote.CONFIG.read_text())
    base=posixpath.join(cfg['remotePath'].rstrip('/'),'research/review_3080_20261010/remote_final_model_budget_root_'+version)
    client=remote.connect();s=client.open_sftp()
    try:
        for name in ('budget_qa.py','BUDGET_ENGINEERING_REGISTRATION.json'):
            with s.open(posixpath.join(base,name),'rb') as f:raw=f.read()
            (actual/name).write_bytes(raw)
    finally:s.close();client.close()
    reg=json.loads((actual/'BUDGET_ENGINEERING_REGISTRATION.json').read_bytes())
    original=D/('final_model_budget_root_'+version)
    local=(original/'budget_qa.py').read_bytes()
    executed=(actual/'budget_qa.py').read_bytes()
    receipt={'read_at_utc':datetime.now(timezone.utc).isoformat(),
      'actual_program_sha256':sha(executed),'registered_program_sha256':reg['program_sha256'],
      'actual_registration_sha256':sha((actual/'BUDGET_ENGINEERING_REGISTRATION.json').read_bytes()),
      'local_registration_unchanged':(original/'BUDGET_ENGINEERING_REGISTRATION.json').read_bytes()==(actual/'BUDGET_ENGINEERING_REGISTRATION.json').read_bytes(),
      'local_prepared_program_sha256':sha(local),'local_prepared_CRLF_count':local.count(b'\r\n'),
      'actual_executed_CRLF_count':executed.count(b'\r\n'),
      'prepared_diff_is_only_CRLF_transport':local.replace(b'\r\n',b'\n')==executed,
      'old_audit_preserved':True,'experiment_reexecuted':False,'source_files_rewritten':False}
    (actual/'ACTUAL_BYTE_IDENTITY.json').write_text(json.dumps(receipt,indent=2))
    assert receipt['local_registration_unchanged'] and receipt['prepared_diff_is_only_CRLF_transport']
    assert receipt['actual_program_sha256']==receipt['registered_program_sha256']
    source=(ROOT/'tools/audit_delivery_final_model_budget_root.py').read_text()
    needle='    ast.parse(text)'
    injection="""    text=text.replace("sha(regp.parent/'budget_qa.py')", "sha(R/'remote_final_model_budget_bundle_root_"+version+"/actual_executed_bytes_r2/budget_qa.py')")
    if version=='v2':
        text=text.replace('INDEPENDENT_ROOT_MODEL_BUDGET_AUDIT.json','INDEPENDENT_ROOT_MODEL_BUDGET_AUDIT_R2.json')
"""
    source=source.replace(needle,injection+needle)
    source=source.replace("out/'INDEPENDENT_ROOT_MODEL_BUDGET_AUDIT.json'","out/('INDEPENDENT_ROOT_MODEL_BUDGET_AUDIT_R2.json' if version=='v2' else 'INDEPENDENT_ROOT_MODEL_BUDGET_AUDIT.json')")
    source=source.replace("out/'ACCEPTED_FINAL_MODEL_BUDGET.json'","out/('ACCEPTED_FINAL_MODEL_BUDGET_R2.json' if version=='v2' else 'ACCEPTED_FINAL_MODEL_BUDGET.json')")
    ast.parse(source)
    ns={'__name__':'independent_actual_byte_budget_r2','__file__':str(Path(__file__).resolve())}
    exec(compile(source,str(Path(__file__).resolve()),'exec'),ns)
    return ns['main'](version)
if __name__=='__main__':sys.exit(main(sys.argv[1]))
