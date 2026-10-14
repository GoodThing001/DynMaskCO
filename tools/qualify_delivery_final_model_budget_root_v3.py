"""Same engineering contract; bind all charged parent work to CPU15 too."""
import ast,json,sys
from pathlib import Path
import qualify_delivery_final_model_budget_root_v2 as base
ROOT=base.ROOT;D=base.D
SOURCE=D/'final_model_budget_root_v3';LOCAL=D/'remote_final_model_budget_bundle_root_v3'
REL='research/review_3080_20261010/remote_final_model_budget_root_v3'

def deploy():
    old=D/'remote_final_model_budget_bundle_root_v2/downloaded/RUN_EXIT.json'
    if not old.exists() or json.loads(old.read_text())['exit_code']!=0:
        raise ValueError('Preserve completed v2 evidence before engineering recovery')
    # Copy the known v2 launcher body into a fresh module, changing only scoped
    # parent resource binding and its executable evidence fields.
    source=Path(base.__file__).read_text(encoding='utf-8')
    marker="    parent=D/'model_budget_integration_root_v1/budget_qa.py';body=parent.read_text()"
    assert source.count(marker)==1
    replacement=marker+'''
    marker_parent="    import psutil\\n    bootstrap(a.root)"
    assert body.count(marker_parent)==1
    body=body.replace(marker_parent,"    import psutil\\n    psutil.Process().cpu_affinity([a.cpu])\\n    bootstrap(a.root)")
    body=body.replace("'provider':provider.identity if provider else None}","'provider':provider.identity if provider else None,'actual_cpu_affinity':psutil.Process().cpu_affinity()}")
    body=body.replace("'source_unchanged':end==reg['source_sha256'],", "'source_unchanged':end==reg['source_sha256'],'actual_charged_parent_cpu_affinity':psutil.Process().cpu_affinity(),")
'''
    source=source.replace(marker,replacement)
    source=source.replace("SOURCE=D/'final_model_budget_root_v2';LOCAL=D/'remote_final_model_budget_bundle_root_v2'",
        "SOURCE=D/'final_model_budget_root_v3';LOCAL=D/'remote_final_model_budget_bundle_root_v3'")
    source=source.replace("REL='research/review_3080_20261010/remote_final_model_budget_root_v2'",
        "REL='research/review_3080_20261010/remote_final_model_budget_root_v3'")
    source=source.replace("TOTAL10S-AND-TIMEOUT-QA-root-v2'","TOTAL10S-AND-TIMEOUT-QA-root-v3'")
    source=source.replace("'maximum_seconds':600}","'maximum_seconds':600,'resource_recovery_scope':'charged parent and child both CPU15; original v2 recorded parent0..15 retained; same weights/masks/fixtures/budget'}")
    ast.parse(source)
    namespace={'__name__':'isolated_final_budget_v3','__file__':str(Path(__file__).resolve())}
    exec(compile(source,str(Path(__file__).resolve()),'exec'),namespace)
    return namespace['deploy']()

if __name__=='__main__':
    action=sys.argv[1]
    if action=='deploy':code=deploy()
    elif action in ('status','download'):
        base.t.LOCAL=LOCAL;base.t.REL=REL;code=getattr(base.t,action)()
    else:raise ValueError('Unknown action')
    sys.exit(code)
