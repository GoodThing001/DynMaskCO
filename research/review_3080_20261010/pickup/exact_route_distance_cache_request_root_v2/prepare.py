import ast,json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;OLD=HERE.parent/'independent_rolling_phase_diagnosis_root_v1'
def change(s,a,b):
    if s.count(a)!=1:raise ValueError('unique source replacement:'+a)
    return s.replace(a,b)
def write(p,s):ast.parse(s);p.open('x',encoding='utf-8').write(s)
def main():
    (HERE/'phase_trace.py').open('x',encoding='utf-8').write((OLD/'phase_trace.py').read_text(encoding='utf-8'))
    s=(OLD/'controller.py').read_text(encoding='utf-8')
    s=change(s,'from energy_complete_saa import EnergyCompleteSAA','from observed_saa import ObservedSAA as EnergyCompleteSAA,save_payload')
    s=change(s,"future_policy='reveal',standby_orders=0)","future_policy='reveal',standby_orders=0,incr_eval=True)")
    s=change(s,'scores={}',"save_payload('SIMULATION_CONTEXT.json',dict(packet=packet,candidates=candidates,selected={k:None if v is None else v['index'] for k,v in selected.items()},scenarios=scenarios,indices=indices,scenario_sha256=scenario_sha,observed_snapshot=observed))\n    scores={}")
    s=change(s,'values.append(value)',"shadow.publish(decision,scenario_index,value)\n            values.append(value)")
    write(HERE/'controller.py',s)
    run=(OLD/'run.py').read_text(encoding='utf-8')
    run=change(run,"HERE/'controller.py',HERE/'run.py',HERE/'audit.py',HERE/'phase_trace.py',HERE/'prepare_sources.py',INPUT,old.HISTORY","HERE/'controller.py',HERE/'run.py',HERE/'audit.py',HERE/'observed_saa.py',HERE/'phase_trace.py',HERE/'prepare.py',INPUT,old.HISTORY")
    run=change(run,"diagnostic_goal='same failed method/input/budget with new bounded incremental stage observations; no utility retry, no source repair or scientific gain'","diagnostic_goal='candidate existingincr_eval route-summary prefilters and delta-ranking; exact action/terminalC0/float.hex equivalence versus originalfalse is REQUIRED, not assumed; no utility selection or scientific gain'")
    run=change(run,"prior_registration_sha256='f82d1871cc8e88d557d1a2c306cf583b5d8e387245aedd69a83c26455db88d44'","prior_registration_sha256='61d71415311a784bdab5787e30920e32955040baa3890ad7766f20a1e713a474'")
    run=change(run,"time_limit=", "time_limit=") if False else run
    run=change(run,"K=10,seed=202610130901","candidate_incr_eval=True,exact_reference_incr_eval=False,partial_prefix_only_cannot_release=True,witness_per_file_cap_bytes=2097152,reference_audit_supervision_cap_s=120.,PyVRP_reference='Wouda,Lan,Kool2024 IJOC36(4)943-955 DOI10.1287/ijoc.2023.0055,route move cache and conservative filtering inspiration only; not implementation replication or C0 equivalence',K=10,seed=202610130901")
    write(HERE/'run.py',run)
    for filename in ('controller.py','run.py','phase_trace.py'):
        p=OLD/filename
    with (HERE/'PARENT_IDENTITIES.json').open('x',encoding='utf-8') as f:json.dump({'prior_files':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (OLD/'controller.py',OLD/'run.py',OLD/'phase_trace.py',OLD/'REGISTRATION.json',OLD/'STAGE_ADJUDICATION.json')},'scope':'new candidate flag plus action witnesses; originalmethod/input/K10/physical contract/acceptance rule/deadline unchanged; equivalence not assumed'},f,indent=2)
if __name__=='__main__':main()
