"""Only instrument the frozen failed controller; do not change its method."""
import ast,hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;OLD=HERE.parent/'independent_rolling_admission_root_v1'
def replace_once(source,a,b):
    if source.count(a)!=1:raise ValueError('replacement identity:'+a)
    return source.replace(a,b)
def main():
    s=(OLD/'controller.py').read_text(encoding='utf-8')
    replacements=[
      ('import numpy as np','import numpy as np\nfrom phase_trace import emit'),
      ("for iteration in range(4):","for iteration in range(4):\n        emit('recourse_pass',iteration)"),
      ("validate_undecided(packet);original_digest=pf.digest(packet)","emit('compute_enter')\n    validate_undecided(packet);original_digest=pf.digest(packet)"),
      ("candidates={};selected={}","emit('generated',value0=len(generated['reject']),value1=len(generated['accept']))\n    candidates={};selected={}"),
      ("for index,plan in enumerate(generated[decision]):","for index,plan in enumerate(generated[decision]):\n            emit('candidate_begin',index,decision)"),
      ("final,recourse=distance_recourse(plan,packet,rp,env,fleet,served,deadline);p=branch_packet(packet,decision,final)","emit('recourse_begin',index,decision)\n            final,recourse=distance_recourse(plan,packet,rp,env,fleet,served,deadline);p=branch_packet(packet,decision,final)\n            emit('recourse_end',index,decision)"),
      ("cert,raw=certificate(p,final);row=", "emit('C0_begin',index,decision)\n            cert,raw=certificate(p,final)\n            emit('C0_end',index,decision,value0=int(cert['qualified']))\n            row="),
      ("selected[decision]=best", "selected[decision]=best\n        emit('decision_plan',decision=decision,value0=int(best is not None))"),
      ("scenarios=shadow.sampler.sample(observed,np.random.default_rng(202610130901),10);scenario_sha=pf.digest(scenarios)","emit('scenario_draw')\n    scenarios=shadow.sampler.sample(observed,np.random.default_rng(202610130901),10);scenario_sha=pf.digest(scenarios)\n    emit('scenario_drawn',value0=len(scenarios))"),
      ("row=selected[decision]\n        if row is None", "emit('score_branch',decision=decision)\n        row=selected[decision]\n        if row is None"),
      ("for ids in indices:","for scenario_index,ids in enumerate(indices):\n            emit('SAA_begin',scenario_index,decision)"),
      ("values.append(value)","values.append(value)\n            emit('SAA_end',scenario_index,decision)"),
      ("return dict(candidates=candidates", "emit('compute_done')\n    return dict(candidates=candidates")]
    for a,b in replacements:s=replace_once(s,a,b)
    ast.parse(s);(HERE/'controller.py').open('x',encoding='utf-8').write(s)
    run=(OLD/'run.py').read_text(encoding='utf-8')
    run=replace_once(run,'from controller import compute','from controller import compute\nimport phase_trace')
    run=replace_once(run,"initial=closure();save('SOURCE_START.json',initial)","phase_trace.initialize(HERE/'PHASE_TRACE.bin')\n    initial=closure();save('SOURCE_START.json',initial)")
    run=replace_once(run,"final=closure();save('SOURCE_END.json',final)","phase_trace.close()\n        final=closure();save('SOURCE_END.json',final)")
    run=replace_once(run,"HERE/'controller.py',HERE/'run.py',HERE/'audit.py',INPUT,old.HISTORY", "HERE/'controller.py',HERE/'run.py',HERE/'audit.py',HERE/'phase_trace.py',HERE/'prepare_sources.py',INPUT,old.HISTORY")
    run=replace_once(run,"save('INPUT.json',encode(p));q=queue.Queue()", "save('INPUT.json',encode(p))\n    with (HERE/'PHASE_TRACE.bin').open('xb') as f:f.write(bytes(32832))\n    q=queue.Queue()")
    run=replace_once(run,"method='current-only independent", "diagnostic_goal='same failed method/input/budget with new bounded incremental stage observations; no utility retry, no source repair or scientific gain',prior_registration_sha256='f82d1871cc8e88d557d1a2c306cf583b5d8e387245aedd69a83c26455db88d44',telemetry_size_bytes=32832,method='current-only independent")
    ast.parse(run);(HERE/'run.py').open('x',encoding='utf-8').write(run)
    with (HERE/'PREPARATION_IDENTITY.json').open('x',encoding='utf-8') as f:json.dump({'original_sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (OLD/'controller.py',OLD/'run.py',OLD/'REGISTRATION.json')},'change':'bounded emitted phase records only; method/input/seed/passes/shadow and deadline identical; setup trace allocation outside component; writer operations counted inside'},f,indent=2)
if __name__=='__main__':main()
