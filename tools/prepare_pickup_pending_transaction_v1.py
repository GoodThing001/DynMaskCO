"""Create exclusive partial-admission adapters without editing decided contracts."""
from pathlib import Path
import hashlib
ROOT=Path(__file__).resolve().parents[1];P=ROOT/'research/review_3080_20261010/pickup';E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem'
def replace_once(s,a,b):
    if s.count(a)!=1:raise ValueError('source anchor count '+repr(a))
    return s.replace(a,b)
def create(name,s):
    with (ROOT/'tools'/name).open('x',encoding='utf-8',newline='\n') as f:f.write(s)
original=E/'scripts/evaluation/coldchain_evaluator_a1.py'
s=original.read_text(encoding='utf-8')
s=replace_once(s,'def evaluate_trace_a1(traces, dataset_instance, contract, accepted_set, rejected_set,','def evaluate_partial_trace(traces, dataset_instance, contract, accepted_set, rejected_set,')
s=replace_once(s,'B, reject_penalty_by_class):','B, reject_penalty_by_class, pending_set):')
s=replace_once(s,'    unaccounted = universe_set - accepted_set - rejected_set','''    # Pending orders retain real features but have no service obligation or
    # rejection loss until a subsequent decision. This is a provisional scope.
    if pending_set & (accepted_set | rejected_set):
        failures.append('pending_overlaps_decided')
    if pending_set - universe_set:
        failures.append('pending_outside_observed_universe')
    if pending_set & served_set:
        failures.append('pending_order_served')
    unaccounted = universe_set - accepted_set - rejected_set - pending_set''')
s=replace_once(s,"        'utility': float(utility),","        'pending_count': len(pending_set),\n        'provisional_not_full_day_utility': True,\n        'utility': float(utility),")
create('pickup_partial_trace_v1.py','# Frozen original evaluator SHA256 '+hashlib.sha256(original.read_bytes()).hexdigest()+'\n'+s)
s=(P/'public_fleet.py').read_text(encoding='utf-8')
start=s.index('def evaluate_known(');end=s.index('\ndef validate_public(',start)
s=s[start:end]
s=replace_once(s,'def evaluate_known(packet,plan):','def evaluate_partial_known(packet,plan):')
s=replace_once(s,'from coldchain_evaluator_a1 import evaluate_trace_a1','from pickup_partial_trace_v1 import evaluate_partial_trace')
s=replace_once(s,'ev=evaluate_trace_a1(', 'ev=evaluate_partial_trace(')
s=replace_once(s,"packet['budget'],packet['penalties'])","packet['budget'],packet['penalties'],set(packet['admission_input']['pending']))")
create('pickup_partial_known_v1.py','import numpy as np\n'+s)
s=(ROOT/'tools/pickup_undecided_public_v1.py').read_text(encoding='utf-8')
s=s[s.index('def validate_undecided('):]
s=replace_once(s,'def validate_undecided(packet):','def validate_partial(packet):')
s=replace_once(s,"if packet['schema']!='pickup-public-undecided-v1'","if packet['schema']!='pickup-public-partial-admissions-v1'")
s=replace_once(s,"    if type(current) is not int or current not in p or snap['served_mask'][current]: raise ValueError('undecided current order')", "    if type(current) is not int or current not in a|r or snap['served_mask'][current]: raise ValueError('current decision missing')\n    if packet['admission_input']['current_decision'] != ('accepted' if current in a else 'rejected'): raise ValueError('decision label')")
create('pickup_partial_validation_v1.py','import numpy as np\n'+s)
s=(P/'common_candidate_saa_root_v2/common_score.py').read_text(encoding='utf-8')
s=replace_once(s,'def score_common(', 'def score_partial_common(')
s=replace_once(s,'    pf.validate_public(packet)','    from pickup_partial_validation_v1 import validate_partial\n    from pickup_pending_transaction_v1 import certificate\n    validate_partial(packet)')
s=replace_once(s,'certificate,_=pf.certificate(packet,plan)','cert,_=certificate(packet,plan)')
s=s.replace("certificate['qualified']","cert['qualified']").replace("'certificate':certificate","'certificate':cert")
s=replace_once(s,"return {'scenario_sha256':scenario_hash", "return {'observed_snapshot':snapshot,'pending_not_rejected':list(packet['admission_input']['pending']),'scenario_sha256':scenario_hash")
create('pickup_partial_common_v1.py',s)
