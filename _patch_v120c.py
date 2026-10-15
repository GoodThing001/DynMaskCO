# -*- coding: utf-8 -*-
p = (r'D:\PyCharm_\MASKCO-Main\C-VRP_Cold-chainVehicleRoutingProblem'
     r'\scripts\tests\d3_l1_v120_bypass_acceptance.py')
s = open(p, encoding='utf-8').read()

# 1. helper: read the guard's own refusal record (the full reason, not a truncated stderr tail)
helper = '''

def guard_refusal_reason(run_dir):
    """The FULL guard refusal, read from the immutable record the worker published.

    Reading stderr would be reading a human-facing tail; the record is the durable evidence, and it
    is what an auditor would read.
    """
    rec = persistence.VerifiedReader(run_dir).latest('WORKER_GUARD_REFUSED') or {}
    return rec.get('reason') or ''

'''
anchor = "\ndef worker_cmd(run_dir):"
assert s.count(anchor) == 1
s = s.replace(anchor, helper + "\ndef worker_cmd(run_dir):")

reps = [
    # 2. N4
    ("    ok = (proc.returncode != 0 and art['MODEL_LOAD'] == 0 and art['SLOT_day'] == 0\n"
     "          and ('consumption_receipt_missing' in proc.stderr\n"
     "               or 'context_missing_field' in proc.stderr))",
     "    reason = guard_refusal_reason(run_dir)\n"
     "    det['guard_reason'] = reason\n"
     "    ok = (proc.returncode != 0 and art['MODEL_LOAD'] == 0 and art['SLOT_day'] == 0\n"
     "          and ('consumption_receipt_missing' in reason\n"
     "               or 'context_missing_field' in reason))"),
    # 3. N5 / N6
    ("    ok = proc.returncode != 0 and 'hmac_mismatch' in proc.stderr and art['MODEL_LOAD'] == 0",
     "    reason = guard_refusal_reason(run_dir)\n"
     "    det['guard_reason'] = reason\n"
     "    ok = proc.returncode != 0 and 'hmac_mismatch' in reason and art['MODEL_LOAD'] == 0"),
    # 4. N7
    ("    ok = proc.returncode != 0 and 'consumption_run_id_mismatch' in proc.stderr \\\n"
     "        and art['MODEL_LOAD'] == 0",
     "    reason = guard_refusal_reason(run_dir)\n"
     "    det['guard_reason'] = reason\n"
     "    ok = proc.returncode != 0 and 'consumption_run_id_mismatch' in reason \\\n"
     "        and art['MODEL_LOAD'] == 0"),
    # 5. N8
    ("    ok = proc.returncode != 0 and 'credential_digest_changed_after_consumption' in proc.stderr \\\n"
     "        and art['MODEL_LOAD'] == 0",
     "    reason = guard_refusal_reason(run_dir)\n"
     "    det['guard_reason'] = reason\n"
     "    ok = proc.returncode != 0 and 'credential_digest_changed_after_consumption' in reason \\\n"
     "        and art['MODEL_LOAD'] == 0"),
    # 6. the context-record glob also matched EXECUTION_CONTEXT_EVIDENCE_*.json
    ("run_dir.glob('EXECUTION_CONTEXT_*.json')", "run_dir.glob('EXECUTION_CONTEXT_[0-9a-f]*.json')"),
]
for old, new in reps:
    n = s.count(old)
    assert n >= 1, ('pattern not found', old[:80])
    s = s.replace(old, new)
    print('replaced %d occurrence(s) of %r' % (n, old[:60]))

open(p, 'w', encoding='utf-8', newline='\n').write(s)
print('OK')
