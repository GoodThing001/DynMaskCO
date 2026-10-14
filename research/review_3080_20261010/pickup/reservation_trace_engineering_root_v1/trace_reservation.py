"""Record an entire budget pre-filter decision without changing its semantics."""
import hashlib
import json


def decision_digest(record):
    return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode('utf-8')).hexdigest()


def traced_decision(policy, env, instance_idx, clock, order, sequence):
    before_decisions = int(policy.reserve_decisions)
    before_rejects = int(policy.reserve_rejects)
    token = {}
    allowed = bool(policy._reserving_allows(env, instance_idx, clock, order, token))
    cls = int(env.dataset['temp_class'][instance_idx, order])
    demand = float(env.demands[instance_idx, order])
    row = {
        'sequence': int(sequence), 'instance_idx': int(instance_idx), 'clock': float(clock),
        'order': int(order), 'class_id': cls, 'demand': demand,
        'budget': float(policy.budget), 'accepted_before': sorted(int(x) for x in policy._accepted),
        'estimated_precool_kwh': float(token['est']), 'revenue': float(token['revenue']),
        'lambda': None if token.get('lambda') is None else float(token['lambda']),
        'clock_bucket': token.get('clock_bucket'), 'reason': str(token['reason']),
        'allowed': allowed,
        'reserve_decisions_before': before_decisions,
        'reserve_decisions_after': int(policy.reserve_decisions),
        'reserve_rejects_before': before_rejects,
        'reserve_rejects_after': int(policy.reserve_rejects),
    }
    row['sha256'] = decision_digest(row)
    return allowed, row
