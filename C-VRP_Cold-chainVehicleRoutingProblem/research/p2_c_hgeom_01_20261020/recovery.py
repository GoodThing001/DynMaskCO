"""Exhaust frozen mask/remove/insert relation; no model or source-day loader."""
from __future__ import annotations
import time


def structure(plans):
    return tuple((int(vid), int(p.anchor_node), float(p.anchor_time).hex(),
                  float(p.anchor_load).hex(), tuple(map(int, p.suffix)))
                 for vid, p in sorted(plans.items()))


def anchors(plans):
    return tuple(x[:4] for x in structure(plans))


def enumerate_closure(shared, view, base, mask, allowed, targets, *, check,
                      cap=5000, deadline=None, hash_fn=None, persist=None):
    """Defaults are fixed production limits; smaller caps are synthetic faults only.

    Preserve a pending terminal exactly like any other pending state. Consequently
    hitting CAP cannot be called exhausted merely because a target was found.
    No hash-only frontier or terminal deduplication is performed.
    """
    start = time.monotonic()
    env = shared.ref.public_env(view)
    initial = shared.remove_mask(base, tuple(mask))
    stack = [(initial, tuple(sorted(mask)), ())]
    records, matches = {}, {h: [] for h in targets}
    expanded = generated = 0
    failure = []
    while stack:
        if deadline is not None and time.monotonic() >= deadline:
            failure.append('UNIT_TIMEOUT')
            break
        plans, rem, path = stack.pop()
        if rem:
            expanded += 1
            acts = sorted(shared.legal_actions_shared(view, env, plans, rem, allowed),
                          key=lambda x: (int(x[0]), x[1].action_id()))
            for c, action, _ in reversed(acts):
                stack.append((shared.apply(plans, action, allowed),
                              tuple(x for x in rem if x != c), path + (action.action_id(),)))
            continue
        generated += 1
        s = structure(plans)
        if s not in records:
            ok, reason = check(plans)
            ok = ok and anchors(plans) == anchors(base)
            h = (hash_fn or shared.signature)(plans)
            records[s] = dict(structure=s, plan_hash=h, legal=ok, check_reason=reason,
                              paths=[path], arrivals=1, plans=plans)
            if h in matches:
                matches[h].append(s)
            if persist:
                persist(records[s])
        else:
            records[s]['paths'].append(path)
            records[s]['arrivals'] += 1
            if persist:
                persist(records[s])
        if len(records) >= cap and stack:
            failure.append('ENUMERATION_CAPPED')
            break
    exhausted = not stack and not failure
    for h, structures in matches.items():
        if not structures:
            failure.append('RECOVERY_NOT_FOUND' if exhausted
                           else 'TARGET_NOT_OBSERVED_IN_PARTIAL_CLOSURE')
        elif len(structures) > 1:
            failure.append('RECOVERY_NOT_UNIQUE')
        elif not records[structures[0]]['legal']:
            failure.append('RECOVERY_ILLEGAL')
    unique = exhausted and all(len(v) == 1 for v in matches.values())
    if not unique:
        failure.append('UNIQUENESS_NOT_PROVED')
    if any(not r['legal'] for r in records.values()):
        failure.append('RECOVERY_ILLEGAL')
    codes = sorted(set(failure))
    status = ('TIMEOUT' if 'UNIT_TIMEOUT' in codes else 'CAPPED' if 'ENUMERATION_CAPPED' in codes
              else 'INVALID' if codes else 'EXHAUSTED')
    return dict(targets_found=sum(bool(v) for v in matches.values()), n_targets=len(targets),
                uniqueness_proved=unique, closure_exhausted=exhausted, closure_status=status,
                legal=all(r['legal'] for r in records.values()), failure_codes=codes,
                distance_relation='NOT_EVALUATED', matches=matches, records=list(records.values()),
                enumeration_cost=dict(wall_s=time.monotonic()-start, expanded_states=expanded,
                                      generated_complete=generated, distinct_complete=len(records),
                                      legal_complete=sum(r['legal'] for r in records.values())))
