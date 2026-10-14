# D3-G0 Evidence B3-AT2B-2a - object-level write exclusion (ISOLATED; ACTIVE / NOT_VERIFIED)

- Registration class: **LATE_PREREGISTRATION_BEFORE_FIRST_EXECUTION** (declared in
  `B3AT2B2A_COUNTEREXAMPLE_SPEC.json`; the draft modules existed before the spec, so this batch's
  standing is NOT equal to a strict implement-after-preregistration batch, and it is never claimed
  to be).
- Evidence: `B3AT2B2A_EXCLUSION_MATRIX.json`; raw `raw/B3AT2B2A_run1..run10_*.txt` (every failure of
  the first eight executions is preserved verbatim and was not deleted after the green run);
  AT2B-1 regression `raw/B3AT2B1_regression_after_2a_*.txt`.
- `strict_online_env.py` is UNMODIFIED (`FROZEN=OK`, sha256 fd17fe77...); `MASKCO_code/` untouched.

## Tenth execution (green) verdicts

FROZEN OK; B2A-DEFAULTOFF IDENTICAL_HISTORY_BEHAVIOUR; B2A-ADVANCE LEGITIMATE_ADVANCE_NOT_BLOCKED;
B2A-REVEAL REVEAL_ALLOWED_WITH_COMMITMENT_FROZEN; B2A-ATTR and B2A-NESTED
REFUSED_AND_STATE_UNCHANGED; B2A-STALE STALE_REFERENCE_INEFFECTIVE; B2A-COMMIT REFUSED;
B2A-POS COMMIT_ALLOWED_INSIDE_SCOPE; B2A-CAND REFUSED_SUFFIX_GROWTH_REQUIRES_DECISION_SCOPE;
B2A-COMPATPOLICY REFUSED_AND_STATE_UNCHANGED (late addition);
B2A-DELETE REFUSED_FOR_UNCOMMITTED_DELETION_AND_ALLOWED_FOR_THE_COMMITTED_HEAD (late addition);
PREREG OK; AUDIT OK.  AT2B-1 regresses green (13 rows unchanged).

## The two security questions the ruling left open, now answered by targeted rows

1. `COMPAT_INSTALL` under a certified policy: the user's source review found that `_authorised()`
   checked `policy` through a parameter that callers never passed, so the rule EXISTED in source but
   was NOT IN FORCE.  Fixed by carrying the policy with the thread-local scope: the compatibility
   scope is now refused when it is OPENED under a certified policy, before any write.  B2A-COMPATPOLICY
   records `expected_refusal_reason = compat_install_not_allowed_under_certified_policy` with the
   authoritative state unchanged and `actual_authoritative_write = false`.
2. Deleting an un-executed, non-revealed customer in the execution scope: the execution scope now
   permits only a ONE-element head consumption whose consumed head equals the vehicle's
   `committed_next`; a middle deletion, a bare head drop without the commitment, and any growth are
   refused (`execution_scope_may_only_consume_the_committed_head`).  B2A-DELETE records both
   directions: refusal for the uncommitted cases and success for the committed head.  DISCLOSED
   RESIDUAL: the binding is to the commitment identity, not yet to a service-completion event object.

## Scope split (all five scopes frozen in the spec)

SCOPE_EXECUTION (committed head only), SCOPE_REVEAL (revocable tail + replan trigger, with the
REVEAL_INVALIDATION V1-V5 contract as the second gate), SCOPE_DECISION_CANDIDATE (detached copies
only; writing the authoritative state is refused), SCOPE_COMPAT_INSTALL (LABELLED isolation
compatibility mode, uncertified, now refused under the certified policy), SCOPE_DECISION_COMMIT
(reserved for a batch in which D8 no longer blocks certification; NOT exercised here).

## Honest residuals and non-claims

- `served_mask` in-place numpy writes are still not intercepted.
- Threat model excludes arbitrary reflection (`object.__setattr__`), ctypes and malicious memory
  modification.
- The compatibility-install mode remains a labelled, uncertified mechanism; its use in this batch
  never counts as a certified install.
- P2 / G-L / G-C NOT_ESTABLISHED; D8 CAPABILITY_GAP_BLOCKING; D3 G0 not passed; G1/G2 not started;
  no production hookup and no real write-back; AT2B-2b not started; algorithmic_gain_claim none;
  I3 and the old D2 not_qualified untouched; stable_online_gain = False.
