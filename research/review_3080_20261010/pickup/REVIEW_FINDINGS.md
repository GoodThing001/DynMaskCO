# Pickup 3080 handover review and causal fleet engineering correction

Written after the engineering diagnostics below. Old user source, protocols, results, I3/D2/risk evidence and final seeds 20260927/28 remain unchanged. This folder is a new diagnostic/tool deliverable, not a replacement formal result.

## Findings supported by actual source

1. `research/pickup_20round_audit/pickup_arm_iterative.py` lines 134–159 run the private full-day continuation, compare realized full-day utility, and feed the winner's sequence into the next reconstruction round. Causal simulation of each branch does not make this selector causal. Private future utility must only enter the evaluator after the actor has frozen its choice. The 7-state zero effect cannot establish that causal iterative MaskCO has no reward region.
2. Lines 103–115 filter clock below six hours and return from the whole hook at the first captured vehicle. The published `size_hist` is a selected late/partial-vehicle observation. It is not a full-day/all-vehicle census.
3. Lines 134 and 159–165 retain the incumbent as best. Best-kept improvement is therefore nonnegative by construction. A negative-best condition cannot trigger a both-arms-worse stopping rule. Report unguarded candidate deltas separately from deployed incumbent-preserving policies.
4. Lines 161–166 drop event/vehicle IDs; pairing at lines 195–199 uses `(instance,clock)`. Multiple vehicle decisions at the same clock can collide. The new key is `(day,event_id,vehicle_id)`; deployment whole-day pairing uses unique day identity.
5. Selected wide states and local counts such as 74/200 favorable states describe inspected state comparisons. They do not estimate representative day utility, a fresh main-gate pass or stability. Width sampling must either be fixed prospectively or treated as engineering coverage.
6. The old StageA uses `ColdChainContractV2()` at line 70, with default distance/time/speed units 1/1/1, while its simulator uses speed 30/20=1.5. Reproducing this operating point diagnoses the old source; it does not waive the mismatch. New actor certificates reject it. New data must use the same `make_c0_contract_v2` units and simulator speed in every arm.
7. Parent source audit reports `strong_rolling` constructed without explicit `hist/k`, whereas own base receives history. A weak mismatched comparator is inadequate attribution. All arms must receive the exact same approved history pool, sampler distribution, K and budget contract; the old one-day history and original I3 200-day pool are distinct identities.

## Executed registered diagnostic

`full_census.py` registered `full_census_diagnostic_v1/DIAGNOSTIC_REGISTRATION.json` before reconstructing the already observed four StageA days (seed 2026120201; one history day seed 2026120202). This is honest diagnostic registration, not a fresh gain experiment. Every event including clock zero and every vehicle are retained: **1,132 events × 15 = 16,980 distinct vehicle records**. Mutable suffix maximum is **9**, accepted-owned noncommitted suffix maximum **10**, jointly owned fleet maximum **11**. At original replanning triggers the respective maxima are 8 and 10. The old maximum-six narrative is unsupported.

These are observed maxima of this finite cohort. Zero observed joint width ≥12 does not establish distribution support, a global upper bound, or a reward threshold. All four terminal raw traces independently reproduce their reported evaluation. The original bare-contract speed mismatch is retained and flagged, so this reproduction is not formally eligible for corrected-physics gain inference.

Registration SHA256: `6a24c6f34485c4866964cca24ef13636ebd97492fdeb22683d509f51a2ce9777`.
Dataset SHA256: `72f01304fe204e958827aeef226d2ad1922f4c2be3418c57c8a60e3757a5f737`.
History dataset SHA256: `475b29339e7502a4a5e51de81ddef9344c6da86e71fa317aced343d0a33d291e`.
`result.json` retains all actual source hashes at start/end, full histograms, per-day evaluation, coverage and replay status. `full_day_all_fleet_trace.jsonl` retains all typed raw physical traces; `private_first_capture.json` is evaluator-only and must not be actor input.

## Runnable causal prototype

`public_fleet.py` exports a strict observed packet from real `StrictOnlineEnv` state. It includes revealed-order fields, current clock/position, actual thermal/cargo manifest and cumulative energy of every vehicle, accepted/rejected status and owned routes; it projects hidden future attributes away. Original public IDs remain stable; sparse visible-ID arrays do not assert a distributional privacy guarantee about the numbering system.

The actor enumerates a declared family: at most five known unserved noncommitted orders across the first nonempty open owner and earliest other open vehicle. Nonpool orders keep ownership and relative order; committed next destination/owner is frozen; pool orders are appended in all permutations/partitions. Own incumbent is an explicit candidate. This is an exact search over that family, **not** a global fleet optimum or admission optimizer.

Each branch uses actual C0 continuation of the known state and immediately closes after known commitments. The certificate preserves accepted services, freezes committed legs, accounts for all vehicle energy, and checks TW/capacity/return/quality/budget with the actual A1 evaluator. The score includes revenue, fuel and rejection loss. It does not use realized hidden future utility. Immediate-close score omits future opportunity cost; a future honest SAA scorer must use a common future law and matched samples, never the private day's realization.

`bounded_select` runs the actor in a subprocess, includes startup/scoring/source verification in its ten-second submission deadline, kills unfinished work and retains incumbent. `install_in_live` repeats actual certification and checks the same deadline immediately before committing. Cleanup wall latency may exceed the submission deadline; no late candidate is installed.

`shared_budget_replanner` and `intervention_hook` are explicit integration sites. Original SAA allows per-order budgets within one reveal event; this wrapper attaches the suffix to the last admission's remaining budget, and route-only events get their original planning start. This is not an entire-batch ten-second guarantee. Parent v3 must choose and enforce one scope identically in every arm; event-total scope needs a shared event start throughout every order's sampling/admission and certification, followed by matched full-day QA. Complete K10 and all-in production timing are still deployment qualification gaps.

## Engineering QA and interpretation

`public_certificate_qa_v2` reuses one previously archived real state (seed 202610101801; clock 1.7700042724609375; committed next 43; mutable known pool 20/17). It generates no fresh research days. Six two-vehicle candidates receive actual hard certificates. Best remains own incumbent: known utility **75.48180642724037**, delta **0**. All-fleet energy is **28.98216639860907 kWh**. Mutating all private future locations, demands, windows and cargo types leaves packet and selection unchanged. Bare-contract speed mismatch is rejected.

Eight CPU regressions pass: full census keys/coverage, independent replay, future mutation, private-data/unit rejection, committed/duplicate-ID protection, actual cross-vehicle certificates/own incumbent, subprocess timeout fallback, strict boolean/finite clocks. V1 QA is preserved; v2 adds stricter clock/order qualification before the tests. Neither QA establishes full-day gain, MaskCO attribution or stable positive online utility. The actor is a nonlearning causal control. A genuine core arm must later integrate full MaskCO encoder+decoder masked reconstruction over the same public action domain, with randomized/discarded-guide and nonlearning controls, exact source/weight identity and all-in deployment accounting.

## Literature basis and limits

Ulmer, Goodson, Mattfeld and Thomas, [On Modeling Stochastic Dynamic Vehicle Routing Problems](https://justin-goodson.com/papers/RouteBasedMDPs.pdf), Sections 2 and 5, model decision states with current vehicle/order information and route plans, distinguish route revision from implemented next movement, and account for changing information over time. This supports explicit public-state and committed-action contracts; it does not establish the current immediate-close score as a sufficient value function or prove any numerical threshold.

Zhang et al., [Solving Large-Scale Dynamic Vehicle Routing Problems with Stochastic Requests](https://arxiv.org/abs/2202.12983), motivates stochastic future-aware routing. Here that motivates a subsequent common-law lookahead control. It does not justify feeding the private realized test future to an online actor. All temperature/quality coefficients in this deliverable are simulation parameters, not physical calibration claims.

## Reproduction

Run in this isolated worktree, using the authoritative extension only as a read-only dependency:

```powershell
python -B research/review_3080_20261010/pickup/test_review_pickup.py --scripts D:/PyCharm_/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/scripts
```

For an independent CPU certificate rerun use a new output directory and the preserved archive; do not overwrite V1/V2:

```powershell
python -B research/review_3080_20261010/pickup/run_certificate_qa.py --scripts D:/PyCharm_/MASKCO-Main/C-VRP_Cold-chainVehicleRoutingProblem/scripts --archive research/pickup_midcourse_probe/engineering_qa_v5 --out research/review_3080_20261010/pickup/public_certificate_qa_server_v1
```

Only `PUBLIC_FLEET_PROTOCOL.json` is prospective. It is written after these engineering results and is not a backdated registration of them. New formal fresh days need the parent's complete v3 registration before generation.
