# Engineering-release hand-back (2026-10-14, updated)

Both lines were taken to an engineering release. No fresh utility day, no training, no final seed. Nothing here is a utility claim.

## 1. Pickup: registration-equals-action identity

| item | value |
|---|---|
| directory | `research/review_3080_20261010/pickup/a1_deployment_rule_identity_root_v3` |
| registered rule | `clock_shadow_price_budget_reserving_admission` |
| producer / audit | 57/0 and 51/0 |
| decisions re-derived | 1000 with 0 mismatches |
| per-day utilities | [1793.888119917363, 1845.8872958533466, 1832.3859690818936, 1799.6103530302644, 1898.560622766614] |
| reproduces the v4 panel rows | yes, all five days identical |

Gate: **PASS** (identity on the sealed log).

## 1b. Pickup: deployability in a current-source online pipeline

| item | value |
|---|---|
| directory | `research/review_3080_20261010/pickup/a1_current_pipeline_deployment_root_v2` |
| prior attempt | `research/review_3080_20261010/pickup/a1_current_pipeline_deployment_root_v1` (11 failures, two mis-scoped harness checks) |
| producer / audit | 77 checks, 0 failed and 54 checks, 0 failed |
| rule rows == v4 candidate | True |
| control rows == v4 control | True |
| paired deltas | [0.0, 102.92300423234701, 24.755533065646887, 0.0, 0.0] (mean 25.53570745959878) |
| decision accounting | {"total": 1000, "admit": 890, "reject_price": 110, "reject_budget": 0, "base_accepted": 553} |
| timeouts | {"rule": [0, 0, 0, 0, 0], "control": [0, 0, 0, 0, 0]} |

Gate: **PASS** (deployable with a contemporaneous control; the scenario_saa drift did not change either arm on these five days).

## 2. Pickup: independent strong-dynamic control and learning attribution

Gate: **NOT PASSED** (not attempted). Requires training runs and multi-round mask-reconstruct-repair under an equal-time request.

## 3. Delivery: actual loaded closure in a complete-model request

| item | value |
|---|---|
| directory | `research/review_3080_20261010/delivery/actual_loaded_closure_request_root_v1` |
| local three-point protocol | 7 pass, 0 fail |
| complete-model request | **not run** |
| server RAM available | 179 GiB (gate 8.0 GiB) |
| GPUs | both RTX 5090 at 95% |

Gate: **NOT PASSED**. Unpassed: the 3080 complete-model loaded closure; the real model native lazy-import surface; the 3080 source identity against the registered shard set; the full 8 s + 2 s request

## 4. Science

Not started: no fresh-day paired stage, no delivery family 3, final seeds 20260927/28 untouched, stable attributable online gain still not established. The +25.5357 five-day paired difference remains a non-learning exploratory lead on five consumed days.

## Source hashes

See `SOURCE_HASHES.json`; the preserved rule-identity result and the drift evidence are in `PRESERVATION_RECORD.json`.
