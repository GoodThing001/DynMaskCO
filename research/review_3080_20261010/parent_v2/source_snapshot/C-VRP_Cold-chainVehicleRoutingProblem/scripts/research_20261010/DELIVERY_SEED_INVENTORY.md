# DELIVERY SEED INVENTORY — delivery line, both research cycles

Purpose: an accurate, conservative inventory of every concretely-used integer seed and every
instance-id range observed in the **delivery** panels of the two research cycles, used as the
disjointness basis for freezing the seed set of protocol **D-TRAINTASK-v1**.

Written by: registration + bookkeeping task, 2026-10-10T06:57:19Z.
No training and no experiment was run to produce this file.

Extension root: `C-VRP_Cold-chainVehicleRoutingProblem/`
Cycle roots searched:

- `results/research_20261010`
- `results/research_20261010_continuation_v2`

Scripts searched:

- `scripts/research_20261010/delivery_*.py`
- `scripts/research_20261010/delivery_*protocol*.json`

---

## 1. Forbidden seeds (never use)

| Seed | Status | Evidence |
|---|---|---|
| `20260927` | FORBIDDEN / protected | `scripts/research_20261010/delivery_traintask_protocol_DRAFT.json` → `protected.final_seeds[0]`; `results/research_20261010/PROGRAM.json` → `validation.old_pickup_final_seeds_forbidden[0]`; `results/research_20261010_continuation_v2/PROGRAM.json` → `validation.old_pickup_final_seeds_forbidden[0]` and `execution.old_pickup_final_seeds_forbidden[0]` |
| `20260928` | FORBIDDEN / protected | same four locations, index `[1]` |

Neither forbidden seed is used by any split registered in the frozen `delivery_traintask_protocol.json`.

---

## 2. Seed table (seed value → script/protocol → role → evidence)

Roles use the vocabulary of the task: **train / dev / pilot / confirm / engineering / control / test / cross-line**.
`mask` = mask-schedule RNG, `anytime` = deadline-driven reconstruction RNG, `arm_order` = arm-order
randomisation RNG, `bootstrap` = paired-bootstrap CI RNG, `lns` = LNS control RNG,
`random_model` = random same-architecture guide RNG.

### 2.1 Panel dataset seeds (which data a panel generated / consumed)

| Seed | Script / protocol | Role | Evidence (file : line or JSON pointer) |
|---|---|---|---|
| `2026101002` | `delivery_equal_schedule.py` (`D-EQ8-v1`) | dev dataset, instances 0–31 | `scripts/research_20261010/delivery_equal_schedule.py:209`; `delivery_eq8_qa_protocol.json:3`; `results/research_20261010/delivery_raw_v1/PRE_DATA_PROTOCOL.json` → `/dataset_seed`; `results/research_20261010_continuation_v2/delivery_raw_v1/PILOT_PROTOCOL.json` → `/dataset_seed` |
| `2026101002` | `delivery_anytime.py` (`D-ANYTIME-v1`) `dataset_seed_dev` | dev dataset | `delivery_anytime.py:65`; `delivery_anytime_protocol.json:5`; `results/research_20261010/delivery_anytime_v1/PROTOCOL.json` → `/dataset_seed_dev` |
| `2026101002` | `delivery_strong_control.py` (`D-CTRL-v1`) `dataset_seed_dev` | dev dataset | `delivery_strong_control.py:59`; `delivery_control_protocol.json:6`; `results/research_20261010/delivery_control_v1/PROTOCOL.json` |
| `2026101002` | `delivery_guide_dose.py` (`D-DOSE-v1`) `dataset_seed_dev` | dev dataset | `delivery_guide_dose.py:69`; `delivery_dose_protocol.json:6`; `results/research_20261010/delivery_dose_v1/PROTOCOL.json` |
| `2026101002` | `delivery_mask_heuristics.py` (`D-MASKHEUR-v1`) `dataset_seed_dev` | dev dataset | `delivery_mask_heuristics.py:60`; `delivery_mask_heuristics_protocol.json:6`; `results/research_20261010/delivery_maskheur_v1/PROTOCOL.json` |
| `2026101002` | `delivery_destroy_operators.py` (`D-DESTROY2-v1`) `dataset_seed_dev` | dev dataset | `delivery_destroy_operators.py:69`; `delivery_destroy2_protocol.json:6`; `results/research_20261010/delivery_destroy2_v1/PROTOCOL.json` |
| `2026101002` | `delivery_mask_fraction.py` (`D-MASKFRAC-v1`) `dataset_seed_dev` | dev dataset | `delivery_mask_fraction.py:53`; `delivery_maskfrac_protocol.json:7`; `results/research_20261010/delivery_maskfrac_v1/PROTOCOL.json` |
| `2026101002` | `delivery_cost_decomposition.py` | dev dataset | `delivery_cost_decomposition.py:59`; `results/research_20261010/delivery_decomp_v1/PROTOCOL.json` |
| `2026101002` | `delivery_independent_recheck.py` | recheck dataset | `delivery_independent_recheck.py:58`; `results/research_20261010/delivery_recheck_v1/PROTOCOL.json` |
| `2026101002` | `delivery_gls_control.py` (GLS control) | control dataset | `delivery_gls_control.py:54`; `results/research_20261010/delivery_gls_v1/PROTOCOL.json` |
| `2026101002` | `test_delivery_destroy_eq8.py` | test fixture only | `test_delivery_destroy_eq8.py:28` |
| `2026101002` | `delivery_guide_misalignment.py` (`D-GUIDE-v1`) — **inherited**, not re-drawn | dev dataset (recorded server decision states from `D-EQ8-v1`) | `delivery_guide_misalignment.py:67` (`parent_panel: D-EQ8-v1`, `state_source: recorded_server_decision_states_taken_as_given`); `results/research_20261010/delivery_guide_misalignment_v1/PROTOCOL.json` carries no dataset seed of its own |
| `2026101013` | `D-ANYTIME-v1`, `D-CTRL-v1`, `D-DOSE-v1`, `D-MASKHEUR-v1`, `D-DESTROY2-v1`, `D-MASKFRAC-v1` `dataset_seed_confirmation` | confirm dataset, instances 1000–1031 | `delivery_anytime.py:66`; `delivery_strong_control.py:60`; `delivery_guide_dose.py:70`; `delivery_mask_heuristics.py:61`; `delivery_destroy_operators.py:70`; `delivery_mask_fraction.py:54`; `delivery_anytime_protocol.json:6`; `delivery_control_protocol.json:7`; `delivery_dose_protocol.json:7`; `delivery_mask_heuristics_protocol.json:7`; `delivery_destroy2_protocol.json:7`; `delivery_maskfrac_protocol.json:8` |
| `2026101141` | `delivery_any50_confirm.py` (`D-ANY50-CONFIRM-v1`) `dataset_seed` | confirm dataset, instances 2000–2031 | `delivery_any50_confirm.py:54,76`; `delivery_any50_confirm_protocol.json:8`; `results/research_20261010/delivery_any50_confirm_v1/PROTOCOL.json` |
| `2026101161` | `delivery_guide_destroy.py` (`D-DESTROY-v1`) `dataset_seed_dev` | dev dataset, instances 3000–3031 | `delivery_guide_destroy.py:73`; `delivery_guide_destroy_protocol.json:7`; `results/research_20261010/delivery_destroy_v1/PROTOCOL.json` |
| `2026101171` | `delivery_guide_destroy.py` (`D-DESTROY-v1`) `dataset_seed_confirmation` | confirm dataset, instances 4000–4031 | `delivery_guide_destroy.py:74`; `delivery_guide_destroy_protocol.json:8`; `results/research_20261010/delivery_destroy_v1/PROTOCOL.json` |
| `202610101701` | `delivery_destroy_eq8.py` (`D-DESTROY-EQ8-v1`) `dataset_seed` | **engineering / pilot**, instances 17000–17031 | `delivery_destroy_eq8.py:24`; `delivery_destroy_eq8_pilot_protocol_DRAFT.json:3`; `delivery_destroy_eq8_qa_protocol.json:3`; `results/research_20261010_continuation_v2/delivery_raw_v1/PILOT_PROTOCOL.json`, `.../delivery_deployment_v1/PRE_DATA_PILOT_TEMPLATE.json` |
| `202610101701` + `202610101702` | `delivery_traintask_data_export.py` (`--seed`, `--mask-seed` defaults) | **engineering only** — the D-TRAINTASK-v1 export feasibility run, one instance (17000), 41 samples | `delivery_traintask_data_export.py:140-141`; `results/research_20261010_continuation_v2/delivery_traintask_export_v1/result.json` → `/seed`, `/instance`, `/mask_seed`, `/isa_protocol_train_split: false` |
| `2026101221` | `delivery_dynamic_suffix.py` (`PROTOCOL['seed']`) | synthetic self-generated instance family (8 instances, seeds `PROTOCOL['seed'] + i`) | `delivery_dynamic_suffix.py:68,211`; `results/research_20261010/delivery_dynamic_suffix_v1/PROTOCOL.json` → `/seed` |

### 2.2 RNG seeds by role

| Seed | Script / protocol | Role | Evidence |
|---|---|---|---|
| `2026101041` | `D-EQ8-v1` `mask_seed`; `D-ANYTIME-v1`, `D-CTRL-v1`, `D-DOSE-v1` `mask_seed_dev` | mask RNG (dev) | `delivery_equal_schedule.py:210`; `delivery_eq8_qa_protocol.json:4`; `delivery_anytime.py:69`; `delivery_anytime_protocol.json:9`; `delivery_strong_control.py:63`; `delivery_control_protocol.json:10`; `delivery_guide_dose.py:73`; `delivery_dose_protocol.json:10`; `results/research_20261010/delivery_deployment_v1/PRE_DATA_PROTOCOL.json` → `/mask_seed`; `results/research_20261010/delivery_raw_v1/pilot/PROTOCOL.json` → `/mask_seed` |
| `2026101052` | `D-ANYTIME-v1`, `D-CTRL-v1`, `D-DOSE-v1` `mask_seed_confirmation` | mask RNG (confirm) | `delivery_anytime.py:70`; `delivery_anytime_protocol.json:10`; `delivery_strong_control.py:64`; `delivery_control_protocol.json:11`; `delivery_guide_dose.py:74`; `delivery_dose_protocol.json:11` |
| `2026101099` | `D-EQ8-v1` `arm_order_seed`; `D-DOSE-v1` `arm_order_seed_dev` | arm-order RNG | `delivery_equal_schedule.py:210`; `delivery_eq8_qa_protocol.json:5`; `delivery_guide_dose.py:75`; `delivery_dose_protocol.json:12`; `results/research_20261010/delivery_deployment_v1/PRE_DATA_PROTOCOL.json` → `/arm_order_seed` |
| `2026101063` | `D-DOSE-v1` `arm_order_seed_confirmation` | arm-order RNG (confirm) | `delivery_guide_dose.py:76`; `delivery_dose_protocol.json:13` |
| `2026101077` | `D-EQ8-v1` `bootstrap_seed` | bootstrap CI RNG | `delivery_equal_schedule.py:211`; `delivery_eq8_qa_protocol.json:6`; `results/research_20261010/delivery_raw_v1/pilot/PROTOCOL.json` → `/bootstrap_seed` |
| `2026101088` | `D-GUIDE-v1` `bootstrap_seed` | bootstrap CI RNG | `delivery_guide_misalignment.py:78`; `results/research_20261010/delivery_guide_misalignment_v1/PROTOCOL.json` → `/bootstrap_seed` |
| `2026101111` | `D-DOSE-v1` `bootstrap_seed` | bootstrap CI RNG | `delivery_guide_dose.py:83`; `delivery_dose_protocol.json:20`; `results/research_20261010/delivery_dose_v1/PROTOCOL.json` |
| `2026101122` | `D-CTRL-v1` `bootstrap_seed` | bootstrap CI RNG | `delivery_strong_control.py:69`; `delivery_control_protocol.json:16`; `results/research_20261010/delivery_control_v1/PROTOCOL.json` |
| `2026101131` | `D-ANYTIME-v1` `anytime_seed_dev`; also `D-MASKHEUR-v1`, `D-DESTROY2-v1`, `D-MASKFRAC-v1` `anytime_seed_dev`; `delivery_cost_decomposition.py`, `delivery_independent_recheck.py`, `delivery_gls_control.py` `anytime_seed` | anytime/arm RNG (dev) | `delivery_anytime.py:71`; `delivery_anytime_protocol.json:11`; `delivery_mask_heuristics.py:64`; `delivery_mask_heuristics_protocol.json:10`; `delivery_destroy_operators.py:73`; `delivery_destroy2_protocol.json:10`; `delivery_mask_fraction.py:57`; `delivery_maskfrac_protocol.json:11`; `delivery_cost_decomposition.py:62`; `delivery_independent_recheck.py:61`; `delivery_gls_control.py:57` |
| `2026101132` | `D-ANYTIME-v1` `anytime_seed_confirmation` | anytime RNG (confirm) | `delivery_anytime.py:72`; `delivery_anytime_protocol.json:12` |
| `2026101133` | `D-ANYTIME-v1` `bootstrap_seed` (+ post-hoc re-analysis) | bootstrap CI RNG | `delivery_anytime.py:78`; `delivery_anytime_protocol.json:18`; `delivery_anytime_posthoc.py:27`; `results/research_20261010/delivery_anytime_v1/posthoc_any50_comparison.json` → `/bootstrap_seed` |
| `2026101143` | `D-ANY50-CONFIRM-v1` `anytime_seed_primary` | anytime RNG (primary arm ANY50) | `delivery_any50_confirm.py:56,77`; `delivery_any50_confirm_protocol.json:10` |
| `2026101144` | `D-ANY50-CONFIRM-v1` `anytime_seed_reference` | anytime RNG (reference arm ANY30) | `delivery_any50_confirm.py:57,77`; `delivery_any50_confirm_protocol.json:11` |
| `2026101151` | `D-ANY50-CONFIRM-v1` `bootstrap_seed` | bootstrap CI RNG | `delivery_any50_confirm.py:62,79`; `delivery_any50_confirm_protocol.json:16`; `results/research_20261010/delivery_any50_confirm_v1/PROTOCOL.json` |
| `2026101163` | `D-DESTROY-v1` `anytime_seed_dev` | anytime RNG (dev) | `delivery_guide_destroy.py:77`; `delivery_guide_destroy_protocol.json:11` |
| `2026101173` | `D-DESTROY-v1` `anytime_seed_confirmation` | anytime RNG (confirm) | `delivery_guide_destroy.py:78`; `delivery_guide_destroy_protocol.json:12` |
| `2026101174` | `D-MASKHEUR-v1` `anytime_seed_confirmation` | anytime RNG (confirm) | `delivery_mask_heuristics.py:65`; `delivery_mask_heuristics_protocol.json:11` |
| `2026101175` | `D-DESTROY2-v1` `anytime_seed_confirmation` | anytime RNG (confirm) | `delivery_destroy_operators.py:74`; `delivery_destroy2_protocol.json:11` |
| `2026101176` | `D-MASKFRAC-v1` `anytime_seed_confirmation` | anytime RNG (confirm) | `delivery_mask_fraction.py:58`; `delivery_maskfrac_protocol.json:12` |
| `2026101181` | `D-DESTROY-v1` `bootstrap_seed` | bootstrap CI RNG | `delivery_guide_destroy.py:82`; `delivery_guide_destroy_protocol.json:16` |
| `2026101211` | `delivery_gls_control.py` `bootstrap_seed` (reused by `audit_delivery_20_rounds.py`) | bootstrap CI RNG | `delivery_gls_control.py:60`; `audit_delivery_20_rounds.py:72,73,77`; `results/research_20261010/delivery_gls_v1/PROTOCOL.json` |
| `2026101231` | `D-MASKHEUR-v1` `bootstrap_seed` | bootstrap CI RNG | `delivery_mask_heuristics.py:68`; `delivery_mask_heuristics_protocol.json:14` |
| `2026101241` | `D-DESTROY2-v1` `bootstrap_seed` (reused by audit) | bootstrap CI RNG | `delivery_destroy_operators.py:77`; `delivery_destroy2_protocol.json:14`; `audit_delivery_20_rounds.py:88`; `results/research_20261010/delivery_destroy2_v1/PROTOCOL.json` |
| `2026101251` | `delivery_cost_decomposition.py` `bootstrap_seed` (reused by audit) | bootstrap CI RNG | `delivery_cost_decomposition.py:65`; `audit_delivery_20_rounds.py:99`; `results/research_20261010/delivery_decomp_v1/PROTOCOL.json` |
| `2026101261` | `D-MASKFRAC-v1` `bootstrap_seed` | bootstrap CI RNG | `delivery_mask_fraction.py:60`; `delivery_maskfrac_protocol.json:14`; `results/research_20261010/delivery_maskfrac_v1/PROTOCOL.json` |
| `202610101702` | `D-DESTROY-EQ8-v1` `mask_seed` (+ engineering export `--mask-seed`) | **engineering / pilot** mask RNG | `delivery_destroy_eq8.py:25`; `delivery_destroy_eq8_pilot_protocol_DRAFT.json:4`; `delivery_destroy_eq8_qa_protocol.json:4`; `delivery_traintask_data_export.py:141`; `results/research_20261010_continuation_v2/delivery_traintask_export_v1/result.json` → `/mask_seed` |
| `202610101703` | `D-DESTROY-EQ8-v1` `bootstrap_seed` | engineering / pilot bootstrap RNG | `delivery_destroy_eq8.py:25`; `delivery_destroy_eq8_pilot_protocol_DRAFT.json:5`; `delivery_destroy_eq8_qa_protocol.json:5` |
| `202610101704` | `D-DESTROY-EQ8-v1` `arm_order_seed` | engineering / pilot arm-order RNG | `delivery_destroy_eq8.py:26`; `delivery_destroy_eq8_pilot_protocol_DRAFT.json:6`; `delivery_destroy_eq8_qa_protocol.json:6` |
| `2026101222` … `2026101228` | `delivery_dynamic_suffix.py` (`PROTOCOL['seed'] + i`) | synthetic instance seeds 2–8 | `delivery_dynamic_suffix.py:211`; `results/research_20261010/delivery_dynamic_suffix_v1/result.json` → `/rows[1..7]/seed` |
| `2026101191` | `pickup_tail_census.py` `bootstrap_seed` | **cross-line** (pickup, not delivery) — listed for namespace hygiene | `pickup_tail_census.py:70`; `results/research_20261010/pickup_tail_census_v1/PROTOCOL.json` |
| `2026101201` | `pickup_clairvoyant_gap.py` `bootstrap_seed` | **cross-line** (pickup, not delivery) | `pickup_clairvoyant_gap.py:74`; `results/research_20261010/pickup_clairvoyant_gap_v1/PROTOCOL.json` |

### 2.3 Small / structural seed components (not date-like)

These are legitimate seed components in the same RNG streams, so they are recorded rather than omitted.

| Seed | Script | Role | Evidence |
|---|---|---|---|
| `41` | `delivery_any50_confirm.py`, `delivery_anytime.py`, `delivery_strong_control.py`, `delivery_cost_decomposition.py`, `delivery_independent_recheck.py`, `delivery_gls_control.py` `lns_seed` | LNS control RNG | `delivery_any50_confirm.py:61`; `delivery_anytime.py:77`; `delivery_strong_control.py:68`; `delivery_cost_decomposition.py:63`; `delivery_independent_recheck.py:62`; `delivery_gls_control.py:58` |
| `42` | `random_model_seed` in `D-DOSE-v1`, `D-DESTROY-v1`, `D-DESTROY-EQ8-v1`, `D-EQ8-v1` | random same-architecture guide RNG | `delivery_guide_dose.py:82`; `delivery_guide_destroy.py:81`; `delivery_destroy_eq8.py:27,247`; `delivery_equal_schedule.py:212` |
| `9001` | `delivery_equal_schedule.py:310`, `delivery_guide_dose.py:143`, `delivery_destroy_eq8.py:284`, `test_delivery_destroy_eq8.py` | synthetic QA fixture mask/instance seed (deliberately independent of scientific seeds) | `delivery_equal_schedule.py:298,310`; `delivery_guide_dose.py:143`; `delivery_destroy_eq8.py:284` |
| `51` | `make_schedule(51, 8, mask_seed, instance_id)` in `delivery_anytime.py`, `delivery_strong_control.py`, `delivery_guide_dose.py`, `delivery_equal_schedule.py` | first component of the `SeedSequence([51, 8, mask_seed, iid])` mask schedule | `delivery_anytime.py:165`; `delivery_strong_control.py:101`; `delivery_guide_dose.py:159`; `delivery_equal_schedule.py:313` |
| `8` | `steps` in every registered delivery panel | number of destroy-repair opportunities | `delivery_anytime_protocol.json:15`; `delivery_dose_protocol.json` → `/steps`; `delivery_destroy_eq8_pilot_protocol_DRAFT.json` → `/steps` |
| `7`, `71` | `test_delivery_equal_schedule.py` | synthetic unit-test schedule literals | `test_delivery_equal_schedule.py:50,61,81,98,108,117,128` |

---

## 3. Instance-id ranges observed per panel

| Instance-id range | Panel(s) | Role | Evidence |
|---|---|---|---|
| `0 … 31` | `D-EQ8-v1` (dev), `D-DOSE-v1` (dev), `D-CTRL-v1` (dev), `D-ANYTIME-v1` (dev), `D-MASKHEUR-v1` (dev), `D-DESTROY2-v1` (dev), `D-MASKFRAC-v1` (dev), cost decomposition, recheck, GLS control, deployment, progress replay | dev / pilot | `delivery_anytime_protocol.json` → `/dev_instance_ids`; `delivery_control_protocol.json` → `/dev_instance_ids`; `delivery_eq8_qa_protocol.json` → `/instance_ids`; `results/research_20261010/delivery_decomp_v1/PROTOCOL.json`; `.../delivery_recheck_v1/PROTOCOL.json`; `.../delivery_gls_v1/PROTOCOL.json`; `.../delivery_deployment_v1/PRE_DATA_PROTOCOL.json` |
| `1000 … 1031` | `D-ANYTIME-v1`, `D-CTRL-v1`, `D-DOSE-v1`, `D-MASKHEUR-v1`, `D-DESTROY2-v1`, `D-MASKFRAC-v1` (confirmation) | confirm | `delivery_anytime_protocol.json` → `/confirmation_instance_ids`; `delivery_control_protocol.json` → `/confirmation_instance_ids`; `delivery_dose_protocol.json` → `/confirmation_instance_ids`; `delivery_mask_heuristics_protocol.json` → `/confirmation_instance_ids`; `delivery_destroy2_protocol.json` → `/confirmation_instance_ids`; `delivery_maskfrac_protocol.json` → `/confirmation_instance_ids` |
| `2000 … 2031` | `D-ANY50-CONFIRM-v1` | confirm | `delivery_any50_confirm_protocol.json` → `/instance_ids`; `results/research_20261010/delivery_any50_confirm_v1/PROTOCOL.json`; `results/research_20261010_continuation_v2/parent_pairing_audit/AUDIT.json` → `/ANY50_confirmation_vs_LNS/unique_instance_ids` |
| `3000 … 3031` | `D-DESTROY-v1` (dev) | dev | `delivery_guide_destroy_protocol.json` → `/dev_instance_ids`; `results/research_20261010/delivery_destroy_v1/PROTOCOL.json` |
| `4000 … 4031` | `D-DESTROY-v1` (confirmation) | confirm | `delivery_guide_destroy_protocol.json` → `/confirmation_instance_ids`; `results/research_20261010/delivery_destroy_v1/PROTOCOL.json` |
| `17000 … 17031` | `D-DESTROY-EQ8-v1` (pilot, QA, deployment, raw) | **engineering / pilot** | `delivery_destroy_eq8_pilot_protocol_DRAFT.json` → `/instance_ids`; `delivery_destroy_eq8_qa_protocol.json` → `/instance_ids`; `results/research_20261010_continuation_v2/delivery_deployment_v1/PRE_DATA_PILOT_TEMPLATE.json` → `/instance_ids`; `.../delivery_raw_v1/PILOT_PROTOCOL.json` → `/instance_ids` |
| `17000` only (1 instance) | D-TRAINTASK-v1 engineering export | **engineering only** | `results/research_20261010_continuation_v2/delivery_traintask_export_v1/result.json` → `/instance: 17000` |
| no instance ids (synthetic seeds `2026101221 … 2026101228`) | `delivery_dynamic_suffix.py` | synthetic self-contained | `results/research_20261010/delivery_dynamic_suffix_v1/result.json` → `/rows[*]/seed` |

---

## 4. Frozen seed set for D-TRAINTASK-v1 and its disjointness check

Frozen file: `scripts/research_20261010/delivery_traintask_protocol.json`

| Field | Value | Role |
|---|---|---|
| `train_seed` | `2026120101` | TRAIN split dataset seed — the only data the trained arm may see |
| `dev_seed` | `2026120102` | DEV split dataset seed (selection; disjoint from train) |
| `confirm_cohort_seeds` | `2026120111`, `2026120112`, `2026120113` | three fresh complete-day CONFIRMATION cohorts |
| `mask_seed_train` | `2026120103` | TRAIN mask-schedule RNG |
| `mask_seed_dev` | `2026120104` | DEV mask-schedule RNG |
| `mask_seed_confirm_cohorts` | `2026120121`, `2026120122`, `2026120123` | one mask RNG per confirmation cohort |

Disjointness: none of these ten integers appears in the inventory above; none appears anywhere in the
searched repository. Repo-wide literal search returned **no matches**:

```
grep -rE '\b2026120101\b|\b2026120102\b|\b2026120103\b|\b2026120104\b|\b2026120111\b|\b2026120112\b|\b2026120113\b|\b2026120121\b|\b2026120122\b|\b2026120123\b' D:\PyCharm_\MASKCO-Main
  -> No matches found
grep -rE '\b2026120[0-9]{3}\b|\b2026121[0-9]{3}\b' C-VRP_Cold-chainVehicleRoutingProblem
  -> No matches found
```

The chosen prefix `20261201xx` / `20261201[23]x` is deliberately outside the two families the
delivery line actually occupied (`2026101002 … 2026101261` and `202610101701 … 202610101704`), so
the disjointness does not depend on a fine-grained reading of those families.

---

## 5. Exact commands / method used

All commands run from the extension root with PowerShell (`pwsh`).
`$env:PYTHONIOENCODING="utf-8"` was set before every Python call.

1. Directory enumeration of the two cycle roots and of `scripts/research_20261010`:

```powershell
Get-ChildItem "C-VRP_Cold-chainVehicleRoutingProblem\results\research_20261010" -Recurse -File
Get-ChildItem "C-VRP_Cold-chainVehicleRoutingProblem\results\research_20261010_continuation_v2" -Recurse -File
Get-ChildItem "C-VRP_Cold-chainVehicleRoutingProblem\scripts\research_20261010"
```

2. Literal seed search in the delivery scripts and protocols, and in the protocols copied into the
   result directories:

```powershell
Select-String -Path scripts\research_20261010\delivery_*.py -Pattern 'seed'
Select-String -Path scripts\research_20261010\pickup_*.py -Pattern 'seed'
Get-ChildItem results\research_20261010,results\research_20261010_continuation_v2 -Recurse -File -Include *.json |
  Select-String -Pattern 'seed'
```

3. A transient read-only Python scanner (run as `python -B`, not retained) that, for every
   `delivery_*.py`, every `delivery_*protocol*.json`, and every `*.json` under the two result roots,
   walked the parsed JSON tree and printed every key whose path contains `seed` (plus integer lists
   under keys containing `instance`/`ids`). Its `json`-based parse skipped files larger than 8 MB and
   failed closed on unparsable files; both classes are reported in §6. The scanner wrote nothing.

4. Cryptographic hashes (Python `hashlib`, `python -B`, run from the extension root):

| File | SHA-256 |
|---|---|
| `scripts/research_20261010/delivery_traintask_protocol_DRAFT.json` | `b2af7b9e45bab1148f27e7069cc69021b673d79c06982da39de78ab6f0673f94` |
| `scripts/research_20261010/delivery_traintask_protocol.json` (frozen, final on-disk) | `4c0c38075235a38aabafa041507a77a5cdb71080e15d18cddd3c4b06f7ec58ce` |
| `results/research_20261010_continuation_v2/PROGRAM.json` | `bc8cfcb51dba940af539e23194fad7c7e7caf681a1e98e68a5b296c7286632d5` |
| `results/research_20261010_continuation_v2/delivery_traintask_export_v1/result.json` | `aa4b73736275f6ed67d0ecab9cabab31fcd87a45a1252abf209d9490eb623a94` |

The recomputed `PROGRAM.json` digest equals the `parent_program_sha256` already recorded in the
DRAFT (`bc8cfcb5…`), so the frozen protocol keeps that value unchanged; no `parent_program_sha256_note`
was needed.

Self-hash rule used inside the frozen file: `SHA-256` over the frozen file's exact bytes with
`frozen_protocol_sha256.value` replaced by the literal ASCII string `PENDING_SELF_SHA`
= `6504e258dc16601e6c579550cabe624210ce7fc10a420e3cd049cf4ee08a9d9c`. A self-referential digest
cannot contain itself, and the verification was re-run after writing:
`sha256(bytes.replace(<recorded value>, b"PENDING_SELF_SHA"))` reproduced the recorded value exactly.

5. DRAFT byte-unchanged check: the DRAFT digest before and after all writes is
   `b2af7b9e45bab1148f27e7069cc69021b673d79c06982da39de78ab6f0673f94`, identical to the value recorded
   in the frozen file and to the value recorded before any write.

---

## 6. What this inventory does and does NOT prove

### It proves

- For the searched files, the complete set of concretely-used integer seeds is enumerated above,
  each with its script/protocol, its role, and an evidence path.
- The ten seeds frozen for `D-TRAINTASK-v1` are **not** equal to any seed in that enumerated set.
- The ten frozen seeds do not occur as literal text anywhere in `D:\PyCharm_\MASKCO-Main` (literal
  repository-wide search returned no matches), which is a stronger statement than set-disjointness
  against the enumerated table alone.
- `202610101701` / `202610101702` are **engineering-only** and are not used by any D-TRAINTASK-v1 split.
- `20260927` / `20260928` are forbidden and are not used by any D-TRAINTASK-v1 split.
- The DRAFT protocol is byte-unchanged.

### It does NOT prove

- **It cannot prove that an unrecorded seed was never used.** The inventory is bounded by the
  archives that were searched. Any seed that was drawn, printed, or discarded without being written
  into a script, protocol, or result file under the searched paths is invisible here. Disjointness is
  therefore established *only with respect to the seeds recorded in the searched archives*.
- Files larger than 8 MB were **not** parsed for seed keys by the scanner:
  `results/research_20261010/delivery_progress_replay_v1/result.json` (44,789,286 B),
  `results/research_20261010/delivery_raw_v1/pilot/result.json` (124,374,830 B),
  `results/research_20261010_continuation_v2/delivery_raw_v1/pilot/result.json` (128,083,639 B).
  Their companion `PROTOCOL.json` files *were* parsed and carry the same registered seeds; the
  per-attempt payloads inside the large files were not read.
- Non-JSON result artifacts (`.jsonl` progress files, `.log` files) were not exhaustively parsed for
  seed integers. The delivery audit document
  (`scripts/research_20261010/DELIVERY_20_ROUND_AUDIT.md`) and the progress files reference the same
  registered seeds listed above, but this inventory does not claim to have enumerated every integer
  appearing in them.
- Integer literals inside generated instance payloads (coordinates, time windows, logits) are not
  seeds and are not enumerated.
- `results/**/delivery_raw_v1/src/**` and `frozen_src/**` are frozen copies of the extension's own
  source tree. Their seeds are identical to the live scripts; they are mirrors, not additional
  seed sources, but they were walked by the scanner.
- The pickup line shares the integer namespace and contributes `2026101191` and `2026101201`. Those
  two are listed for hygiene only; the `pickup_*.py` scripts otherwise take their dataset seed from a
  previously recorded gate config (`cfg['seed']`) rather than from a literal in these files, so the
  pickup dataset seed value is not enumerated here.
- Nothing in this file is a performance claim. No training and no experiment was run; no result of
  any kind is asserted.

---

## 7. Files created by this task

| File | SHA-256 |
|---|---|
| `scripts/research_20261010/delivery_traintask_protocol.json` | `4c0c38075235a38aabafa041507a77a5cdb71080e15d18cddd3c4b06f7ec58ce` |
| `scripts/research_20261010/DELIVERY_SEED_INVENTORY.md` | this file |

The DRAFT `scripts/research_20261010/delivery_traintask_protocol_DRAFT.json` and everything under
`results/research_20261010_continuation_v2/` (and `results/research_20261010/`) were **not modified**.
