# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**MaskCO** is the official ICLR 2026 implementation of a Neural Combinatorial Optimization (NCO) framework using masked generation (TSP / CVRP / MIS). The unmodified upstream implementation lives in `MASKCO_code/`.

**DynMaskCO-CC** is an isolated extension in `C-VRP_Cold-chainVehicleRoutingProblem/` that extends MaskCO to dynamic cold-chain vehicle routing: CVRPTW (time windows) → Cold-Chain (temperature classes) → Dynamic (reveal_time + EDoD). It is a **pickup-to-depot** problem (vehicles leave empty, pick up cold-chain orders, return to depot), not depot-to-customer delivery.

This is a combined workspace, not a single package:

| Area | Purpose |
|---|---|
| `MASKCO_code/` | Upstream TSP/CVRP/MIS source (never modify) |
| `C-VRP_Cold-chainVehicleRoutingProblem/` | DynMaskCO main research workspace (the active area) |
| `CC_Compare/` | Learned-baseline reproduction; only the `dcc_vrp/` adapter layers are edited |

## ⚠️ Current Status (2026-09-19)

> **Only authoritative status entry:** `C-VRP_Cold-chainVehicleRoutingProblem/项目当前状态.md` + `docs/当前规划/结果与进度/执行进度表.md`. When research status conflicts with anything in this file, those win.

- **DEV-GATE v4 = GO** (9×128=1152 instances, run_id `20260908-054550`): nine-cell equal-weight Δ = **−0.2771**, grouped-bootstrap 95% CI [−0.2848, −0.2696]. Output: `results/o0cc/dev_gate_v4_official/`. Pilot oracle result, **not** a learned-method claim.
- **路线 B** (M0/M1 direct, no CAL-PHYS→VAL gate). CAL-PHYS is a literature-supported simulator setting; ORACLE-VAL is optional.
- **M0/M1 layer implemented and run on server** on **m0_scale**: TRAIN-64 (root_seed 9701) / CAL-16 (9702) / DEV-CHECK-16 (9703), 50-node R1 EDoD=0.5. (Earlier m0dev 8/4 is superseded.)
- **De-index ablation done**: ID-sensitivity is real and strong — renumbering non-depot nodes flips scores (max |Δ| ≈ 0.008–0.027). `--deindex` masks `ACTION_ID_CHANNELS=(0,2,4,5)` (customer/slot_anchor/predecessor/successor) + `FLEET_ID_CHANNELS=(0,4)` (vehicle_node/committed_next) and removes the sensitivity — but does **not** improve utility.
- **Three-group (feature-local / H-only / M1) and A/B/C/D compact ablation, 3 seeds each**: none establish positive utility. A (explicit-only) is best (net ≈ −0.0016, Spearman .33); B/C/D degrade (Spearman .33→.13→.02→.07). **B/C/D fit poorly in a short budget (30-step loss 31–59 vs A's 0.30)** → A→B degradation is an optimization/budget confound, **not** "encoder has no info". Defensible claim: "current recipe (Huber + mutable-edge recon, 2000-step budget) fails". Do **not** write "encoder untransferable" / "recon harmful" / "capacity ruled out".
- **X_state (candidate-vehicle-cargo state) wired + tested**: `extract_vehicle_cargo_state` + `resolve_target_vid` (in `coldchain_visible_features.py`) resolve target vehicle and read per-zone cargo quality. 2×2 (F_H/F_R/S_H/S_R) shows state has **no increment** — legal candidates with cargo ≈58–67% (not the padding-diluted "17%"), yet S−F diff ≈0 even in the has-cargo subset.
- **Decision-loss 2×2 + soft label done; branch closed**: KEEP-ranking gives exploratory τ=0 improvement (2/3 seeds) but no positive net at CAL-selected τ\*; three-way decomposition shows pick_loss (wrong candidate) and accept_loss (accept/reject) both ≈0.010–0.017. Soft label (`F_soft`) lowers margin error (0.035→0.023) but is insufficient to improve actual net gain (magnitude conflict is not the sole confirmed bottleneck). Per stop rule: **no further loss-branch extension, no encoder, no state expansion; recorded as negative (recipe closed, not "MaskCO infeasible"); current DEV is not a held-out test.**
- Frozen scale profile: `o0cc-pilot-devmean-equal-v2` (`results/o0cc/scale_v2/objective_profile.json`), contract hash `0f00af5d…`.
- Frozen encoder for M0/M1: `ckpts/r1_5_baseline/typed_v1_edge/phase3c/seed42/step50000.ckpt` (loads as `DynamicColdChainModel`, 7D input, `edge_feat=None`).
- Current source identity: `compute_sha256 = f83ad3c485327d8c…` (15 files), control `3c77ecf5…`, analysis `a56f4e55…`.
- **④ interface and ⑤ closed-loop/attribution results recorded** — frozen feature-local models have not established positive utility. FULL−ONE and FULL−EVENT-ONE show additional degradation for F_R_s42; the corresponding F_H intervals cross zero. EVENT-ONE itself has not established a gain. Reported reveal-tail invalidation explains checked cases, not all zero-gain actions; equal next dispatch does not imply equivalent future execution. Do not reinstate the unsupported “baseline undo → repeated reapplication” mechanism. See the results table §1.5d–f and D-050–D-052.
- **EP-v1.2 closed under the effort limit (D-056)** — local summaries confirm A/B/proxy select infinity and DEV has no interventions; distance delta is +0.041983. B has no detected conflicting-input groups in these samples, which does not prove a lossless representation or rule out representation as a major bottleneck. CAL seed43 favors B over A, the other two do not. Source review found WAIT not advanced to H, committed load double-counted in projection summaries, and training/generation seeds coupled. Record these limits without automatically fixing and rerunning the closed recipe. Numerical reproduction is reported; server byte identity was not independently verified. Preserve assets as requiring validation before reuse. See `C-VRP_Cold-chainVehicleRoutingProblem/docs/当前规划/决策与审计/event_plan两轮收束与后续决策.md`.
- **N0-R → N1/N2 (cold-chain search reference + MaskCO conditional reconstruction; concluded negative, no learning increment)** — N0-R (forbid LNS from using JF1-H-F's reserved idle vehicle, `--guard`) reached the exploratory N1 entry (R: CAL ΔJ −0.0686, DEV −0.0232, service 16/16). N1/N2 trained a MaskCO conditional joint-repairer (frozen `DynamicColdChainModel` encoder + trainable `MaskCODecoder` + insertion head, cross-entropy on teacher insertion steps) and closed the loop: MaskCO does **not** beat regret-2 R (mixed vs baseline, R better on both splits). Diagnostics (evidence bounds corrected 2026-09-20): (a) train/deploy padding mismatch — s42 first 20 training records' first-step choice flips 6/20 (pad length = total nodes; whether it equals training `Nv_max` still to cross-check); (b) fixed-state — learned model reproduces regret-2's **customer set** 100% / full-plan hash 60%, ~440× slower (compile/steady not separated, learner timing excludes encoder); **no `J_vis` computed, so fixed-state quality is unjudged**. Conclusion: closed-loop has no learning increment; the diagnostics do **not** prove "the student can only copy R". The follow-up no-training enhanced-repair validation (interleaved two-customer cross-vehicle swap after R's full repair, full certification + full `J_vis`) also concluded negative — fixed-state clustered-bootstrap mean Δ(J_S−J_R)=+0.0058 (CI>0) and closed-loop swap_effect>0 on CAL/DEV; this does **not** prove "R is the deployable upper bound" nor "no better local repair exists" (228 states improved; S's candidate set doesn't contain R's). Stop the imitate-regret-2 distillation route, **not** turning to a non-learning paper; next = MaskCO direct-objective-training limited prototype (reuse pretrained decoder, policy-gradient on full-repair reward). Server GPU now needs `jax-cuda12-plugin` + `nvidia-*-cu12` + `flax==0.10.4` (encoder's `SwiGLU` breaks on 0.10.7).
- **MaskCO direct-objective-training (M-pre / M-trained, Steps 1–5; concluded negative, then corrected training)** — reuse the **pretrained CVRP decoder** (`MASKCO_code/ckpts/cvrp100.ckpt`, 512-dim; NOT the extension's 256-dim encoder-only `step50000.ckpt`) + a cold-chain adapter `C` + a directed insertion head `δ` (both zero-init last layer), train via REINFORCE on full-repair reward `r=(J_vis(P0)−J_vis(P))/s_TRAIN`. Step 5 closed-loop (CAL/DEV, budget 4s): unified-policy M-trained did **not** beat M-pre (CAL −0.010 / DEV +0.015, CI cross zero) nor regret-2 R (CAL +0.073 significant / DEV +0.037). **Two collector bugs found & fixed (2026-09-21)**: (a) saved **live `vehicles`/`served_mask` references** → simulator mutated them, network input read terminal (not event-time) state (60.8% of vehicle fields mismatched); (b) global `max_states` → 256 states from only 4 instances. **Corrected training running** (seed42, 1000 updates, 8×4, same network/reward/lr); old checkpoint marked `mpre_reinforce_s42_defective`. See `docs/当前规划/工程实现/MaskCO直接目标训练工作包.md`.
- **Work ownership** — the user decides direction and may delegate implementation+execution to the assistant; the assistant implements code, runs experiments (locally and on the server via `python _srv.py`, credentials read from `.vscode/sftp.json`), and writes result docs; the user reviews and corrects docs. Routine implementation choices within the recorded protocol do not need repeated method approval.

## Hard Invariants

1. **Never modify `MASKCO_code/`.** All extension code and path adapters live in `C-VRP_Cold-chainVehicleRoutingProblem/`.
2. **No `__init__.py`** in CVRPTW subdirectories — imports are done via `sys.path.insert` to avoid shadowing parent packages.
3. **strict-online / causal**: non-anticipatory; use `coord_normalize_visible` (never `coord_normalize`, which leaks future nodes); frozen executed prefix; committed invariance; exact-once order service.
4. **Cost semantics**: `distance_cost` is always pure travel distance; the cold-chain composite objective uses a **separate** `coldchain_cost` field. Never mix or substitute them.
5. **Version separation** (established after the DEV-GATE drift incident):
   - `COMPUTE_FILES` (15 files, in `formal_gate_contract.py`) = files that decide oracle/eval/env/contract behavior. Resume requires identical hash; changing any one means re-running 1152 instances.
   - `CONTROL_FILES` = driver / instance validation / summary rebuild — evolve independently.
   - `ANALYSIS_FILES` = aggregator — evolve independently.
6. **Runner self-hash**: `run_action_oracle.py` hashes its own source at process **startup** and writes that into instance records + manifest. Never re-read disk at the end (that was the drift bug).
7. **Data roles** are disjoint by seed: DEV-CAL(777) / DEV-PROTO(7781) / DEV-GATE(7782) / VAL128 / TEST. CAL-PHYS fitting must not read DEV-GATE/VAL/TEST method effects.
8. **Ownership partition** (M0/M1 online accept/certify): validate the full state partition `committed ⊎ planned suffix ⊎ explicit deferred` — not just "suffix covers universe". `validate_full_partition` in `dynmaskco_cc_context.py` is the reference. Never use `validate_ownership` with a universe that includes committed/deferred customers (they are legitimately outside the suffix and would be falsely flagged "missing").
9. **Input consistency** (M0/M1): the training, online-replanner, and diagnosis paths must produce **identical** input tensors for the same state+action (DEFER/KEEP included). Do not hand-construct action/coldchain features in one path and use `extract_action_features`/`cc_state_dict` in another. Verify per-field equality when adding features.
10. **Teacher sampling**: default `--context-sampling earliest` truncates at the first event (all contexts at t=0), which starves the model of committed-leg/deferred/late-reveal states. Use `--context-sampling spread` for real TRAIN/DEV export.

## Common Commands

All paths below are relative to `C-VRP_Cold-chainVehicleRoutingProblem/` (run from there — several tests use `results/…`-relative paths resolved against the extension root).

### Tests (pure NumPy, no GPU/JAX needed locally)

```bash
# Control-chain / identity suites (run from the extension root)
python scripts/tests/test_formal_gate_contract.py      # roles/manifest/seed/hash/registry/stat-plan
python scripts/tests/test_contract_manifest.py          # cold-chain contract serialize/round-trip
python scripts/tests/test_dev_gate_control.py           # instance validation / cell summary / aggregator identity
python scripts/tests/test_control_chain.py              # DEV-GATE driver end-to-end (resume + drift negatives)
python scripts/tests/test_aggregate_cross_cell.py       # aggregator 9-cell negatives
python scripts/tests/test_val_control_chain.py          # VAL archive end-to-end + startup-lock negatives

# Compute-layer regression
python scripts/tests/test_coldchain_contract.py         # C0 functional gate (20/20)
python scripts/tests/test_sequential_oracle.py          # sequential oracle synthetic (6/6)
```

Run a single test function: `python -c "import sys; sys.path.insert(0,'scripts/tests'); import test_contract_manifest as t; sys.exit(0 if t.test_file_roundtrip() else 1)"`.

### Formal gate (DEV-GATE / O0-CC VAL)

```bash
# Generic driver (dev_gate; --role val is blocked — VAL is archive-only)
python scripts/evaluation/run_formal_gate.py --role dev_gate \
    --manifest <exact manifest> --profile <profile.json> --workers 8 --out <fresh dir>

# DEV-GATE thin entry (legacy CLI)
python scripts/evaluation/run_dev_gate.py \
    --data-dir data/baseline/50_node/dev_gate \
    --profile results/o0cc/scale_v2/objective_profile.json --workers 8 --out results/o0cc/dev_gate

# VAL entry — ONLY from a sealed archive (self-contained, verified)
python scripts/evaluation/run_o0cc_val.py \
    --archive <sealed archive> --expected-bundle-sha256 <64-hex> --workers 8 --out <fresh dir>
```

### Freeze / verify the source archive

```bash
python scripts/evaluation/freeze_source_archive.py \
    --role val --manifest <VAL_MANIFEST.json> --profile <profile.json> \
    --split-registry <registry.json> --contract <contract.json> \
    --statistical-plan <plan.json> --out <archive dir>

python -B scripts/evaluation/verify_source_archive.py --archive <archive dir>
```

### Oracle runner (single cell/shard, internal `--role regression`)

```bash
python scripts/evaluation/run_action_oracle.py \
    --data <npz> --capacity 50 --num_vehicles 25 --objective coldchain \
    --objective-profile <profile.json> --coldchain-contract <contract.json> \
    --local --role regression --workers 8 --out <dir>
```

### Data split generation

```bash
python scripts/data/generate_o0cc_split.py \
    --split-role val --seed <root_seed> --num-instances 128 \
    --profile <profile.json> --disjoint-manifest <other_manifest> --out <dir>
```

### M0/M1 data & model (server GPU; JAX/Flax needed)

```bash
# Fresh M0 TRAIN/DEV base instances (root seeds 9601/9602, R1 EDoD=0.5)
python scripts/data/generate_m0_teacher_split.py --train-instances 8 --dev-instances 4 --out data/m0dev

# Teacher candidate export (spread across events; default 'earliest' is the t=0 trap)
python scripts/expert/export_coldchain_teacher.py \
    --data data/m0dev/dcc_50_r1_edod05_train_teacher.npz --dataset-role train_teacher \
    --objective coldchain --objective-profile results/o0cc/scale_v2/objective_profile.json \
    --num-vehicles 25 --capacity 50 --max-instances 8 --max-contexts-per-instance 12 \
    --context-sampling spread --manifest data/m0dev/M0_TEACHER_MANIFEST.json \
    --out results/m0dev/train_spread

# Feature-only utility probe (train + eval) — the M0 representation probe
python scripts/training/train_coldchain_utility_probe.py \
    --teacher-dir results/m0dev/train --data data/m0dev/dcc_50_r1_edod05_train_teacher.npz \
    --num-steps 2000 --batch-size 8 --seed 42 --out results/m0dev/probe_feature_only
python scripts/training/run_coldchain_utility_probe.py \
    --ckpt results/m0dev/probe_feature_only/probe.ckpt \
    --teacher-dir results/m0dev/dev --data data/m0dev/dcc_50_r1_edod05_dev_teacher.npz \
    --out results/m0dev/eval_feature_only

# M1 (DynMaskCO-CC) joint training: frozen encoder + masked decoder + utility head
python scripts/training/train_dynmaskco_cc.py \
    --teacher-dir results/m0dev/train --data data/m0dev/dcc_50_r1_edod05_train_teacher.npz \
    --ckpt ckpts/r1_5_baseline/typed_v1_edge/phase3c/seed42/step50000.ckpt \
    --num-steps 2000 --batch-size 8 --seed 42 --out results/m0dev/m1_2000

# M1 / feature-only online gate (baseline vs method on the strict-online trajectory)
python scripts/evaluation/run_dynmaskco_cc_gate.py \
    --data data/m0dev/dcc_50_r1_edod05_dev_teacher.npz --capacity 50 --num_vehicles 25 \
    --objective coldchain --objective-profile results/o0cc/scale_v2/objective_profile.json \
    --max-instances 4 --K 1 --m1-ckpt results/m0dev/m1_2000/m1.ckpt \
    --encoder-ckpt ckpts/r1_5_baseline/typed_v1_edge/phase3c/seed42/step50000.ckpt \
    --out results/m0dev/gate_m1k1

# Offline utility diagnosis (g_pred vs counterfactual g_true) + input-consistency check
python scripts/evaluation/run_utility_diagnosis.py \
    --data data/m0dev/dcc_50_r1_edod05_dev_teacher.npz \
    --feature-only-ckpt results/m0dev/probe_feature_only/probe.ckpt \
    --objective coldchain --objective-profile results/o0cc/scale_v2/objective_profile.json \
    --max-contexts 12 --out results/m0dev/diag_fo
python scripts/evaluation/run_input_consistency.py \
    --teacher-dir results/m0dev/train --data data/m0dev/dcc_50_r1_edod05_train_teacher.npz \
    --max-contexts 4 --out results/m0dev/input_consistency
```

M0/M1 tests (JAX + NumPy; run locally — JAX is installed):
`test_coldchain_reader_features.py`, `test_coldchain_teacher_dataset.py`, `test_coldchain_utility_probe.py`, `test_dynmaskco_cc.py` (no-op parity), `test_dynmaskco_cc_accept.py` (deterministic accept), `test_m1_partition_fix.py` (ownership partition), `test_m1_jit_equiv.py` (real-checkpoint JIT vs non-JIT), `test_deindex_features.py` (de-index + renumbering invariance), `test_pre_prepare_collector.py` (pre-prepare context collector cache/拒绝语义).

### De-index ablation + compact A/B/C/D (m0_scale, the current active data)

Active data is now `data/m0_scale/` (`generate_m0_teacher_split.py --train-instances 64 --cal-instances 16 --dev-check-instances 16`), **not** the older m0dev 8/4. The newer scripts (all take `--deindex`):

- `scripts/training/train_feature_only_local.py` — feature-only-local (70-dim: context 25 + action 16 + local 29); `--deindex` masks the ID channels.
- `scripts/training/train_m1_compact.py` — compact A/B/C/D ablation (`--group A|B|C|D`; A=explicit, B=+compact H endpoint, C=+compact Z recon=0, D=+recon); `--fit-check` overfits a subset to separate wiring vs optimization.
- `scripts/training/train_coldchain_utility_probe.py` — `--context-mode encoder_real` (H-only) + `--train-all` (use all TRAIN-64, no internal dev split).
- `scripts/evaluation/run_threeway_eval.py` — unified offline eval (`--model feature_local|encoder_real|m1|compact|state_2x2`): margin scan (instance-equal net gain, CAL-locked τ\*, regret, harmful ratio) + Spearman/MAE. Reads the model's `s` from the checkpoint (never re-estimates).
- `scripts/evaluation/run_id_sensitivity.py` — renumbering invariance check (fix depot, permute other nodes, compare scores/winner/KEEP).
- `scripts/evaluation/run_candidate_gain_space.py` — hindsight `g* = max(0, max_a[J_keep − J_a])` + 4-category candidate coverage.
- `scripts/evaluation/run_margin_scan.py` — fixed CAL selection rule (argmax over CAL instance-equal net gain, restricted to `service_feasible` & `label_complete` & finite; tie-break smaller τ). `τ=∞` = "keep baseline", not a gain.

- `scripts/training/train_state_2x2.py` — the decision-loss 2×2 trainer (small MLP over the 70/83-dim explicit features, no encoder): `--use-state` (append the 13-dim X_state), `--use-rank` (hard-sign KEEP ranking), `--soft-rank` (soft-label ranking `softplus(m/T)−σ(y/T)(m/T)`); `--fit-n` overfits a subset; writes full config to `manifest.json`. `s` = std(δ) over **full TRAIN** supervision (computed before any `--fit-n` slice).
- `scripts/evaluation/diag_v2.py` / `diag_v3.py` / `agg_fsoft.py` — offline diagnostics (instance-equal selection outcomes, τ\* weighted decomposition + identity check, has-cargo stratification, three-way gain decomposition full/fixed/actual → pick_loss/accept_loss, margin error). No training; reuse the checkpoints' predictions.

Key new extractors in `scripts/data/coldchain_visible_features.py`: `extract_vehicle_cargo_state(snapshot, vid, plan, …)` (13-dim target-vehicle state: anchor_time/remaining-capacity from the **plan** (not snapshot's ready/load), per-zone temp/zone-load/remaining-quality, `state_valid` + `has_cargo` separate) + `resolve_target_vid(action, plans, allowed_vehicle_ids)` (NEW_ROUTE → `_match_new_route`, DEFER → None). Wired into `train_state_2x2.py`'s `build_x_state` (not yet into the online replanner).

### feature-local 70-dim adapter + ⑤ attribution (m0_scale, 2026-09-18)

The frozen 70-dim `ScoringMLP` scorer (`train_state_2x2.py` `model.ckpt`, no encoder) is deployed online via `feature_local_replanner.py` with a `PrePrepareContextCollector` for the pre-prepare context. All these take `--deindex` (required, read from training `manifest.json`):

```bash
# ④ interface reconciliation (feature/score/writeback/causality — all PASS)
python scripts/evaluation/run_feature_local_reconcile.py \
    --teacher-dir results/m0_scale/dev_check \
    --data data/m0_scale/dcc_50_r1_edod05_dev_check_teacher.npz \
    --max-contexts 8 --deindex --out results/m0_scale/iface_reconcile
python scripts/evaluation/run_feature_local_score_check.py \
    --ckpt results/m0_scale/state2x2_full_v1/F_H_s42/model.ckpt \
    --teacher-dir results/m0_scale/dev_check \
    --data data/m0_scale/dcc_50_r1_edod05_dev_check_teacher.npz \
    --max-contexts 4 --deindex --out results/m0_scale/iface_score
python scripts/evaluation/run_feature_local_writeback_check.py \
    --teacher-dir results/m0_scale/dev_check \
    --data data/m0_scale/dcc_50_r1_edod05_dev_check_teacher.npz \
    --max-contexts 2 --deindex --out results/m0_scale/iface_writeback

# ⑤ K=1 closed-loop (baseline vs FeatureLocalReplanner; --tau inf = no-op control, must == baseline)
python scripts/evaluation/run_feature_local_gate.py \
    --ckpt results/m0_scale/state2x2_full_v1/F_H_s42/model.ckpt \
    --data data/m0_scale/dcc_50_r1_edod05_dev_check_teacher.npz \
    --capacity 50 --num-vehicles 25 --objective coldchain \
    --objective-profile results/o0cc/scale_v2/objective_profile.json \
    --deindex --K 1 --tau 0.005 --max-instances 16 --out results/m0_scale/gate_fl

# ⑤ attribution: B/ONE/FULL (ONE = max_total_accepts=1, first accept then baseline)
python scripts/evaluation/run_feature_local_onefull.py \
    --ckpt results/m0_scale/state2x2_full_v1/F_H_s42/model.ckpt \
    --data data/m0_scale/dcc_50_r1_edod05_dev_check_teacher.npz \
    --capacity 50 --num-vehicles 25 --objective coldchain \
    --objective-profile results/o0cc/scale_v2/objective_profile.json \
    --deindex --tau 0.005 --max-instances 16 --out results/m0_scale/onefull_F_H_s42

# ⑤ attribution: ĝ vs g_B counterfactual (samples 1st/5th/10th accepts, ≤24 decision points)
python scripts/evaluation/run_feature_local_gB.py \
    --ckpt results/m0_scale/state2x2_full_v1/F_H_s42/model.ckpt \
    --data data/m0_scale/dcc_50_r1_edod05_dev_check_teacher.npz \
    --capacity 50 --num-vehicles 25 --objective coldchain \
    --objective-profile results/o0cc/scale_v2/objective_profile.json \
    --deindex --tau 0.005 --max-instances 4 --out results/m0_scale/gB_F_H_s42

# ⑤ behavior summary (aggregates onefull instances/*.json decision logs)
python scripts/evaluation/run_behavior_summary.py \
    --gate-dir results/m0_scale/onefull_F_H_s42 --out results/m0_scale/onefull_F_H_s42/behavior_summary.json
```

Key facts that bite: the 70-dim model is a plain MLP over `ctx25 + action16 + local29`, **no encoder/mask**. The online context must be **pre-prepare** (via `PrePrepareContextCollector` registered to `env.snapshot_hook`), never reconstructed from post-`prepare_decision_point` vehicles — `prepare_decision_point` bumps `ready_time` (idle→clock, ready→max) and the pre-bump value is unrecoverable. `deindex` is a required constructor arg (no default). The `ĝ vs g_B` diagnostic must install `P_before`/`P_after` explicitly via `_snapshot_with_force` (never `rollout_baseline` from the original snapshot, which rebuilds the pre-modification incumbent).

## Architecture

### Formal gate control chain (DEV-GATE / VAL share one core)

- `scripts/evaluation/formal_gate_contract.py` — the single shared contract: roles, manifest validation (9 cells R1/C1/RC1 × EDoD 0.2/0.5/0.8, 128 instances, shared seeds), hash helpers, frozen-compute plan + resume validation, cross-split disjointness, split registry, statistical-plan validation, and sealed-archive verification (`verify_archive` / `verify_active_code`).
- `scripts/evaluation/run_formal_gate.py` — the generic driver (`run()`): validate manifest/profile/contract/registry → write `pre_run_manifest.json` (first run requires empty dir; resume requires identical frozen identity) → per-cell run `run_action_oracle.py --role regression` (resume only missing instances, hard-fail on error/protocol/service-fail) → rebuild canonical cell summary → aggregate.
- `scripts/evaluation/run_dev_gate.py` / `run_o0cc_val.py` — thin entries fixing `role=dev_gate` / `role=val`. VAL entry verifies the archive (bundle + active-code hash) before running.
- `scripts/evaluation/aggregate_cross_cell.py` — nine-cell equal-weight grouped bootstrap (by internal seed group, not 1152 independent rows), with `regression`/`dev_gate`/`val` modes and formal blocking checks.
- `scripts/evaluation/instance_validation.py` + `cell_summary.py` — shared strict instance acceptance and canonical cell summary rebuild (atomic publish + `canonical.COMPLETE` marker).

### Identity / freeze model

- A run's frozen identity = `compute/control/analysis` hashes + `data_sha256` (per-cell NPZ) + `profile_hash` + (VAL) `val_identity` (manifest hash / root_seed / scene ids / contract / profile / registry / stat-plan hashes) + (explicit contract) `contract_identity` (raw + effective contract hash + contract/profile file hashes).
- `freeze_source_archive.py` builds a self-contained archive: 15+6+1 source files, profile, manifest, and — for VAL — the 9 NPZ, `contract.json`, `statistical_analysis_plan.json`, and a normalized bundle-relative `split_registry.json` + `registered_splits/`. It writes `ARCHIVE_SEALED` + a seal manifest with a `bundle_sha256`.
- `verify_source_archive.py` / `formal_gate_contract.verify_archive` check exact file set + per-file SHA-256 + bundle recompute + seal↔源码冻结清单 cross-validation + ARCHIVE_SEALED.

### Cold-chain layer

- `scripts/coldchain/coldchain_contract.py` — versioned physical + objective contract (`ColdChainContract`), `ObjectiveProfile` (named scale/λ), `from_manifest` / `load_coldchain_contract` / `validate_contract_manifest` (strict schema + hash re-check). Calibrated contracts are loaded explicitly via `--coldchain-contract`; the implicit `default_pilot_contract()` remains the legacy pilot default.
- `scripts/coldchain/coldchain_state.py` — the single pure state transition (`transition_segment`; pickup enters cargo, return unloads).
- `scripts/evaluation/hard_gate.py` — shared service/hard gate (`hard_vector_from_outcome` / `service_ok` / `outcome_protocol_error` / `select_improving`).
- `scripts/simulation/jf1h_repair.py` — **JF1-H-F** hard-feasible baseline (`make_continuation()` factory + `export_state`/`restore_state` + urgent-defer).

### Oracle (O0-D / O0-CC)

`scripts/evaluation/run_action_oracle.py` compares baseline (JF1-H-F) vs a greedy **sequential** action-space oracle on the same strict-online protocol, plus a **local** single-customer baseline-state oracle. Instance-level multiprocessing (`--workers`), per-instance non-degeneration + `PROTOCOL_ERROR` checks.

### M0/M1 data & model layer (DynMaskCO-CC)

The active research area. The pipeline is: **teacher data → utility learning (M0 probe) → masked reconstruction + utility (M1) → online replanner → offline diagnosis**.

- **Teacher data** — `scripts/expert/export_coldchain_teacher.py` runs the JF1-H-F baseline, collects decision-point snapshots, and for each (snapshot × customer) enumerates all candidate actions + counterfactual rollouts, writing a versioned dataset (`contexts.jsonl` + `candidates.jsonl` + `manifest.json`, fail-closed). `scripts/data/coldchain_teacher_dataset.py` reads it (groups candidates by `context_id`, separates `legal_mask` from `supervision_mask`); `scripts/data/coldchain_visible_features.py` builds the visible-order/fleet/action features and candidate-local features (info-boundary aware — unrevealed orders are masked).
- **M0 feature-only probe** — `scripts/models/coldchain_utility_head.py` (`UtilityHead`, masked-mean order/fleet context + action). `scripts/training/train_coldchain_utility_probe.py` trains Huber regression `y = -(J_a − J_keep)/s` (s estimated on TRAIN only); `run_coldchain_utility_probe.py` is the eval entry (MAE / Spearman / top1-agreement). This is a representation probe, **not** the final method.
- **M1 model** — `scripts/models/dynmaskco_cc.py` (`DynMaskCOModel`: frozen `DynamicColdChainModel` encoder + trainable `MaskCODecoder` + `ScoringMLP`; `M1Scorer` wraps decode→gather→head in one persistent `jax.jit`). `scripts/models/dynmaskco_cc_graph.py` builds the visible-node index, symmetric plan adjacency, event mask `M = ⋁(A_a ≠ A_0)` + `A_in`, timestep, and endpoint resolution. `scripts/simulation/dynmaskco_cc_context.py` holds the shared state-restore / incumbent / full-partition validation used by training, online, and diagnosis.
- **Online replanners** — `scripts/simulation/dynmaskco_cc_replanner.py` (M1: JF1-H-F incumbent → K rounds of event-mask/reconstruct/score/certify/write-back) and `feature_only_replanner.py` (the same flow with the M0 head). Both subclass `JF1HRepairReplanner`; `model=None`/`scorer=None` = no-op == baseline.
- **70-dim feature-local adapter (④⑤, `feature_local_replanner.py`)** — `FeatureLocalReplanner` (JF1HRepairReplanner subclass) scores `ScoringMLP(70)` over `ctx25+action16+local29`; `PrePrepareContextCollector` (registered to `env.snapshot_hook`) saves the pre-prepare 25-dim context keyed by `(inst, event, clock)` — missing key raises, never falls back to post-prepare. Accept rule `g = s·(f(a)−f(KEEP/DEFER)) > τ`; `max_total_accepts=1` = the ONE strategy. Per-decision logging records `P_before`/`P_after` (full plans) for the ĝ-vs-g_B counterfactual. Empty-plan write-back uses `env.has_future_reveal` (`[]` WAIT vs `[0]` return), matching teacher/JF1-H-F. `plan_to_dict`/`plan_distance` are module-level helpers for the decision log.
- **Gate + diagnosis** — `scripts/evaluation/run_dynmaskco_cc_gate.py` runs baseline vs method on 4 DEV instances (per-instance JSON + online stats + split timing). `run_utility_diagnosis.py` pairs model-predicted improvement `g_pred = s·[f(a)−f(KEEP)]` against counterfactual `g_true = J_KEEP − J_a`; `run_input_consistency.py` verifies train/online/diagnosis tensors are identical.

Key wiring facts that bite if forgotten: the M1 masked decoder's adjacency is a **soft 0/1 attention bias** (not a hard mask), and `edge_feat=None` (no edge input) in this dev stage. Node padding: `train_dynmaskco_cc.py` pads nodes **globally** (fixed JIT shape); the compact `dynmaskco_cc_compact.py` masks padded nodes via a `node_valid` bias (−1e9) + zeroed output. The older `dynmaskco_cc.py` decoder does **not** mask padded nodes, so its scores shift under padding (verified max |Δ| ≈ 0.008–0.027) — don't trust its per-context numbers as padding-invariant.

### event_plan_v2 (event-level complete-plan evaluation — concluded, kept as assets)

Event-level plan evaluation supersedes the per-modification 70-dim scorer: from one decision-point incumbent P0, generate ≤8 full-fleet candidate plans (G_simple_r1, 7 deterministic requests) and score each candidate's counterfactual terminal gain `g_B = J(P0→B) − J(P→B)` under a common JF1-H-F continuation. Pipeline: **export (labels+features) → train `PlanEvalMLP` → online replanner → fixed-state selection + CAL→DEV closed loop**. Core modules (all under `scripts/`):

- `simulation/event_plan_candidate.py` — `generate_simple_candidates_r1` (request/plan-bound, KEEP-deduped) + `plan_distance` (includes depot-return leg; distinguishes WAIT vs RETURN).
- `simulation/event_plan_projection.py` — visible execution-consequence projection: advances the **currently-known** plan via C0 `transition_segment` to a D/Q/E/J proxy (`project_plan`/`proxy_cost`), no future reveals. Physical engine behind the B representation and the non-learned S_proxy control.
- `simulation/event_plan_features.py` — `extract_plan_delta` (A, 100-dim structural summary) vs `extract_plan_consequence` (B, 529-dim consequence).
- `simulation/event_plan_replanner.py` — `EventPlanReplanner` (`mode=distance|proxy|learned`, `representation=A|B`).
- `training/train_event_plan.py` — `PlanEvalMLP` over `[context25 + plan_feature]`, Huber on `(f(P)−f(P0)) − g_B/s`, instance→event→candidate balanced sampling.
- `evaluation/run_event_plan_export.py` (TRAIN/CAL data: g_B + both features + hard vectors + real deferred) / `run_event_plan_select.py` (fixed-state selection table) / `run_event_plan_closed_loop.py` (CAL threshold grid {0,0.005,0.02,∞} → service-first lock τ → DEV) / `run_event_plan_gate.py` (single mode/tau).

Result is a concluded negative (see Current Status); the modules are research assets, not an active recipe.

### CC-LNS-RH + reserved protection + MaskCO conditional reconstruction (N0 → N1/N2, concluded negative)

The current research line: a **non-learning cold-chain LNS reference (CC-LNS-RH)** proves the framework target, then **MaskCO learns to generate joint-repair candidates** against it. The pipeline, in dependency order (all under `scripts/`):

- `simulation/visible_state.py` — read-only cold-chain view `VisibleState` (order-level cargo manifest + per-vehicle temps/zone_load) + the visible-scenario evaluator `evaluate_visible_plan` (no-new-orders completion via C0 `transition_segment`, returns `J_vis`). Unused idle vehicles are NOT dispatched (no phantom pre-cooling).
- `simulation/cc_lns_replanner.py` — `CCLNSReplanner`: JF1-H-F incumbent → remove mask → regret-2 reconstruct → `J_vis` accept. `guard_reserved=True` (N0-R) forbids LNS from using the highest-id idle vehicle that JF1-H-F still reserves. `run_cc_lns_gate.py` is its closed-loop gate.
- `simulation/repair_teacher.py` + `expert/export_repair_teacher.py` — per-step insertion labels `P_before → mask → P_partial → steps → P_target` from a more-thorough regret-2 search on the guarded R trajectory (384 records).
- `training/train_dynmaskco_cc_repair.py` — conditional insertion-step training: frozen encoder + trainable `MaskCODecoder` + insertion `ScoringMLP`, softmax cross-entropy over legal insertions (NOT the old g_B regression). Reuses `models/dynmaskco_cc.py` (`MaskCODecoder`, `ScoringMLP`, `gather_endpoints`).
- `simulation/dynmaskco_cc_repair_replanner.py` — `RepairReplanner` + `greedy_reconstruct` (decode Z once per candidate, then score insertion steps); `evaluation/run_repair_gate.py` is the closed-loop gate.
- `evaluation/diag_n1_table{1,2}.py` — table-1 (padding consistency / label replay / skip counting) and table-2 (fixed-state learned-vs-regret-2) diagnostics.
- `simulation/cc_swap.py` + `evaluation/run_cc_lns_swap_{fixed_state,gate}.py` — **enhanced-repair validation (no training; the next work package after N1/N2)**: limited two-customer cross-vehicle swap after R's full repair, full certification + full `J_vis`. `cc_lns_swap_search(swap=False)` is behaviorally identical to `cc_lns_search`; `swap=True` adds swap candidates (TW pre-check + capacity via try/except). Fixed-state (Q1: does swap beat R from the same public state?) + closed-loop (Q2: does it persist online?). See `docs/当前规划/决策与审计/N1N2收束与增强修复验证决策.md`.

Key wiring facts: the decoder is a **soft 0/1 attention bias** over the plan adjacency, and `edge_feat=None`. The frozen encoder loads as `DynamicColdChainModel` (7D input) via `train_fleet_head.load_base_model`; server needs `jax-cuda12-plugin` + `nvidia-*-cu12` + `flax==0.10.4` (0.10.7 breaks `SwiGLU`). Train/deploy padding is a known inconsistency (see Current Status): the training pads nodes to a fixed Nv (unmasked) while the online replanner uses the actual visible Nv, flipping ~30% of insertion choices.

### M-pre / M-trained (MaskCO direct-objective-training, Steps 1–5, concluded negative)

Reuse the **pretrained CVRP decoder** (not the extension's 256-dim encoder) + cold-chain adapter + directed head, trained by REINFORCE on full-repair reward. Modules (all under `scripts/`):

- `models/mpre.py` — `load_cvrp_model`/`encode_cvrp`/`decode_edge_logits`: loads `MASKCO_code/ckpts/cvrp100.ckpt` (512-dim `CVRPModel`, 3D coord+demand input), produces **symmetric** edge logits `features@features.T`.
- `models/mpre_trained.py` — `MpreTrainedModel` = frozen CVRP encoder (`init_proj/encoder/depot_bias`) + trainable decoder (`mid_proj/timestep_embedder/decoder/final_proj/final_norm`) + `c_adapter` + `delta_head` (both **zero-init last layer** → init scores == M-pre). `partition()` splits train/frozen params via `nnx.split` path predicates; `edge_insertion_scores(L, pred, c, succ)` adds `L[pred][c]+L[c][succ]` and deletes `L[pred][succ]` **only if pred≠succ** (empty-route pred==succ==depot must NOT subtract the masked diagonal sentinel ≈−1.7e38).
- `data/repair_state.py` — `extract_repair_state_v1` (repair_state_v1 schema: F_NODE=10, F_ACTION_EXPLICIT=20 incl. target-vehicle `X_state` 13-dim + slack valid bits + order-level cargo table). **`freeze_vehicles` deep-copies the collector's vehicle list.**
- `simulation/mpre_policy.py` — `stepwise_sample`/`replay_log_prob` (sampling/replay separation; `deterministic=True` = argmax for eval), `enumerate_legal_actions`, `validate_repair_plan` (full-plan reconciliation: vehicle set/anchor/exact-once/non-mask order/protected — NOT customer-set).
- `simulation/mtrained_replanner.py` — `PolicyReplanner` + `cc_lns_policy_search` (unified softmax/argmax policy), `load_trained_model` (merges trained `train_state` into `MpreTrainedModel`).
- `training/train_mpre_reinforce.py` — REINFORCE (baseline = mean of other K−1 samples; reward `(J_vis(P0)−J_vis(P))/s_TRAIN`; functional JIT + fixed-capacity padding; `CollectProbe` with **per-instance quota + `freeze_vehicles`**).
- `evaluation/run_step5_gate.py` — 4-way B/R/M-pre/M-trained closed-loop; `evaluation/diag_collector_state.py` — collector freeze/quota diagnostic.

**Key wiring facts that bite:**
- **Collectors must freeze mutable state** — save `freeze_vehicles(vehicles)` + `np.array(served_mask, copy=True)` at event time; the simulator mutates `VehicleState` fields in-place afterward (60.8% mismatch when not frozen).
- **Per-instance quota, not global `max_states`** — a global cap concentrates all states in the first few instances.
- **Fixed-capacity padding is NOT lossless** for deterministic argmax search: padding to Nv_max/M_max introduces ~1e-5 float32 noise (XLA reassociation) that flips argmax in near-ties and diverges the search. Use unpadded (varying shapes, per-shape JIT recompile) for exact deterministic results; padding only when approximation is acceptable.
- **Checkpoint dims**: pretrained CVRP decoder is **512-dim**; the extension's frozen `step50000.ckpt` is **256-dim encoder-only**. There is no adapted bridge checkpoint on the server.

## Environment

- **Local (Windows, Python 3.12)**: JAX 0.5.0 / Flax 0.10.4 (NNX) / optax are installed, so the M0/M1 training (small MLP heads) runs locally on CPU; the control-chain/identity test suites need only NumPy. The frozen `DynamicColdChainModel` encoder + real teacher data are **server-only**.
- **Server (Linux, Python 3.10.19)**: `/home/hzeng/project/MASKCO-Main/` (credentials in `.vscode/sftp.json`), conda `MASKCO_env` (jax 0.6.2 / flax 0.10.4 / optax 0.2.4), 2× RTX 5090. GPU access requires `pip install jax-cuda12-plugin==0.6.2 nvidia-*-cu12` (the default jax is CPU-only), and `flax` must stay at **0.10.4** (0.10.7 breaks `SwiGLU` → `'ArrayImpl' object has no attribute 'value'`). Run remote commands via the local helper `python _srv.py "<cmd>"` (reads credentials from `sftp.json`, never inline the password — the auto-mode classifier flags plaintext credentials in the transcript). The GPUs are **shared** with other users' jobs — for M0/M1 runs use `CUDA_VISIBLE_DEVICES=1` and cut memory/compile contention with:
  ```bash
  export XLA_PYTHON_CLIENT_PREALLOCATE=false   # avoid JAX's 75% upfront allocation (else OOM under contention)
  export XLA_FLAGS=--xla_gpu_autotune_level=0  # skip slow GEMM autotuning (fine for the full 256-dim decoder)
  ```
  ⚠️ `--xla_gpu_autotune_level=0` triggers a `XlaRuntimeError: CANCELLED: Too small divisible part of the contracting dimension` on small MLPs (H-only probe) and batch-8 compact decoders — **`unset XLA_FLAGS` for those runs**.
  For any job >5 min use `tmux` (not `nohup`) — MobaXterm SSH drops kill foreground jobs. Frozen M0/M1 encoder: `ckpts/r1_5_baseline/typed_v1_edge/phase3c/seed42/step50000.ckpt`.
- **File upload/download to server**: `python _sftp_put.py <local_rel>` / `python _sftp_get.py <remote_rel>` (both take **repo-relative** paths; remote base `/home/hzeng/project/MASKCO-Main/` is hardcoded in the scripts). ⚠️ Git Bash converts `/home/...` CLI args to Windows paths, so **never pass absolute remote paths as args** — pass only relative local paths. `cvrp100.ckpt` (~305 MB) is NOT on the server and must be uploaded.
