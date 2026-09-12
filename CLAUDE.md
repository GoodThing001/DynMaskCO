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

## ⚠️ Current Status (2026-09-12)

> **Only authoritative status entry:** `C-VRP_Cold-chainVehicleRoutingProblem/项目当前状态.md` + `docs/当前规划/结果与进度/执行进度表.md`. When research status conflicts with anything in this file, those two win.

- **DEV-GATE v4 = GO** (9×128=1152 instances, run_id `20260908-054550`): nine-cell equal-weight Δ = **−0.2771**, grouped-bootstrap 95% CI [−0.2848, −0.2696] (does not cross 0). Output: `results/o0cc/dev_gate_v4_official/`.
- **Next sequence** (Task 3 decision A): **CAL-PHYS (physical calibration) → new DEV-CAL/profile → new DEV-GATE re-confirm → freeze O0-CC VAL → one-shot VAL → M0**. DEV-GATE GO ≠ O0-CC closed; **only VAL GO permits M0**.
- The pre-calibration numbers (pilot `quality`/`energy`, old `−2.668`/`−10.9%` O0-D) are pilot/historical, never paper claims.
- Frozen scale profile: `o0cc-pilot-devmean-equal-v2` (`results/o0cc/scale_v2/objective_profile.json`), contract hash `0f00af5d…`.
- Current source identity: `compute_sha256 = f83ad3c485327d8c…` (15 files), control `3c77ecf5…`, analysis `a56f4e55…`.

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

## Environment

- **Local (Windows, Python 3.12)**: the control-chain/identity test suites run with only NumPy. No GPU, JAX, or C++ extension needed.
- **Server (Linux, Python 3.10.19)**: `/home/hzeng/project/MASKCO-Main/`, conda `MASKCO_env`, 2× RTX 5090, GPU 0 free. For any job >5 min use `tmux` (not `nohup`) — MobaXterm SSH drops kill foreground jobs.
