# DynMaskCO

**Causal and Feasibility-Preserving Masked Generation for Dynamic Cold-Chain Vehicle Routing**

DynMaskCO extends the [MaskCO](https://github.com/ai4co/maskco) (ICLR 2026) masked-generation paradigm for neural combinatorial optimization from *static* routing to *causal, feasibility-preserving online optimization* for the **dynamic cold-chain vehicle routing problem (DCC-VRP)** — capacitated vehicle routing with time windows, temperature classes, and progressively revealed (online) orders.

> This directory is an **isolated extension workspace**. All new code lives here; the original MaskCO sources under `../` are never modified.

> **Current status (2026-09-03):** The paper line is fixed as a MaskCO-based method for dynamic cold-chain logistics. JF2/HFR are frozen negative results and P0-R/P0-S/P0-A/P0-U are complete. The current blocking task is **C0 cargo-manifest/cold-chain state/unit/trace-evaluator closure**, followed by O0-D/O0-CC. See [`项目当前状态.md`](项目当前状态.md).

> **Operational protocol:** the main benchmark is dynamic cold-chain **pickup-to-depot**, not depot-to-customer delivery. Vehicles leave the depot empty; an order enters the cargo manifest at `service_finish`; quality is tracked until `return_arrival`; each vehicle then unloads and closes. The main setting uses homogeneous multi-compartment, shared-capacity, single-trip vehicles with no reload.

---

## Abstract

We investigate how MaskCO's masked generation can be extended from static combinatorial optimization to online dynamic cold-chain routing under a causal (non-anticipatory) information model. The planned DynMaskCO-CC method has three connected components:

1. **Causal Dynamic Masked Generation** — *Visibility-Gated Attention* (future orders are masked from the encoder with an attention bias of $-10^9$), *Visible-Only Normalization* (`coord_normalize_visible`, statistics computed only over revealed nodes), *Event-Driven Masking* (only affected route segments are reconstructed), *Frozen Prefix* (executed edges are immutable), and *Online Sequence Training* (progressive visibility reveal).
2. **Cold-chain-aware fleet–route masked recourse** — event-affected mutable decisions are masked and reconstructed as permutation-aware, full-fleet actions; accepted actions must pass route/fleet/cold-chain certificates while executed and committed decisions remain frozen.
3. **Counterfactual terminal-utility alignment** — masked action reconstruction is supervised by common-continuation rollout outcomes over distance, trace-level quality loss, and refrigeration energy instead of static route-structure imitation.

The first causal routing infrastructure exists. The authoritative trace-level cold-chain state and objective are the current C0 work item; the README does not claim that this planned component is already validated.

---

## Current Research Status

The old Phase-3c numbers (`14.77`, `15.45`, `14.08`, and the associated feasibility claims) were produced before the strict-online P0 audit and are retained only as historical results.

Under the corrected strict-online protocol on R1 EDoD=0.5:

| Method | Distance Cost | Complete | Status |
|---|---:|---:|---|
| OR-Tools-RH | 23.04 | 100% | strong baseline |
| JF1-H | 24.50 | 100% | current development baseline |
| JF2 canonical | 25.91 | 100% | frozen negative result |
| HFR-M0 g_only | 24.97 | 100% | grouping signal, Gate FAIL |
| HFR-M0 full | 29.95 | 99.2% | F3 / service Gate FAIL |

The historical `Complete` column covers customer service and depot return under the distance protocol. It does not yet include the v4 cargo-manifest and `delivered_to_depot` checks, which are part of C0.

HFR-M0 learned real structural signal (G AUROC 0.727 and improved route reconstruction loss), but that signal did not improve downstream online cost. This motivates changing the **MaskCO reconstruction target** from static structure to utility-preferred executable actions; cost-aware preference remains an internal alignment mechanism.

P0-R, P0-S, P0-A, and P0-U are complete. The next stage is C0, which must unify physical units, temperature-dependent quality, refrigeration energy, cold-chain snapshots, and the execution-trace evaluator. O0-D and O0-CC follow; M0 is only a frozen-representation probe, while M1 is the minimum complete DynMaskCO-CC method with event masking and iterative reconstruction.

See [`项目当前状态.md`](项目当前状态.md), the [`documentation index`](docs/README.md), the [`implementation blueprint`](docs/当前规划/代码实现蓝图.md), and the [`results index`](results/README.md).

---

## Installation

```bash
# 1. Parent MaskCO environment (JAX 0.5.0, Flax 0.10.4, Triton 3.1.0, PyTorch-CPU, NumPy 1.26.4)
cd .. && sh install.sh && cd lib && make && cd ..

# 2. CVRPTW / cold-chain C++ extension (EDD repair + TW-aware 2-opt + quality cost)
pip install pybind11
cd "C-VRP_Cold-chainVehicleRoutingProblem/scripts/lib" && make && cd ../..
```

Python dependencies for this sub-directory are listed in [`requirements.txt`](requirements.txt).

Set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` (training) or `0.5` (inference). GPU 0 is the default.

---

## Historical Reproduction Quick Start

The legacy DynMaskCO-v1 reproduction entry point is [`scripts/run_method_freeze.sh`](scripts/run_method_freeze.sh). It reproduces historical phases and is not the current C0/O0 development entry point:

```bash
# Smoke run (~10 min, 1 seed × 2K steps)
bash "C-VRP_Cold-chainVehicleRoutingProblem/scripts/run_method_freeze.sh" smoke

# Full Phase-3c training + evaluation (5 seeds)
bash "C-VRP_Cold-chainVehicleRoutingProblem/scripts/run_method_freeze.sh" phase3c

# Evaluate an existing checkpoint only
bash "C-VRP_Cold-chainVehicleRoutingProblem/scripts/run_method_freeze.sh" eval
```

### Step-by-step

```bash
# 3a. Generate DCC data (R1, 50 nodes, EDoD=0.5)
python -u "C-VRP_Cold-chainVehicleRoutingProblem/scripts/data/generate_coldchain_data.py" \
    --problem_size 50 --num_instances 1280 --type R1 --capacity 50 --edod 0.5 \
    --output <out_train.npz>

# 3b. Train (Phase 3c K=5 online-sequence)
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python -u "C-VRP_Cold-chainVehicleRoutingProblem/scripts/training/train_dynamic_cc.py" \
    --gpu_id 0 --num_nodes 50 --capacity 50 --model_config softcap_fn \
    --encoder_input_dim 7 --peak_lr 1e-3 --batch_size 64 --num_steps 50000 \
    --masking_mode spatio_temporal --online_seq_training --online_seq_steps 5 \
    --data <train.npz> --logdir <logdir> --savedir <savedir> \
    --optimizer_type adamw --weight_decay 1e-2

# 3c. Evaluate with the resource-beam decoder
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 \
python -u "C-VRP_Cold-chainVehicleRoutingProblem/scripts/decoding/cvrptw.py" \
    --capacity 50 --penalty 3. --data <test.npz> --ckpt <step50000.ckpt> \
    --keep_rate 0.3 --two_opt_steps 4 --batch_size 8 --runs 8 --cycles 40 \
    --sampling_steps 2 --seed 42 --threads_over_batches 1 \
    --enable_resource_decoder --beam_width 16 --enable_tw_aware_2opt_py
```

---

## Repository Structure

```
C-VRP_Cold-chainVehicleRoutingProblem/
├── scripts/
│   ├── models/       # TSPModel → CVRPModel → CVRPTWModel → ColdChainModel → DynamicColdChainModel
│   ├── data/         # Solomon CVRPTW + cold-chain/DCC generators and loaders
│   ├── training/     # train_cvrptw / train_coldchain / train_dynamic_cc + auto_train
│   ├── decoding/     # cvrptw.py (main decoder), resource_beam/mask, maskco_dynamic, thermal_state
│   ├── lib/          # C++ extension (EDD repair, TW-aware & quality-aware 2-opt)
│   ├── simulation/   # event-driven rolling-horizon simulator
│   ├── baselines/    # OR-Tools (clairvoyant + rolling-horizon), Solomon audit, EURO adapter
│   ├── analysis/     # constraint ablation, edge influence, mechanism/toy analysis
│   └── tests/        # no-future-leakage / cardinality / target-label leakage / data QC
├── docs/
│   ├── 当前规划/     # authoritative method, implementation, evaluation and progress docs
│   ├── 实验记录/     # evidence tracking
│   └── 论文参考/     # reference papers
├── data/             # datasets (regenerable; git-ignored)
├── ckpts/            # checkpoints (regenerable; git-ignored)
├── logs/             # training/eval logs (git-ignored)
└── archive/          # superseded code / scripts / historical docs
```

A detailed file index is in [`scripts/README.md`](scripts/README.md). Historical phases and superseded experiments are preserved under [`archive/`](archive/).

---

## Reproduction Notes

- **Two objective names are kept separate** — `distance_cost` is always pure travel distance; C0 introduces a separate `coldchain_cost` built from frozen distance/quality/energy scales and weights. Hard infeasibility is never mixed into either scalar (see [`评估口径.md`](docs/当前规划/评估口径.md)).
- **Authoritative checkpoints** (5-seed frozen + typed/typed-edge) live on the training server and are regenerable via `run_method_freeze.sh`; see [`docs/服务器说明.md`](docs/服务器说明.md).
- **Causality discipline**: use `coord_normalize_visible` (visible-node statistics), never the parent `coord_normalize` (which leaks future nodes).

## Citation

```bibtex
@misc{dynmaskco,
  title  = {{DynMaskCO}: Causal and Feasibility-Preserving Masked Generation for
            Dynamic Cold-Chain Vehicle Routing},
  author = {},
  year   = {2026},
  note   = {ICLR 2027 submission (in preparation)}
}
```

## License

To be determined. (The parent MaskCO repository retains its own license.)
