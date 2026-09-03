# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## ⚠️ 当前状态（2026-09-03）：论文主线锁定为 MaskCO → 动态冷链；当前 C0 冷链闭环

> **唯一状态入口**：[`项目当前状态.md`](项目当前状态.md)。本文件保留工程规则和历史背景；若研究进度或权威路径冲突，以该状态入口为准。

> **v4 运营契约**：主实验是 dynamic cold-chain pickup-to-depot，不是 depot-to-customer 配送。车辆空载出发，`service_finish` 取货入舱并开始品质计时，`return_arrival` 卸货并关闭；同构多温舱、共享总容量、单次行程、无 reload。C0 必须实现订单级 cargo manifest。

**核心证据链（R1 EDoD=0.5，strict-online）**：

| 方法 | `distance_cost` | complete |
|------|------|----------|
| OR-Tools-RH（OR-joint） | 23.04 | 100% |
| **JF1-H（heuristic joint）** | **24.50** | 100% |
| JF1-R（real logits joint） | 26.84 | 97.7% |
| model guarded (H2-G) | 27.10 | 100% |
| NN (H1) | 27.66 | 100% |
| model raw (H2) | 28.20 | 100% |
| JF1-S（shuffle joint） | 29.10 | 96.1% |
| model_shuffle (H3) | 30.19 | 100% |

- **R1.7 归因闭环**：H2-G < H1（−2.0%）、H3-G ≈ H1 → learned preference 提供全部 2.1% 收益（`local_first_divergence = 61.3%`）。
- **B0 三层**（256 train，paired）：`G_seq = −1.227%（sequencing 饱和）、G_fleet = 17.891%（fleet 主导）`（P0-R 修复后复核 2026-09-02）。
- **JF1 四路**：JF1-H（24.50，−11.4%）最好；JF1-R（26.84）比 JF1-H 差 9.5% 且 complete 掉 97.7% → 联合分配收益来自搜索空间拓扑，不是 learned preference（R4）。
- **JF2（exact-vehicle CE）No-Go**：M0-v2 Gate A FAIL（25.91 vs 24.50，110 恶化）→ 2×2 oracle 证明 partition×sequencing 交互（interaction **−1.7430**，P0-R 修复后复核，CI [−2.387, −1.104]，四路 100% complete）。
- **HFR-M0 实现完成 + Gate A FAIL**：训练端结构信号可学，但没有任何 service-equivalent 变体超过 JF1-H。严格结论是旧 structural target 未建立 downstream utility，不是所有 route signal 本质无用。

**v4 方向定论**：冻结 JF2/HFR 的旧结构监督，但不放弃 MaskCO。论文主线固定为 **DynMaskCO-CC：基于 MaskCO 的动态冷链效用对齐掩码重构**。Cost-aware preference 只作为内部监督机制；M0 frozen encoder + MLP 仅为表征探针，最终 M1 必须包含 event mask、masked action reconstruction 与 iterative refinement。当前先完成 C0（pickup manifest、单位、温度、品质、能耗、snapshot 与 trace evaluator），再做 O0-D/O0-CC。

**权威状态**：`项目当前状态.md` + `docs/当前规划/科研方法创新主控文档.md` + `docs/当前规划/代码实现蓝图.md` + `docs/当前规划/执行进度表.md`。

**本文档下方「Phase 0 Baseline 15.45 / 100% feasible」及 Phase A-D 路线图均为 P0 审计前旧数字，作废待重建。**

---

## Project Overview

**DynMaskCO** extends the MaskCO (ICLR 2026) masked-generation paradigm for neural combinatorial optimization from *static* routing to *causal, feasibility-preserving online optimization* for the **dynamic cold-chain vehicle routing problem (DCC-VRP)**.

This is an **isolated extension workspace** within the larger MaskCO repository. All DynMaskCO code lives in `C-VRP_Cold-chainVehicleRoutingProblem/`; the parent MaskCO sources are never modified.

Key characteristics:
- **Causal (non-anticipatory)**: Model cannot see future orders (visibility-gated attention)
- **Feasibility-preserving**: 100% TW/capacity feasibility via resource-aware beam search + 2-opt repair
- **Cold-chain specific**: Temperature classes, quality loss, time-window constraints
- **JAX/Flax**: All models use JAX for fast GPU inference and training

---

## Quick Start Commands

### Environment Setup
```bash
# Activate environment (server)
source /home/hzeng/envs/MASKCO_env/bin/activate

# Or create new environment
conda create -n maskco_env python=3.10
conda activate maskco_env
pip install -r requirements.txt
cd lib && make  # Build C++ extensions
```

### Training

**Main training script** (typed embedding with 5 seeds):
```bash
# Smoke test (1 seed, 2000 steps, ~10min)
bash scripts/run_typed_retrain.sh smoke

# Full training (5 seeds, 50K steps, ~5h on GPU)
bash scripts/run_typed_retrain.sh full

# With edge features variant
USE_EDGE=1 bash scripts/run_typed_retrain.sh full
```

**Training entry point**: `scripts/training/train_dynamic_cc.py`
- Model: `scripts/models/DynamicColdChainModel.py`
- Data loader: `scripts/data/ColdChainDataloader.py`

### Evaluation

**Phase 0 baseline evaluation** (9-cell × 5-seed = 45 runs):
```bash
bash scripts/phase0_freeze_baseline_adapted.sh
```

**Evaluate existing checkpoints**:
```bash
bash scripts/run_typed_retrain.sh eval
```

**Single model evaluation**:
```bash
python scripts/decoding/cvrptw.py \
  --ckpt ckpts/path/to/checkpoint.ckpt \
  --data data/baseline/50_node/test/dcc_50_r1_edod05_test.npz \
  --capacity 50 \
  --beam_width 16 \
  --batch_size 128 \
  --augment_level 0
```

### Testing

**Toy instance test** (8-node, quick verification):
```bash
python scripts/analysis/toy_instance_test.py \
  --ckpt ckpts/path/to/checkpoint.ckpt \
  --num_nodes 8 \
  --seed 42
```

**Constraint ablation** (verify TW/capacity handling):
```bash
bash scripts/run_constraint_ablation.sh
```

---

## Architecture Overview

### Model Hierarchy (5-layer inheritance)

```
BaseModel (MaskCO parent repo)
  ↓
CVRPTWModel (scripts/models/CVRPTWModel.py)
  ├── Adds: TW constraints, capacity, distance matrix
  ├── Encoder: Transformer over node features
  └── Decoder: Autoregressive masked generation
  ↓
ColdChainModel (scripts/models/ColdChainModel.py)
  ├── Adds: Temperature classes (3), quality loss physics
  ├── Node features: [x, y, demand, tw_start, tw_end, temp_class]
  └── Quality physics: thermal_state.py (arrhenius decay)
  ↓
DynamicColdChainModel (scripts/models/DynamicColdChainModel.py)
  ├── Adds: Causal masking, progressive revelation
  ├── Key: coord_normalize_visible (uses only revealed nodes)
  └── Visibility: future_mask prevents attention to unrevealed orders
  ↓
typed_v1 / typed_v1_edge (current baseline)
  ├── Adds: Type-specific embeddings (5 classes: depot/frozen/chilled/ambient/mixed)
  └── typed_v1_edge: + edge features (energy_mat from distance/time/TW-compatibility)
```

**Critical invariant**: Always use `coord_normalize_visible` for normalization in dynamic models (not `coord_normalize` which leaks future information).

### Data Flow

```
Training:
data/baseline/50_node/train/*.npz (9 domains × 1000 instances)
  ↓ [balanced sampling: P(type)=1/3, P(EDoD|type)=1/3]
ColdChainDataloader.py
  ↓ [event-driven masking, progressive revelation]
train_dynamic_cc.py
  ↓ [Phase 3c: online_seq K=5 training]
DynamicColdChainModel
  ↓ [greedy self-labeling or expert labels]
ckpts/p0_fix/typed_v1_edge/phase3c/seed{42,123,999,2025,2026}/step50000.ckpt

Inference:
data/baseline/50_node/test/*.npz (9-cell matrix)
  ↓
cvrptw.py (decoder entry)
  ↓ [beam_width=16, 2-opt repair]
resource_beam.py (feasibility-preserving beam search)
  ↓ [TW filter, capacity check, quality tracking]
results/phase0_baseline_freeze/matrix_summary.csv
```

### Key Modules

**Decoding** (`scripts/decoding/`):
- `cvrptw.py`: Main evaluation entry point
- `resource_beam.py`: Resource-aware beam search (capacity + TW feasibility)
- `maskco_dynamic.py`: Dynamic masking logic (causal, visible-only)
- `thermal_state.py`: Cold-chain physics (Arrhenius decay, quality loss)

**Training** (`scripts/training/`):
- `train_dynamic_cc.py`: Main training loop (Phase 3c online seq)

**Data** (`scripts/data/`):
- `ColdChainDataloader.py`: Event-driven data loading
- `generate_coldchain_data.py`: Dataset generation (Solomon-based)

**Baselines** (`scripts/baselines/`):
- `ortools_rolling_horizon.py`: OR-Tools rolling-horizon baseline

**Analysis** (`scripts/analysis/`):
- `constraint_ablation.py`: TW/capacity ablation experiments
- `edge_influence.py`: Edge harmful rate analysis
- `mechanism_analysis.py`: Interpretability tools

---

## Data Organization

### Directory Structure (Post-2026-08-23 Reorganization)

```
data/
├── baseline/              # Official baseline (Phase 0 frozen)
│   ├── 50_node/
│   │   ├── train/  (9)   # R1/C1/RC1 × EDoD 0.2/0.5/0.8
│   │   ├── val/    (9)   # Validation (model selection)
│   │   └── test/   (9)   # Test (final reporting only)
│   └── 100_node/test/ (3)
├── experimental/          # Ablation/transfer/quality experiments
├── archive/               # Deprecated datasets (pre-Phase 0)
└── expert_labels/         # Phase B expert distillation (future)
```

**Critical rules**:
- **Phase 0 baseline is frozen**: `data/baseline/50_node/test/` (9 files × 128 instances)
- **train/val/test strict separation**: 
  - Training: `baseline/50_node/train/`
  - Model selection / hyperparameter tuning: `baseline/50_node/val/`
  - Final reporting only: `baseline/50_node/test/` (run once after all decisions locked)
- **Never use `archive/` in active scripts**: Old paths like `data/p0_fix/` are deprecated

### Naming Convention

```
dcc_{size}_{type}_edod{XX}_{split}[_{variant}].npz
```

Examples:
- `dcc_50_r1_edod05_train.npz` (standard training, 50-node R1, EDoD=0.5)
- `dcc_50_r1_edod05_test_tempswap.npz` (temperature ablation)
- `dcc_100_r1_edod05_test.npz` (100-node cross-scale)

### Data Integrity

Checksum file: `data/baseline/DATA_MANIFEST.sha256` (30 files frozen on 2026-08-24)

Verify integrity:
```bash
cd data/baseline
sha256sum -c DATA_MANIFEST.sha256
```

---

## Evaluation Protocol

### Phase 0 Frozen Baseline (2026-08-24)

**Model**: `typed_v1_edge` (type-specific embeddings + edge features)

**Results** (9-cell × 5-seed = 45 runs):
- Overall mean: **15.45 ± 3.30**
- R1 EDoD=0.5: **19.22 ± 0.18**
- C1 EDoD=0.5: **11.38 ± 0.13**
- RC1 EDoD=0.5: **15.99 ± 0.05**
- Feasibility: **100%** (45/45 runs)

**Configuration**:
- Train seeds: [42, 123, 999, 2025, 2026]
- Decode seed: 42
- Beam width: 16
- Augment level: 0 (DynamicAugment has bug at level 2)
- Batch size: 128
- 2-opt steps: 100

### Metrics (from evaluation_contract.md)

**Primary metrics**:
1. **Service-first**: TW feasibility (100% required), capacity feasibility (100% required)
2. **Cost**: Total distance traveled (infeasible instances excluded, not penalized)
3. **Num unsalable**: Count of orders with quality_loss > 0.10 (cold-chain specific)

**Secondary metrics**:
- Gap_ref: `(J^π - J_ref) / J_ref` (vs anticipatory baseline)
- Conditional Recourse Regret: At event epoch e, compare online policy vs clairvoyant recourse from that point

**Reporting**:
- Always report 5-seed mean ± std
- Use paired t-test for statistical significance (p < 0.05)
- Report wall-clock time (ms), not just GPU forward pass

---

## Phase A-D Roadmap (`archive/docs/优化历程/v1/优化方案.md`) —— ⚠️ 已暂停（D-057，2026-08-26）

> **已作废**：Phase A-D（≤14.7 等目标基于旧 Phase 0 数字 15.45）。v4 当前路线为 strict-online P0 → C0 冷链闭环 → O0-D/O0-CC → M0 probe → M1 DynMaskCO-CC；以本文档顶部权威入口为准。

### Phase A: Edge-State Representation (6-8 weeks, P0)
**Goal**: ≤ 14.7 (−5% from Phase 0's 15.45)
- A1: 5D static pairwise edge features + EdgeBiasProjector (multi-head edge bias)
- A2: Dynamic resource context (h_current + h_depot)
- A3: Constraint-sparse dual attention
- A4: Final fusion (Edge + Resource + Dual)

### Phase B: Expert Distillation (4-6 weeks, P0)
**Goal**: ≤ 14.0 (−5% from Phase A)
- B1-a: OR-Tools frozen-prefix recourse labels
- B1-b: DAgger (student prefix + OR-Tools recourse)
- B2: Preference learning (pairwise comparison)
- B3: Neural LNS (learned large neighborhood search)

### Phase C: Stochastic Lookahead (6-8 weeks, P1)
- C1: Scenario generation (future order distributions)
- C2: Multi-objective optimization (cost/quality/robustness)

### Phase D: Scaling (4-6 weeks, P2)
- 100/200-node instances
- Real road network (OSRM integration)

---

## Common Development Patterns

### Adding a New Model Variant

1. **Create model class** in `scripts/models/`:
```python
from DynamicColdChainModel import DynamicColdChainModel

class MyNewModel(DynamicColdChainModel):
    def setup(self):
        super().setup()
        # Add new components
        self.my_new_module = ...
    
    def encode(self, x, visible_mask):
        # Use coord_normalize_visible, NOT coord_normalize
        x_norm = self.coord_normalize_visible(x, visible_mask)
        # ... rest of encoding
```

2. **Register in training script** (`scripts/training/train_dynamic_cc.py`):
```python
from MyNewModel import MyNewModel
# Update model instantiation
```

3. **Train with new config**:
```bash
# Modify run_typed_retrain.sh or create new script
MODEL_CONFIG="my_new_config" bash scripts/run_typed_retrain.sh smoke
```

### Adding Edge Features

Example from `typed_v1_edge`:
```python
# In model setup
if use_edge_feat:
    self.edge_weight_proj = nn.Dense(embed_dim)

# In encoder
if use_edge_feat:
    energy_mat = compute_energy_matrix(coords, tw_start, tw_end, dist_mat)
    edge_feats = self.edge_weight_proj(energy_mat)
    # Add to attention logits
```

### Causality Debugging

If model sees future orders (violates causality):
1. Check `future_mask` is applied in attention: `attn_mask = combine_masks(padding_mask, future_mask)`
2. Verify normalization uses `coord_normalize_visible`, not `coord_normalize`
3. Check data loader: `reveal_time` must be respected in masking

### Feasibility Debugging

If TW/capacity violations occur:
1. Enable filters: `--enable_tw_filter --enable_tw_repair_edd`
2. Check resource beam: `resource_beam.py` should reject infeasible actions
3. Verify 2-opt repair: `--enable_tw_aware_2opt_py --two_opt_steps 100`

---

## Documentation Structure

All documentation in `docs/`:

**Core references**:
- `项目当前状态.md`: current status and the only navigation authority
- `docs/当前规划/评估口径.md`: Metrics, statistical testing, reporting standards
- `docs/当前规划/总索引.md`: Quick reference for current documents
- `archive/docs/优化历程/v1/`: Historical Phase A-D roadmap and progress

**Technical specs**:
- `docs/当前规划/实验细节.md`: Complete formulas, architecture, hyperparameters
- `docs/当前规划/理论形式化.md`: Theorems, proofs (causality, feasibility)
- `docs/当前规划/基线对比.md`: Baseline comparisons (OR-Tools, RRNCO, etc.)

**Data**:
- `data/数据集组织规范.md`: Dataset structure and naming conventions
- `data/README.md`: Quick reference

---

## Git Workflow

**Current baseline**: `typed_v1_edge` (Phase 0 frozen 2026-08-24)

**Checkpoints**: `ckpts/p0_fix/typed_v1_edge/phase3c/seed{42,123,999,2025,2026}/step50000.ckpt`

**Branch strategy** (for Phase A-D):
- Create feature branch: `feature/phase-a-edge-state-v2.1`
- Commit frequently with descriptive messages
- Never commit large binaries (checkpoints, data files) — use `.gitignore`

**Data version control**:
- Checksum: `data/baseline/DATA_MANIFEST.sha256`
- Metadata: `results/phase0_baseline_freeze/baseline_frozen.json` (Git commit, timestamp, config)

---

## Key Constraints and Invariants

### Causality (Non-Anticipatory)
- **Never** use `coord_normalize` in dynamic models (uses all nodes including future)
- **Always** use `coord_normalize_visible` (uses only revealed nodes)
- **Verify**: `future_mask` applied in all attention layers

### Feasibility
- **100% TW/capacity feasibility required** (not negotiable)
- Infeasible solutions are excluded from cost metrics (not penalized)
- Use `--enable_resource_decoder --beam_width 16` for beam search with feasibility

### Evaluation
- **Never use test set** during model development / hyperparameter tuning
- Use `val/` for all Go/No-Go decisions, model selection
- Only run `test/` once after Phase is completely frozen

### Data Paths
- **Never reference** `data/p0_fix/` (deprecated, moved to `archive/`)
- **Always use** `data/baseline/` for official datasets
- Scripts should use relative paths from project root

---

## Troubleshooting

### Common Issues

**1. DynamicAugment ValueError**
- **Symptom**: `ValueError` in `decoding/utils.py` line 84 when `augment_level=2`
- **Workaround**: Use `augment_level=0` (Phase 0 configuration)
- **Status**: Bug under investigation

**2. Slow evaluation (batch_size=8)**
- **Fix**: Increase `batch_size=128` for 16× speedup
- **Trade-off**: Slightly higher memory usage

**3. GPU not used**
- **Fix**: Add `CUDA_VISIBLE_DEVICES=0` before python command
- **Check**: `nvidia-smi` should show python process

**4. Checkpoint not found**
- **Check path**: Checkpoints moved from `ckpts/p0_fix/` (not deleted, just moved in doc references)
- **Verify**: `ls -lh ckpts/p0_fix/typed_v1_edge/phase3c/seed*/step50000.ckpt`

**5. Data file not found**
- **Old path**: `data/p0_fix/dcc_50_r1_edod05_test.npz` (deprecated)
- **New path**: `data/baseline/50_node/test/dcc_50_r1_edod05_test.npz`
- **Migration**: Run `bash scripts/migrate_data_paths.sh` to update old scripts

---

## Performance Expectations

### Training Time (GPU)
- Smoke test (1 seed, 2K steps): ~10 minutes
- Full training (5 seeds, 50K steps): ~5 hours
- Phase 3c online seq K=5: adds ~20% overhead vs static training

### Inference Time
- Single instance (50-node, beam=16): ~2-3 seconds
- Batch of 128 instances: ~3 minutes
- Full 9-cell evaluation (45 runs): ~3-4 hours on GPU

### Model Size
- typed_v1_edge checkpoint: ~219 MB per seed
- 5-seed ensemble: ~1.1 GB total

---

## References

**Paper**: MaskCO (ICLR 2026) - https://openreview.net/forum?id=psUjNnLhl9

**Related work**:
- RRNCO (ICLR 2026): Route reuse in NCO
- CaDA: Constraint-aware dual attention
- OR-Tools: Google's optimization library (rolling horizon baseline)

**Server**: Training server details in `docs/服务器说明.md`

---

## Contact and Support

For questions about:
- **Method / architecture**: See `docs/当前规划/实验细节.md` (complete formulas)
- **Evaluation protocol**: See `docs/当前规划/评估口径.md` (metrics definitions)
- **Roadmap**: See `docs/当前规划/科研方法创新主控文档.md`
- **Progress**: See `docs/当前规划/执行进度表.md`
- **Data**: See `data/数据集组织规范.md` (dataset structure)
