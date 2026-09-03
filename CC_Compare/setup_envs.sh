#!/usr/bin/env bash
# =============================================================================
# CC_Compare 三方法环境搭建（服务器, RTX 5090 Blackwell = sm_120）
#
# 关键约束：
#   - RTX 5090 (sm_120) 需要 torch >= 2.7（首个支持 Blackwell 的版本）。
#   - CaDA/RRNCO/PIP 原始 requirements 都 pin 了旧 torch（2.0.x / 2.6 CPU），
#     必须升级到 torch>=2.7，并处理 torchrl/tensordict 新旧 API。
#
# 用法：
#   bash CC_Compare/setup_envs.sh cc       # ★推荐：CaDA+RRNCO 共用一个环境（PIP 不跑，跳过）
#   bash CC_Compare/setup_envs.sh clone <base_env>  # ★5090 推荐：从已有 CUDA torch 环境克隆后只装增量
#   bash CC_Compare/setup_envs.sh cada     # 只建 CaDA 环境
#   bash CC_Compare/setup_envs.sh rrnco    # 只建 RRNCO 环境
#   bash CC_Compare/setup_envs.sh pip      # 只建 PIP 环境（单车辆，最终不用于对比）
#   bash CC_Compare/setup_envs.sh all      # 全部
#
# 为什么不复用 MASKCO_env：MASKCO_env 的 torch 是 CPU 版（install.sh: torch==2.6.0 --index-url
#   .../whl/cpu），而 CaDA/RRNCO 都是 PyTorch-GPU 模型，必须 CUDA torch；且在该 env 里再装
#   rl4co/torchrl/tensordict 会污染 JAX 0.5.0 + tensorflow_cpu 2.19 的冻结依赖（numpy 等共享
#   传递依赖），破坏 DynMaskCO 权威结果的可复现性。CaDA/RRNCO 底层都是 torch CUDA +
#   torchrl/tensordict，天然可共用一个环境，故 `cc` 是推荐的单环境方案。
#
# clone 目标：从服务器上「已在 5090 跑通 CUDA torch ≥2.7」的母环境克隆，克隆后再装增量
#   （rl4co 会拉 torchrl/tensordict）。注意：被 clone 的 base 必须是 CUDA torch ≥2.7（Blackwell
#   sm_120 首个支持版本）；若是 CPU torch 或 torch<2.7，克隆也救不了，仍需换 torch wheel。
#
# 依赖选择说明：
#   torch 用 cu128 wheel 源（torch 2.7+ 支持 sm_120）。服务器已装 CUDA 13.1，
#   cu128 的 torch 能在其上运行（driver 向后兼容）；如遇问题可换 cu130。
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"          # MASKCO-Main/
CC="$ROOT/CC_Compare"
PYTHON=3.11
TORCH_INDEX="https://download.pytorch.org/whl/cu128"

TARGET="${1:-all}"

info()  { echo -e "\n\033[1;34m==>\033[0m $*"; }
warn()  { echo -e "\033[1;33m[!]\033[0m $*"; }

# ---- CaDA ----
setup_cada() {
  info "CaDA 环境 (cada)"
  conda create -y -n cada python=$PYTHON
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate cada
  # 核心依赖（不用 CaDA 的 requirements.txt —— 那是作者 pip freeze 的脏快照，
  # 内含 CUDA 11.7 torch 与大量 @file:// 本地路径，不可直接装）。
  pip install --upgrade pip
  pip install torch torchvision --index-url "$TORCH_INDEX"
  pip install torchrl tensordict einops numpy pyyaml tqdm
  # torchrl 新旧 API 兼容补丁已内置在 dcc_vrp/_torchrl_compat.py，无需额外处理。
  warn "CaDA 需确认 checkpoint：50/result/*/checkpoint-300.pt 是否存在（官方权重）"
  python - <<'PY'
import torch, torchrl, tensordict
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| torchrl", torchrl.__version__)
PY
}

# ---- RRNCO ----
setup_rrnco() {
  info "RRNCO 环境 (rrnco)"
  conda create -y -n rrnco python=$PYTHON
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate rrnco
  pip install --upgrade pip
  pip install torch torchvision --index-url "$TORCH_INDEX"
  # 用项目 pyproject 安装（拉取 rl4co>=0.5.1 及 torchrl/tensordict 依赖）
  pip install -e "$CC/RRNCO"
  # HuggingFace 下载器（checkpoint/数据）
  pip install "huggingface-hub[cli]>=0.31.2"
  warn "RRNCO checkpoint 需下载：cd CC_Compare/RRNCO && python scripts/download_hf.py --no-data"
  python - <<'PY'
import torch, rl4co, torchrl
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| rl4co", rl4co.__version__)
PY
}

# ---- CaDA + RRNCO 共用环境（推荐）----
setup_cc() {
  info "CaDA + RRNCO 共用环境 (cc_compare)"
  conda create -y -n cc_compare python=$PYTHON
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate cc_compare
  pip install --upgrade pip
  pip install torch torchvision --index-url "$TORCH_INDEX"
  # rl4co 会拉取 torchrl/tensordict；再补 CaDA 的纯 Python 依赖（einops/pyyaml/tqdm）
  pip install -e "$CC/RRNCO"
  pip install einops pyyaml tqdm
  # HuggingFace 下载器（RRNCO checkpoint）
  pip install "huggingface-hub[cli]>=0.31.2"
  warn "CaDA checkpoint: 50/result/*/checkpoint-300.pt（手动下）; RRNCO: cd CC_Compare/RRNCO && python scripts/download_hf.py --no-data"
  python - <<'PY'
import torch, rl4co, torchrl, tensordict, einops, yaml
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| rl4co", rl4co.__version__)
PY
}

# ---- 从已有 CUDA torch 环境克隆（5090 推荐：clone-then-prune 工作流）----
setup_clone() {
  local BASE="${1:-}"
  if [ -z "$BASE" ]; then
    echo "用法: $0 clone <base_env_name>   （base 必须是已在 5090 跑通 CUDA torch>=2.7 的母环境）" >&2
    exit 1
  fi
  info "从 '$BASE' 克隆 cc_compare 环境"
  conda create -y -n cc_compare --clone "$BASE"
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate cc_compare
  pip install --upgrade pip
  # 只装增量（base 已含 CUDA torch>=2.7）：rl4co 会拉 torchrl/tensordict；再补 CaDA 纯 Python 依赖
  pip install -e "$CC/RRNCO"
  pip install einops pyyaml tqdm
  pip install "huggingface-hub[cli]>=0.31.2"
  # 硬校验：必须是 CUDA torch（否则 5090 跑不动，需要换 torch wheel，克隆白搭）
  python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda)
assert torch.version.cuda, "CUDA torch 缺失：base 是 CPU torch 或 torch<2.7，需换 wheel 而非克隆"
assert torch.cuda.is_available(), "torch.cuda.is_available()=False：驱动/CUDA 不匹配"
import rl4co, torchrl, tensordict, einops, yaml
print("rl4co", rl4co.__version__, "| torchrl", torchrl.__version__, "| tensordict", tensordict.__version__)
PY
}

# ---- PIP-constraint ----
setup_pip() {
  info "PIP-constraint 环境 (pip)"
  conda create -y -n pip python=$PYTHON
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate pip
  pip install --upgrade pip
  pip install torch torchvision --index-url "$TORCH_INDEX"
  # PIP README 要求（跳过 tensorflow/wandb 等重依赖，评估不需要）
  pip install matplotlib tqdm pytz scikit-learn tensorboard_logger pandas
  warn "PIP 是单车辆 TSPTW/TSPDL，无法适配 DCC-VRP，见 PIP-constraint/dcc_vrp/README.md"
  python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.version.cuda)
PY
}

case "$TARGET" in
  cc)     setup_cc ;;
  clone)  setup_clone "${2:-}" ;;
  cada)   setup_cada ;;
  rrnco)  setup_rrnco ;;
  pip)    setup_pip ;;
  all)    setup_cada; setup_rrnco; setup_pip ;;
  *) echo "用法: $0 {cc|clone <base>|cada|rrnco|pip|all}"; exit 1 ;;
esac

info "完成。下一步（逐方法核对）："
echo "  CaDA : bash $ROOT/CC_Compare/CaDA/dcc_vrp/README.md 里的 run 命令"
echo "  RRNCO: 先下载 checkpoint，再跑 run_dcc.py"
echo "  PIP  : 跳过（单车辆，不适用）"
