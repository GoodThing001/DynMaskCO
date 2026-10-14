#!/bin/bash
# A-v1 正式批次 v2 · GPU 腿 —— **已整体迁移到 3080**（2026-10-13，用户指示「把 GPU 腿整体搬到 3080」）。
#
# 为什么迁移：1003 的两张卡被他人 20 个进程长期占 ~89%（实测 GPU0/1 各 89% 利用、
# 12.3/11.7 GB 已用），而 3080（RTX 3080 Ti）自 MAPT 训练结束后空闲。
#
# 为什么可以迁移（身份前提已实测）：把 1003 的共享计算文件同步到 3080 后，3080 用当前代码重算出
# **逐字节相同**的 contract_hash=e25a12104b20c3ea… / data_sha256=8767a03d787c0e0f… /
# data_meta=72a805fd5bc0b2b7…（工具 CC_Compare/tools/verify_contract_identity.py 可在 3080 复验）
# ⇒ 与 1003 新基线配对合法。
#
# 本脚本在 1003 上**只写完成标记**，真实 GPU 腿在 3080 上执行
# （输出落在 3080 的 results/a1_ext_formal2_3080/，完成后回传到 1003 同名 results/a1_ext_formal2/<臂>/）。
# 这样编排的等待逻辑不变，且不会与 3080 抢同一输出目录。
set -u
RUN=/tmp/xformal_v2
mkdir -p "$RUN"
MARK="$RUN/gpu.done"
{
  echo "GPU_LEG_MOVED_TO_3080 $(date -u)"
  echo "见 CC_Compare/tools/_gpu_leg_3080_*.cmd 与 README §13"
  echo ALLDONE
} > "$MARK"
cat "$MARK"
