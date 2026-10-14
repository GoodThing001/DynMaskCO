#!/bin/bash
# 只读：补 MVMoE md5、CO-enriched 官方模型目录、MAPT 训练/checkpoint 状态
set -u
cd /home/hzeng/project/MASKCO-Main || exit 1
echo "== MVMoE ckpt md5（RESULTS.md 记录 md5 977ba947…）=="
md5sum CC_Compare/MVMoE/pretrained/pomo_vrptw_n100/epoch-5000.pt
echo "== CO-enriched training/models 目录 =="
ls -la CC_Compare/CO-enriched-ML/training/models/ 2>&1 | head -12
echo "== CO-enriched A-v1 权重 sha256 =="
sha256sum CC_Compare/CO-enriched-ML/a1_accept_v1/artifacts_a1/a1_weights.npz \
         CC_Compare/CO-enriched-ML/a1_accept_v1/artifacts/official_weights.npz 2>&1 | cut -c1-80
echo "== MAPT RESULTS.md §3/§5/§7 是否已回填（看标题行）=="
grep -n "回填" CC_Compare/MAPT_code/a1_accept_v1/RESULTS.md | head -8
echo "== MAPT checkpoint（1003 侧）=="
ls -la CC_Compare/MAPT_code/a1_accept_v1/*.pt 2>&1
