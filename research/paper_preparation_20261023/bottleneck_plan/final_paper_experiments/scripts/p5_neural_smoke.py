# -*- coding: utf-8 -*-
"""P5 神经驱动冒烟（远端）：冻结 f0_r0 模型 + 标准特征前向验证。"""
import json
import shlex
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = next(p for p in HERE.parents if (p / "CC_Compare/configs/methods.json").exists())
sys.path.insert(0, str(ROOT / "C-VRP_Cold-chainVehicleRoutingProblem/tools"))
import action_calibration_t2_resume as transport

PYTHON = "/home/hzeng/envs/MASKCO_env/bin/python"

SCRIPT = r'''
import json, math, os, sys, time
from pathlib import Path
import numpy as np
P5 = Path("/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5")
sys.path.insert(0, str(P5))
PROJECT = Path("/home/hzeng/project/MASKCO-Main")
origin = PROJECT / "_maskco_cross_dataset_20261119"
sys.path.insert(0, str(origin))
import compare
access, core, source, locked = compare.setup("SLOT")
# 冻结模型加载（c205 仅作作用域案例；权重为 f0_r0）
start = time.perf_counter()
policy, scope = compare.load("SLOT", access, core, source, locked, "c205")
load_s = time.perf_counter() - start
print("model loaded in %.2fs" % load_s, flush=True)
# 标准 CVRP 实例 X-n101-k25
from standard_vrp.core.standard_evaluator import CVRPInstance
from standard_vrp.core.standard_features import StandardFeatureBuilder, window_indices
text = (P5/"cvrplib/instances/X-n101-k25.vrp").read_text(encoding="utf-8")
inst = CVRPInstance.parse_vrp(text)
fb = StandardFeatureBuilder(inst, "cvrp")
# 共同初解（NN 见证）→ 部分路线
from standard_vrp.solver.standard_pool_search import nn_witness
routes = nn_witness(inst)
removed = set(np.random.default_rng(1).choice(np.arange(1, inst.dimension+1), size=40, replace=False).tolist())
partial = tuple(tuple(i for i in r if i not in removed) for r in routes)
partial = tuple(r for r in partial if r)
batch = fb.build(partial, removed)
print("features:", batch["features"].shape, "adjacency:", batch["adjacency"].shape, flush=True)
# 冻结前向（CPU JAX，scope 内）
from profit_neural import predict
with access.restricted_scope(scope):
    t0 = time.perf_counter()
    logits = predict(policy, batch)
    fwd_s = time.perf_counter() - t0
print("forward: %.3fs  logits shape %s" % (fwd_s, logits.shape), flush=True)
shifted = logits - logits.max(axis=1, keepdims=True)
heat = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
print("heat finite:", bool(np.all(np.isfinite(heat))), "row sums ~0:", float(np.abs(heat.sum(axis=1)).max()), flush=True)
# 被遮客的边分排名样例
rids = [i for i in removed if i <= inst.dimension]
top = {int(i): int(np.argsort(-heat[i])[0]) for i in rids[:8]}
print("sample masked-customer top partner:", top, flush=True)
json.dump(dict(load_s=load_s, fwd_s=fwd_s, shape=list(logits.shape), finite=bool(np.all(np.isfinite(heat)))),
          open(str(P5/"neural_smoke.json"), "w"), indent=2)
print("SMOKE DONE", flush=True)
'''

def main():
    transport.record = lambda *a, **k: None
    client, _ = transport.connect(ROOT / ".vscode/sftp.json")
    try:
        with client.open_sftp() as s:
            with s.open("/home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5/p5_neural_smoke.py", "w") as f:
                f.write(SCRIPT)
        out = transport.remote_exec(client, "cd /home/hzeng/project/MASKCO-Main/_maskco_final_paper_p5 && tmux new-session -d -s p5_neural_smoke '" +
                                    "JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= " + PYTHON + " -B p5_neural_smoke.py > p5_neural_smoke.log 2>&1'")
        print("tmux:", out)
    finally:
        client.close()


if __name__ == "__main__":
    main()
