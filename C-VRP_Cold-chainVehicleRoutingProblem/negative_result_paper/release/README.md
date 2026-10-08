# B 论文复现材料发布包（release bundle）

> 本目录是论文的**待发布复现材料索引**，尚未上传公开位置，也不包含大文件本体或 `docs/`。不能仅凭本目录完成从零复现；上传与独立执行核验后才能称为公开复现包。

## 1. 身份清单（SHA256 全量见 `SHA256SUMS.txt`）

| 文件 | 大小 | SHA256 前 16 |
|---|---|---|
| cvrp100.ckpt（预训练 MaskCO CVRP 解码器） | 305 MB | 03615b08 |
| mpre_reinforce_s42_clean/model.ckpt | 89.6 MB | f5acf83f |
| mpre_reinforce_s43_clean/model.ckpt | 89.6 MB | e5364075 |
| mpre_reinforce_s44_clean/model.ckpt | 89.6 MB | f0b276e1 |
| dcc_50_r1_edod05_train_teacher.npz（TRAIN-64, seed 9701） | ~1 MB | c70258c4 |
| dcc_50_r1_edod05_cal_teacher.npz（CAL-16, 9702） | ~1 MB | c35fa683 |
| dcc_50_r1_edod05_dev_check_teacher.npz（DEV-CHECK-16, 9703） | ~1 MB | 66573ddc |
| dcc_50_r1_edod05_heldout.npz（留出-16, 9704） | 420 KB | 57e46022 |
| dcc_50_c1_edod05_confirm16.npz（C1 确认-16, 9751） | 423 KB | 1ff4f160 |
| results/o0cc/scale_v2/objective_profile.json（o0cc-pilot-devmean-equal-v2） | KB 级 | 63e56212 |
| v1 模块 ×6（`v1_modules/`） | KB 级 | 见 SUMS |

**训练身份**：`results/m0_scale/mpre_reinforce_s42_clean/training_identity.json` 记录 12 个关键源文件 SHA + 训练 summary + state_manifest（s42）。**已知边界（如实声明）**：
- `mpre_trained.py` 训练时版本（SHA `e88c7d0d…`）源码已丢失，本包使用当前版（`c1935274…`）；现有审计记载其加载归档 checkpoint 后逐位复现了固定状态评估数字。**完整评估链仍需外部干净环境验收**；重新训练不能承诺逐位复现。
- s43/s44 未生成 `training_identity.json`（训练 summary/state_manifest 均在）。
- step5 闭环的 `instances/*.json` 逐实例日志未保留；`per_instance.csv` + `summary.json` 足以重建主表。

## 2. 复现步骤（本地，Windows/Python 3.12 或 Linux/Python 3.10，JAX+Flax+NumPy+matplotlib+Pillow）

```bash
# 0) 从 §3 位置下载 10 个大文件，按上表路径放置并核对 SHA256

# 1) 建立独立的 v1 shadow 环境（当前工作树默认物理含 precool/昼夜扩展）
#    固定状态：使用本目录 v1_modules/ 中全部 6 个文件；其中
#    coldchain_state.py/coldchain_contract.py 放入 scripts/coldchain/，
#    strict_online_env.py/dynmaskco_cc_context.py/cc_lns_replanner.py/
#    mtrained_replanner.py 放入 scripts/simulation/。
#    闭环：按《端到端复现审计》的原运行身份，只 pin v1 的
#    coldchain_state.py/coldchain_contract.py，其余模块使用该次闭环记录的版本。
#    两种运行身份不同；发布前须提供可执行的独立环境与逐文件 SHA 核验，
#    不要在当前主工作树上直接覆盖模块。

# 2) 固定状态主表（示例：seed42 CAL）
python scripts/evaluation/run_fixed_state_quality.py \
    --data data/m0_scale/dcc_50_r1_edod05_cal_teacher.npz \
    --cvrp-ckpt ../MASKCO_code/ckpts/cvrp100.ckpt \
    --model-ckpt results/m0_scale/mpre_reinforce_s42_clean/model.ckpt \
    --objective-profile results/o0cc/scale_v2/objective_profile.json \
    --split cal --out results/m0_scale/fsq_clean_cal

# 3) 配对分析（candidate/accepted 两列，实例聚类 bootstrap）
python scripts/evaluation/analyze_fixed_state.py \
    --per-state results/m0_scale/fsq_clean_cal/per_state.json --out analysis.json

# 4) 闭环（留出 16 实例，每次重规划调用预算 4.0s；服务器 GPU）
python scripts/evaluation/run_step5_gate.py \
    --data data/heldout/dcc_50_r1_edod05_heldout.npz \
    --cvrp-ckpt ../MASKCO_code/ckpts/cvrp100.ckpt \
    --model-ckpt results/m0_scale/mpre_reinforce_s42_clean/model.ckpt \
    --objective coldchain --objective-profile results/o0cc/scale_v2/objective_profile.json \
    --budget 4.0 --max-instances 16 --out results/m0_scale/step5_heldout_s42

# 5) 诊断链（纯 NumPy；脚本在 negative_result_paper/scripts/，逐条见其 docstring）
#    run_headroom_census / run_terminal_headroom / run_causal_headroom /
#    run_value_predictability / run_perishability_* / run_execution_trace /
#    run_future_sampling_oracle / analyze_complementarity

# 6) 三张图
cd negative_result_paper && python scripts/make_figures.py
```

**依赖**：jax 0.5–0.6、flax 0.10.4、optax、numpy、matplotlib、Pillow；服务器 GPU 需 `jax-cuda12-plugin`（flax 必须 0.10.4，0.10.7 破坏 SwiGLU）。

## 3. 大文件上传目标（发布时填写）

| 文件 | 建议上传位置 |
|---|---|
| cvrp100.ckpt | Zenodo 记录 1（附原 MaskCO Google Drive 链接 https://drive.google.com/drive/folders/1Y9kN7H5qvlsgbnOKih6hpkpcF2MHUreI） |
| 3× model.ckpt + 5× npz + profile | Zenodo 记录 1（同一 record 分文件，含 C1 确认数据） |
| 本目录（README+SUMS+scripts+results JSON）及 `negative_result_paper/docs/` | 随论文仓库 / Zenodo 记录 2 |

**Zenodo 上传步骤（保留 DOI 草稿，投稿时解禁）**：
1. 登录 zenodo.org → New upload → 拖入 §1 的 10 个大文件 + 本目录及论文文档；
2. 上传完成后逐文件分别核对同一种算法的摘要值（Zenodo 页面 MD5 对本地 MD5；本地 SHA256 对本清单 SHA256）；
3. 填写 metadata（标题=论文名 + “reproduction package”，类型 dataset，license 建议 CC-BY-4.0 或 MIT，related identifier 关联论文）；
4. 保存为 draft → 拿到预留 DOI → 把 DOI 填回本文件与论文初稿 §5；
5. 论文被录用后 Publish。

**大文件当前所在地（服务器，未公开前从这取）**：`/home/hzeng/project/MASKCO-Main/` 下 `MASKCO_code/ckpts/cvrp100.ckpt`、`C-VRP_Cold-chainVehicleRoutingProblem/results/m0_scale/mpre_reinforce_s{42,43,44}_clean/model.ckpt`、`C-VRP_Cold-chainVehicleRoutingProblem/data/{m0_scale,heldout}/*.npz`、`C-VRP_Cold-chainVehicleRoutingProblem/results/o0cc/scale_v2/objective_profile.json`。

上传后把访问 URL 回填本文件 §3 与论文初稿 §5。

## 4. 与论文的对应

- **统一 v1 主表**：`results/09_unified_v1/`（本目录已含 9 个 JSON）——表 1/表 2 的唯一机器可读来源，由 2026-09-25 全量 v1 重跑 + `analyze_fixed_state.py` 生成。
- **C1 独立确认**：`results/10_confirm_c1/`（本目录已含 5 个 JSON）+ 数据 `data/confirm_c1/dcc_50_c1_edod05_confirm16.npz`（SHA 1ff4f160…，root seed 9751）——表 4/表 5 的数据来源。预注册的三项联合成功判据**未通过**，请连同原协议和结果一起公开。
- 主表（表 1/2）与诊断链数字的逐项复现核对见《端到端复现审计》（docs/01_论文/）。
- 论文文本与图注见《论文完整初稿_2026-09-25》（docs/01_论文/）；三张图由 `make_figures.py` 从本目录数据生成。
- 全部结论只针对声明的仿真协议（50 节点 / EDoD=0.5 / pickup-to-depot；几何覆盖 R1 与 C1 两档），不涉及实车或真实食品数据。
