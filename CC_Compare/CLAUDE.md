# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 用途与边界

`CC_Compare/` 是对比 baseline 工作区（独立于 `C-VRP_Cold-chainVehicleRoutingProblem/` 与 `MASKCO_code/`，后者绝不修改）。所有外部方法在统一 **strict-online 协议** 下接入项目的 `StrictOnlineEnv` 事件引擎，最终由项目 C0 evaluator 重放并报告 D/Q/E/J_CC——**不信任任何方法自报指标**。

三份权威 README（先读）：
- `CC_Compare/README.md` — 方法清单、下载/冻结状态
- `common/README.md` — 公共 baseline contract 架构与口径
- `PyVRP/dcc_vrp/README.md` — PyVRP-RH-D（已冻结）完整说明

## 核心架构（跨文件才能看懂的部分）

**公共合同层 `common/`**（所有 baseline 共用，纯 NumPy，不依赖任何方法包）：

```
adapter（方法侧）          bridge（契约层）            runner（执行+记录层）
ExternalReplanner.propose  → BridgeReplanner        → RecordingEnv(StrictOnlineEnv)
  (DecisionView)            写保护+proposal校验      事件推进/commit/审计/落盘
NativeReplanner.plan      → NativeBridgeReplanner   → 同 runner（无 view/proposal）
  (env/vehicles 直写)       同款写保护
```

- `_bootstrap.py` — 路径引导（把项目 scripts 各子目录插进 sys.path，调用 `project_paths.validate_layout()`）；所有模块先 import 它
- `method_adapter.py` — `DecisionView`（结构隔离未来信息：adapter 拿不到 env/dataset/future 客户）+ `PlanProposal` + 两种 bridge（写 committed/非 replan 车/状态字段/环境数组 → 抛 `ContractViolation`）；Native bridge 另有**环境不可变 fingerprint**（实例数组 + capacity/tw_speed）与条件式 repair 属性透传（无 repair 层的 native 方法不暴露，_eval 不崩溃）
- `strict_online_runner.py` — `run_instance`/`run_batch`：RecordingEnv 捕获 commit 动作；事件/动作记录装配（execution 层 COMMIT/CLOSE/WAIT + plan_diff 层 KEEP/DEFER/INSERT/NEW_ROUTE）；未来泄漏静态检查；`code_hash` 绑定计算链（逻辑相对路径 `common/`、`adapter/`、`project/` 命名空间）
- `ownership_audit.py` — 公共审计（duplicate/committed_and_suffix/served_in_plan/future_in_plan 等）；`record_validation.py` — NaN/Inf/缺字段/交叉一致性拒绝
- 关键语义：**可变池 = visible unserved − 非 replan 车辆 committed_next − 非 replan 车辆完整 frozen tail**（B1.1 P0 修复）；无 replan 事件 pool/protected 为 None（分区校验跳过）；有 repair 层的原生 adapter 输出 `repair_ownership_violations`/`repair_terminal_unresolved` 独立字段

**已冻结方法**：
- `PyVRP/dcc_vrp/` — **PyVRP-RH-D**（protocol revision 1 + identity manifest revision 4）。冻结配置见 `FROZEN_CONFIG.json`（MaxIterations=300、seed=0、无 warm start/fallback、distance 目标 + C0 重放、动态 coverage prize、界限 penalty、WAIT/CLOSE 规则）。身份链单向绑定：`SOURCE_MANIFEST.json`（identity rev4，生成器 `tools/gen_source_manifest.py` 可重跑复验）← `FROZEN_CONFIG.json` ← `FREEZE_SEAL.json`（无循环哈希）。**任何配置变化必须升 revision**
- `OR-Tools/dcc_vrp/` — **OR-Tools-RH-D**（protocol revision 1 + identity manifest revision 1；OR7 预算扫描选定 `solution_limit=30`，九 cell 等权 pure distance 17.3956）。三层身份（compute/control/analysis）+ `SOURCE_MANIFEST.json` ← `FROZEN_CONFIG.json` ← `FREEZE_SEAL.json`（seal revision 3 含证据包）。核验 `verify_freeze.py` + `verify_or7_freeze.py`。候选预算选择器 `or7_selector.py`（预注册 0.5%/1% 等价阈值 + bootstrap + 追加 300）
- `internal/dcc_vrp/jf1hf_adapter.py` — JF1-H-F 经 NativeReplanner 接入（项目 `make_continuation()` 零改动包装）

**进行中**：
- `RRNCO/dcc_rh_v4/` — **RRNCO-Ordering-RH-D**（R0.5 Stage A 离线骨架完成，Stage B 真模型验证待 DEV-GATE 结束后上服务器）。判定 `PROVISIONAL_GO_RRNCO_ORDERING_RH_D`；R1 三层身份冻结未授权。见 `RRNCO/dcc_rh_v4/CLAUDE.md` 与 `R0_ADAPTABILITY_REPORT.md`

**环境**：`cc_pyvrp` conda env（Python 3.12 + pyvrp==0.14.0，**不装进 MASKCO_env**）；python 路径 `D:/AA_Py/A_Anaconda/envs/cc_pyvrp/python.exe`。1.0.0a0 main 源码仅 reference_only。

## 常用命令

```bash
# 测试（必须用 cc_pyvrp env python；纯本地）
"D:/AA_Py/A_Anaconda/envs/cc_pyvrp/python.exe" CC_Compare/common/tests/test_baseline_contract.py   # 单个测试即可跑
for t in CC_Compare/common/tests/test_*.py; do "D:/AA_Py/A_Anaconda/envs/cc_pyvrp/python.exe" $t; done
# PyVRP 套件：CC_Compare/PyVRP/dcc_vrp/tests/test_*.py（testutil.py 是公共辅助，非测试）
# JF1-H-F：CC_Compare/internal/dcc_vrp/tests/test_jf1hf_integration.py

# DEV-PROTO 证据级回归（输出目录必须为空；9x2 行为 Gate / 9x32 协议稳定性）
"D:/AA_Py/A_Anaconda/envs/cc_pyvrp/python.exe" CC_Compare/PyVRP/dcc_vrp/run_devproto_9x2.py \
    --iterations 300 --instances-per-cell 2 --out CC_Compare/PyVRP/dcc_vrp/results/<新目录>

# 身份复验（SOURCE_MANIFEST 重算）
"D:/AA_Py/A_Anaconda/envs/cc_pyvrp/python.exe" CC_Compare/PyVRP/dcc_vrp/tools/gen_source_manifest.py
```

## 关键规则

1. **绝不修改外部仓库原代码**（`PyVRP/pyvrp/`、`RRNCO/rrnco/`、`CaDA/50/` 等）；适配代码只在 `dcc_vrp/` 下
2. **DEV-GATE v4 运行期间服务器保持安静**：本地操作、不上传、不碰冻结链/归档/输出目录；DEV-PROTO 数据 npz 只能只读下载（sha256 必须与 DEV_MANIFEST 核对）
3. **协议口径**：`objective='coldchain'` 必须显式冻结 profile（拒绝 INVALIDATED v1，`scale_v2/objective_profile.json`）；测试可用 `default_pilot_profile()`（仅协议回归，非证据）
4. **证据链**：正式运行产物必须是完整 record 落盘（原子写）+ 盘上复验 + `canonical.COMPLETE` 绑定 artifact hash；汇总从落盘重算，不信内存结果
5. **确定性协议**：固定 seed + 固定迭代数，同 seed 两次运行 `decision_hash` 一致（排除计时与身份 hash 字段）；`decision_hash`=行为确定性，`artifact_hash`=复现身份
6. 预算敏感性（300/1000/3000 差异）只作报告，不解释成方法优劣；不按 J_CC 选参数
7. 根 `.gitignore` 已加精确大文件忽略（checkpoints/pretrained/data/results + 扩展名）；SFTP 同步同样有 ignore 清单，权重/数据需单独处理
8. 项目代码（`C-VRP_Cold-chainVehicleRoutingProblem/scripts/`）**只读使用**：导入、调用、子类化（如 RecordingEnv），绝不修改
