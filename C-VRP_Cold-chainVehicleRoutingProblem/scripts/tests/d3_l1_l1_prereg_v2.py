#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""L1-v2 EXECUTION PREREGISTRATION BUILDER + VERIFIER (task 1).

The L1-v1 execution pre-registration (`A1_D3_L1_L1_FORMAL_EXECUTION_PREREGISTRATION_V1.md`, §7) still
registers the OLD 34-path F-only watch set, while the V2 signing package freezes 42 L1 paths.  A
pre-registration whose §7 contradicts the identity the package freezes cannot be the basis of a
signature, so this tool emits the NEXT version.

It does three things and nothing else:

  1. `--build` derives the CURRENT frozen identity and writes a MACHINE-READABLE manifest of the
     watch set with the complete per-file SHA-256 list, the budgets, the state binding and the source
     identity digest;
  2. `--render` writes the L1-v2 pre-registration markdown, with §7 generated FROM that manifest so
     the document and the identity cannot disagree;
  3. `--verify` re-derives everything from disk and REFUSES if the markdown, the manifest and the
     live identity have drifted apart.

The science is untouched: the research question, the 64 states in original publication order, the two
arms, V3 = NOT_RUN, the budgets and the order-confound rule are the SAME objects, re-registered under
a new version because the identity underneath them changed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXT = HERE.parent.parent
REPO = EXT.parent
for _p in ('scripts', 'scripts/simulation', 'scripts/tests', 'scripts/data'):
    sys.path.insert(0, str(EXT / _p))

RUNNER = EXT / 'scripts/simulation/d3_l1_l1_formal_run.py'
EV = REPO / 'research/review_3080_20261010/d3_g0_evidence_root_v1/a1_d3_l1_prereg_root_v1'
L1 = EV / 'l1_gates'

# ---- the NEW version ------------------------------------------------------------------------
PREREG_VERSION = 'V2'
RUN_ID = 'd3l1-l1-two-arm-20261016-l1v2'
PREREG_MD = EV / 'A1_D3_L1_L1_FORMAL_EXECUTION_PREREGISTRATION_V2.md'
PREREG_V1_MD = EV / 'A1_D3_L1_L1_FORMAL_EXECUTION_PREREGISTRATION_V1.md'
MANIFEST = L1 / 'L1_V2_PREREGISTRATION_MANIFEST.json'
OLD_PACKAGE = L1 / 'L1_FORMAL_EXECUTION_SIGNING_PACKAGE.json'
V2_PACKAGE = L1 / 'L1_FORMAL_EXECUTION_SIGNING_PACKAGE_V2.json'
INVALIDATION = L1 / 'L1_FORMAL_EXECUTION_SIGNING_PACKAGE_V1_INVALIDATION.json'


def sha(path):
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def sha_bytes(raw):
    return hashlib.sha256(raw).hexdigest()


def load(p):
    p = Path(p)
    return json.loads(p.read_text(encoding='utf-8')) if p.is_file() else None


def entry_module():
    spec = importlib.util.spec_from_file_location('l1_entry_prereg', str(RUNNER))
    module = importlib.util.module_from_spec(spec)
    sys.modules['l1_entry_prereg'] = module
    spec.loader.exec_module(module)
    return module


def repo_rel(p):
    return Path(p).resolve().relative_to(REPO.resolve()).as_posix()


def build_manifest():
    """The frozen identity, flattened into the manifest the pre-registration is generated from."""
    module = entry_module()
    ident = module.build_l1_identity(RUN_ID, created_utc='2026-10-16T00:00:00Z')
    entries = []
    for path, digest in sorted(ident['watch'].items()):
        live = sha(path)
        entries.append({'path': repo_rel(path), 'sha256_frozen': digest, 'sha256_on_disk': live,
                        'matches': digest == live})
    source = [e for e in entries if e['path'].startswith('C-VRP_Cold-chainVehicleRoutingProblem/'
                                                        'scripts/')]
    assets = [e for e in entries if e not in source]
    drift = [e['path'] for e in entries if not e['matches']]
    return {
        'schema': 'd3_l1_l1_preregistration_manifest',
        'prereg_version': PREREG_VERSION,
        'run_id': RUN_ID,
        'identity_schema': ident['schema'],
        'n_watch_total': len(entries),
        'n_watch_source': len(source),
        'n_watch_non_source_assets': len(assets),
        'watch': entries,
        'watch_source': source,
        'watch_non_source_assets': assets,
        'n_drifted': len(drift),
        'drifted': drift,
        'watch_derivation': ident['watch_derivation'],
        'source_identity_sha256': _source_identity_sha256(ident['watch']),
        'state_binding_sha256': ident['state_binding_sha256'],
        'forward_targets_sha256': ident['forward_targets_sha256'],
        'dependency_identity_sha256': ident['dependency_identity_sha256'],
        'n_states': ident['state_list']['n_states'],
        'budget': ident['budget'],
        'wall_clock_scope': module.WALL_CLOCK_SCOPE,
        'arm_order': list(module.ARM_ORDER),
        'order_confound_rule': module.ORDER_CONFOUND_RULE,
        'expected_states': module.EXPECTED_STATES,
        'critical_l1_sources': ident['watch_derivation']['critical_l1_sources'],
        'identities': ident['identities'],
        'contracts': ident['contracts'],
        'labels': ident['labels'],
        'verdicts': ident['verdicts'],
        'execution_command': (
            'python scripts/simulation/d3_l1_l1_formal_run.py --supervise '
            '--out-dir <frozen L1 run directory> --authorization <signed credential> '
            '--preregistration '
            'research/review_3080_20261010/d3_g0_evidence_root_v1/a1_d3_l1_prereg_root_v1/'
            'A1_D3_L1_L1_FORMAL_EXECUTION_PREREGISTRATION_V2.md'),
        'supersedes': {
            'preregistration_v1': repo_rel(PREREG_V1_MD),
            'preregistration_v1_sha256': sha(PREREG_V1_MD),
            'why': 'V1 §7 registers a 34-path F-ONLY watch set, while the frozen identity now carries '
                   '%d L1 paths.  A pre-registration that contradicts the identity cannot be the '
                   'basis of a signature, so the NEXT version is created rather than edited.'
                   % len(entries),
            'not_deleted': True,
            'not_edited': True,
        },
        'v2_engineering_acceptance': {
            'signing_package_V2': {'path': repo_rel(V2_PACKAGE), 'sha256': sha(V2_PACKAGE)},
            'blocker_acceptance': {
                'path': 'research/review_3080_20261010/d3_g0_evidence_root_v1/'
                        'a1_d3_l1_prereg_root_v1/l1_gates/L1_SIGN_BLOCKER_ACCEPTANCE.json',
                'verdict': (load(L1 / 'L1_SIGN_BLOCKER_ACCEPTANCE.json') or {}).get('verdict'),
                'n_pass': (load(L1 / 'L1_SIGN_BLOCKER_ACCEPTANCE.json') or {}).get('n_pass'),
                'n_fail': (load(L1 / 'L1_SIGN_BLOCKER_ACCEPTANCE.json') or {}).get('n_fail')},
            'blocker_independent_audit': {
                'path': 'research/review_3080_20261010/d3_g0_evidence_root_v1/'
                        'a1_d3_l1_prereg_root_v1/l1_gates/L1_SIGN_BLOCKER_INDEPENDENT_AUDIT.json',
                'verdict': (load(L1 / 'L1_SIGN_BLOCKER_INDEPENDENT_AUDIT.json') or {}).get('verdict'),
                'n_pass': (load(L1 / 'L1_SIGN_BLOCKER_INDEPENDENT_AUDIT.json') or {}).get('n_pass'),
                'n_fail': (load(L1 / 'L1_SIGN_BLOCKER_INDEPENDENT_AUDIT.json') or {}).get('n_fail')},
            'arm_order_acceptance': {
                'path': 'research/review_3080_20261010/d3_g0_evidence_root_v1/'
                        'a1_d3_l1_prereg_root_v1/l1_gates/L1_ARM_ORDER_ACCEPTANCE.json',
                'verdict': (load(L1 / 'L1_ARM_ORDER_ACCEPTANCE.json') or {}).get('verdict'),
                'n_pass': (load(L1 / 'L1_ARM_ORDER_ACCEPTANCE.json') or {}).get('n_pass'),
                'n_fail': (load(L1 / 'L1_ARM_ORDER_ACCEPTANCE.json') or {}).get('n_fail')},
            'supervised_synthetic_run': {
                'path': 'research/review_3080_20261010/d3_g0_evidence_root_v1/'
                        'a1_d3_l1_prereg_root_v1/l1_gates/L1_SUPERVISED_SYNTHETIC_RUN.json',
                'verdict': (load(L1 / 'L1_SUPERVISED_SYNTHETIC_RUN.json') or {}).get('verdict')},
            'run_entry_acceptance': {
                'path': 'research/review_3080_20261010/d3_g0_evidence_root_v1/'
                        'a1_d3_l1_prereg_root_v1/l1_gates/L1_RUN_ENTRY_ACCEPTANCE.json',
                'verdict': (load(L1 / 'L1_RUN_ENTRY_ACCEPTANCE.json') or {}).get('verdict'),
                'n_pass': (load(L1 / 'L1_RUN_ENTRY_ACCEPTANCE.json') or {}).get('n_pass'),
                'n_fail': (load(L1 / 'L1_RUN_ENTRY_ACCEPTANCE.json') or {}).get('n_fail')},
        },
        'human_approval': 'NOT FILLED: this manifest is not an approval and carries no signature',
        'science_ready': False, 'stable_online_gain': False, 'algorithmic_gain_claim': 'none',
        'tool_sha256': sha(Path(__file__)),
    }


def _source_identity_sha256(watch):
    """The code binding the credential will declare: one digest over the whole watched set."""
    return sha_bytes(json.dumps({str(k): str(v) for k, v in sorted(watch.items())},
                                sort_keys=True).encode())


def render(manifest, *, manifest_sha256=None):
    b = manifest['budget']
    # The document names the manifest by digest and the manifest names the document by digest, so the
    # PAIR validates itself: neither can be edited without the other check failing.  The manifest's own
    # digest is computed over the manifest WITHOUT this field (see `manifest_body_sha256`), so the
    # mutual reference has no fixed point to chase.
    manifest_line = (
        "- 机器可读清单（本文件 §7 由它生成）：`l1_gates/L1_V2_PREREGISTRATION_MANIFEST.json`\n"
        "  - 清单 sha256（对**不含本行摘要字段**的清单本体计算）：`%s`"
        % (manifest_sha256 or '(pending)'))
    rows = '\n'.join('| `%s` | `%s` |' % (e['path'], e['sha256_frozen'])
                     for e in manifest['watch'])
    src_rows = '\n'.join('| `%s` | `%s` |' % (e['path'], e['sha256_frozen'])
                         for e in manifest['watch_source'])
    asset_rows = '\n'.join('| `%s` | `%s` |' % (e['path'], e['sha256_frozen'])
                           for e in manifest['watch_non_source_assets'])
    acc = manifest['v2_engineering_acceptance']
    return f"""# A1-D3-L1 正式机制比较执行版预登记（七要素）· L1-v2

- 轮次：`A1-D3-L1-MECHANISM-PREREG`（**正式执行版**）
- 状态：`EXECUTION_PREREG_FROZEN / READY_FOR_USER_SIGNATURE`（**本文件不授权、也未执行正式比较**）
- 绑定运行：`run_id = {manifest['run_id']}`
- 取代：`A1_D3_L1_L1_FORMAL_EXECUTION_PREREGISTRATION_V1.md`（SHA `{manifest['supersedes']['preregistration_v1_sha256']}`）
  —— **原件逐字节保留、不改写**。取代理由：V1 §7 登记旧的 **34 条 F-only** watch 集，
  而冻结身份现为 **{manifest['n_watch_total']} 条 L1 watch**；预登记与身份矛盾时**不得**沿用其 SHA 作为签署依据。
- 机器可读清单（本文件 §7 由它生成，二者不可能不一致）：
{manifest_line}
- 上位（**判据原样继承，不事后修改**）：
  `A1_D3_L1_MECHANISM_DIAGNOSTIC_PREREGISTRATION.md`（唯一研究问题、三判定分离、分叉语义、失败分母）、
  `A1_D3_L1_TWO_ARM_EXECUTION_PREREGISTRATION_V1_18.md`（共享输入单因素、六类反例、可重放验收）
- 前置事实：**V1.22 真实 F 前向已执行并通过（64/64，独立裁决 6/6，独立只读复核 `VERIFIED`）**，
  一次性授权回执恰一条。**本预登记不重跑 F 资格批次。**
- 不改变：D3 G0、D8、E2、L2/L3 资格；I3 正式负结果；D2 `not_qualified`；**`stable_online_gain = False`**

---

## 0. L1 的唯一研究问题（原样继承）

> 在**完全相同的合法动作集合**中，F（真实 MaskCO 权重评分）与 D-Geo（`−Δd` 评分）
> 是否产生**不同的可执行修复选择**？

L1 **只**回答"是否存在**可归因的动作分歧**"。L1 的分歧**不得**被追认为增益、显著性、质量或在线效用；
那是 L2/L3，须各自独立授权。

## 1. 要素① 状态来源

| 项 | 冻结内容 |
|---|---|
| 状态集合 | **复用 V1.9 冻结的 {manifest['n_states']} 个 dev_check 状态**，台账**发布顺序**（`event='PUBLISHED'`），不重排 |
| 数量 | `{manifest['n_states']}`，且 `EXPECTED_STATES = {manifest['expected_states']}` 由裁决器强制 |
| 身份 | `state_binding_sha256 = {manifest['state_binding_sha256']}` |
| 前向目标 | `forward_targets_sha256 = {manifest['forward_targets_sha256']}` |
| 空池 | **原样保留**并记 `empty_pool`；该状态留在分母 |
| 失败状态 | **全部留在分母**；不替换、不补采、不重跑 |
| 开发用途标注 | 这些状态**已用于开发**，L1 结论**只**支持"在这些状态上两评分器的机制差异"，**不是**独立科学验证；L2 必须另用事前登记的新鲜开发日 |

状态的物化路径与 V1.10 一致（同一 `materialize` 码路），不新写并行实现。

## 2. 要素② 唯一比较（单因素）

```
mask_scope      = B                      # build_pool_B：按 mutable_ids 计划后缀在采样前过滤
eps_rel         = 0.0                    # EPS_D_PRIMARY_REL
eps_abs         = 0.0                    # EPS_D_PRIMARY_ABS
tie_break       = legal_order_first      # F 的历史 first-maximum argmax 语义
F:      s_F(a) = restored MaskCO logits          a_F = first argmax_{{a in legal}} s_F(a)
D-Geo:  s_D(a) = -incremental_distance(a)        a_D = first argmax_{{a in legal}} s_D(a)
```

**两臂必须共享（机器可检验）**：初始快照、`P0`、strict-online 可见信息、**B 池**、
**初始有序 legal**（环境枚举器原序）、`mask`、合同、`tie_break`、**可用资源**。
**唯一声明差异** = `scorer_id`（`DECLARED_VARYING_FIELDS = ('scorer_id',)`）。

### 2.1 两臂执行顺序与共享 10 秒预算（**本次新增的事前登记规则**）

两臂**顺序执行**，顺序冻结为 `{manifest['arm_order']}`，且**共享同一个单调计时起点**
（`wall_clock_scope = {manifest['wall_clock_scope']}`）。

**机械后果**：第二臂的剩余时间 = `{b['per_state_wall_clock_s']} s −（第一臂已花时间）`，
故**第二臂的时间预算取决于第一臂的成本**。这是**仪器属性**，不是任一评分器的属性。

**事前解释规则（不得事后依据结果调整）**：

```
{manifest['order_confound_rule']}
```

- **V1 不受该顺序影响**：`V1` 比较**第一步**，而第一步之前**没有任何评分调用**，
  故此时第二臂花费为 0、任何截止都不可能触发。**V1 在构造上顺序不变**；
  隔离合成反例 O4 用同一状态在两种顺序下复算确认：两臂都产出第一步时，
  **所选动作与首步 legal 指纹完全相同**。
- **V2 可能受该顺序影响**：`V2` 比较**逐臂完成情况**，而这正是共享截止所约束的对象，
  故**第二臂的 `timeout` 可能由顺序造成**。隔离合成反例对（O2/O3）：同一状态、同一掩码、
  同一有序 legal、两臂自身成本与调用上限全部不变，**唯一变化的是哪一臂先跑** ——
  O2 两臂完成，O3 第二臂 `timeout` 且**回退原计划**。
- **顺序还可能使 V1 变为 `NOT_COMPARABLE`**：若第一臂吃掉全部额度，第二臂在**第一次评分调用**
  即被 fail-closed 拒绝，于是**没有第一步可比较**。该情形记为 `NOT_COMPARABLE`
  （含义恰为"两臂无法比较"），**不是**评分器差异；O4 单独记录此情形。
- **禁止的解读**：由执行顺序造成的超时差异**不得**被解释为评分器机制优势；
  该解读由 `classify_order_confound` 机械排除，并在裁决中列出受影响状态。

**最小公平性处理 = 上述事前解释规则 + 机械分类**。**不放宽**任何上限、**不增加**正式运行次数、
**不改动**任一臂的评分器或调用上限。受影响状态**全部保留在 64 状态分母中**，不剔除、不替换、不重跑。

三项机械化检查（沿用 V1.18 协议，不新造）：①两臂收到**同一个 state 对象**；
②state 摘要运行前后三处相等；③首步 legal 指纹两臂相等且等于共享值。

**分叉语义**（原样继承）：两臂首步不同后，后续部分计划与 legal 可自然不同；**只在首步检查身份**，
分叉后各自的 legal 指纹**逐条记录、永不用于判 FAIL**。

## 3. 要素③ 三判定（独立报告，永不合并）

| 判定 | 定义 | 取值 |
|---|---|---|
| **V1 首步动作分歧** | 两臂首步六字段动作是否不同 | `DIVERGENT` / `IDENTICAL` / `NOT_COMPARABLE` / `NOT_APPLICABLE_EMPTY_MASK` / `NOT_ATTEMPTED` |
| **V2 完整修复** | **每臂**是否走完整个掩码且 `failure_reason is None`（**不含认证**）；**并附 `v2_order_attribution`** | 逐臂布尔 + `failure_reason` + 顺序归因 |
| **V3 可认证计划** | **本次选择 `NOT_RUN`**：不提供认证器、不做正式 C0 认证、**不安装任何计划** | `NOT_RUN`（记录"无认证器"这一事实） |

**V3 = NOT_RUN 的直接后果（必须如实表述）**：本次**只允许**讨论
「候选动作分歧」与「**未认证**的完整修复差异」；
**不得**称为"已认证的可执行计划差异"。任何"可执行/可认证"措辞在本轮均不成立。

## 4. 要素④ 日志（逐槽与逐步记录组）

| 记录组 | 必需字段 |
|---|---|
| 状态身份 | `day_id`、`position`、`state_hash`、`clock`、可见信息指纹 |
| 模型身份 | F checkpoint SHA、backbone SHA、训练元数据、`label`（`HISTORICAL_BACKBONE_UNVERIFIED`） |
| 共同条件 | `mask_order`、`mask_sha256`、`mutable_ids`、`protected_ids`、`pool_B`、`contract_hash`、首步 `legal_sha256` |
| 逐步动作 | 两臂 `steps` 六字段动作、`records`（含 `legal_order`、`canonical_keys`、`score_exact_ties`）、**逐步 legal 指纹** |
| **逐步评分对应** | **逐次评分调用**记录：该步 legal 指纹、logits 长度、logits 摘要、dtype ⇒ 使"跨步复用"可被检出 |
| 执行结果 | 每臂 `failure_reason`、`n_steps`、`plan_after`、`plan_hash_after`、`returned_original_plan`、`diag` |
| **顺序归因** | `arm_order`、`wall_budget_source`、`wall_clock_scope`、`v2_order_attribution`（每状态） |
| 资源 | 每臂 `BudgetGate` 快照（调用数、墙钟、是否越限）与 `Budget` 六项计数 |
| 重放 | `replay_digest`：audit / V1 / V2 / V3 / plan_after / plan_hash 去计时后取摘要 |

**JAX 首次编译与模型初始化单独计时并保留**；**不得**据此声称满足 L3 完整 8+2 秒在线门。

## 5. 要素⑤ 失败语义与停止规则（fail-closed，全部留在分母）

- 失败词表沿用 `d3_geo_adapter.FAILURE_REASONS`：`no_legal` / `nan` / `timeout` / `rollback` /
  `cert_fail` / `score_shape` / `score_len` / `geo_len` / `no_certifier`；本轮另加
  `budget_calls`（调用数越限，**fail-closed 回退原计划**）。
- 任一失败臂**返回调用方原计划**；部分计划只作 `diag.partial_plan_at_failure` 证据；**不得安装半修复计划**。
- 超时按 `timeout` 记账并保留；**不得**把失败记成"两臂一致"，**不得**记成"某臂落后"。
- **超时状态的顺序归因**（§2.1）：`v2_order_attribution` 取
  `NOT_ORDER_ATTRIBUTABLE` / `ORDER_ATTRIBUTABLE` / `ORDER_INDETERMINATE`；
  后两者**不得**读作机制差异，但**状态照常留在分母**。
- 空池 ⇒ V1 `NOT_APPLICABLE_EMPTY_MASK`、零步、计划不变，**该状态留在分母**。
- **禁止**：为改善结果重跑、追加状态、剔除失败状态、放宽上限、拼接任何历史记录。

## 6. 要素⑥ 预算与资源门（**由 V1.22 实测推导，含明确缺口**）

### 6.1 实测事实（V1.22，`REAL_F_FORWARD_ADJUDICATION_V122.json`）

| 量 | 值 | 含义 |
|---|---|---|
| 单状态跨度 max | 0.422 s | **仅**一次状态内的**两次首步**评分，**不是**多步 repair 成本 |
| worker 墙钟 | 33.0 s | 128 次评分 + 初始化/编译 |
| 初始化+编译 | 6.688 s | 单独计时 |
| 记录前向数 | 129 | 1 warm-up + 128 |

⇒ 单次 F 调用成本（由墙钟反推）：`(33.0 − 6.688) / 128 ≈ 0.2057 s`；**最坏单次**取
`max(0.422/2, 0.2057) = 0.211 s`。

### 6.2 步数会计（本次冻结的核心）

```
m = 掩码步数 = min(2, |B 池|)          # 与 V1.10 已验收输入物化所用的掩码规则一致
每状态 F 评分调用数 = m                  # 每步一次
每状态 D-Geo 评分调用数 = m              # 每步一次
F 调用上限 = 64 × m = 128
D-Geo 调用上限 = 64 × m = 128
```

**两臂上限完全相同**；**不因 D-Geo 更快而给 F 任何额外时间或调用额度**。

### 6.3 冻结上限（**本次修复未改动任何一条**）

| 上限 | 值 | 推导 |
|---|---|---|
| 每臂每状态评分调用 | `m = {b['per_arm_per_state_scorer_calls']}` | 步数会计（精确量，非估计） |
| 每臂总评分调用 | `{b['per_arm_total_scorer_calls']}` | `64 × m` |
| 单决策（每状态）墙钟 | **{b['per_state_wall_clock_s']} s** | **沿用既有 D3 单决策口径，不放宽**；两臂**共享**该额度（§2.1） |
| 初始化+编译 | `{b['initialize_compile_s']} s` | 沿用既有工程上限（V1.22 实测 6.688 s） |
| 整批墙钟 | `{b['batch_wall_clock_s']} s` | 沿用既有工程上限；`64 × 2 × 0.211 s ≈ 27 s`（F 侧）+ D-Geo（向量读取，量级更小） |

### 6.4 **明确的缺口（不得用乘数掩盖，也不得为此开未授权试跑）**

1. **第 2…m 步的单位成本从未实测**：V1.22 每状态**只在首步**评分一次，多步 repair 无直接计时样本。
   由于掩码消耗后合法集变小，**首步成本是每次调用的上界**，但**总调用数与墙钟不因此被确定**。
2. **不得**为填补该缺口而运行任何真实 F 试跑（那需要另行授权）；**本预登记不编造乘数**。
3. 若签署人要求更紧的墙钟上限，可用的**唯一**合法手段是**降低 `m`**（减少步数），而不是猜测单步耗时。
4. **离线机制诊断不得冒称通过在线时限**；既有 D3 的 10 s 单决策口径是**工程上限**，
   **不是** L3 的 8+2 秒发布门。
5. **顺序混杂**（§2.1）的残余不确定性如实记录：第一臂成本与第二臂剩余额度**实际消耗**，
   只有在真实运行中才能观察到；本预登记给出的**不是**对其数值的预测，而是**解读规则**。

## 7. 要素⑦ 源码与资产身份（**导出闭包，逐文件 SHA**）

- **运行期 watch 集 = {manifest['n_watch_total']} 条**（{manifest['n_watch_source']} 个源码文件
  + {manifest['n_watch_non_source_assets']} 项非源码资产），由
  `scripts/data/d3_l1_closure.py` 从 **L1 正式入口**出发按传递本地 import 图导出。
  **V1 §7 登记的 34 条 F-only watch 集已作废**：它冻结了决定 F 前向的代码，
  **一条**决定 L1 比较的代码都没有冻结（`l1_sources_in_closure == []`）。
- 闭包模块自身也是被冻结路径之一（自指必须被覆盖，否则闭包可被改写而不被检出）。
- **源码身份摘要** `source_identity_sha256 = {manifest['source_identity_sha256']}`
  —— 对整个 watch 集取一次规范摘要，凭据以此**绑定执行代码**。
- **逐文件 SHA（{manifest['n_watch_total']} 条，完整列表）**：

| 路径（仓库相对） | SHA-256 |
|---|---|
{rows}

- **非源码资产（{manifest['n_watch_non_source_assets']} 项，显式根）**：

| 路径（仓库相对） | SHA-256 |
|---|---|
{asset_rows}

- **关键 L1 决策模块（必须在册，缺一即拒绝）**：

{chr(10).join('- `%s`' % c for c in manifest['critical_l1_sources'])}

- **明确排除**：测试 / 验收 / 证据 / 只读审计工具按 `EXCLUDED_WITH_REASON` **逐条记录理由**后排除；
  并行 `CC_Compare` 提交不属于 D3 身份。
- 第三方库版本以运行身份 `packages` 记录，不以文件 SHA 表达。
- **冻结校验同源**：`d3_l1_closure.cross_check_watch_set` 由身份与冻结门共用；
  闭包任一文件缺失即 `NOT_READY`，**不存在**静默回退到 F-only watch 的分支。
- **消费前实时校验**：监督器对**磁盘实时重算**的 watch 集授权，
  故冻结之后被修改的代码在**消费一次性凭据之前**即被阻断；消费后再校验一次。
- `--build` 时漂移 = **{manifest['n_drifted']}** 条（{json.dumps(manifest['drifted']) if manifest['drifted'] else '无'}）。

## 8. 执行边界与消费规则

**授权**：以 `run_id = {manifest['run_id']}` 在上述冻结身份与预算下执行**一次**
{manifest['n_states']} 状态、最多 `m=2` 步、每臂最多 {b['per_arm_total_scorer_calls']} 次评分的
**F−D-Geo 同掩码机制比较**（离线、不安装、V3 = `NOT_RUN`）。

**唯一执行命令**：

```
{manifest['execution_command']}
```

**不授权**：E2 训练、L2、L3、40 日批次、最终测试、任何安装/写入真实状态、任何效用或显著性主张、
C0 正式预算裁定。

**消费规则**：PASS / FAIL / 超时 / 中途强杀**均消费**本次唯一机会；不重跑、不追加状态、
不拼接任何历史记录；崩溃时保留原件、不静默重启整批。

## 9. 本次可形成与不可形成的科学结论

**可以形成**：
1. 在完全相同的初始有序 legal 上，F 与 D-Geo **首步动作是否分歧**（含分歧字段与分叉位置）；
2. **每臂**是否产生**完整（未认证）修复**，并**区分**该差异是机制差异还是**顺序/共享额度**所致（§2.1）；
3. 失败 / 超时 / 回滚 / 空池的**分布**，且全部状态留在分母；
4. **顺序归因计数**：`ORDER_ATTRIBUTABLE` / `ORDER_INDETERMINATE` / `NOT_ORDER_ATTRIBUTABLE` 各多少状态。

**不可以形成**：
1. 任一修复是**可认证**的（V3 = `NOT_RUN`）；
2. 任何**决策质量**或**在线效用**差异；
3. 任何**显著性**、**学习贡献**或 **MaskCO** 结论；
4. 关于**历史训练期 backbone** 的任何陈述；
5. 把 `ORDER_ATTRIBUTABLE` 状态的 V2 差异当作**评分器机制优势**。

## 10. Claim limits

1. 本文件是**规格 + 工程资格**；产出时 L1 正式比较次数 = **0**。
2. 复用的 {manifest['n_states']} 状态**曾用于开发**，**不是**独立科学验证；L2 必须另用新鲜开发日。
3. 离线机制诊断**不满足** L3 在线发布门。
4. C0 正式预算**继续作为独立待裁定项**（`NON_BLOCKING_UNRESOLVED`），本轮不假装已解决。
5. V1.22 的 PASS 只证明**评分接口**在真实输入上可用，**不**蕴含任何分歧或增益。
6. I3 正式负结果（−144.680，95% CI [−191.109, −100.125]）、D2 `not_qualified`、D3 G0 未通过、
   D8 `CAPABILITY_GAP_BLOCKING`、E2 无合格结果、`stable_online_gain = False` **全部不变**。
7. **本文件替代 V1 作为签署对象**；V1 原件、旧签署包与 V2 证据**全部逐字节保留**。
8. **人工批准字段不由任何工具代填**；本文件本身**不是**授权。
"""


def manifest_body_sha256(manifest):
    """The manifest's identity digest, computed WITHOUT the mutual-reference fields.

    `manifest['preregistration']` records the markdown it generated (whose own text names this
    manifest), and `manifest['manifest_body_sha256']` records this value.  Excluding both breaks the
    cycle, so the digest is stable and the pair still validates itself.
    """
    body = {k: v for k, v in manifest.items()
            if k not in ('preregistration', 'manifest_body_sha256')}
    return sha_bytes(json.dumps(body, sort_keys=True, default=repr).encode())


def verify():
    """Re-derive everything and refuse if the markdown, the manifest and the identity disagree."""
    problems = []
    manifest = load(MANIFEST)
    if manifest is None:
        return ['the manifest is absent: %s' % MANIFEST]
    live = build_manifest()
    if manifest['n_watch_total'] != live['n_watch_total']:
        problems.append('watch count drifted: manifest %d vs live %d'
                        % (manifest['n_watch_total'], live['n_watch_total']))
    if manifest['source_identity_sha256'] != live['source_identity_sha256']:
        problems.append('source_identity_sha256 drifted: manifest %s vs live %s'
                        % (manifest['source_identity_sha256'], live['source_identity_sha256']))
    if manifest['state_binding_sha256'] != live['state_binding_sha256']:
        problems.append('state_binding_sha256 drifted')
    if manifest['budget'] != live['budget']:
        problems.append('budget drifted: manifest %s vs live %s'
                        % (manifest['budget'], live['budget']))
    if manifest['run_id'] != live['run_id']:
        problems.append('run_id drifted: manifest %s vs live %s'
                        % (manifest['run_id'], live['run_id']))
    if live['n_drifted']:
        problems.append('%d watched path(s) drifted on disk: %s'
                        % (live['n_drifted'], live['drifted'][:6]))
    # MUTUAL BINDING: the manifest's recorded body digest must match the manifest on disk ...
    recorded_body = manifest.get('manifest_body_sha256')
    actual_body = manifest_body_sha256(manifest)
    if recorded_body != actual_body:
        problems.append('the manifest body digest does not match the manifest on disk: %s != %s'
                        % (recorded_body, actual_body))
    # ... and the markdown must name BOTH the recorded body digest and the manifest file
    md_path = PREREG_MD
    if not md_path.is_file():
        problems.append('the V2 pre-registration markdown is absent: %s' % md_path)
    else:
        md = md_path.read_text(encoding='utf-8')
        if recorded_body and recorded_body not in md:
            problems.append('the markdown does not carry the manifest body digest %s' % recorded_body)
        if 'L1_V2_PREREGISTRATION_MANIFEST.json' not in md:
            problems.append('the markdown does not name the manifest file')
        for e in live['watch']:
            if ('| `%s` | `%s` |' % (e['path'], e['sha256_frozen'])) not in md:
                problems.append('the markdown does not carry the frozen row for %s' % e['path'])
                break
        # the count appears in §7 with the source/asset split, e.g. "watch 集 = 42 条**（38 个源码文件"
        if ('watch 集 = %d 条' % live['n_watch_total']) not in md:
            problems.append('the markdown does not carry the live watch count %d'
                            % live['n_watch_total'])
        if ('%d 个源码文件' % live['n_watch_source']) not in md:
            problems.append('the markdown does not carry the live source-file count %d'
                            % live['n_watch_source'])
        if live['source_identity_sha256'] not in md:
            problems.append('the markdown does not carry source_identity_sha256')
        if RUN_ID not in md:
            problems.append('the markdown does not carry run_id %s' % RUN_ID)
        if '34 条 F-only' not in md:
            problems.append('the markdown does not record the superseded 34-path V1 watch set')
        # and the markdown's own digest must be the one the manifest recorded
        md_rec = manifest.get('preregistration') or {}
        if md_rec.get('sha256') != sha(md_path):
            problems.append('the markdown digest does not match the manifest record: %s != %s'
                            % (md_rec.get('sha256'), sha(md_path)))
        # the superseded V1 file must still hash to what the manifest recorded
        if sha(PREREG_V1_MD) != manifest['supersedes']['preregistration_v1_sha256']:
            problems.append('the V1 pre-registration has CHANGED: %s != %s'
                            % (sha(PREREG_V1_MD),
                               manifest['supersedes']['preregistration_v1_sha256']))
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build', action='store_true')
    ap.add_argument('--render', action='store_true')
    ap.add_argument('--verify', action='store_true')
    ap.add_argument('--all', action='store_true')
    args = ap.parse_args()
    if not any((args.build, args.render, args.verify, args.all)):
        raise SystemExit('pass --build, --render, --verify or --all')
    if args.build or args.all:
        manifest = build_manifest()
        MANIFEST.parent.mkdir(parents=True, exist_ok=True)
        manifest['manifest_body_sha256'] = manifest_body_sha256(manifest)
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=repr),
                            encoding='utf-8')
        print(json.dumps({'manifest': str(MANIFEST.relative_to(REPO)),
                          'sha256': sha(MANIFEST),
                          'manifest_body_sha256': manifest['manifest_body_sha256'],
                          'run_id': manifest['run_id'],
                          'n_watch_total': manifest['n_watch_total'],
                          'n_watch_source': manifest['n_watch_source'],
                          'n_watch_non_source_assets': manifest['n_watch_non_source_assets'],
                          'n_drifted': manifest['n_drifted'],
                          'source_identity_sha256': manifest['source_identity_sha256'],
                          'state_binding_sha256': manifest['state_binding_sha256'],
                          'budget': manifest['budget']}, indent=1, default=repr))
    if args.render or args.all:
        manifest = load(MANIFEST)
        if manifest is None:
            raise SystemExit('build the manifest first')
        PREREG_MD.write_text(render(manifest, manifest_sha256=manifest['manifest_body_sha256']),
                             encoding='utf-8', newline='')
        print(json.dumps({'preregistration': str(PREREG_MD.relative_to(REPO)),
                          'sha256': sha(PREREG_MD),
                          'bytes': PREREG_MD.stat().st_size}, indent=1, default=repr))
        # the manifest records the markdown it generated, so the pair is self-consistent
        manifest['preregistration'] = {'path': repo_rel(PREREG_MD), 'sha256': sha(PREREG_MD),
                                      'bytes': PREREG_MD.stat().st_size}
        MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=repr),
                            encoding='utf-8')
        print('manifest updated with the rendered markdown digest; new sha256', sha(MANIFEST))
    if args.verify or args.all:
        problems = verify()
        print(json.dumps({'verify': 'OK' if not problems else 'FAILED',
                          'n_problems': len(problems), 'problems': problems[:8]},
                         indent=1, default=repr))
        return 0 if not problems else 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
