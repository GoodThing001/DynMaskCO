# superseded_v1 — 首次 T2 链，已作废，数字不得引用

**为什么作废**：首次 `verify_camr_static.py` 的 `F1_row_level_consistency` 用了 `plan_hash` 长度 = 64 位的假设，而 `action_contract.plan_hash` 返回 `sha256(...)[:16]`。这是**校验器自身的判据错误**，不是分析结果错误。修正该判据后，`verify_camr_static.py` 的字节发生变化，而它列在 T1 的 `source_file_list` 内。

**处置**（按授权裁定"源码修复后必须重新冻结 T2 并重跑全部前置自检，不得混用不同源码版本"）：

1. 首次链的全部产物原样保留在本目录：`run_identity.json`、`synthetic_regression.json`、`preflight.json`、`pass_a/b_run1/2.json`、`verification.json`；
2. T2 重新冻结（新 `run_identity.json`）；
3. 全部前置自检重跑：合成回归 → T2 → 退化闸门 → Pass A/B 两次 → 独立复核；
4. 正式结论只引用包根目录下的新产物。

**没有改变的东西**：`registration.json`（T1）未动；`camr_shared.py`、`run_pass.py`、`degeneration_gate.py`、`synthetic_regression.py`、`freeze_identity.py`、`register.py` 未动；Pass A/B 的分析口径未动。首次链的 19/19 结果与新链逐字相同（两次完整运行均逐字一致），但**首次链的产物不作为证据引用**。
