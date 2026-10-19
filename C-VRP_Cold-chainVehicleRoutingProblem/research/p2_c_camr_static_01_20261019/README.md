# P2-C-CAMR-STATIC-01 — 读我（含 L1-B 撤回）

**`REPORT.md` 中的 L1-B 结论（"MaskCO-greedy 被 distance-SGBS 覆盖 19/19"）已被撤回。**
`run_pass.py:155-156` 取了错误的比较对象（`sd` 而非 `sm`），只读复算的正确结果是 **0/19 覆盖、19/19 未覆盖**。

- 勘误与只读裁决：[ERRATA_L1B_01.md](ERRATA_L1B_01.md)
- 机器可读回执：[readonly_l1b_adjudication.json](readonly_l1b_adjudication.json)（8/8，含自身来源身份）
- `REPORT.md` **保持原样未改**，其 L1-B 段落只作为被撤回的记录存在。

L1-A（19/19 合法且可复现的行为差异）不受影响。质量、物理可行性、在线效用、可安装性均未建立；本包不自动授权任何后续工作。
