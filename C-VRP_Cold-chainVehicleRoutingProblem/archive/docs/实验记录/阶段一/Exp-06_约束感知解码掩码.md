# 实验 Exp-06: 约束感知解码掩码

- **日期**：2026-07-29 ⏭️ 跳过
- **原因**：Exp-07 (TW-preserving 2-opt) 和 Phase 2 (EDD 修复 + TW-aware 2-opt) 已覆盖 TW 约束处理，效果远超单纯的解码掩码
- **结论**：`--enable_tw_filter` 已在候选边阶段实现 TW 过滤，但 C++ 2-opt 会覆盖过滤效果。资源集中在 TW-aware 局部搜索

→ 详见 `00_Phase总结.md`
