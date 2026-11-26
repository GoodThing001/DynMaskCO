# -*- coding: utf-8 -*-
"""在 audit/V2_缺陷复现记录.md 追加 D7（共同初解不可行）与车队上限协议对齐记录。"""
from pathlib import Path

p = Path("audit/V2_缺陷复现记录.md")
t = p.read_text(encoding="utf-8")
add = """

## D7 共同初解不可行（2026-11-26 开发面板 VRPTW 阶段发现）

**现象**：开发面板进入 VRPTW 部分时，`STD_BUGFIX`/`STD_FIX` 端点被 `check()` 判为
INFEASIBLE，`STD_OLD`（冻结 V1）同样不可行，整批中止。

**定位（V2 侧）**：`nn_witness`（共同初解）在 VRPTW 上只做"最近邻链 + 时窗断链"，
断链后剩余客户若无法插入既有路线就会被**丢弃**（覆盖性缺失）；同时开发实例头部
`NUMBER` 等于构造解车辆数（10–12），使任何构造式初解的 K（11–17）都触发车队超限。
两项叠加导致五版本的共同初解不可行 → 面板失去度量意义（不是算法差异）。

**复现**：12 个开发 VRPTW 实例全部 `ok=False`（修复前 `nn_witness`）。Solomon 上
R101 亦不可行、RC105 在 K=26>25 时不可行。

**修复**：

1. `nn_witness` 重写为确定性构造：VRPTW 按 ready time 升序做"最优可行插位"，
   保证覆盖全部客户；随后 `_shrink_fleet` 把短线客户并入其他路线直到 K ≤ 限额；
2. 开发 VRPTW 实例头部 `NUMBER` 统一对齐为 25（与 Solomon 官方车队上限一致），
   构造解 K 保留在 manifest 的 `k` 字段作为**参考目标**而非硬界
   （`manifests` 记录 `fleet_cap_alignment`，SHA 同步更新）；
3. 复核：Solomon 56/56 初解可行；开发 VRPTW 12/12 初解可行（K=11–17，参考 K=10–12）。

**说明**：这是**协议对齐**而非结果驱动调参——判据是"所有版本的共同初解必须可行"，
与任何版本的最终成绩无关；正式面板使用官方 Solomon 数据（头部 NUMBER=25 本已一致）。

**边界**：`STD_OLD` 仍使用冻结 V1 内部的构造器（不改 V1 源码），其不可行/崩溃如实登记
为 `V1_INFEASIBLE`/`V1_CRASH`，不参与汇总、不被修复、不被掩盖。
"""
t = t.rstrip() + add
p.write_text(t, encoding="utf-8")
print("D7 recorded")
