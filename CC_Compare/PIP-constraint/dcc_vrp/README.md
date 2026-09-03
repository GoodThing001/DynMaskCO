# PIP-constraint → DCC-VRP 适配说明（诚实结论）

## 结论：无法直接对比，不参与 DCC-VRP 主协议

PIP-constraint（NeurIPS 2024，[repo](https://github.com/jieyibi/PIP-constraint)，[arXiv 2410.21066](https://arxiv.org/abs/2410.21066)）是一个**单车辆** TSP 约束处理框架，只支持两个问题：

| 问题 | 车辆数 | 容量约束 | 时间窗 |
|------|:---:|:---:|:---:|
| **TSPTW**（TSP + 时间窗） | 1 | 无 | ✅ |
| **TSPDL**（TSP + deadline） | 1 | 无 | ✅（截止） |
| **DCC-VRP（我们的协议）** | 多 | ✅ 容量 50 | ✅ |

三行对比即可看出**根本不兼容**：PIP 的三个实现（POMO+PIP / AM+PIP / GFACS+PIP）的解码器都是「单车辆访问所有节点一次」的 TSP 式自回归解码，没有车辆分配、没有 depot 回程分隔、没有容量状态。

我们的 DCC-VRP 是 **CVRPTW**（多车辆 + 容量 50 + 时间窗）：
- 平均 50 客户 × 需求 ≈ 5 ≈ 总需求 250 ≫ 容量 50，**必须多辆车**；
- 单车辆 TSPTW 无法表达「容量耗尽必须回 depot 再出发」这一核心约束。

因此 **PIP 无法用「改数据」的方式适配** —— 需要重写其解码器（加容量状态、加多车辆 depot 回程、加可行 mask），等价于重新实现一个 CVRPTW 求解器，远超「基线复现」范畴。

## 处理方式

1. **主协议对比不纳入 PIP**：`基线对比.md` 的方法清单里对 PIP 标注「单车辆 TSPTW/TSPDL，无法适配多车辆 CVRPTW」。
2. **论文数据对比仍可用**：PIP 论文报告的 TSPTW/TSPDL 最优性 gap（vs LKH3）见 `MAPT/README.md` 的方法汇总表，仅作「约束处理范式」相关工作的数据参考，不与我们的 cost 直接比（问题定义不同）。
3. **可选参考桥接**（`dcc_data_tsptw.py`）：若坚持要在 PIP 上跑 DCC 数据，只能退化为**单车辆 TSPTW 松弛**（丢弃容量与多车辆、service 置 0），这会改变问题定义、结果**不可比**，仅作「验证 PIP 代码可运行」用。

## 若将来要做单车辆 TSPTW 消融

DCC 数据里没有单车辆可解的实例（总需求 ≫ 容量），需要另行用 `POMO+PIP/generate_data.py --problem=TSPTW` 生成标准 TSPTW 数据；PIP 的 checkpoint 也需按其 README 训练（无官方预训练 TSPTW 权重随仓库提供，需自查 `pretrained/`）。

> 一句话：**PIP 是单车辆方法，我们的协议是多车辆 CVRPTW，硬约束对不上，跳过。**
