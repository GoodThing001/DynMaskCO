# CC_Compare 方法论文对照

> 用途：为 `CC_Compare/` 里 clone 的对比方法仓库补齐原论文 PDF。
> 这些是 `docs/论文参考/对比方法/对比方法/`（17 篇总表）**未覆盖**的方法。
> 下载日期：2026-09-22，来源 arXiv（`https://arxiv.org/pdf/<id>`）。

## 方法 → PDF 对照

| CC_Compare 文件夹 | 方法 | 出处 | arXiv | 本地 PDF |
|------|------|------|------|------|
| `CO-enriched-ML/` | CO-Enriched ML（原生动态 VRPTW） | Transportation Science 2024 | [2304.00789](https://arxiv.org/abs/2304.00789) | `2304.00789.pdf` |
| `RouteFinder/` | RouteFinder: Towards Foundation Models for VRP | TMLR 2025 / ICML 2026 J2C | [2406.15007](https://arxiv.org/abs/2406.15007) | `2406.15007.pdf` |
| `Learning-to-Delegate/` | Learning to Delegate for Large-scale Vehicle Routing | NeurIPS 2021 | [2107.04139](https://arxiv.org/abs/2107.04139) | `2107.04139.pdf` |
| `DeepACO/` | DeepACO: Neural-enhanced Ant Systems | NeurIPS 2023 | [2309.14032](https://arxiv.org/abs/2309.14032) | `2309.14032.pdf` |
| `Omni-VRP/` | Towards Omni-generalizable Neural Methods for VRP | ICML 2023 | [2305.19587](https://arxiv.org/abs/2305.19587) | `2305.19587.pdf` |
| `Sym-NCO/` | Sym-NCO: Leveraging Symmetricity for NCO | NeurIPS 2022 | [2205.13209](https://arxiv.org/abs/2205.13209) | `2205.13209.pdf` |
| `SGBS/` | Simulation-guided Beam Search for NCO | NeurIPS 2022 | [2207.06190](https://arxiv.org/abs/2207.06190) | `2207.06190.pdf` |
| `Learn-Improvement-Heuristics/` | Learning Improvement Heuristics for Routing | IEEE TNNLS 2022 | [1912.05784](https://arxiv.org/abs/1912.05784) | `1912.05784.pdf` |
| `PyVRP/` | PyVRP: A High-Performance VRP Solver Package | INFORMS JOC 2024 | [2403.13795](https://arxiv.org/abs/2403.13795) | `2403.13795.pdf` |

## 求解器库（无单一方法论文）

| CC_Compare 文件夹 | 说明 | 引用方式 |
|------|------|------|
| `OR-Tools/` | Google OR-Tools，约束求解/路由库，无单一可引用论文 | 引用官方文档 <https://developers.google.com/optimization>；或 Perron & Furnon 相关技术文档 |
| `PyVRP/` | 有论文（上表），归入方法类而非纯库 | `2403.13795.pdf`（JOC 2024, DOI 10.1287/ijoc.2023.0055） |

## 备注

- 主表候选 `CO-enriched-ML` 与 `RouteFinder` 的原文**此前缺失**，现已补齐（前两个是动态 VRPTW / VRP 基础模型的主基线）。
- 与 `对比方法/对比方法/`（17 篇总表）不重复；两者合起来覆盖 `CC_Compare/` 全部可引用方法。
- 引用时注意口径：PyVRP/OR-Tools 是求解器，走「同协议适配 + C0 重放」而非原论文数值直接对标。
