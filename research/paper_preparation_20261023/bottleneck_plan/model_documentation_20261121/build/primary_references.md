## 基础方法与分车思想

- Chen, L., Li, Y., and Yan, J. **MaskCO: Masked Generation Drives Effective Representation Learning and Exploiting for Combinatorial Optimization**. ICLR, 2026. [正式论文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d85816bf41e54d9b0847e02249f4fcc6-Abstract-Conference.html)；[作者代码](https://github.com/Thinklab-SJTU/MaskCO)。本项目的掩码生成网络基础。
- Beasley, J. E. **Route First—Cluster Second Methods for Vehicle Routing**. Omega, 1983, 11(4):403–408. [DOI](https://doi.org/10.1016/0305-0483(83)90033-6)。给定访问顺序再分车的经典思想。
- Vidal, T. **Technical note: Split algorithm in O(n) for the capacitated vehicle routing problem**. Computers & Operations Research, 2016, 69:40–47. [正式论文](https://www.sciencedirect.com/science/article/pii/S0305054815002798)。原文的线性复杂度依赖CVRP结构；本项目按冷链路线逐段评价的复杂度在正文单独说明。

## 外部比较方法

- Ropke, S., and Pisinger, D. **An Adaptive Large Neighborhood Search Heuristic for the Pickup and Delivery Problem with Time Windows**. Transportation Science, 2006, 40(4):455–472. [正式论文](https://pubsonline.informs.org/doi/abs/10.1287/trsc.1050.0135)。
- Wouda, N. A., and Lan, L. **ALNS: a Python implementation of the adaptive large neighbourhood search metaheuristic**. Journal of Open Source Software, 2023, 8(81):5028. [软件论文](https://joss.theoj.org/papers/10.21105/joss.05028)。
- Mladenović, N., and Hansen, P. **Variable neighborhood search**. Computers & Operations Research, 1997, 24(11):1097–1100. [DOI](https://doi.org/10.1016/S0305-0548(97)00031-2)。
- Hottung, A., Wong-Chung, K., and Tierney, K. **Neural Deconstruction Search for Vehicle Routing Problems**. Transactions on Machine Learning Research, 2025. [正式论文](https://openreview.net/forum?id=bCmEP1Ltwq)。
- Kool, W., van Hoof, H., and Welling, M. **Attention, Learn to Solve Routing Problems!** ICLR, 2019. [正式论文](https://openreview.net/forum?id=ByxBFsRqYm)。
- Kwon et al. **POMO: Policy Optimization with Multiple Optima for Reinforcement Learning**. NeurIPS, 2020. [正式论文](https://proceedings.neurips.cc/paper_files/paper/2020/hash/f231f2107df69eab0a3862d50018a9b2-Abstract.html)。
- Ye et al. **DeepACO: Neural-enhanced Ant Systems for Combinatorial Optimization**. NeurIPS, 2023. [正式论文](https://proceedings.neurips.cc/paper_files/paper/2023/file/883105b282fe15275991b411e6b200c5-Paper-Conference.pdf)。
- Bi et al. **Towards Efficient Constraint Handling in Neural Solvers for Routing Problems**. ICLR, 2026. [CaR正式论文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/1413fbda783a5bad7947c2f7396ba64b-Abstract-Conference.html)。
- Wouda, N. A., Lan, L., and Kool, W. **PyVRP: A High-Performance VRP Solver Package**. INFORMS Journal on Computing, 2024, 36(4):943–955. [DOI](https://doi.org/10.1287/ijoc.2023.0055)。
- Zhou et al. **MVMoE: Multi-Task Vehicle Routing Solver with Mixture-of-Experts**. ICML, 2024, PMLR 235:61804–61824. [正式论文](https://proceedings.mlr.press/v235/zhou24c.html)。
- Li et al. **CaDA: Cross-Problem Routing Solver with Constraint-Aware Dual-Attention**. ICML, 2025, PMLR 267:35438–35456. [正式论文](https://proceedings.mlr.press/v267/li25bi.html)。
- Berto et al. **RouteFinder: Towards Foundation Models for Vehicle Routing Problems**. Transactions on Machine Learning Research, 2025. [论文页](https://openreview.net/forum?id=QzGLoaOPiY)；[作者引用](https://github.com/ai4co/routefinder)。
- Son et al. **RRNCO: Towards Real-World Routing with Neural Combinatorial Optimization**. ICLR, 2026. [正式论文](https://proceedings.iclr.cc/paper_files/paper/2026/hash/9446c291a8744a125a0bda5b18f4d5a1-Abstract-Conference.html)。

## 数据论文与原始发布

- Hülagü, S., Ciancio, C., Dabia, S., and Dullaert, W. **An Exact Algorithm for the Vehicle Routing Problem with Time Windows and Perishable Products**. Transportation Science, 2026, 60(5):675–695. [DOI](https://doi.org/10.1287/trsc.2025.0262)；[作者数据](https://github.com/SelinHulagu/vrptwq-solomon-instances)。
- Solomon, M. M. **Algorithms for the Vehicle Routing and Scheduling Problems with Time Window Constraints**. Operations Research, 1987, 35(2):254–265. [DOI](https://doi.org/10.1287/opre.35.2.254)。
- Nisyrios, E. **Dynamic PDPCDPG with Rolling-Horizon**. Mendeley Data, V2, 2026. [原始数据发布](https://data.mendeley.com/datasets/93w8ynrk59/2)。
- Wen, M., Larsen, J., Clausen, J., Cordeau, J.-F., and Laporte, G. **Vehicle Routing with Cross-Docking**. Journal of the Operational Research Society, 2009, 60(12):1708–1718. [作者机构记录](https://orbit.dtu.dk/en/publications/vehicle-routing-with-cross-docking-2/)。
- Uchoa, E., Malheiros, I., Queiroga, E., Martinelli, R., and Subramanian, A. **Generation of the XTW and XLTW Instances**. CVRPLIB技术说明, 2026. [官方说明](https://galgos.inf.puc-rio.br/cvrplib/uploads/files/vrptw_challenge/instances.pdf)；[数据入口](https://galgos.inf.puc-rio.br/cvrplib/en/vrptw/instances)。
- Uchoa et al. **New benchmark instances for the Capacitated Vehicle Routing Problem**. European Journal of Operational Research, 2017, 257(3):845–858. [DOI](https://doi.org/10.1016/j.ejor.2016.08.012)。这是X坐标与需求底座的来源。
- Homberger, J., and Gehring, H. **Two Evolutionary Metaheuristics for the Vehicle Routing Problem with Time Windows**. INFOR, 1999, 37(3):297–318. [DOI](https://doi.org/10.1080/03155986.1999.11732386)。
- Homberger, J., and Gehring, H. **A two-phase hybrid metaheuristic for the vehicle routing problem with time windows**. European Journal of Operational Research, 2005, 162(1):220–238. [DOI](https://doi.org/10.1016/j.ejor.2004.01.027)；[SINTEF基准托管](https://www.sintef.no/projectweb/top/vrptw/homberger-benchmark/)。
- Bazelmans, ter Braake, Schrotenboer, van Lieshout, and Van Woensel. **The Vehicle Routing Problem with Pickups, Deliveries, and Scheduled Linehauls**. European Journal of Operational Research, 2026, 330(3):995–1008. [DOI](https://doi.org/10.1016/j.ejor.2025.07.070)。2025年8月23日在线发表，正式卷期为2026年。[作者实例库](https://github.com/siemterbraake/vrppdl-algorithms-instances/tree/7bbbd3e975eb77915aa255217af771e0fdb08e65)。
- FER–LARES, University of Zagreb. **A Multi-City Benchmark for Vehicle Routing in Real Urban Environments**. IEEE ITSC, 2026. [研究组官方论文说明](https://www.lares.fer.hr/lares/about_lares/news?%40=33uop)；[RWVRP作者数据与字段说明](https://github.com/unizg-fer-lares/RWVRP_benchmark/tree/2c3a976f2212923db31ac86aea11483930c64b14)。城市门店位置来源为OpenStreetMap贡献者，需求、时窗和服务为作者建模；[OpenStreetMap署名及许可](https://www.openstreetmap.org/copyright)。
