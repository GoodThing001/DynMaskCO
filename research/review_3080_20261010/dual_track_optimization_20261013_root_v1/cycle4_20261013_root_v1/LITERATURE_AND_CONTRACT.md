# 双线第四轮：独占源码租约与缓存耗时边界

2026-10-13。工程循环4/12，Arbor n9/n10；工程树不是科学族账本。本轮两线实际阶段都已执行和独立核查，没有新模型训练、科学数据、服务器/GPU或共享MD上传。配送第2族FAIL封闭，第3族未启动。两位执行代理在启动实验前遇到模型容量错误，主代理在本地串行完成本轮组件，未重复派发。

## 原论文、官方资料与采用范围

Niels A. Wouda、Leon Lan、Wouter Kool，2024，*PyVRP: a high-performance VRP solver package*，INFORMS Journal on Computing 36(4):943–955；DOI [10.1287/ijoc.2023.0055](https://doi.org/10.1287/ijoc.2023.0055)，[arXiv:2403.13795v2全文](https://arxiv.org/html/2403.13795v2)。实际阅读§4.3、§4.3.2和§5.3，复核了路线算子缓存及扩展成本评价的讨论。取货继续借鉴“复用不变评价”的机制；本轮先对已存时钟和重复key做可检验诊断。没有安装或执行PyVRP，不复制其静态距离/软约束评价；冷链完整物理、接拒、承诺前缀和原串行浮点算术继续独立保留。下一候选仅拟在原插入循环中惰性复用同一路线的原始base距离，尚未登记/执行，不根据旧效用扩大缓存或修改搜索邻域。科学放行仍需精确语义、完整请求、独立强动态对照及正式学习归因。

Russell W. Bent、Pascal Van Hentenryck，2004，*Scenario-Based Planning for Partially Dynamic Vehicle Routing with Stochastic Customers*，Operations Research 52(6):977–987；[DOI 10.1287/opre.1040.0124及出版页](https://pubsonline.informs.org/doi/10.1287/opre.1040.0124)。实际阅读出版页摘要、作者和期刊元数据，未读完整PDF。其多场景/共识机制提供独立动态接拒设计线索；当前成对共同SAA不是该文MSA复现，原最大服务数量也不同于完整冷链效用。本轮没有执行新接拒策略，论文收益不迁移为MaskCO证据。

配送读[Microsoft CreateFileW参数说明](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-createfilew)与[GetFileInformationByHandle](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-getfileinformationbyhandle)接口说明，检验新独占副本的读共享句柄、文件身份与退出释放边界。只锁本轮复制件，不锁全局依赖、原源码或团队文件；单文件句柄不等完整目录或运行库不变保护。另读[Python3.12 open换行说明](https://docs.python.org/3.12/library/functions.html#open)：默认写模式换行会按系统转换。官网当前3.12.15与实际3.12.7须区分；解释本次已存CRLF与冻结writer未指定newline相符，不重写原字节或改原失败判据。

## 固定边界

本地实际Python3.12.7/psutil5.9.0/CPU7，只运行无模型组件；未下调完整JAX模型CPU15/RAM≥8GiB门。取货权威终局epoch64/callback80/pending空不重跑。没有旧34编译候选、9099审计、10计时请求/100前向/220恢复或其它旧QA重跑。保留全部旧FAIL、缺13800字节、缺逐推进ledger和原科学注册字节。
