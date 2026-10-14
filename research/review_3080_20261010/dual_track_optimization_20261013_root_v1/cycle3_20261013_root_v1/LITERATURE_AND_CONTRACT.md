# 双线第三轮：实际依赖守卫与精确距离缓存

记录日期：2026-10-13。第三工程循环，生产与独立验收完成后才追加实际结果；未经证据的候选不安装。

## 针对瓶颈的论文依据

Niels A. Wouda、Leon Lan、Wouter Kool，2024，**PyVRP: a high-performance VRP solver package**，INFORMS Journal on Computing 36(4):943–955，DOI [10.1287/ijoc.2023.0055](https://doi.org/10.1287/ijoc.2023.0055)，[arXiv 2403.13795v2](https://arxiv.org/html/2403.13795v2)，[期刊软件快照](https://github.com/INFORMSJoC/2023.0055/blob/main/README.md)。本轮读取§4.3和§4.3.2，沿用前轮读取的§5.3范围，没有运行PyVRP。本轮借鉴路线邻域缓存思路，缓存原逐边累加结果而不改排序；不是其C++路线操作实现。论文的静态距离目标和软约束搜索不能替代本项目原完整冷链硬约束、承诺/揭示、浮点算术或实际预算，原任务效应不能迁移。

Russell W. Bent、Pascal Van Hentenryck，2004，**Scenario-Based Planning for Partially Dynamic Vehicle Routing with Stochastic Customers**，Operations Research 52(6):977–987，DOI [10.1287/opre.1040.0124](https://pubsonline.informs.org/doi/10.1287/opre.1040.0124)。阅读范围仍是出版摘要和书目，本轮未新增全文阅读；本项目配对接受/拒绝K10保持原C0效用，依旧不属于原consensus MSA复现。其他四篇学习/迭代参考及阅读范围见[第一轮记录](双线共同迭代_论文依据与本轮工程证据_2026-10-13.md)，本轮没有据此选择模型/生成科学数据。

工程参考：[Python 3.12 functools官方文档](https://docs.python.org/3.12/library/functools.html#functools.lru_cache)。本轮读取lru_cache的参数键、类型、边界容量、返回值引用和纯函数适用限制；当前文档页标3.12.15，实际工程解释器仍必须按3.12.7封存。本候选是自定义有界缓存，不声明直接采用该decorator；不能缓存动态可行性/品质/随机/计时函数，也不能依靠同对象指针认定距离矩阵未变。

配送身份守卫属于工程设计，并非新NCO论文算法。它重算全部实际文件与全活参数SHA，不用mtime代替字节身份；实际依赖集合少于旧全注册树时必须明记差异。若未加载文件变化漏检，原全文件门仍FAIL，不能靠“实际尚未使用”改写旧注册。参考[JAX官方浮点FAQ](https://docs.jax.dev/en/latest/faq.html#jit-changes-the-exact-numerics-of-outputs)的范围不变，不据此放宽评分逐位门。

## 科学放行边界

本轮工程验证与任何完整请求尝试均需准确新登记、真实资源、原件保留和独立核查。组件PASS不等已训同初始化CE/同任务随机归因、真正独立强动态、完整等耗时MaskCO掩码重构迭代或多新鲜日校正确认。第2族FAIL封闭；第3族尚未启动。原取货终局callback80/pending空没有81可继续，不做旧日或旧已合格QA。
