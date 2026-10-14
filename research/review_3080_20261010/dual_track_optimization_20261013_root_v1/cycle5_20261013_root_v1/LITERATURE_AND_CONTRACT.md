# 双线第五轮：动作匹配评分头更新与精确插入计算复用

2026-10-13。Arbor工程循环5/12，配送n12和取货n11；工程树不是科学族账本。先读当前状态、交接和WORK_STATUS，以第四轮实际原件为起点。所有旧FAIL保持；配送第2族永久封闭、第3族未启动。新组件准确登记优先于本轮概要，实际结果在完成后追加。实际计算共享CPU7串行，先取货生产/独立核查，再配送；模型主干CPU15/RAM≥8GiB门不降低。

## 原论文及阅读范围

Lvda Chen、Yang Li、Junchi Yan，2026，*MaskCO: Masked Generation Drives Effective Representation Learning and Exploiting for Combinatorial Optimization*，ICLR 2026正式会议论文。[官方出版页](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d85816bf41e54d9b0847e02249f4fcc6-Abstract-Conference.html)、[正式PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/d85816bf41e54d9b0847e02249f4fcc6-Paper-Conference.pdf)、[官方实现](https://github.com/Thinklab-SJTU/MaskCO)。本轮实际读摘要、§3部分解/选择算子定义与§4.1变量BCE训练。针对配送目标与插入动作错位，本轮接通当前客户/车槽/位置/前后锚点对应的CE更新工程。原论文变量BCE不同于本工程的每客户可行动作组CE，不能称原训练复现；暂存latent意味着主干冻结，没有新76M梯度/前向，两个工程步骤不能升级正式已训同初始化CE或同任务随机资格。原论文静态任务结果不迁移为冷链在线收益。正式放行还需完整掩码重构与多次迭代、完整C0/硬约束、公平预算和独立归因控制。

Stéphane Ross、Geoffrey J. Gordon、J. Andrew Bagnell，2011，*A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*，AISTATS 2011，PMLR 15:627–635。[官方元数据](https://proceedings.mlr.press/v15/ross11a.html)、[PDF](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf)、arXiv:1011.0686。实际读§2.1、§3及Algorithm3.1，区分专家状态监督与由学习策略产生的状态分布。未来拟在预登记新科学族中固定学习轨迹采集/专家查询/聚合顺序，防止训练与部署分布错位；本轮没有执行DAgger或采集新轨迹。论文验证集挑策略不采用，沿用本项目固定最后一步及校正确认；理论条件不自动覆盖有界近似教师、冷链不可逆承诺和完整10秒预算。

Stéphane Ross、J. Andrew Bagnell，2014，*Reinforcement and Imitation Learning via Interactive No-Regret Learning*，arXiv:1406.5979v1预印本，网页写明当时NIPS 2014审稿中；[arXiv官方页](https://arxiv.org/abs/1406.5979)、[DOI 10.48550/arXiv.1406.5979](https://doi.org/10.48550/arXiv.1406.5979)。本轮仅读官方摘要和版本/状态元数据，未读算法全文。其利用动作成本的思路提供后续完整冷链cost-to-go监督线索，尚未采用/执行；不以教师动作标签虚构成本偏好，不据此宣称实现AggreVaTe或具有论文保证。

Niels A. Wouda、Leon Lan、Wouter Kool，2024，*PyVRP: a high-performance VRP solver package*，INFORMS Journal on Computing 36(4):943–955；[DOI 10.1287/ijoc.2023.0055](https://doi.org/10.1287/ijoc.2023.0055)、[arXiv:2403.13795v2全文](https://arxiv.org/html/2403.13795v2)。沿用第四轮实际已读§4.3/§4.3.2/§5.3的成本复用机制依据。本轮取货只拟在一次原插入vehicle循环中复用完全相同的原base距离，不改成本delta重分组、局部邻域、物理评价或incr_eval=False，不安装/执行PyVRP。局部省调用数不等公平提速，更不等独立强动态接拒或学习收益；工程合格后完整K10和等耗时部署仍需另登记。

## 冻结与资格边界

旧二十轮/I3/D2/183/C1、上游与比较仓库、所有失败/封存工件和最终取货20260927/28不修改。本轮不重跑旧已合格QA、34编译候选/9099审计、60首前向、旧计时请求/100前向、旧缓存请求/103局部组件或原取货全日/50后续/31107/终局。取货权威epoch64/callback80/pending空不变，没有callback81。新局部独立参考执行仅针对新登记机制/固定未执行工程分支；不把局部新参考说成历史原件重放完备。

完整实际学习提议与正式同初始化已训CE/同任务训练随机、独立强动态接拒、完整C0等耗时掩码重构迭代、多新鲜研究日/种子与校正确认、科学全加载依赖和完整整请求预算仍是科学放行条件。稳定、可归因核心在线增益未建立；禁止把损失下降或工程PASS当收益。
