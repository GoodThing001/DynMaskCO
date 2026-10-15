# V1.11：真实输入 F 前向工程资格（零比较、零安装）

2026-10-15，V1.10 64/64 输入检查和独立1619/1619通过后，执行前追加。本任务属于用户“核查通过则继续”的工程延续，不启动明确禁止的F−D-Geo比较、训练、L2/L3、40日或最终种子。

输入只用 V1.10 已保存的64个 original 输入，原序全部留分母，禁止改掩码、搜索输入或看评分选状态。模型checkpoint固定f5acf83fa2519020aeea274df5ecf740a6ad809b215d2d270209b7901903a7f5，当前骨干固定03615b08a55125a8fe4a6dad1df9ab592b56f7ed9d21b54d2562070b3c6d38db。必须经load_mpre_trained→partition→nnx.merge恢复train_state；使用训练元数据seed42、F_NODE10、F_ACTION_EXPLICIT20，检查恢复后83个训练叶子逐位一致，当前重建冻结encoder只按当前文件身份声明。

评分采用现有 mtrained_score_fn(model,Nv_max=30,M_max=26)，维度来自已验收64输入的最大值（Nv27–30、M6–26），不依评分结果决定。单进程CPU顺序执行，JAX float32和模型默认均不改；不启用缓存评分变体，不改原权重和上游。

事前固定调用数：以第1状态作一次初始化/编译前向；然后每状态完全相同输入恰好两次前向，最多129次。编译/初始化单独记录，最多60s；批含初始化/最终写盘最多180s；每状态两次重复评分跨度最多10s，**这些是离线接口检查的工程上限，不宣称通过L3完整在线10s发布门**。独立Job进程树监督和父进程截止；原件逐状态落盘。任何身份、结构、finite、shape或重复逐类型逐bits不一致，保留失败，后续槽NOT_ATTEMPTED_AFTER_FAILURE，不重试、不追加状态；超时/资源异常同样保留。原始比较规则、NaN payload和dtype规则不放宽。

记录：所有原始输入SHA、P0/有序legal指纹、原评分完整numpy数组及其dtype/shape/bytes、两次时长/逐位结果、历史first-argmax一致性（仅同一评分向量的规则审计，不应用动作）、源码真实origin与起止SHA、包/硬件、训练元数据、checkpoint起止SHA。原始与隐藏扰动输入在V1.10已逐位相等，故本阶段不额外调用隐藏分支F，不调用D-Geo。

标签强制保留F_RECONSTRUCTED_WITH_CURRENT_BACKBONE / HISTORICAL_BACKBONE_UNVERIFIED；通过只得真实输入评分接口工程资格。C0完整认证、D8/E2及科学执行资格继续BLOCKED；stable_online_gain=False。无效用、显著性、学习贡献或行为分歧主张。
