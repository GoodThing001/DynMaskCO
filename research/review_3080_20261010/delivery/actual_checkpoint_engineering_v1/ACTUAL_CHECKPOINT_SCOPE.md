# 真实完整 MaskCO 架构一步工程资格

这里仅验证正式权重的完整神经路径、原 TRAIN 夹具的 C0 投影、保留边与重载。工程注册 `ACTUAL_FULLSTEP_ENGINEERING_REGISTRATION.json` SHA 为 `e06823ec5b2f09fb84b091e4182b38d8a9635d120e1f0c62299499a596bd95a7`；研究模型资格永久 false，不可把工程更新的权重拿去 DEV 选择收益。旧配送产物、上游与在跑树保持只读。

`ACTUAL_CHECKPOINT_INSPECTION.json` 已从 SHA `03615b08…` 的真实 305,061,150 字节权重独立读取：encoder 16 层、decoder 6 层、512 维、8 heads、3 输入；234 参数叶、76,259,840 个参数，原计算配置 bfloat16、工程运行按旧正式 inference adapter 使用 float32。`step=0` 是权重卡的元数据；是否等同随机初始化由实际参数哈希比较，而非这个数字决定。

`ACTUAL_TEACHER_C0_REPLAY.json` 已从旧 TRAIN 的第一条认证记录读完整 target，按原冻结 generator 的原 seed/instance 再生旧夹具：51 节点，16 条保留边，全部交付/完整/硬可行通过，C0 完整 cost 独立复算误差 0，缺客户负向对照确实触发。这些是旧数据的工程夹具，不是新科学样本或最优教师证据。

本地实际注册资源门因空闲 RAM 不足 8GiB 退出 1，留档于 `local_resource_gate_v1/`；没有进行模型优化。这是资源门未过，不是模型失败。下一步由根执行端在已核实空闲的 3080 CPU 上运行；原生 Windows JAX 仍是 CPU，不能写成 GPU 训练。Linux 两块 5090 当前被团队占用，不做中断或抢占。

`actual_full_step_qa.py` 会真正加载原架构与全部权重，以客户行正边 log-softmax CE 做 1 固定 SGD 步。对全部 234 参数求梯度并记录 encoder/decoder 有限梯度、实际参数变化；输入真实 51×3 可见坐标/需求，读入实际保留邻接。输出随后走现有 regret-2 重构与 C0 完整认证，保存全部路线、插入步骤、manifest/trace 和成本分解。原有 logit 加权插入属于项目的混合投影，不是论文原解码算法的逐字复现；末端完整硬认证与保留边均须通过。

权重保存为新 `ENGINEERING_ONLY_FULL_FINAL.pkl`，重载后实际 logits 要逐位相同，原输入 checkpoint 与 38 项源码起止 SHA 必须不变。新程序自身两个 SHA 已 pin。该过程不测 10 秒总预算、不给效用裁决；random 对照仅核实际配置 rng888 的同架构初始化身份，不冒称完成 random42 比较；本一步就是 CE-only reconstruction，CE+偏好与 CE-only 科学消融没有执行。

[MaskCO 官方论文](https://proceedings.iclr.cc/paper_files/paper/2026/file/d85816bf41e54d9b0847e02249f4fcc6-Paper-Conference.pdf) 支持部分解条件重构与完整 Transformer 路径；本入口的具体 CE 来自实际 `training/train_cvrp.py`，区别于论文一般 BCE 式。3 输入原权重没有直接看温区、TW、品质率；10 维扩展及其单位/货物域、真实目标教师、强对照和动态在线资格仍待实现注册。工程通过不能代替这些资格，更不能证明稳定正向。

```powershell
python research/review_3080_20261010/delivery/actual_checkpoint_engineering_v1/actual_full_step_qa.py --root C-VRP_Cold-chainVehicleRoutingProblem --upstream MASKCO_code --checkpoint MASKCO_code/ckpts/cvrp100.ckpt --registration research/review_3080_20261010/delivery/actual_checkpoint_engineering_v1/ACTUAL_FULLSTEP_ENGINEERING_REGISTRATION.json --out research/review_3080_20261010/delivery/actual_checkpoint_engineering_v1/REMOTE_QA_NEW --cpu 15
```

根执行端须先确认该 CPU 与内存可用，完整日志、真实远端 exit 和新目录均保存。只可上传源码/实验 JSON，本文 MD 不上传。
