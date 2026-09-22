# CC_Compare 仓库冻结清单（下载于 2026-09-08 18:14:02）

| 文件夹 | 上游仓库 | commit | 默认分支 | 许可证 | 下载方式 |
|---|---|---|---|---|---|
| PyVRP | PyVRP/PyVRP | 4b2aa285f34efb8e2ff1995a87b834414c1c5c17 | main | MIT (LICENSE.md) | shallow (depth 1) |
| CO-enriched-ML | tumBAIS/euro-meets-neurips-2022 | e90e91008ff2c447f5c6a5b1102efac8060bcce1 | main | MIT | shallow (depth 1) |
| RouteFinder | ai4co/routefinder | cc3ab078650e8db62c0562a60f421defe38d5ae3 | main | MIT | shallow (depth 1) |
| MVMoE | RoyalSkye/Routing-MVMoE | af29e5af0595f94f3ecc3bc46d72df1089a62682 | main | MIT | shallow (depth 1) |
| Learning-to-Delegate | mit-wu-lab/learning-to-delegate | 4b54d65e8809d628897d7367ced321f6a5e886cc | main | 无 LICENSE 文件，待核实 | shallow (depth 1) |
| POMO | yd-kwon/POMO | d7c3d6ea580499a53e874fe9e065f69e799a8551 | master | 无 LICENSE 文件，待核实 | shallow (depth 1) |
| DeepACO | henry-yeh/DeepACO | 9a756a3fe6cf9627ecc166110a02fdbb61f2a0d7 | main | MIT | shallow (depth 1) |
| AttentionModel | wouterkool/attention-learn-to-route | c9abf41ac2f878a55b20dc7e829bc942bb999631 | master | MIT | shallow (depth 1) |
| Omni-VRP | RoyalSkye/Omni-VRP | 8950e159ed4fb6cd17da5b53aa12dcc9f3e98772 | main | MIT | shallow (depth 1) |
| Sym-NCO | alstn12088/Sym-NCO | bbd2f16548bb005e0e9b638766c3cc92127a9fc5 | main | 无 LICENSE 文件，待核实 | shallow (depth 1) |
| SGBS | yd-kwon/SGBS | d8038171628e2592358865ba5268e09f8e1d4ce7 | main | MIT | shallow (depth 1) |
| Learn-Improvement-Heuristics | WXY1427/Learn-Improvement-Heuristics-for-Routing | ed96abeffd95f9910e4b3af1ef933dce3995c0da | main | 无 LICENSE 文件，待核实 | shallow (depth 1) |

## 附加冻结资产（下载于 2026-09-08）

- **RRNCO checkpoints**：`RRNCO/checkpoints/{rcvrp,rcvrptw,atsp}/epoch_199.ckpt`（来源：服务器已有副本，上游 ai4co/real-routing-nco）
- **CaDA checkpoints**：`CaDA/50/result/2024-1111-1139/checkpoint-300.pt`、`CaDA/100/result/2024-1121-1355/checkpoint-300.pt`（来源 HF `Goodyee/CaDA` checkpoint.zip，经 hf-mirror，CRC 校验通过）
- **CaDA data**：`CaDA/data/{lib_data,synthetic_data}`（来源 HF `Goodyee/CaDA` data.zip，经 hf-mirror）
- **RouteFinder checkpoints**：`RouteFinder/checkpoints/{50,100}/`（rf-transformer/rf-pomo/rf-moe，来源 HF `ai4co/routefinder`，经 hf-mirror）
- **MVMoE / POMO / AttentionModel / Omni-VRP / CO-enriched-ML**：权重仓库自带，无需外部下载
- **Learning-to-Delegate 权重**：另需 10GB Dropbox zip（非优先，未下载）
