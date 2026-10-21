# 公开论文案例表：QDVRP2021

来源：Stellingwerf等，*The quality-driven vehicle routing problem: Model and application to a case of cooperative logistics*，DOI[10.1016/j.ijpe.2020.107849](https://doi.org/10.1016/j.ijpe.2020.107849)。[项目机构原文](https://transsonic.nl/documents/Stellingwerf_EtAl_IJPE_2021_The_Quality_Driven_Vehicle_Routing_Problem.pdf)。

[tables_2_3_14.json](tables_2_3_14.json)转录原文Table2–3（p6）及Table14（p11），关键原页已渲染视觉核对；原始PDF SHA写在JSON内，核验见[研究记录](../../../docs/static_delivery_research_20261020/dataset_product_review/verification.json)。

- 0是中央配送中心，1—7是超市链配送中心；需求单位kg，距离km，速度km/h。
- 保留3类鲜切蔬菜Product1/2/3，未擅自补品种或品质指标标签。总需求12671kg，三产品分别6955/2649/3067kg。
- 距离对称，速度有向，不做对称化。速度表未打印对角值，以null标记未使用，不伪造观测。
- 数据来自发表案例表，不是随机客户生成，也不是作者发布的完整代码/原始订单包。矩阵直接使用，不反推坐标。
- 原文没有在这些表内给客户时间窗和商品售价/采购价，均留null；未将本案例冒称Solomon或完整利润基准。
- 同时转录原文温度/车辆/作业数值假设，保留275K/293K原约定。没有给它补上番茄、鱼或旧RRP车型身份。

此轮用于数据选型和后续模型核验，未计算路线、食品利润或作者算法复现成绩。复核工具：[build_dataset_product_review_20261020.py](../../../tools/evidence/analyze/build_dataset_product_review_20261020.py)。
