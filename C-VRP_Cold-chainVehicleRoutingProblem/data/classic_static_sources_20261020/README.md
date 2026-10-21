# 静态经典数据：原Solomon与作者VRPTWQ改编版

文档版本：[全项目文档索引](../../docs/文档索引与版本状态.md)。

2026-10-20。原件只读使用，不调用生成器，不增加、复制或缩放客户，不凭坐标反造业务需求/产品标签。

## 原Solomon

来源：[SINTEF 100-customer经典页](https://www.sintef.no/projectweb/top/vrptw/100-customers/)，原论文Solomon(1987)，DOI10.1287/opre.35.2.254。官方备用[ZIP](https://www.sintef.no/globalassets/project/top/vrptw/solomon/solomon-100.zip)的SHA256为`8a0a72cbe6b7f8f9988ace4ebde0378ec34943acaaac47f2c408915e41887747`。56份、每份100客户+仓库，原路径`solomon_100/instances/In/`，逐件SHA见`solomon_100/FILES_SHA256.json`。

原目标为先最少车辆、再最短距离；距离与时间按双精度，参考距离显示两位小数。原数字是经典抽象单位，不直接当km、kg、min，不引入温度、产品、售价。BKS页面原HTML本地保存并绑定SHA。RC101参考当前为1696.95，遵循页面舍入勘误；不同车数不报距离gap。

## 作者品质约束改编版

来源：[SelinHulagu/vrptwq-solomon-instances](https://github.com/SelinHulagu/vrptwq-solomon-instances)，对应[An Exact Algorithm for the Vehicle Routing Problem with Time Windows and Perishable Products](https://doi.org/10.1287/trsc.2025.0262)。固定commit`58eadb8fe16a354b29877f402d17350ce639795d`，原ZIP/README和逐文件SHA保留。

56份、每份100客户+仓库。作者README声明坐标km、需求kg、时窗/服务min；容量20,000kg、速度60km/h、距离费1€/km，其他品质与制冷参数在README另列。作者已做坐标×3.5、需求×100、24h时域等改编；本项目直接使用作者原文件，不自行重建。文件在`VRPTWQ_Solomon/instances/vrptwq_instances/`。

这是带明确单位的**数值基准**，并非番茄/鲻鱼业务实测。初始品质400、最大下降80等不是实际食品指标，没有商品收入和采购价。不能直接把论文热学设定、电费系数、18m³机组和20t载重拼为同车。论文全文本轮直取403，只核机构摘要和作者README，不宣称其精确算法已复现。

后续[数值情景审计](../../docs/static_delivery_research_20261020/net_benefit_scenario/README.md)发现：按README的60km/h、90分钟服务、0—1440分钟仓库及本轮返仓约束，56例中9例存在单客户直达/返仓时间下界冲突。结论限定于本轮明确运行域，不由此断言作者论文出错。原件和SHA保持，冲突不通过改坐标/时窗修复；六类各选首个通过必要条件的原例开展试算，47例通过必要条件也不等于全都已证明可行。

未见独立数据LICENSE文件，不由代码MIT或期刊文章许可推出数据再分发许可。此目录用于本地研究核验；公开发布范围应单独确认。

总审计`DATASET_AUDIT.json`含两套逐实例计数、需求、时域和SHA。原RRP30例仍在原目录，保留为已有冷冻运输成本基准，不混入新目标训练/测试。
