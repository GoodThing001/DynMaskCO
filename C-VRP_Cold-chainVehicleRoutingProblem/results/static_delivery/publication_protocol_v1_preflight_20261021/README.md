# 组合模型v1：数据与可行性预检

本目录为2026-10-21协议诊断，不是MaskCO/ALNS性能比较或训练结果。

- [模型与数据尺度](../../../docs/static_delivery_research_20261021/组合模型与数据尺度_v1.md)
- [摘要](summary.json)：56原实例，47可行，9例有必要条件冲突证据；4组几何、4折；188份完整路线集合独立重算。
- [源文件清单和排除证据](dataset_manifest.json)、[几何四折](geometry_folds.json)。
- [可行路线与货损约束证书](feasibility_witnesses.json)：近邻路线只证明可行性/约束作用；双客合并只证明非退化，不声称全局最优。
- [六种条件预检](sensitivity_preflight.json)、[源代码与配置SHA](source_code_sha256.json)。

不得把39例找到低品质路线解释为39个最优解的约束必然绑定，不得把47个时间窗变体或多种子当成47组独立几何。
