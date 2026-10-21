# 公开资料获取

从扩展根目录运行，使用python -B；各工具的原数据、输出目录和业务逻辑保持原身份。

- [acquire_dataset_product_review_20261020.py](acquire_dataset_product_review_20261020.py)
- [acquire_direct_value_evidence_20261020.py](acquire_direct_value_evidence_20261020.py)
- [acquire_parameter_origins_20261020.py](acquire_parameter_origins_20261020.py)
- [acquire_publication_model_evidence_20261021.py](acquire_publication_model_evidence_20261021.py)
- [acquire_static_evidence_20261020.py](acquire_static_evidence_20261020.py)
- [acquire_static_evidence_fallback_20261020.py](acquire_static_evidence_fallback_20261020.py)
- [acquire_static_supplement_20261020.py](acquire_static_supplement_20261020.py)
- [acquire_static_vehicle_20261020.py](acquire_static_vehicle_20261020.py)

[返回工具分类](../README.md)。历史命令中的旧位置见[迁移清单](../../../results/workspace_organization_20261021/migration_manifest.json)。

其中acquire_static_supplement与acquire_static_vehicle包含顶层下载流程，不能作为无副作用模块导入；本次最终核验对这两份仅做静态检查。
