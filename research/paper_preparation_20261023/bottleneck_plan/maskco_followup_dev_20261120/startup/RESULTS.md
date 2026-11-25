# 冻结 MASKCO 首次请求的编译复用

| 首次进程入口到两个热图 | 测量一（秒） | 测量二 | 测量三 | 中位墙钟秒 | 中位 CPU 秒 |
|---|---:|---:|---:|---:|---:|
| 原加载 | 61.89 | 60.35 | 61.45 | 61.45 | 61.42 |
| 复用编译缓存 | 13.10 | 13.02 | 12.71 | 13.02 | 13.02 |

预先构建缓存用时 65.05 秒，写入 31 个编译文件，共 599.9 KiB。复用后首次进程耗时中位数下降 78.81%，CPU 下降 78.80%。训练例 r101 两个固定掩码在全部进程中的非对角边分最大绝对差 0，每行前八名一致。

原科研服务器已在冻结训练例上完成预编译。在该服务器的相同 CPU 环境下，启动原求解入口前执行这四行即可复用缓存。

```bash
export JAX_ENABLE_COMPILATION_CACHE=true
export JAX_COMPILATION_CACHE_DIR=/home/hzeng/project/MASKCO-Main/_maskco_unified_external_20261120/startup02/compiled
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
```

独立进程仍要读取模型权重。复现用参数见[计时代码](startup_measure.py)和[计时及数值记录](results/startup02/summary.json)。

## 开发例的首次请求

三种既有来源各一例、每例两个种子，每块用同一物理核。对照均重新启动独立求解进程。

| 实现 | 首个 60 秒已交付可行车队 | 启动至在线搜索中位秒 | 缓存 MASKCO 减对照的 6 块平均 U |
|---|---:|---:|---:|
| 原加载 MASKCO | 3/6 | 59.88 | 未完整配对 |
| 缓存 MASKCO | 6/6 | 12.36 | +0.00 |
| 免权重加载的同规则距离 | 6/6 | 0.60 | -3.53 |
| ALNS | 6/6 | 1.97 | +37.33 |

统一 60 秒搜索过程中记录每个获准解的实际接受时间，截取首个 60 秒已可交付的车队。原加载用时过长时仍保留该块，并记为未及时交付。

[首次请求逐例数据](results/service01/paired_service.csv) · [配对方案](首次请求复核.md)
