# 归档目录说明

本目录保存已封存的冻结版本与历史资产。**只增不改不删**：正式冻结包只允许整目录移动，不允许修改内部任何文件；历史文档保存证据，不消除重复。

## 当前唯一正式冻结包

`正式冻结包/O0-CC_DEV-GATE/o0cc_devgate_2dfff7ae_20260908_v4/`

| 身份 | 值 |
|---|---|
| archive_content_sha256（19 源码，相对路径+hash） | `6e4db0c259c0c924843424290a06f2a831be85b7587c7f2b32894b01c790ac52` |
| bundle_sha256（29 文件：源码+输入+证据+清单） | `1bbcc978b0f4a60f82389629dc709cbc47373be38bba89d28395b21f9bc5e466` |
| compute_sha256（15 计算文件） | `2dfff7ae8948ce6eea2bb7b2b60fdae6924ca294652fffff66a7dee0dca7b888` |
| control_sha256 | `49019c4adf0cd71c80a81c81897fd7479c2c18e8bdf767904de90865073b848e` |
| analysis_sha256 | `cd266de12c267aa44ef1bae9a72d5a1b4dfda3a13b44e326ae02a9e5c339ca30` |
| runner_sha256 | `b74ea7cb5c5b2b7b5497aa16a752cf688a41219ab2edce3703643d7c3181b322` |
| profile_hash | `df1f7a6e935f00a02f00be8e66abec64abcbb13c1021006fc1f6c1701eb325b6` |

## 旧冻结包（为何作废）

| 目录 | 状态 |
|---|---|
| `诊断与作废冻结包/frozen_version_v1/` | 早期不完整冻结方案（8 文件 + basename 键），已取代 |
| `诊断与作废冻结包/o0cc_devgate_2dfff7ae_20260908/` | 缓存污染诊断版：封存后产生 `.pyc`/`__pycache__`，**禁止正式使用** |
| `诊断与作废冻结包/o0cc_devgate_2dfff7ae_20260908_v2/` | 完整封存但验证器未归档；已由 v3 取代 |
| `诊断与作废冻结包/o0cc_devgate_2dfff7ae_20260908_v3/` | 清单用 Windows 默认 GBK 编码写出，Linux 验证器读取失败；已由 v4 取代 |

## 正式结果能否引用

只有 `正式冻结包/O0-CC_DEV-GATE/o0cc_devgate_2dfff7ae_20260908_v4/` 是当前正式候选冻结版本。其余均为诊断/作废，不得用于正式 DEV-GATE 判定。

## 验证命令

```bash
# 服务器核验时必须先核验验证器自身（SHA-256 见下），再用它核验归档
python -B 正式冻结包/O0-CC_DEV-GATE/o0cc_devgate_2dfff7ae_20260908_v4/evidence/verify_source_archive.py \
    --archive 正式冻结包/O0-CC_DEV-GATE/o0cc_devgate_2dfff7ae_20260908_v4 \
    --expected-bundle-sha256 1bbcc978b0f4a60f82389629dc709cbc47373be38bba89d28395b21f9bc5e466
```

> 以上路径相对本 `archive/` 目录。验证器自身 SHA-256：
> `c8f5c37c12e661345414067f03b2b843c95dc441b457af6b9310cb2c93bc9394`

## 禁止事项

- **禁止原地修改**正式冻结包内的任何文件；源码变化必须创建新版本目录。
- **禁止自动同步**（VSCode SFTP `autoUpload` 必须关闭）覆盖归档。
- **禁止在归档内输出结果**；正式运行结果必须写到归档外的新目录。
- **禁止在归档内运行未加 `-B`/`PYTHONDONTWRITEBYTECODE=1` 的 Python**，避免生成 `.pyc` 污染归档。
- 服务器封存后执行 `chmod -R a-w <归档目录>`（Windows 只读属性不随传输保留）。
