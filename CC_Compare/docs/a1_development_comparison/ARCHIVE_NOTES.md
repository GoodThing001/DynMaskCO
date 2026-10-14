# 归档说明：A-v1 开发期对比（development，非最终测试结果）

本目录是 **Markdown 文档的入库快照**。它存在的唯一原因：`CC_Compare/results/**` 被
`.gitignore` 忽略（有意——实验证据不入库），导致交付表与 README 在仓库里**看不到**。
按用户 2026-10-14 裁决，采用**文档与结果分离**：

| 类别 | 去向 | 是否入 Git |
|---|---|---|
| `README.md`、`A-v1对比方法总表.md`、`读表注意与已知限定.md` | **本目录** | ✅ 入 Git |
| `A-v1对比方法总表.xlsx`、`dataset.json`、效用图 PNG、`TIMEOUT_SCAN.md`、`V2批次结果.md`、`README_SECTION2_paste.md` | **Release 附件** | ❌ 不入 Git（`results/**` 忽略规则**保持不变**） |
| 审计交付物（资格矩阵 / 适配边界 / 独立复核） | `CC_Compare/audit/` | ✅ 入 Git |

## 名称与主张边界

`ARTIFACT_MANIFEST.json` 的 `kind` = `development-evidence-package`。
**本包不是最终测试结果，也不声称 SOTA。** 发布时名称须含 `development`，
**不得**使用暗示"最终测试结果"或"已证明最优"的标题。

## 归档内容不是"结果本身"，而是"当时的呈现"

本目录是**快照**：真正的数字来源是 `dataset.json`（Release 附件，含 SHA-256）与各臂的
`gate.json`。快照与 `results/` 下的活文件**会随重渲染而分叉**，这是设计如此；
`ARTIFACT_MANIFEST.json` 为每份快照记录 SHA-256，`--verify` 可校验快照未被改动。

## 必读的限定

引用本目录任何数字前，先读 `读表注意与已知限定.md`。至少包含：

- **主结论口径以主档 `p_c=(5,10,15)` 为准**；`p_c=0` 是敏感性结果。
- **配对必须同 batch 同源**：I3 主档三元组是 `456.317 / 600.997 / −144.680`；
  `631.4` 属**另一套协议**（外部臂配对基线 `a1_c1fresh_reveal_40|cond_hist`），**不得**并列。
  机器检查：`CC_Compare/tools/check_pairing_identity.py`。
- **Ours 行同样受 CPU 争用影响**（A3 19.83 / I3 37.33 次超时每天），且 **A3 配对身份未通过**。
- **三层级不可混**：可执行 / 可正式裁决 / 已建立优势 —— 前两层合格**不等于**第三层成立。

## 重新生成

```bash
# 1) 采集 + 渲染（服务器侧采集，本地渲染；也可全本地）
python CC_Compare/tools/collect_compare_table.py --json CC_Compare/results/a1_compare_table/dataset.json
python CC_Compare/tools/render_compare_table.py \
    --data CC_Compare/results/a1_compare_table/dataset.json \
    --outdir CC_Compare/results/a1_compare_table
# 2) 刷新本目录快照 + 清单
python CC_Compare/tools/make_artifact_manifest.py
python CC_Compare/tools/make_artifact_manifest.py --verify
```
