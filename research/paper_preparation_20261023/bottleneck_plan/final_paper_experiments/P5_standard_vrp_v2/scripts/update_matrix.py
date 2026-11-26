from pathlib import Path
p = Path("audit/V2_原始机制与标准适配差异矩阵.csv")
t = p.read_text(encoding="utf-8-sig")
rows = [
    "试选选项集合,三选项[-d,heat,heat_cal],两选项[heat,heat_cal]（V2 首轮）,偏离（D6a）,恢复为三选项,ranking_counts 三选项均被采用[0,2,1]",
    "试选计分对象,repair_batch 返回值（保留部分重构）,超时整次丢弃（V2 首轮）,偏离（D6b）,恢复为保留部分重构,试选 outcomes 非空且被采用",
    "SA 温度,2.5*max(.10,1-elapsed/s),额外乘 |best|/n（V2 首轮）,偏离（D6c）,恢复锁版公式,MECHANISM_PARITY §3.1",
    "内核内接受与record更新,repair_batch 内部（record 先更新）,池搜索层近似（V2 首轮）,偏离（D6c）,恢复为内核内,test_repair_batch_related_and_acceptance",
    "related 掩码构造,偶数次迭代按亲和度平方随机抽样,缺失（V2 首轮）,偏离（D6b）,恢复为 repair_batch 内实现,test 与冒烟 mask_mode_counts",
]
if not t.endswith("\n"):
    t += "\n"
t += "\n".join(rows) + "\n"
p.write_text(t, encoding="utf-8-sig")
print("matrix rows added")
