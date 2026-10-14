"""Expose newest reports at current entries, and finalize update identities."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';R=ROOT/'research/review_3080_20261010'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    manifest=R/'current_entries_before_multistep_teacher_telemetry_20261011/UPDATE_IDENTITY.json';m=json.loads(manifest.read_text())
    top='\n> 2026-10-11最新工程阶段：配送8多步旧状态/944候选，实际2016项环境封存及独立9670项通过；取货整请求预算日志已更正，完整事务运行仍待。无新训练/确认；稳定MaskCO核心在线增益仍未建立。详见本页最新节和两线新报告。\n'
    for rel in ('项目当前状态.md','00_项目导航.md','docs/当前规划/决策与审计/三线新增交接_独立核查修正与下一阶段_2026-10-10.md'):
        p=E/rel;txt=p.read_text(encoding='utf-8');head,sep,rest=txt.partition('\n');p.write_text(head+'\n'+top+rest,encoding='utf-8')
    reports={'01_取货线':'取货_整请求硬预算日志更正_2026-10-11.md','02_配送线':'配送_多步插入教师工程验收与环境闭包_2026-10-11.md'}
    for folder,name in reports.items():
        for index in ('文档索引.md','实验结果索引.md','README.md'):
            p=E/folder/index;txt=p.read_text(encoding='utf-8');link='\n- [2026-10-11最新工程资格与更正](文档/结果与进度/'+name+')\n';p.write_text(txt+link,encoding='utf-8')
    for x in m['files']:x['after_sha256']=sha(E/x['path'])
    with manifest.open('w') as f:json.dump(m,f,indent=2)
    print(json.dumps({'current_entry_headers':3,'indexed_reports':2,'identity_entries':len(m['files'])}))
if __name__=='__main__':main()
