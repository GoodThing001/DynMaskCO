"""Append finite actual-decoder oracle evidence without extending its scope."""
import hashlib,json,shutil
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];E=ROOT/'C-VRP_Cold-chainVehicleRoutingProblem';R=ROOT/'research/review_3080_20261010';DOC=E/'docs/当前规划/决策与审计/三线新增交接_独立核查修正与下一阶段_2026-10-10.md'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=R/'delivery/insertable_teacher_oracle_probe_root_v1/RESULT.json';a=json.loads(p.read_text());assert a['failed']==0 and a['checks_count']==18 and a['all_two_partial_teacher_edge_targets_reproduced_by_this_oracle']
    entries=[E/r['path'] for r in json.loads((R/'current_entries_before_insertable_teacher_correction_20261011/UPDATE_IDENTITY.json').read_text())['files']]
    backup=R/'current_entries_before_finite_teacher_oracle_closure_20261011';backup.mkdir(exist_ok=False)
    section='''

## 2026-10-11：实际插入解码器的有限可执行性探针

在两份同一旧工程状态上，以已知教师目标的无向边取1、其它边取0构造理想logit，保持实际部署的标准化、guide_weight=1、原mask和原`reconstruct`代码。两个部分掩码目标和两个全部保留边边界均精确重现教师邻接及完整C0成本，18项0失败，所有客户完整服务、保留边不变。原件与实际生产解码器源码哈希在前后核对一致。

这比v3单纯骨架必要条件多了一步：证明所测两个部分掩码标签存在可交给实际贪心插入流程的边分数。它不证明任意状态或全部目标可达，也未执行任何学习模型、训练或新数值日。将`delivery/insertable_teacher_oracle_probe_root_v1/RESULT.json`作为有限存在性证据，不能升级为“模型已学会”或MaskCO归因；后续仍需完整科学源闭包、多步教师状态资格及数据前新协议。取货父投影/现场安装事务仍待，负结果31项必要门维持，无稳定在线增益。
'''
    manifest=[]
    for f in entries:
        saved=backup/f.relative_to(E);saved.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,saved);old=f.read_text(encoding='utf-8')
        if f.suffix=='.json':
            s=json.loads(old);s['updated']=datetime.now(timezone.utc).isoformat();s['delivery'].update(finite_teacher_oracle_checks=18,finite_teacher_oracle_failures=0,finite_teacher_targets_reproduced=2,finite_teacher_oracle_identity=sha(p),all_states_greedy_decoder_reachability_proven=False,specific_learned_greedy_decoder_teacher_reachability_qualified=False,next='seal complete science source/interpreter closure and multistep teacher-state engineering, then new pre-data registered hypothesis; two finite oracle cases are not learned gain');new=json.dumps(s,ensure_ascii=False,indent=2,allow_nan=False)
        else:
            new=old.replace('共同K10/完整C0184项','共同K10/完整C0，184项')+(section if f==DOC else '')
        f.write_text(new,encoding='utf-8');manifest.append({'path':f.relative_to(E).as_posix(),'before_sha256':sha(saved),'after_sha256':sha(f)})
    (backup/'UPDATE_IDENTITY.json').write_text(json.dumps({'files':manifest,'shared_MD_uploaded':False,'frozen_science_modified':False},indent=2));print(json.dumps({'local_entries_updated':len(manifest),'finite_actual_decoder_oracle_checks':18,'stable_online_gain':False}))
if __name__=='__main__':main()
