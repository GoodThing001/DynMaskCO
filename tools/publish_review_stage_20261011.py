"""Update local current entries only, preserving their previous bytes."""
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import os
import shutil

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / 'C-VRP_Cold-chainVehicleRoutingProblem'
DOC = EXT / 'docs/当前规划/决策与审计/三线新增交接_独立核查修正与下一阶段_2026-10-10.md'
REVIEW = ROOT / 'research/review_3080_20261010'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    backup = REVIEW / 'current_entries_before_review_20261011'
    backup.mkdir(exist_ok=False)
    now = datetime.now(timezone(timedelta(hours=8))).isoformat()
    entries = [EXT / '项目当前状态.md', EXT / '00_项目导航.md',
        EXT / 'docs/当前规划/结果与进度/执行进度表.md',
        EXT / 'docs/当前规划/结果与进度/三线持续研究_执行进度_2026-10-10.md',
        EXT / 'docs/当前规划/决策与审计/方向A_总核查与下一阶段路线图_2026-09-30.md']
    entries += [EXT / line / name for line in ['01_取货线','02_配送线','03_负结果线']
                for name in ['README.md','实验结果索引.md','文档索引.md']]
    provenance = []
    for p in entries:
        if not p.exists():
            continue
        relative = p.relative_to(EXT)
        saved = backup / relative
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, saved)
        link = os.path.relpath(DOC, p.parent).replace('\\','/')
        header = (f'> **2026-10-11 新增交接独立核查与修正（当前有效）**：'
          f'取货396项数值复核通过，但旧Stage A真实未来接受、漏车普查和物理单位缺陷已明确；'
          f'新全日普查1132事件/16980车记录、因果联合证书及8项总控回归通过，完整模型接线继续。'
          f'配送原训练仅head，负结果保留；完整masked合成QA已在3080 CPU通过，真实76M参数工程正在资格验证。'
          f'负结果当前入口已修正，62项对抗回归及183封存/14JSON独立通过，29headline零失败不等于科学资格。'
          f'稳定MaskCO在线增益尚未建立；新v3追加登记、三线继续、30分钟ACTIVE。文档仅本地。'
          f'[本轮完整依据与未决项]({link})。下方较早状态按历史理解。\n\n')
        p.write_text(header + p.read_text(encoding='utf-8'), encoding='utf-8')
        provenance.append({'path':relative.as_posix(),'before_sha256':sha(saved),'after_sha256':sha(p)})
    p = EXT / 'results/research_20261010_continuation_v2/WORK_STATUS.json'
    saved = backup / 'WORK_STATUS_before.json'
    shutil.copy2(p,saved)
    status=json.loads(p.read_text(encoding='utf-8'))
    status['updated']=now
    status['schema']='current-research-status-v3-review'
    status['current_review_document']=str(DOC.relative_to(EXT)).replace('\\','/')
    status['append_only_registration']='results/research_20261010_continuation_v2/review_followup_v3/CONTINUATION_V3_REGISTRATION.json'
    status['append_only_registration_sha256']='2184df0054789f1866b959fa462e4cf660ba1b372553b04ac7292f64df1b6035'
    status['pickup'].update(stage='causal_public_fleet_qualified_actual_masked_integration_in_progress',
        new_handover_numeric_checks=396,new_handover_numeric_failures=0,
        old_stage_a_causal_qualified=False,full_census_events=1132,full_census_vehicle_records=16980,
        finite_cohort_max_mutable=9,finite_cohort_max_owned=10,
        old_bare_contract_physics_qualified=False,independent_root_engineering_tests=8,
        independent_root_four_day_terminal_replay=True,
        next='actual full MaskCO public proposal plus all-arm matched history/units/request-budget qualification before fresh research data')
    status['delivery'].update(stage='old_head_training_negative_preserved_actual_full_masked_engineering',
        prior_train_scope='frozen_encoder_decoder_edge_head_only',old_head_trained_cycles=200,
        old_head_certified_pool_cycles=505,actual_original_parameter_elements=76259840,
        synthetic_full_masked_remote_3080_cpu_exit=0,
        next='actual checkpoint full-network engineering, C0 teacher/projection, then registered full training; tiny QA is not research eligibility')
    status['reproduction'].update(stage='current_v3_installed_independently_tested_and_archives_verified',
        independent_adversarial_and_compatibility_tests=62,original_package_files_unchanged=183,
        formal_json_verified=14,actual_support_cli_exit=2,
        headline_checks=29,headline_failures=0,
        scientific_qualification=False,all_numeric_dimensions_qualified=False,
        numeric_dimension_gap='impossibility-related composed upper bound lacks raw instance binding; not zeroed')
    status['resource_preflight_3080_20261011']={'available_ram_bytes':13375819776,
        'backend':'JAX_CPU','actual_checkpoint_sha256':'03615b08a55125a8fe4a6dad1df9ab592b56f7ed9d21b54d2562070b3c6d38db'}
    status['automation']='maskco-i3 ACTIVE each30min, latest three-line review prompt applied'
    p.write_text(json.dumps(status,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    provenance.append({'path':str(p.relative_to(EXT)),'before_sha256':sha(saved),'after_sha256':sha(p)})
    with (backup / 'LOCAL_CURRENT_UPDATE_PROVENANCE.json').open('x',encoding='utf-8') as f:
        json.dump({'updated':now,'current_entries_only':True,'shared_MD_uploaded':False,
                   'old_protocol_or_result_modified':False,'files':provenance},f,ensure_ascii=False,indent=2)
    print(json.dumps({'current_entries_updated':len(provenance),'old_protocol_or_result_modified':False,'server_MD_uploaded':False}))


if __name__=='__main__':
    main()
