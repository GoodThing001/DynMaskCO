"""Append the current two-track iteration after executor evidence is complete."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]
TREE = ROOT / 'research/review_3080_20261010'
STAGE = TREE / 'dual_track_optimization_20261013_root_v1'
EXT = ROOT / 'C-VRP_Cold-chainVehicleRoutingProblem'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def create(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8', newline='\n') as f:
        f.write(v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, indent=2) + '\n')

def record():
    result = json.loads((STAGE / 'EXECUTOR_EVIDENCE.json').read_text(encoding='utf-8'))
    assert set(result) == {'delivery', 'pickup'}
    for arm in result.values():
        assert arm['stable_online_gain'] is False
        for rel, h in arm['evidence_hashes'].items():
            assert sha(ROOT / rel) == h, rel
    auto_path = Path(os.environ.get('CODEX_HOME', str(Path.home()/'.codex'))) / 'automations/maskco/automation.toml'
    auto_bytes = auto_path.read_bytes()
    auto = tomllib.loads(auto_bytes.decode('utf-8'))
    assert auto['status'] == 'ACTIVE' and auto['rrule'] == 'FREQ=MINUTELY;INTERVAL=30'
    assert '两线都要实质推进' in auto['prompt'] and '参考论文必须写入记录文档' in auto['prompt']
    create(STAGE / 'AUTOMATION_AFTER.toml', auto_bytes.decode('utf-8'))
    report = EXT / 'docs/当前规划/决策与审计/双线共同迭代_论文依据与本轮工程证据_2026-10-13.md'
    text = (STAGE/'LITERATURE_AND_METHOD_RECORD.md').read_text(encoding='utf-8')
    text += '\n## 本轮实际执行收口\n\n'
    for line, arm in result.items():
        text += ('### ' + ('配送' if line=='delivery' else '取货') + '\n\n' + arm['summary_zh'] + '\n\n')
        text += '证据目录：`' + arm['directory'] + '`。下一动作：' + arm['next_zh'] + '\n\n'
    text += '两线均未建立稳定、可归因MaskCO核心在线正向增益。自动任务已实读核定每30分钟ACTIVE、共同推进两线并强制论文记录；旧自动化全文已保存。\n'
    create(report, text)
    note = '\n> 2026-10-13 用户要求两线共同推进、增加每轮实质优化与论文记录：本轮新增配送中间算术诊断和取货独立接受/拒绝滚动接拒组件，实际结果及界限见[双线共同迭代记录](docs/当前规划/决策与审计/双线共同迭代_论文依据与本轮工程证据_2026-10-13.md)。工程假设树不增加科学族计数；第2族FAIL封闭、第3族未启。稳定、可归因在线增益未建立；完整科学控制、等耗时迭代与真实强动态仍待，取货原终局80/pending空不重跑。\n'
    entries = [EXT/'项目当前状态.md', EXT/'docs/当前规划/决策与审计/三线新增交接_独立核查修正与下一阶段_2026-10-10.md', EXT/'01_取货线/README.md', EXT/'02_配送线/README.md', EXT/'results/research_20261010_continuation_v2/WORK_STATUS.json']
    backup = TREE / 'current_entries_before_dual_track_iteration_20261013'
    backup.mkdir(exist_ok=False)
    identities = []
    for p in entries:
        raw = p.read_bytes()
        q = backup / p.relative_to(EXT)
        q.parent.mkdir(parents=True, exist_ok=True)
        with q.open('xb') as f: f.write(raw)
        if p.suffix == '.json':
            data = json.loads(raw.decode('utf-8-sig'))
            data['updated'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            for line, arm in result.items():
                data[line]['dual_track_iteration_20261013'] = arm
                data[line]['next'] = arm['next_zh']
            data['automation'].update(primary_objective='取货与配送共同推进，至少一线稳定可归因核心在线正向增益', both_tracks_required=True, literature_record_required=True, substantive_stages_per_iteration=True)
            data['latest_dual_track_iteration_root'] = STAGE.relative_to(ROOT).as_posix()
            data['latest_literature_record'] = report.relative_to(EXT).as_posix()
            new = json.dumps(data, ensure_ascii=False, indent=2) + '\n'
        else:
            local_note = note
            if p != EXT/'项目当前状态.md':
                rel = os.path.relpath(report, p.parent).replace('\\','/')
                local_note = note.replace('docs/当前规划/决策与审计/' + report.name, rel)
            new = local_note + raw.decode('utf-8-sig')
        assert p.read_bytes() == raw
        p.write_text(new, encoding='utf-8', newline='\n')
        identities.append({'path':p.relative_to(EXT).as_posix(),'before_sha256':sha(q),'after_sha256':sha(p)})
    create(backup/'UPDATE_IDENTITY.json', {'files':identities, 'report_sha256':sha(report)})
    receipt = {'executors':result,'report':report.relative_to(EXT).as_posix(),'report_sha256':sha(report),'automation_id':auto['id'],'automation_status':auto['status'],'interval_minutes':30,'automation_before_sha256':sha(STAGE/'AUTOMATION_BEFORE.toml'),'automation_after_sha256':sha(STAGE/'AUTOMATION_AFTER.toml'),'scientific_family3_started':False,'stable_online_gain':False,'shared_MD_uploaded':False,'final_pickup_seeds_used':False,'record_tool_sha256':sha(Path(__file__))}
    create(STAGE/'ITERATION_RECEIPT.json',receipt)
    create(TREE/'reproduction/DUAL_TRACK_ITERATION_BOUNDARY_20261013.json',receipt)
    print(json.dumps({'recorded':True,'both_tracks':True,'stable_online_gain':False,'entries':len(entries)}))

if __name__ == '__main__':
    record()
