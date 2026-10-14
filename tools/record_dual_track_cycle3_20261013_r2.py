"""Record only completed, immutable engineering evidence for both tracks."""
from pathlib import Path
import datetime
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / 'C-VRP_Cold-chainVehicleRoutingProblem'
TREE = ROOT / 'research/review_3080_20261010'
STAGE = TREE / 'dual_track_optimization_20261013_root_v1/cycle3_20261013_root_v1'

def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(262144), b''):
            h.update(block)
    return h.hexdigest()

def create(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8', newline='\n') as f:
        f.write(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2) + '\n')

def record():
    spec = json.loads((STAGE / 'COMPLETED_EVIDENCE_INPUT.json').read_text(encoding='utf-8'))
    assert set(spec['tracks']) == {'delivery', 'pickup'}
    for evidence in spec['tracks'].values():
        assert evidence['stable_online_gain'] is False
        assert evidence['scientific_family3_started'] is False
        assert evidence['candidate_installed'] is False
        for rel, expected in evidence['evidence_hashes'].items():
            path = ROOT / rel
            assert path.resolve().is_relative_to(TREE.resolve())
            assert digest(path) == expected, rel
    report = EXT / 'docs/当前规划/决策与审计/双线第三轮_实际依赖守卫与精确距离缓存_2026-10-13.md'
    text = (STAGE / 'LITERATURE_AND_CONTRACT.md').read_text(encoding='utf-8')
    for name, evidence in spec['tracks'].items():
        text += '\n## ' + ('配送' if name == 'delivery' else '取货') + '实际阶段\n\n'
        text += evidence['summary_zh'] + '\n\n下一步：' + evidence['next_zh'] + '\n\n'
        text += '证据目录：`' + evidence['directory'] + '`。详细绑定见本轮 `COMPLETED_EVIDENCE_INPUT.json`。\n'
    text += '\n稳定、可归因核心在线增益尚未建立。工程资格不等正式已训控制、独立强动态接拒、全C0等耗时迭代或完整请求/科学依赖资格；第2族FAIL封闭、第3族未启。原取货权威终局callback80/pending空不重跑。\n'
    create(report, text)
    entries = [EXT / '项目当前状态.md', EXT / 'docs/当前规划/决策与审计/三线新增交接_独立核查修正与下一阶段_2026-10-10.md', EXT / '01_取货线/README.md', EXT / '02_配送线/README.md', EXT / 'results/research_20261010_continuation_v2/WORK_STATUS.json']
    backup = TREE / 'current_entries_before_dual_track_cycle3_20261013'
    backup.mkdir(exist_ok=False)
    identities = []
    for p in entries:
        before = p.read_bytes()
        saved = backup / p.relative_to(EXT)
        saved.parent.mkdir(parents=True, exist_ok=True)
        with saved.open('xb') as f:
            f.write(before)
        if p.suffix == '.json':
            status = json.loads(before.decode('utf-8-sig'))
            assert status['stable_online_gain_established'] is False
            status['updated'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            for name, evidence in spec['tracks'].items():
                status[name]['dual_track_cycle3_20261013'] = evidence
                status[name]['next'] = evidence['next_zh']
            status['latest_dual_track_cycle_root'] = STAGE.relative_to(ROOT).as_posix()
            status['latest_literature_record'] = report.relative_to(EXT).as_posix()
            status['latest_delivery_stage_receipts_root'] = spec['tracks']['delivery']['directory']
            status['latest_stage_receipts_root'] = spec['tracks']['pickup']['production_directory']
            new = json.dumps(status, ensure_ascii=False, indent=2) + '\n'
        else:
            link = os.path.relpath(report, p.parent).replace('\\', '/')
            note = '\n> 2026-10-13 双线第三轮：实际依赖守卫与精确距离缓存新候选均完成实际工程执行，原失败与资格边界保留，结果见[第三轮记录](' + link + ')。沿用PyVRP原论文缓存机制依据；工程循环3/12不是科学族计数。稳定可归因在线增益尚未建立，第2族封闭、第3族未启；不重复旧QA或取货终局。\n'
            new = note + before.decode('utf-8-sig')
        assert p.read_bytes() == before, 'Entry changed during recording: ' + str(p)
        p.write_text(new, encoding='utf-8', newline='\n')
        identities.append({'path': p.relative_to(EXT).as_posix(), 'before_sha256': digest(saved), 'after_sha256': digest(p)})
    create(backup / 'UPDATE_IDENTITY.json', {'files': identities, 'report_sha256': digest(report)})
    receipt = {'schema': 'dual-track-cycle3-record-v1', 'tracks': spec['tracks'], 'report': report.relative_to(EXT).as_posix(), 'report_sha256': digest(report), 'tool_sha256': digest(Path(__file__)), 'cycle': 3, 'scientific_family3_started': False, 'stable_online_gain': False, 'shared_MD_uploaded': False, 'final_pickup_seeds_used': False, 'automation_status_last_verified': 'ACTIVE', 'automation_interval_minutes': 30}
    create(STAGE / 'ITERATION_RECEIPT.json', receipt)
    create(TREE / 'reproduction/DUAL_TRACK_CYCLE3_BOUNDARY_20261013.json', receipt)
    print(json.dumps({'recorded': True, 'both_tracks': True, 'stable_online_gain': False, 'entries': len(entries)}))

if __name__ == '__main__':
    record()
