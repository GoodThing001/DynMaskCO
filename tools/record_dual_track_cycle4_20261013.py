"""Record completed cycle4, preserving all prior state-entry bytes."""
from pathlib import Path
import datetime
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[1]
EXT = ROOT / 'C-VRP_Cold-chainVehicleRoutingProblem'
TREE = ROOT / 'research/review_3080_20261010'
STAGE = TREE / 'dual_track_optimization_20261013_root_v1/cycle4_20261013_root_v1'

def digest(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(262144), b''):
            h.update(block)
    return h.hexdigest()

def create(p, x):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8', newline='\n') as f:
        f.write(x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, indent=2) + '\n')

def main():
    spec = json.loads((STAGE / 'COMPLETED_EVIDENCE_INPUT.json').read_text(encoding='utf-8'))
    assert set(spec['tracks']) == {'delivery', 'pickup'}
    for evidence in spec['tracks'].values():
        assert not evidence['stable_online_gain'] and not evidence['scientific_family3_started'] and not evidence['candidate_installed']
        for rel, expected in evidence['evidence_hashes'].items():
            p = ROOT / rel
            assert p.resolve().is_relative_to(TREE.resolve()) and digest(p) == expected, rel
    report = EXT / 'docs/当前规划/决策与审计/双线第四轮_独占源码租约与缓存耗时边界_2026-10-13.md'
    text = (STAGE / 'LITERATURE_AND_CONTRACT.md').read_text(encoding='utf-8')
    for name, evidence in spec['tracks'].items():
        text += '\n## ' + ('配送' if name == 'delivery' else '取货') + '实际阶段\n\n'
        text += evidence['summary_zh'] + '\n\n下一步：' + evidence['next_zh'] + '\n\n'
        text += '原件目录：`' + evidence['directory'] + '`；全部逐SHA身份见本轮COMPLETED_EVIDENCE_INPUT.json。\n'
    text += '\n稳定、可归因MaskCO核心在线增益尚未建立。正式已训同初始化CE/同任务训练随机、独立强动态接拒、完整冷链等耗时掩码重构迭代、多新鲜研究日与校正确认、完整请求及科学依赖仍未合格。Arbor最佳模型保持None；没有科学合并或正向宣告。maskco保持ACTIVE每30分钟。\n'
    create(report, text)
    entries = [EXT / '项目当前状态.md', EXT / 'docs/当前规划/决策与审计/三线新增交接_独立核查修正与下一阶段_2026-10-10.md', EXT / '01_取货线/README.md', EXT / '02_配送线/README.md', EXT / 'results/research_20261010_continuation_v2/WORK_STATUS.json']
    before = {p: p.read_bytes() for p in entries}
    backup = TREE / 'current_entries_before_dual_track_cycle4_20261013'
    backup.mkdir(exist_ok=False)
    for p, raw in before.items():
        saved = backup / p.relative_to(EXT)
        saved.parent.mkdir(parents=True, exist_ok=True)
        with saved.open('xb') as f:
            f.write(raw)
    identities = []
    for p, raw in before.items():
        if p.suffix == '.json':
            status = json.loads(raw.decode('utf-8-sig'))
            assert status['stable_online_gain_established'] is False
            status['updated'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            for name, evidence in spec['tracks'].items():
                status[name]['dual_track_cycle4_20261013'] = evidence
                status[name]['next'] = evidence['next_zh']
            status['latest_dual_track_cycle_root'] = STAGE.relative_to(ROOT).as_posix()
            status['latest_literature_record'] = report.relative_to(EXT).as_posix()
            status['latest_delivery_stage_receipts_root'] = spec['tracks']['delivery']['directory']
            status['latest_stage_receipts_root'] = spec['tracks']['pickup']['production_directory']
            new = json.dumps(status, ensure_ascii=False, indent=2) + '\n'
        else:
            link = os.path.relpath(report, p.parent).replace('\\', '/')
            note = '\n> 2026-10-13 双线第四轮：配送新独占源码租约真实边界试验完成，但生产10/3FAIL和独立951/6FAIL及起始依赖缺口保持；另27项只读诊断不升级资格。取货已存缓存/时钟诊断独立277项0失败，不能据hit数认定提速。原论文与官方资料、实际阅读范围及后续见[第四轮记录](' + link + ')。工程循环4/12不是科学族计数；稳定可归因在线增益未建立，第2族封闭、第3族未启，取货callback80/pending空不重跑。\n'
            new = note + raw.decode('utf-8-sig')
        assert p.read_bytes() == raw, 'Shared entry changed during recording: ' + str(p)
        p.write_text(new, encoding='utf-8', newline='\n')
        identities.append({'path': p.relative_to(EXT).as_posix(), 'before_sha256': digest(backup / p.relative_to(EXT)), 'after_sha256': digest(p)})
    create(backup / 'UPDATE_IDENTITY.json', {'files': identities, 'report_sha256': digest(report)})
    receipt = {'schema': 'dual-track-cycle4-record-v1', 'tracks': spec['tracks'], 'report': report.relative_to(EXT).as_posix(), 'report_sha256': digest(report), 'tool_sha256': digest(Path(__file__)), 'cycle': 4, 'scientific_family3_started': False, 'stable_online_gain': False, 'shared_MD_uploaded': False, 'final_pickup_seeds_used': False, 'automation_status_last_verified': 'ACTIVE', 'automation_interval_minutes': 30}
    create(STAGE / 'ITERATION_RECEIPT.json', receipt)
    create(TREE / 'reproduction/DUAL_TRACK_CYCLE4_BOUNDARY_20261013.json', receipt)
    print(json.dumps({'recorded': True, 'both_tracks': True, 'stable_online_gain': False, 'entries_backed_up': len(entries)}))

if __name__ == '__main__':
    main()
