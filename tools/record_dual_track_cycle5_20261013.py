"""Record dual-track cycle 5 after both executor evidence sets are complete."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]
TREE = ROOT / 'research/review_3080_20261010'
STAGE = TREE / 'dual_track_optimization_20261013_root_v1'
CYCLE = STAGE / 'cycle5_20261013_root_v1'
EXT = ROOT / 'C-VRP_Cold-chainVehicleRoutingProblem'
REPORT_NAME = '双线第五轮_动作匹配评分头更新与精确插入计算复用_2026-10-13.md'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def create(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8', newline='\n') as f:
        f.write(v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, indent=2) + '\n')


def record():
    result = json.loads((CYCLE / 'EXECUTOR_EVIDENCE.json').read_text(encoding='utf-8'))
    assert set(result) >= {'delivery', 'pickup'}
    for line in ('delivery', 'pickup'):
        arm = result[line]
        assert arm['stable_online_gain'] is False
        assert arm['scientific_family3_started'] is False
        assert arm['candidate_installed'] is False
        for rel_path, h in arm['evidence_hashes'].items():
            assert sha(ROOT / rel_path) == h, rel_path
    assert result['stable_online_gain'] is False
    assert result['shared_MD_uploaded'] is False
    assert result['final_pickup_seeds_used'] is False

    auto_path = Path(os.environ.get('CODEX_HOME', str(Path.home() / '.codex'))) / 'automations/maskco/automation.toml'
    auto_bytes = auto_path.read_bytes()
    auto = tomllib.loads(auto_bytes.decode('utf-8'))
    assert auto['status'] == 'ACTIVE' and auto['rrule'] == 'FREQ=MINUTELY;INTERVAL=30'
    assert '两线都要实质推进' in auto['prompt'] and '参考论文必须写入记录文档' in auto['prompt']
    create(CYCLE / 'AUTOMATION_AFTER.toml', auto_bytes.decode('utf-8'))

    report = EXT / 'docs/当前规划/决策与审计' / REPORT_NAME
    text = (CYCLE / 'LITERATURE_AND_CONTRACT.md').read_text(encoding='utf-8')
    text += '\n## 本轮实际执行收口\n\n'
    text += ('两线各完成一个有界实现并实际执行、独立核查与证据收取；共享 CPU 串行，先取货生产/独立核查，再配送。'
             'v1 失败的配送尝试与全部旧 FAIL、旧封存原件均未被改写。\n\n')
    for line, label in (('pickup', '取货'), ('delivery', '配送')):
        arm = result[line]
        text += '### ' + label + '\n\n' + arm['summary_zh'] + '\n\n'
        text += '证据目录：`' + arm['directory'] + '`。'
        if arm.get('preserved_failed_attempt'):
            text += '保留失败尝试：`' + arm['preserved_failed_attempt'] + '`。'
        text += '下一动作：' + arm['next_zh'] + '\n\n'
    text += ('两线均未建立稳定、可归因 MaskCO 核心在线正向增益：取货只证明省掉 13 次重复原 base 求值且实测未提速，'
             '配送只把评分头更新/保存/重载接通且暴露 Adam 归一化放大 float32 算术、跨库轨迹不可复现的机制边界。'
             '第2族 FAIL 封闭、第3族未启动；工程循环 5/12 不是科学族计数。自动任务实读核定每30分钟 ACTIVE、'
             '共同推进两线并强制论文记录。\n')
    create(report, text)

    note = ('\n> 2026-10-13 双线第五轮：取货新增惰性复用原 base 路线距离的有界组件（固定未执行 scenario index8 前四订单，'
            '生产 7/0、独立 155/0，路线距离求值 109→96 省 13 次重复 base，但实测 0.0270015 秒未提速，不称加速）；'
            '配送把动作匹配评分头更新/固定顺序/保存/重载接通（v1 生产 57/2FAIL 保留，只读诊断证明差异仅 encodings.cp437 懒加载；'
            'v2 生产 60/0、独立审计 97/2FAIL），四叶 196929 参数与 construct_head(seed=20261012) 逐位相同、前向 5.066e-7、'
            '重载逐位一致、可用内存 1.94GiB 低于 8GiB 主干门故未加载主干，机制诊断确认跨库轨迹差异是 Adam 归一化放大的 float32 效应。'
            '详细取舍、论文依据与科学放行边界见[第五轮记录](docs/当前规划/决策与审计/'
            + REPORT_NAME + ')。工程循环5/12不是科学族计数；稳定可归因在线增益未建立，第2族封闭、第3族未启，'
            '取货 callback80/pending空不重跑。\n')
    entries = [EXT / '项目当前状态.md',
               EXT / 'docs/当前规划/决策与审计/三线新增交接_独立核查修正与下一阶段_2026-10-10.md',
               EXT / '01_取货线/README.md', EXT / '02_配送线/README.md',
               EXT / 'results/research_20261010_continuation_v2/WORK_STATUS.json']
    backup = TREE / 'current_entries_before_dual_track_cycle5_20261013'
    backup.mkdir(exist_ok=False)
    identities = []
    for p in entries:
        raw = p.read_bytes()
        q = backup / p.relative_to(EXT)
        q.parent.mkdir(parents=True, exist_ok=True)
        with q.open('xb') as f:
            f.write(raw)
        if p.suffix == '.json':
            data = json.loads(raw.decode('utf-8-sig'))
            data['updated'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
            for line, arm in result.items():
                if line in ('delivery', 'pickup'):
                    data[line]['dual_track_cycle5_20261013'] = arm
                    data[line]['next'] = arm['next_zh']
            data['automation'].update(primary_objective='取货与配送共同推进，至少一线稳定可归因核心在线正向增益',
                                      both_tracks_required=True, literature_record_required=True,
                                      substantive_stages_per_iteration=True,
                                      cycle5_verified_20261013=True)
            data['latest_dual_track_cycle_root'] = CYCLE.relative_to(ROOT).as_posix()
            data['latest_dual_track_iteration_root'] = STAGE.relative_to(ROOT).as_posix()
            data['latest_literature_record'] = report.relative_to(EXT).as_posix()
            new = json.dumps(data, ensure_ascii=False, indent=2) + '\n'
        else:
            local_note = note
            if p != EXT / '项目当前状态.md':
                rel = os.path.relpath(report, p.parent).replace('\\', '/')
                local_note = note.replace('docs/当前规划/决策与审计/' + REPORT_NAME, rel)
            new = local_note + raw.decode('utf-8-sig')
        assert p.read_bytes() == raw
        p.write_text(new, encoding='utf-8', newline='\n')
        identities.append({'path': p.relative_to(EXT).as_posix(),
                           'before_sha256': sha(q), 'after_sha256': sha(p)})
    create(backup / 'UPDATE_IDENTITY.json', {'files': identities, 'report_sha256': sha(report)})

    receipt = {'executors': {k: v for k, v in result.items() if k in ('delivery', 'pickup')},
               'report': report.relative_to(EXT).as_posix(), 'report_sha256': sha(report),
               'automation_id': auto['id'], 'automation_status': auto['status'], 'interval_minutes': 30,
               'automation_before_sha256': sha(STAGE / 'AUTOMATION_BEFORE.toml'),
               'automation_after_sha256': sha(CYCLE / 'AUTOMATION_AFTER.toml'),
               'cycle': 5, 'cycle_root': CYCLE.relative_to(ROOT).as_posix(),
               'scientific_family3_started': False, 'stable_online_gain': False,
               'shared_MD_uploaded': False, 'final_pickup_seeds_used': False,
               'GPU_used': False, 'server_used': False,
               'record_tool_sha256': sha(Path(__file__))}
    create(CYCLE / 'ITERATION_RECEIPT.json', receipt)
    create(TREE / 'reproduction/DUAL_TRACK_CYCLE5_BOUNDARY_20261013.json', receipt)
    print(json.dumps({'recorded': True, 'both_tracks': True, 'stable_online_gain': False,
                      'entries': len(entries), 'report_sha256': receipt['report_sha256']}))


if __name__ == '__main__':
    record()
