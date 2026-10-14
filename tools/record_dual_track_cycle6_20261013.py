"""Record dual-track cycle 6 after both executor evidence sets are complete."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]
TREE = ROOT / 'research/review_3080_20261010'
STAGE = TREE / 'dual_track_optimization_20261013_root_v1'
CYCLE = STAGE / 'cycle6_20261013_root_v1'
EXT = ROOT / 'C-VRP_Cold-chainVehicleRoutingProblem'
REPORT_NAME = '双线第六轮_评分头泛化与全场景惰性复用_2026-10-13.md'

LITERATURE = """# 双线第六轮：动作匹配评分头泛化与全场景惰性复用

2026-10-13。Arbor工程循环6/12，配送n13和取货n14；工程树不是科学族账本。两线各完成一个有界实现、实际执行、独立核查与证据收取，共享CPU7串行。所有旧FAIL与封存原件保持；配送第2族永久封闭、第3族未启动。

## 原论文及实际阅读范围

Lvda Chen、Yang Li、Junchi Yan，2026，*MaskCO: Masked Generation Drives Effective Representation Learning and Exploiting for Combinatorial Optimization*，ICLR 2026正式会议论文。[官方出版页](https://proceedings.iclr.cc/paper_files/paper/2026/hash/d85816bf41e54d9b0847e02249f4fcc6-Abstract-Conference.html)、[正式PDF](https://proceedings.iclr.cc/paper_files/paper/2026/file/d85816bf41e54d9b0847e02249f4fcc6-Paper-Conference.pdf)、[官方实现](https://github.com/Thinklab-SJTU/MaskCO)。本轮实际读摘要、§3部分解/选择算子定义与§4.1变量BCE训练。针对配送目标与插入动作错位，本轮把该思路落成每客户可行动作组CE，并在固定留出划分上检验迁移。原论文变量BCE不同于本工程的每客户可行动作组CE，也不能把原论文静态任务收益迁移为冷链在线收益；本轮结论是该方法在已封存latent上不迁移，属本项目自测，不是对原论文方法的否定。

Diederik P. Kingma、Jimmy Ba，2015，*Adam: A Method for Stochastic Optimization*，ICLR 2015（第3届国际学习表征会议，San Diego），arXiv:1412.6980v9。[arXiv官方页](https://arxiv.org/abs/1412.6980)、[DOI 10.48550/arXiv.1412.6980](https://doi.org/10.48550/arXiv.1412.6980)。本轮实际读arXiv官方摘要页全文（含版本历史、会议与DOI元数据），未读算法全文或收敛证明。摘要明确该方法"对梯度的对角缩放不变"并"适合非常嘈杂和/或稀疏的梯度"，这正好解释本轮实测：逐坐标归一化使更新只依赖梯度符号，在float32舍入地板上符号跨库不可复现时会整步±lr摆动。论文的regret/收敛保证不迁移为本组件的可复现性保证，也不据此宣称任何在线收益。

Stéphane Ross、Geoffrey J. Gordon、J. Andrew Bagnell，2011，*A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*，AISTATS 2011，PMLR 15:627–635。[官方元数据](https://proceedings.mlr.press/v15/ross11a.html)、[PDF](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf)、arXiv:1011.0686。沿用上一轮实际读§2.1、§3及Algorithm3.1的范围：区分专家状态监督与由学习策略产生的状态分布。本轮没有执行DAgger或采集新轨迹，本轮的头-only留出负结果不改变该路线仍待独立登记的判断。

Niels A. Wouda、Leon Lan、Wouter Kool，2024，*PyVRP: a high-performance VRP solver package*，INFORMS Journal on Computing 36(4):943–955；[DOI 10.1287/ijoc.2023.0055](https://doi.org/10.1287/ijoc.2023.0055)、[arXiv:2403.13795v2全文](https://arxiv.org/html/2403.13795v2)。沿用此前实际已读§4.3/§4.3.2/§5.3的成本复用机制依据。本轮取货把同循环base距离复用扩到整条固定场景并与未修改原实现配对，只借"复用不变评价"的机制，不安装/执行PyVRP；实测省调用但不提速，正好说明论文的复用机制必须在有真实评价开销时才有意义。

## 冻结与资格边界

旧二十轮/I3/D2/183/C1、上游与比较仓库、全部失败/封存工件和最终取货20260927/28不修改。本轮不重跑第四/五轮组件、34编译候选/9099审计、旧计时请求/100前向、旧缓存请求/103局部或原取货全日/50后续/31107/终局。取货权威epoch64/callback80/pending空不变，没有callback81。每个新阶段在各自REGISTRATION.json先登记后执行；cycle6的CYCLE_SCOPE.json明确标注为执行后的追溯汇总，不是绑定登记。

完整实际学习提议与正式同初始化已训CE/同任务训练随机、独立强动态接拒、完整C0等耗时掩码重构迭代、多新鲜研究日/种子与校正确认、科学全加载依赖和完整整请求预算仍是科学放行条件。稳定、可归因核心在线增益未建立；禁止把损失下降或工程PASS当收益。
"""


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def create(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8', newline='\n') as f:
        f.write(v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, indent=2) + '\n')


def record():
    result = json.loads((CYCLE / 'EXECUTOR_EVIDENCE.json').read_text(encoding='utf-8'))
    for line in ('delivery', 'pickup'):
        arm = result[line]
        assert arm['stable_online_gain'] is False
        assert arm['scientific_family3_started'] is False
        assert arm['candidate_installed'] is False
        for rel_path, h in arm['evidence_hashes'].items():
            assert sha(ROOT / rel_path) == h, rel_path
    assert result['stable_online_gain'] is False and result['shared_MD_uploaded'] is False

    auto_path = Path(os.environ.get('CODEX_HOME', str(Path.home() / '.codex'))) / 'automations/maskco/automation.toml'
    auto_bytes = auto_path.read_bytes()
    auto = tomllib.loads(auto_bytes.decode('utf-8'))
    assert auto['status'] == 'ACTIVE' and auto['rrule'] == 'FREQ=MINUTELY;INTERVAL=30'
    assert '两线都要实质推进' in auto['prompt'] and '参考论文必须写入记录文档' in auto['prompt']
    create(CYCLE / 'AUTOMATION_AFTER.toml', auto_bytes.decode('utf-8'))

    report = EXT / 'docs/当前规划/决策与审计' / REPORT_NAME
    text = LITERATURE + '\n## 本轮实际执行收口\n\n'
    text += ('两线各完成一个有界实现、实际执行、独立核查与证据收取；共享 CPU7 串行，先取货后配送。'
             '两个阶段的绑定登记都是各自 REGISTRATION.json，且都在该阶段执行之前写入。'
             '失败的取货 v1 与上一轮配送 v1 原样保留，未被改写。\n\n')
    for line, label in (('pickup', '取货'), ('delivery', '配送')):
        arm = result[line]
        text += '### ' + label + '\n\n' + arm['summary_zh'] + '\n\n'
        text += '证据目录：`' + arm['directory'] + '`。'
        if arm.get('preserved_prior_attempt'):
            text += '保留失败尝试：`' + arm['preserved_prior_attempt'] + '`。'
        text += '下一动作：' + arm['next_zh'] + '\n\n'
    text += ('本轮两线都得到负向但可归因的结论：配送证明在已封存 latent 上做头-only 动作匹配不迁移（留出 CE 反而升高），'
             '并量化了归一化优化器在 float32 舍入地板上轨迹不可复现而功能结果可复现；取货证明把冗余路线求值削减 29.7% '
             '既不提速、也不改变任何精确门，因此该微优化线封闭。稳定、可归因核心在线增益仍未建立：第2族 FAIL 封闭、'
             '第3族未启动；工程循环 6/12 不是科学族计数。自动任务实读核定每30分钟 ACTIVE、共同推进两线并强制论文记录。\n')
    create(report, text)

    note = ('\n> 2026-10-13 双线第六轮：配送在固定预登记 20/9 划分上检验评分头动作匹配迁移（生产 62/0、独立审计 72/2FAIL），'
            '结论是留出 CE 反而升高（训练 1.2509→0.2172/0.2344，留出 1.3735→1.8231/1.5245，留出 argmax 2/9→4–5/9），'
            '并量化 float32 与 float64 梯度符号一致率仅 0.4978、Adam 更新饱和到 ±lr 而只有功能结果可复现；'
            '取货把惰性 base 复用扩到整条封存 scenario8（166 单/71 可插），距离求值 1042→733 省 309 次重复 base(29.7%)，'
            '能跑的精确门全部通过，但实测 0.2704/0.3976 秒未提速，生产 1 项与审计 4 项 FAIL 及缺陷分类全部按原样保留。'
            '详细取舍、论文依据与科学放行边界见[第六轮记录](docs/当前规划/决策与审计/'
            + REPORT_NAME + ')。工程循环6/12不是科学族计数；稳定可归因在线增益未建立，第2族封闭、第3族未启，'
            '取货 callback80/pending空不重跑。\n')
    entries = [EXT / '项目当前状态.md',
               EXT / 'docs/当前规划/决策与审计/三线新增交接_独立核查修正与下一阶段_2026-10-10.md',
               EXT / '01_取货线/README.md', EXT / '02_配送线/README.md',
               EXT / 'results/research_20261010_continuation_v2/WORK_STATUS.json']
    backup = TREE / 'current_entries_before_dual_track_cycle6_20261013'
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
                    data[line]['dual_track_cycle6_20261013'] = arm
                    data[line]['next'] = arm['next_zh']
            data['automation'].update(primary_objective='取货与配送共同推进，至少一线稳定可归因核心在线正向增益',
                                      both_tracks_required=True, literature_record_required=True,
                                      substantive_stages_per_iteration=True, cycle6_verified_20261013=True)
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
               'automation_after_sha256': sha(CYCLE / 'AUTOMATION_AFTER.toml'),
               'cycle': 6, 'cycle_root': CYCLE.relative_to(ROOT).as_posix(),
               'scientific_family3_started': False, 'stable_online_gain': False,
               'shared_MD_uploaded': False, 'final_pickup_seeds_used': False, 'GPU_used': False, 'server_used': False,
               'record_tool_sha256': sha(Path(__file__))}
    create(CYCLE / 'ITERATION_RECEIPT.json', receipt)
    create(TREE / 'reproduction/DUAL_TRACK_CYCLE6_BOUNDARY_20261013.json', receipt)
    print(json.dumps({'recorded': True, 'both_tracks': True, 'stable_online_gain': False,
                      'entries': len(entries), 'report_sha256': receipt['report_sha256']}))


if __name__ == '__main__':
    record()
