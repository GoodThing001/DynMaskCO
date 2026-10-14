"""Seal and summarize only saved phase traces; never replay a workload."""
import datetime
import hashlib
import json
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TREE = ROOT / 'research/review_3080_20261010'
STAGE = TREE / 'delivery/phase_timed_model_execution_root_v1'
RAW = STAGE / 'downloaded'
EXT = ROOT / 'C-VRP_Cold-chainVehicleRoutingProblem'


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def create(p, value):
    with p.open('x', encoding='utf-8', newline='\n') as f:
        f.write(value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def seal():
    files = [STAGE / 'DOWNLOAD_MANIFEST.json', STAGE / 'EXECUTION_REGISTRATION.json']
    files += sorted(RAW.rglob('*'))
    create(STAGE / 'SAVED_TIMING_RECORDING_REGISTRATION.json', {
        'registered_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'tool_sha256': sha(Path(__file__)),
        'inputs': {str(p.relative_to(ROOT)).replace('\\', '/'): sha(p) for p in files if p.is_file()},
        'method': 'Exact rational sums of already independently parsed completed spans only; open spans excluded. No model, parser, C0 or original QA replay. Verify downloaded raw hashes; do not label timing/recovery PASS as complete scientific request.',
        'new_scientific_family': False,
    })


def record():
    reg = read(STAGE / 'SAVED_TIMING_RECORDING_REGISTRATION.json')
    assert sha(Path(__file__)) == reg['tool_sha256']
    assert all(sha(ROOT / p) == h for p, h in reg['inputs'].items())
    manifest = read(STAGE / 'DOWNLOAD_MANIFEST.json')
    assert len(manifest) == 141
    for item in manifest:
        p = RAW / item['name']
        assert p.stat().st_size == item['size'] and sha(p) == item['sha256']
    a = read(RAW / 'INDEPENDENT_TIMED_REQUEST_AUDIT.json')
    r = read(RAW / 'RESULT.json')
    assert a['checks_count'] == 277 and a['failed'] == 0 and not a['failure_names']
    assert a['new_timed_evidence_forward_replays'] == 34
    assert len(r['receipts']) == 2 and r['selected_count'] == 0
    assert all(read(RAW / f)['exit_code'] == 0 for f in ('RUN_EXIT.json', 'AUDIT_RUN_EXIT.json', 'CHAIN_EXIT.json'))
    start, end = (read(RAW / f) for f in ('TIMED_AUDIT_SOURCE_START.json', 'TIMED_AUDIT_SOURCE_END.json'))
    assert start['files'] == end['files'] and start['module_files'] == end['module_files']
    rows = []
    for x, receipt in zip([x for x in a['phase_summaries'] if x['role'] == 'WORKER'], r['receipts']):
        phases = x['phase_totals']
        totals = {k: Fraction(*v['exact_completed_seconds']) for k, v in phases.items()}
        assert phases['7']['completed'] == 17 and phases['7']['open'] == 1
        assert phases['9']['completed'] == 17 and not x['closed']
        assert all(k not in phases for k in ('12', '13', '14', '15', '16', '17'))
        assert receipt['worker_reaped'] and receipt['actual_worker_returncode'] == 1
        assert not receipt['selected'] and receipt['elapsed'] < 10 and not receipt['natural_exit']
        model = totals['4'] + totals['7']
        evidence = totals['8'] + totals['10'] + totals['11']
        rows.append({
            'arm': receipt['arm'], 'elapsed_seconds': receipt['elapsed'],
            'completed_insertions': 17, 'remaining_customers': 8,
            'unfinished_score_phase': 7, 'unfinished_score_step': 17,
            'completed_phase_exact_seconds': {k: [v.numerator, v.denominator] for k, v in totals.items()},
            'completed_phase_seconds': {k: float(v) for k, v in totals.items()},
            'completed_encode_and_score_seconds': float(model),
            'completed_evidence_write_and_transition_hash_seconds': float(evidence),
            'model_to_evidence_ratio': float(model / evidence),
            'dependency_and_parameter_gates_reached': False,
            'selected': False, 'original_plan_retained': True,
            'killed_actual_returncode': receipt['actual_worker_returncode'],
            'whole_request_union_is_completed_work': False,
        })
    scope = {
        'registration_sha256': sha(STAGE / 'EXECUTION_REGISTRATION.json'),
        'requests': 2, 'raw_independent_checks': 277, 'raw_independent_failures': 0,
        'new_saved_forward_replays': 34, 'optimizer_steps': 0,
        'download_files': 141, 'download_bytes': sum(i['size'] for i in manifest),
        'all_download_hashes_match': True, 'actual_owned_processes_exited_before_download': True,
        'run_exit': read(RAW / 'RUN_EXIT.json'), 'audit_exit': read(RAW / 'AUDIT_RUN_EXIT.json'),
        'chain_exit': read(RAW / 'CHAIN_EXIT.json'),
        'rows': rows, 'max_abs_numpy_head_error': a['max_abs_numpy_head_error'],
        'numpy_head_bound_only': 2e-5,
        'independent_loaded_files': len(start['files']), 'independent_module_files': len(start['module_files']),
        'independent_source_start_end_equal': True,
        'missing_completed_source_end': a['missing_completed_source_end'],
        'raw_replay_qualified': True, 'cancel_and_original_fallback_observed': True,
        'complete_chain_qualified': False, 'full_request_budget_qualified': False,
        'full_scientific_loaded_closure_qualified': False, 'formal_controls_qualified': False,
        'learned_output_qualified': False, 'scientific_family3_started': False, 'stable_online_gain': False,
        'physical_live_installation': False, 'GPU_used': False, 'shared_MD_uploaded': False,
        'final_pickup_seeds_used': False, 'old_QA_repeated': False,
        'diagnosis': 'Completed encoder plus synchronized decoder/head dominates the recorded work. Phase7 combines decoder/head/host conversion and does not isolate JIT/native dispatch. End dependency and parameter gates not reached; their prospective cost remains unknown. This supports further compute-scheduling diagnosis, not an IO-removal speed counterfactual or benefit claim.',
    }
    create(STAGE / 'STAGE_ADJUDICATION.json', scope)
    report = EXT / '02_配送线/文档/结果与进度/配送_实际分阶段模型计时与预算瓶颈验收_2026-10-13.md'
    lines = ['# 配送实际分阶段模型计时与预算瓶颈验收', '',
        '执行前登记SHA为`' + scope['registration_sha256'] + '`。在3080 CPU15、实际可用RAM11.244358GiB、负载0.8%、无观察到本人旧研究树后，仅派发一个独占PY/JSON包。固定旧首项key[12,3]/25masked，两initial臂同headseed20261012、随机888、0更新。没有教师标签/成本选样、旧00评分复用、样本模型/C0预热或旧QA重跑。', '',
        '两个新请求均使用19阶段有界原始遥测，执行8秒、回收预留2秒。实际各完成17/25插入，下一评分处仍open，进程被kill/reap；0输出选择，原完整计划保留。并未完成候选全C0、发布前依赖/参数门或响应发布。', '',
        '| 臂 | 含回收请求秒 | 已完成encoder秒 | 已完成decode/head秒 | NPZ/JSON写及转移SHA秒 |',
        '|---|---:|---:|---:|---:|']
    for row in rows:
        t = row['completed_phase_seconds']
        lines.append('| %s | %.9f | %.9f | %.9f | %.9f |' % (row['arm'], row['elapsed_seconds'], t['4'], t['7'], row['completed_evidence_write_and_transition_hash_seconds']))
    lines += ['',
        '已完成模型段约6.88/6.90秒，相对于已完成证据写入和转移哈希约0.288/0.284秒占主要耗时。phase7仍组合decode/head/同步/host转换，尚不能唯一归因JIT、native计算或调度；尚未到达的全依赖枚举与参数核对不可当作零成本。open段不算完成，嵌套/并集不冒充完整请求，不据此承诺移除日志即可过8秒。', '',
        '生产driver自然退出0、44.441238500秒，独立审计自然0、45.476819800秒，监督自然0、89.935322762秒；两个模型worker真实kill返回1单独保留，不能写自然退出0。新独立277项0失败，包括34已保存完整前向/cache/JAX评分、原234叶76263424参数/fresh888/同seed四叶头/10D归一/当前锚点/原局部过滤与regret/17次实际转移、原incumbent C0、四原trace字节和精确区间。NumPy头最大差1.013278961e-6仅对事前新头跨库2e-5；参数/特征/latent/JAX要求精确。原JAX/Transformer与C0共享实现，不是另一物理或模型实现。', '',
        '独立实际9016文件/1019module起止一致；生产各起始8903文件/1029module，但两个被kill请求没有生产完整SOURCE_END，不补造末端闭包。确认实际监督/worker树全部退出后，141原件30426046字节有界只读下载逐SHA一致。两臂均在本10秒窗内回收只属本负载取消观察，不等完整10秒候选交付、现场车辆安装或任意负载硬实时。', '',
        '下一优先按新事前工程分解评分中decode、head及同步/host转换，再据实际证据实现保持模型/动作/参数语义的调度改进和独立一致性验收；不放宽8秒、不预热旧样本追速度、不按标签/成本换头。保持所有source/参数放行门在请求计时内。正式已训同初始化CE/同任务训练随机、各控制/等耗时MaskCO迭代/独立强动态接拒/科学完整依赖仍待。第2族FAIL封闭、第3族未启动；取货已权威终局64/callback80/pending空，不跑callback81或旧无模型正常QA。稳定可归因MaskCO核心在线正向增益仍未建立。', '']
    create(report, '\n'.join(lines))
    next_action = 'Two NEW phase-traced initial model requests completed17/25 insertions each then killed/reaped within10s; no output chosen, incumbent retained. Independent277/0 replays34 saved forward states. Completed encode+decode/head6.88/6.90s vs completed NPZ/JSON/transitionSHA0.288/0.284s; phase7 combines decode/head/sync/host, unique JIT/native cause not identified. Full dependency/parameter gates NOT reached, their cost unknown; no scientific closure/budget upgrade. Do NOT rerun these2 requests,34 forwards,277 audit or old QA. Next preregister bounded compute-phase split and semantics-preserving model scheduling engineering with independent equality and unchanged8+2/all end gates in window; no sample warmup/label-cost selection. Formal trained CE/task random/strongdynamic/equal-time iterative science pending, family2closed/family3notstarted, pickupterminal64/pendingempty/no81.'
    note = '\n> 2026-10-13 新实际分阶段模型请求：固定旧key[12,3]两initial臂各完成17/25插入后8秒取消，本人kill1/reap、含回收8.095167/8.121825秒，0选择原计划保留。独立277/0及34新保存前向精确重放，141原件30,426,046字节下载逐SHA闭合。完成encoder+decode/head约6.88/6.90秒、写入与转移SHA约0.288/0.284秒；phase7仍组合，未到全依赖/参数门，不能称完整10秒。生产缺自然SOURCE_END保持；正式科学控制/稳定增益仍False，第2族封闭/第3族未启，取货终局64不重跑。下一新事前评分计算细分/语义保持调度工程，不重复本2/34/277或旧QA。\n'
    entries = [EXT / '项目当前状态.md', EXT / 'docs/当前规划/决策与审计/三线新增交接_独立核查修正与下一阶段_2026-10-10.md', EXT / '02_配送线/README.md', EXT / 'results/research_20261010_continuation_v2/WORK_STATUS.json']
    backup = TREE / 'current_entries_before_phase_timed_model_20261013'
    backup.mkdir(exist_ok=False)
    changed = []
    for p in entries:
        raw = p.read_bytes()
        q = backup / p.relative_to(EXT)
        q.parent.mkdir(parents=True, exist_ok=True)
        q.open('xb').write(raw)
        if p.suffix == '.json':
            w = json.loads(raw.decode('utf-8-sig'))
            w['updated'] = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
            w['delivery'].update(phase_timed_model_component=scope, next=next_action)
            w['delivery']['active_phase_timed_stage']['status'] = 'completed_all_actual_processes_exited_raw_replay_pass_full_request_not_qualified'
            w['latest_delivery_stage_receipts_root'] = str(STAGE.relative_to(ROOT)).replace('\\', '/')
            new = json.dumps(w, ensure_ascii=False, indent=2) + '\n'
        else:
            new = note + raw.decode('utf-8-sig')
            if p == EXT / '02_配送线/README.md':
                new += '\n- [实际分阶段模型计时与预算瓶颈验收](文档/结果与进度/' + report.name + ')\n'
        assert p.read_bytes() == raw
        p.write_text(new, encoding='utf-8', newline='\n')
        changed.append({'path': str(p.relative_to(EXT)), 'before_sha256': sha(q), 'after_sha256': sha(p)})
    create(backup / 'UPDATE_IDENTITY.json', {'files': changed, 'report': str(report.relative_to(EXT)), 'shared_MD_uploaded': False})
    create(TREE / 'reproduction/PHASE_TIMED_MODEL_COMPONENT_EVIDENCE_20261013.json', scope)
    create(STAGE / 'LOCAL_RECORDING_RECEIPT.json', {'entries': len(changed), 'report_sha256': sha(report), 'next': next_action, 'raw_input_hashes_unchanged': all(sha(ROOT / p) == h for p, h in reg['inputs'].items())})
    print(json.dumps({'recorded': True, 'rows': rows, 'stable_online_gain': False}, ensure_ascii=False))


if __name__ == '__main__':
    {'seal': seal, 'record': record}[sys.argv[1]]()
