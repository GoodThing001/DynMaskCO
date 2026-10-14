"""Collect completed cycle4 receipts without rerunning experimental workloads."""
from pathlib import Path
import datetime
import hashlib
import json
import psutil

ROOT = Path(__file__).resolve().parents[1]
TREE = ROOT / 'research/review_3080_20261010'
STAGE = TREE / 'dual_track_optimization_20261013_root_v1/cycle4_20261013_root_v1'
D = TREE / 'delivery/local_source_lease_boundary_root_v1'
P = TREE / 'pickup/cache_saved_boundary_diagnosis_root_v1'

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(262144), b''):
            h.update(b)
    return h.hexdigest()

def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))

def create(p, x):
    with p.open('x', encoding='utf-8', newline='\n') as f:
        f.write(x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, indent=2) + '\n')

def main():
    STAGE.mkdir(exist_ok=False)
    for directory in (D, P):
        reg = read(directory / 'REGISTRATION.json')
        for name, expected in reg['programs'].items():
            assert sha(directory / name) == expected
        assert sha(ROOT / 'tools/engineering_stage_closure_v1.py') == reg['helper_sha256']
    for rel, expected in read(P / 'REGISTRATION.json')['inputs'].items():
        assert sha(ROOT / rel) == expected
    d = read(D / 'RESULT.json'); da = read(D / 'INDEPENDENT_AUDIT.json')
    recovery = read(D / 'SAVED_FAILURE_DIAGNOSIS_R2.json')
    p = read(P / 'RESULT.json'); pa = read(P / 'INDEPENDENT_AUDIT.json')
    assert (d['checks_count'], d['failed'], da['checks_count'], da['failed']) == (10, 3, 951, 6)
    assert (recovery['checks_count'], recovery['failed']) == (27, 0)
    assert (pa['checks_count'], pa['failed']) == (277, 0) and p['loaded_source_closed']
    identities = []
    for directory in (D, P):
        for name in ('RUN_EXIT.json', 'AUDIT_RUN_EXIT.json'):
            receipt = read(directory / name)
            assert receipt['reaped'] and not receipt['timed_out']
            assert receipt['exit_code'] == (1 if directory == D else 0)
            expected = {k: receipt[k] for k in ('pid', 'create_time', 'cmdline', 'owner')}
            try:
                proc = psutil.Process(expected['pid'])
                actual = {'pid': proc.pid, 'create_time': proc.create_time(), 'cmdline': proc.cmdline(), 'owner': proc.username()}
                assert actual != expected, 'Original stage process still active'
                observation = {'same_original_process_alive': False, 'PID_reused_identity': actual}
            except psutil.NoSuchProcess:
                observation = {'same_original_process_alive': False, 'observation': 'PID absent'}
            except psutil.AccessDenied:
                observation = {'same_original_process_alive': None, 'observation': 'AccessDenied; rely only on Popen owned-handle reaped receipt'}
            identities.append({'receipt': (directory / name).relative_to(ROOT).as_posix(), 'expected': expected, **observation})
    create(STAGE / 'PROCESS_RECEIPT_INSPECTION.json', {'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'processes': identities, 'no_process_killed': True, 'global_no_research_process_assertion': False})
    automation = Path('C:/Users/mmc/.codex/automations/maskco/automation.toml')
    auto_text = automation.read_text(encoding='utf-8')
    assert 'status = "ACTIVE"' in auto_text and 'rrule = "FREQ=MINUTELY;INTERVAL=30"' in auto_text
    create(STAGE / 'AUTOMATION_READ_ONLY_IDENTITY.json', {'sha256': sha(automation), 'status': 'ACTIVE', 'interval_minutes': 30, 'automation_modified': False})
    create(D / 'STAGE_ADJUDICATION.json', {'production_checks': 10, 'production_failures': 3, 'independent_checks': 951, 'independent_failures': 6, 'saved_only_diagnosis_checks': 27, 'saved_only_diagnosis_failures': 0, 'original_FAIL_and_exit1_preserved': True, 'source_qualified': False, 'full_runtime_guard_qualified': False, 'full_request_budget_qualified': False, 'stable_online_gain': False, 'old_probes_or_audit_repeated': False})
    create(P / 'STAGE_ADJUDICATION.json', {'independent_checks': 277, 'independent_failures': 0, 'saved_only_boundary_diagnosis_qualified': True, 'inner_cache_time_attribution_qualified': False, 'larger_cache_acceleration_established': False, 'request_or_original_QA_repeated': False, 'full_request_budget_qualified': False, 'stable_online_gain': False})
    literature = '''# 双线第四轮：独占源码租约与缓存耗时边界

2026-10-13。工程循环4/12，Arbor n9/n10；工程树不是科学族账本。本轮两线实际阶段都已执行和独立核查，没有新模型训练、科学数据、服务器/GPU或共享MD上传。配送第2族FAIL封闭，第3族未启动。两位执行代理在启动实验前遇到模型容量错误，主代理在本地串行完成本轮组件，未重复派发。

## 原论文、官方资料与采用范围

Niels A. Wouda、Leon Lan、Wouter Kool，2024，*PyVRP: a high-performance VRP solver package*，INFORMS Journal on Computing 36(4):943–955；DOI [10.1287/ijoc.2023.0055](https://doi.org/10.1287/ijoc.2023.0055)，[arXiv:2403.13795v2全文](https://arxiv.org/html/2403.13795v2)。实际阅读§4.3、§4.3.2和§5.3，复核了路线算子缓存及扩展成本评价的讨论。取货继续借鉴“复用不变评价”的机制；本轮先对已存时钟和重复key做可检验诊断。没有安装或执行PyVRP，不复制其静态距离/软约束评价；冷链完整物理、接拒、承诺前缀和原串行浮点算术继续独立保留。下一候选仅拟在原插入循环中惰性复用同一路线的原始base距离，尚未登记/执行，不根据旧效用扩大缓存或修改搜索邻域。科学放行仍需精确语义、完整请求、独立强动态对照及正式学习归因。

Russell W. Bent、Pascal Van Hentenryck，2004，*Scenario-Based Planning for Partially Dynamic Vehicle Routing with Stochastic Customers*，Operations Research 52(6):977–987；[DOI 10.1287/opre.1040.0124及出版页](https://pubsonline.informs.org/doi/10.1287/opre.1040.0124)。实际阅读出版页摘要、作者和期刊元数据，未读完整PDF。其多场景/共识机制提供独立动态接拒设计线索；当前成对共同SAA不是该文MSA复现，原最大服务数量也不同于完整冷链效用。本轮没有执行新接拒策略，论文收益不迁移为MaskCO证据。

配送读[Microsoft CreateFileW参数说明](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-createfilew)与[GetFileInformationByHandle](https://learn.microsoft.com/en-us/windows/win32/api/fileapi/nf-fileapi-getfileinformationbyhandle)接口说明，检验新独占副本的读共享句柄、文件身份与退出释放边界。只锁本轮复制件，不锁全局依赖、原源码或团队文件；单文件句柄不等完整目录或运行库不变保护。另读[Python3.12 open换行说明](https://docs.python.org/3.12/library/functions.html#open)：默认写模式换行会按系统转换。官网当前3.12.15与实际3.12.7须区分；解释本次已存CRLF与冻结writer未指定newline相符，不重写原字节或改原失败判据。

## 固定边界

本地实际Python3.12.7/psutil5.9.0/CPU7，只运行无模型组件；未下调完整JAX模型CPU15/RAM≥8GiB门。取货权威终局epoch64/callback80/pending空不重跑。没有旧34编译候选、9099审计、10计时请求/100前向/220恢复或其它旧QA重跑。保留全部旧FAIL、缺13800字节、缺逐推进ledger和原科学注册字节。
'''
    create(STAGE / 'LITERATURE_AND_CONTRACT.md', literature)
    tracks = {
        'delivery': {'directory': D.relative_to(ROOT).as_posix(), 'stable_online_gain': False, 'scientific_family3_started': False, 'candidate_installed': False,
            'raw_six_mutations_denied': True, 'source_qualified': False, 'independent_checks': 951, 'independent_failures': 6, 'saved_only_diagnosis_checks': 27, 'saved_only_diagnosis_failures': 0, 'full_request_budget_qualified': False,
            'summary_zh': '配送n9固定四份实际stdlib源码/既存pyc的新独占复制件，真实跨进程写入、truncate、replace、rename、delete和pyc写入均拒绝，四文件SHA和volume/file-index逐字保持。目录新增实际成功，完整目录manifest拒绝；释放读共享句柄后写入成功，单独所有者持有时拒绝、实际os._exit(17)后操作系统释放句柄且写入成功。生产10项3FAIL/自然exit1、独立951项6FAIL/自然exit1原件保留。27项另登记只读诊断通过，只确认已存错误码与换行预期差异：CRT路径errno13且winerror缺省，replace=5，rename/delete=32；replace拒绝的唯一原因未隔离，不能写所有失败都是sharing32。新增源实际CRLF，原writer默认newline，不能写原LF。父180→183文件/84→85映射新增gbk源码、既存pyc与kernel.appcore.dll缺起始身份，既有SHA无变化但源资格FAIL，首次加载触发未唯一定位；原源门不放宽。组件2.1728451000526547秒仅局部租约操作，不是3080模型请求/完整守卫或加速。所有实际子进程由本人Popen回收，未锁/改全局源，没有实际模型/C0/训练。',
            'next_zh': '本租约/951及27诊断不重跑追PASS；不要用局部租约替换旧完整文件hash门。下一在新事前工程明确目录命名空间与完整依赖覆盖，并把实际native/编码器预初始化和early-launch清理封存到准确新工作负载；未证明全树充分性前保留原完整守卫。配送继续逐原语精确调度与完整8+2、参数/依赖发布前核验及正式同初始化已训CE/同任务随机，不调成本或选头。'},
        'pickup': {'directory': P.relative_to(ROOT).as_posix(), 'production_directory': P.relative_to(ROOT).as_posix(), 'stable_online_gain': False, 'scientific_family3_started': False, 'candidate_installed': False,
            'saved_only_diagnosis_qualified': True, 'independent_checks': 277, 'independent_failures': 0, 'full_request_budget_qualified': False,
            'summary_zh': '取货n10只解析已存phase trace、三完整分支时钟、原缓存源码AST与旧局部103查询日志，不调用旧parser/路线/物理C0/模型或重跑超时请求。新stdlib二进制parser与另进程byte-slice参考、binary64精确有理数分段/并集及key频数核验277项0FAIL，两监督自然exit0且回收。三core+最终D守卫段2.4186700000427668/1.966973299975507/1.90200880006887秒，core后至marker .04794570000841022/.03296480005717575/.030149699988302588秒；前者含原仿真/缓存/物理/矩阵守卫，后者含序列化发布等组合，不能单列缓存或IO耗时。完整已保存前缀132000查询/70259hit/61741miss/61485eviction；局部103查询38不同key/65重复只说明该小日志有重复。完整请求没有key流、缓存内部没有计时，不能推容量最优、工作集或扩大缓存会加速。生产及独立各186实际文件/87映射起止一致，仅本无模型诊断负载，不升级完整科学依赖。原完整K10超时、只完成三reject/accept未执行的边界保持。',
            'next_zh': '不重跑277或旧缓存请求/103局部QA，不调容量追速度。下一另事前固定新的未执行局部分支，检验原插入循环不变base路线距离的惰性精确复用，保留首trial后base求值顺序、原串行累加/全部可行性/完整C0，不引入O(1)重分组或incr_eval=True。局部语义合格后才另登记完整K10有界请求与公平计时；仍优先正式模型提议、同初始化已训CE/同任务已训随机及真正独立强动态接拒，禁止旧无模型全日重复。'}
    }
    for name, directory in (('delivery', D), ('pickup', P)):
        tracks[name]['evidence_hashes'] = {q.relative_to(ROOT).as_posix(): sha(q) for q in sorted(directory.rglob('*')) if q.is_file()}
    create(STAGE / 'COMPLETED_EVIDENCE_INPUT.json', {'tracks': tracks, 'cycle': 4, 'engineering_not_scientific_family_ledger': True, 'builder_sha256': sha(Path(__file__))})
    print(json.dumps({'both_tracks_completed': True, 'delivery_overall_FAIL_preserved': True, 'pickup_independent': '277/0', 'stable_online_gain': False}))

if __name__ == '__main__':
    main()
