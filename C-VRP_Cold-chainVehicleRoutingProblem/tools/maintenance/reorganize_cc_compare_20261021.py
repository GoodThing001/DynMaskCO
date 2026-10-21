"""Build and verify the user-approved seven-method CC_Compare layout.

The old tree is renamed intact into a local recovery directory, including its
nested MAPT repository. No recursive deletion, Git-index operation, download,
upstream edit, training or server operation is performed.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys

SCRIPTS = next(p / 'scripts' for p in Path(__file__).resolve().parents
               if (p / 'scripts/project_paths.py').is_file())
sys.path.insert(0, str(SCRIPTS))
from project_paths import EXTENSION_ROOT as EXT, WORKSPACE_ROOT as ROOT
from organize_workspace_20261021 import preservation_checks

CC = ROOT / 'CC_Compare'
ARCHIVE = EXT / 'archive/cc_compare_reorganization_20261021'
LEGACY = ARCHIVE / 'legacy/CC_Compare'
STAGED = ARCHIVE / 'prepared_CC_Compare'
REPORT = EXT / 'results/cc_compare_organization_20261021'
PLAN = REPORT / 'migration_manifest.json'
UPSTREAM = EXT / 'comparisons/static_coldchain/upstream'
KEEP = ['ALNS', 'VNS', 'OR-Tools', 'PyVRP', 'RouteFinder', 'DeepACO', 'CaR']
SOURCES = {'ALNS': 'ALNS', 'OR-Tools': 'OR-Tools', 'PyVRP': 'PyVRP-0.14.0',
           'RouteFinder': 'RouteFinder', 'DeepACO': 'DeepACO', 'CaR': 'CaR'}
DOCUMENTS = [ROOT / n for n in ['AGENTS.md', 'README.md', '工作区导航.md', 'GITHUB_SYNC.md', '.gitignore']]
DOCUMENTS += [EXT / n for n in ['README.md', '00_项目导航.md', 'archive/README.md',
               'results/README.md', 'tools/maintenance/README.md', 'docs/文档索引与版本状态.md',
               'comparisons/static_coldchain/README.md', 'comparisons/static_coldchain/方法适用性.md']]


def io_path(path):
    """Windows extended paths keep deep legacy trees readable after relocation."""
    value = os.path.abspath(path)
    if os.name == 'nt' and not value.startswith('\\\\?\\'):
        value = '\\\\?\\' + value
    return Path(value)


def plain_path(path):
    value = str(path)
    return Path(value[4:] if value.startswith('\\\\?\\') else value)


def guarded(path):
    path = Path(os.path.abspath(path))
    if path == ROOT or not path.is_relative_to(ROOT):
        raise ValueError('Target must be inside the intended workspace')
    if path.relative_to(ROOT).parts[0].lower() in {'.git', '.vscode', '.codex', '.cursor', '.claude', 'maskco_code'}:
        raise ValueError('Protected root')
    for item in [path, *path.parents]:
        try:
            info = io_path(item).lstat()
        except FileNotFoundError:
            info = None
        if info and info.st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            raise ValueError('Reparse path refused')
        if item == ROOT:
            break
    return path


def sha(path):
    h = hashlib.sha256()
    with io_path(path).open('rb') as f:
        while data := f.read(4 * 1024 * 1024):
            h.update(data)
    return h.hexdigest()


def relative(path):
    return Path(path).relative_to(ROOT).as_posix()


def write(path, text):
    path = guarded(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode('utf-8'))


def save(path, value):
    write(path, json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def git_state(folder):
    def query(*args):
        return subprocess.check_output(['git', '--no-optional-locks', '-c', 'core.longpaths=true', '-C', str(folder), *args]).decode('utf-8')
    return dict(head=query('rev-parse', 'HEAD').strip(),
                status=query('status', '--porcelain', '--untracked-files=all'))


def tree_rows(folder, *, hash_files=True):
    rows = []
    for base, dirs, files in os.walk(io_path(guarded(folder)), followlinks=False):
        base = plain_path(base)
        for name in dirs:
            guarded(base / name)
        for name in sorted(files):
            p = guarded(base / name)
            info = io_path(p).stat()
            row = dict(path=p.relative_to(folder).as_posix(), bytes=info.st_size)
            if hash_files:
                row['sha256'] = sha(p)
            rows.append(row)
            if hash_files and len(rows) % 5000 == 0:
                print(json.dumps(dict(phase='hash_inventory', files=len(rows))), flush=True)
            elif not hash_files and len(rows) % 10000 == 0:
                print(json.dumps(dict(phase='metadata_inventory', files=len(rows))), flush=True)
    return sorted(rows, key=lambda row: row['path'])


def validate_tree(folder, rows, *, hashes=True, allow_new_runs=False):
    expected = {r['path']: r for r in rows}
    actual = tree_rows(folder, hash_files=False)
    actual_paths = {r['path'] for r in actual}
    assert set(expected) <= actual_paths, 'Original tree member missing'
    extra = actual_paths - set(expected)
    assert not extra or (allow_new_runs and all(p.startswith(('results/', 'checkpoints/')) for p in extra)), 'Unexpected tree member'
    for index, row in enumerate(actual, 1):
        if row['path'] not in expected:
            continue
        old = expected[row['path']]
        assert row['bytes'] == old['bytes'], row['path']
        if hashes:
            assert sha(folder / row['path']) == old['sha256'], row['path']
            if index % 10000 == 0:
                print(json.dumps(dict(phase='verify_file_sha', files=index)), flush=True)


def copy_identical(source, destination, rows):
    source, destination = guarded(source), guarded(destination)
    assert not destination.exists(), destination
    expected = sha(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    assert sha(destination) == expected
    rows.append(dict(source=relative(source), destination=destination.relative_to(STAGED).as_posix(),
                     bytes=source.stat().st_size, sha256=expected))


def checkpoint_destination(method, path):
    parts = path.parts
    if method == 'CaR' and parts[:2] == ('pretrained', 'CVRP') and '100' in path.as_posix():
        return Path('checkpoints/CaR') / Path(*parts[1:])
    if method == 'CaR' and parts[:2] == ('pretrained', 'CVRPBLTW') and '100' in path.as_posix():
        return Path('checkpoints/CaR') / Path(*parts[1:])
    if method == 'DeepACO' and parts[0] == 'pretrained' and len(parts) == 3 and parts[1] in {'cvrp', 'cvrp_nls'} and path.name == 'cvrp100.pt':
        return Path('checkpoints/DeepACO') / Path(*parts[1:])
    return None


def build_layout():
    copied, sources, excluded = [], [], []
    pins = json.loads((EXT / 'comparisons/static_coldchain/SOURCE_PINS.json').read_text(encoding='utf-8'))
    by_name = {r['name']: r for r in pins}
    binary = {'.pt', '.pth', '.ckpt', '.pkl', '.npz', '.npy', '.zip', '.gz', '.whl', '.so', '.dll', '.exe', '.a', '.o', '.pdb', '.lib'}
    for method, folder in SOURCES.items():
        origin = UPSTREAM / folder
        pin = by_name[folder]
        provenance = origin / 'SNAPSHOT_PROVENANCE.json'
        assert sha(provenance) == pin['provenance_sha256']
        recorded = json.loads(provenance.read_text(encoding='utf-8'))
        for row in recorded['files']:
            assert sha(origin / row['path']) == row['sha256'], (method, row['path'])
        sources.append(dict(method=method, snapshot_name=folder, upstream_repo=pin['repo'],
                            commit=pin['commit'], original_provenance=relative(provenance),
                            original_provenance_sha256=sha(provenance), full_snapshot_remains_at=relative(origin)))
        for row in tree_rows(origin, hash_files=False):
            rel = Path(row['path'])
            path = origin / rel
            checkpoint = checkpoint_destination(method, rel)
            if checkpoint is not None:
                copy_identical(path, STAGED / checkpoint, copied)
                continue
            if rel.parts[0] in {'data', 'pretrained', 'checkpoints', '__pycache__'} or path.suffix.lower() in binary:
                excluded.append(dict(method=method, path=rel.as_posix(), reason='Bundled non-current benchmark data, unrelated weights, or rebuildable binary retained only in original evidence snapshot'))
                continue
            copy_identical(path, STAGED / 'models' / method / rel, copied)
    # Preserve the existing four RouteFinder weights without importing legacy adapters.
    for path in sorted((CC / 'RouteFinder/checkpoints').rglob('*.ckpt')):
        rel = path.relative_to(CC / 'RouteFinder/checkpoints')
        copy_identical(path, STAGED / 'checkpoints/RouteFinder' / rel, copied)
    data_manifest = EXT / 'results/static_delivery/publication_protocol_v1_preflight_20261021/dataset_manifest.json'
    data = json.loads(data_manifest.read_text(encoding='utf-8'))
    assert len(data['files']) == 56 and sum(r['eligible'] for r in data['files']) == 47
    data_rows = []
    for row in data['files']:
        source = EXT / row['path'].replace('\\', '/')
        assert sha(source) == row['sha256']
        dest = Path('data/vrptwq_100') / source.name
        copy_identical(source, STAGED / dest, copied)
        data_rows.append(dict(case=row['name'], path=dest.relative_to('data').as_posix(),
                              original_frozen_path=relative(source), sha256=row['sha256'],
                              eligible=row['eligible'], geometry_sha256=row['geometry_sha256']))
    copy_identical(EXT / 'results/static_delivery/publication_protocol_v1_preflight_20261021/geometry_folds.json',
                   STAGED / 'data/geometry_folds.json', copied)
    save(STAGED / 'data/DATASET_MANIFEST.json', dict(identity=data['protocol'], files=data_rows,
         eligible=47, source_cases=56, originals_modified=False, customer_instances_generated=False,
         role='Byte-identical local mirror; all executed solvers continue reading original frozen paths'))
    for name in ['publication_model_v1.json', 'publication_experiment_freeze_v1.json', 'algorithm_candidate_v0.json']:
        copy_identical(EXT / 'configs/static_delivery' / name, STAGED / 'configs' / name, copied)
    save(STAGED / 'configs/SOURCE_PINS.json', sources)
    methods = []
    statuses = {'ALNS': 'full_objective_development_adapter', 'VNS': 'full_objective_development_adapter',
                'OR-Tools': 'distance_routing_control_only', 'PyVRP': 'distance_routing_control_only',
                'RouteFinder': 'candidate_requires_frozen_profit_adaptation_and_training',
                'DeepACO': 'candidate_requires_frozen_profit_adaptation_and_training',
                'CaR': 'candidate_requires_frozen_profit_adaptation_and_training'}
    for method in KEEP:
        methods.append(dict(method=method, model_dir='models/' + method, status=statuses[method],
            formal_comparison_qualified=False,
            runtime_implementation=(relative(EXT / 'scripts/static_profit/profit_search.py') + (':vns_search' if method == 'VNS' else ':alns_search')) if method in {'ALNS', 'VNS'} else None,
            pretrained_quality_profit_claim=False))
    save(STAGED / 'configs/methods.json', dict(date='2026-10-21', selected_by_user=KEEP,
        selection_basis='User-approved active workspace scope, not removal based on observed test performance',
        model_data_analysis_freeze_unchanged=True, outer_test_enabled=False, methods=methods,
        canonical_evaluator=relative(EXT / 'comparisons/static_coldchain/adapters/publication_model_v1_strict.py')))
    save(STAGED / 'models/VNS/IMPLEMENTATION.json', dict(kind='Custom full-objective VNS, no external upstream repository',
         source=relative(EXT / 'scripts/static_profit/profit_search.py'), function='vns_search',
         separate_source_copy=False, reason='Keep one implementation shared with the frozen-objective evaluator and existing baseline'))
    save(STAGED / 'checkpoints/ASSETS.json', [r for r in copied if r['destination'].startswith('checkpoints/')])
    save(STAGED / 'configs/OMITTED_VENDOR_ASSETS.json', excluded)
    save(STAGED / 'results/catalog.json', dict(kind='Pointers to original experiment evidence; no scores recomputed',
        current_state=relative(EXT / '项目当前状态.md'),
        development_results=relative(EXT / 'results/static_delivery/algorithm_dev_20261021/README.md'),
        dev04_summary=relative(EXT / 'results/static_delivery/algorithm_dev_20261021/remote/dev04/descriptive_summary.json'),
        original_legacy_results_after_migration=relative(LEGACY / 'results'), formal_test_executed=False))
    write_docs()
    return copied, sources, excluded


def write_docs():
    docs = {
        'README.md': '''# 静态冷链对比方法工作区

当前仅保留用户确认的7项：ALNS、VNS、OR-Tools、PyVRP、RouteFinder、DeepACO、CaR。一级子目录按职责分类，各方法位于models内。

| 目录 | 内容 |
|---|---|
| [data](data/README.md) | 56份作者100客户原实例的同字节镜像、47例纳入身份与几何划分 |
| [models](models/README.md) | 6套固定作者源码及1个自定义VNS实现入口 |
| [scripts](scripts/README.md) | 统一经典基线运行入口、目录与SHA核验入口 |
| [configs](configs/README.md) | 已有模型/分析配置镜像、方法清单与固定源码来源 |
| [docs](docs/README.md) | 保留范围、方法资格与目录规则 |
| [results](results/README.md) | 原始实验结果入口；新运行使用独立子目录 |
| [checkpoints](checkpoints/README.md) | 与当前候选有关的已有预训练权重、原路径和SHA |

ALNS/VNS已有同目标开发适配；OR-Tools/PyVRP仍属距离路由控制；其余三项为待适配候选。所有方法尚不能据此宣称正式优胜。当前MASKCO在[主算法目录](../C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_profit/README.md)，不重复复制。

原有20项适用性审查是历史候选池，当前活动名单以[methods.json](configs/methods.json)为准。旧动态方法、适配器、历史结果和MAPT内嵌Git仓库已整体移出活动目录；原字节和未跟踪文件均可恢复，见[迁移报告](../C-VRP_Cold-chainVehicleRoutingProblem/results/cc_compare_organization_20261021/README.md)。

本次只调整目录与入口；现行训练、评价、远程任务和历史manifest继续使用原冻结路径，不重写已有实验身份。作者源码按原字节保存；原包中的非当前数据和无关权重不放入活动模型树，完整来源仍保存在原证据快照。
''',
        'CLAUDE.md': '''# 对比方法目录规则

先读[README](README.md)、[方法清单](configs/methods.json)与根AGENTS.md。仅保留已确认7项，子目录按data/models/scripts/configs/docs/results/checkpoints组织。第三方源码位于models内，保持只读。

当前运行仍委托扩展目录的严格评价与搜索；不得把预训练原目标成绩冒充冷链净效益成绩。历史材料见迁移报告，不恢复为活动方法。
''',
        'models/README.md': '''# 方法源码与实现入口

| 方法 | 身份 |
|---|---|
| [ALNS](ALNS) | 官方固定ALNS源码；完整目标适配器仍在扩展profit_search中 |
| [VNS](VNS/IMPLEMENTATION.json) | 自定义移位/交换/反转/合并搜索；指向唯一现行实现，不复制算法代码 |
| [OR-Tools](OR-Tools) | 固定官方源码，经典路由控制，不是完整净效益求解器 |
| [PyVRP](PyVRP) | 固定0.14.0对应源码，不与另一开发版本混用 |
| [RouteFinder](RouteFinder) | 候选，需完整目标适配与验证 |
| [DeepACO](DeepACO) | 候选，需完整目标适配与验证 |
| [CaR](CaR) | 候选，需品质约束与完整目标适配 |

作者源码字节不修改。权重集中于checkpoints，当前比较数据集中于data；原作者包内的小型测试fixture保持其源码结构。移出的原基准数据与无关权重见[资产排除记录](../configs/OMITTED_VENDOR_ASSETS.json)，完整原快照仍在扩展证据目录。直接运行作者脚本时需显式指定集中后的数据/权重路径，尚无净效益适配的候选不提供虚假的统一求解接口。
''',
        'data/README.md': '''# 统一比较数据

vrptwq_100保存56份作者原100客户TXT的同字节镜像；[数据清单](DATASET_MANIFEST.json)保留9例排除与47例纳入标志，不能把56例全部作为可行训练集。[geometry_folds.json](geometry_folds.json)保持原四组几何隔离身份。

现行求解仍读取扩展目录原冻结数据，本镜像只提供集中浏览和后续适配所需路径。不生成客户、不改坐标、需求、窗口或服务时间，不重复定义划分。
''',
        'configs/README.md': '''# 比较配置

- [methods.json](methods.json)：用户确认的7项及当前运行资格。
- [SOURCE_PINS.json](SOURCE_PINS.json)：6个官方源码快照的commit、来源与原件SHA。
- [publication_model_v1.json](publication_model_v1.json)：现行冻结模型的同字节镜像。
- [publication_experiment_freeze_v1.json](publication_experiment_freeze_v1.json)：现行分析计划的同字节镜像。
- [algorithm_candidate_v0.json](algorithm_candidate_v0.json)：开发候选，非最终方法锁定。
- [OMITTED_VENDOR_ASSETS.json](OMITTED_VENDOR_ASSETS.json)：未复制到活动树的原包数据、无关权重与二进制。

镜像不替代原FREEZE_LOCK中的权威路径；任何新方法或检查点须继续满足原分析计划的正式测试门槛。
''',
        'scripts/README.md': '''# 统一运行与核验入口

从工作区根目录运行：

```powershell
python -B CC_Compare/scripts/run_classical.py --help
python -B CC_Compare/scripts/verify_layout.py
```

run_classical.py仅委托现有static_profit入口的classical子命令，执行同目标ALNS/VNS。示例参数为--fold 0 --role validation --case c205 --seconds 10 --iterations 100000 --out CC_Compare/results/new_validation；每次使用新输出目录。它不开放正式外层测试。

其余方法的资格见[方法配置](../configs/methods.json)，尚未完成适配者不伪装为可运行的利润基线。verify_layout.py核对目录、官方源码、数据镜像、权重与科研冻结；加--full-archive可同时重新核验旧目录全部原件SHA。
''',
        'docs/README.md': '''# 方法范围与目录约定

2026-10-21用户明确选择保留ALNS、VNS、OR-Tools、PyVRP、RouteFinder、DeepACO、CaR。其余旧方法仅保留在独立历史恢复树，移出当前活动范围；这不是按测试结果挑选或否定这些方法的学术价值。

当前模型是静态多车品质净效益，旧动态取货/接拒协议及成绩不直接复用。7项保留不等于7项已完成同目标适配：ALNS/VNS为开发基线，OR-Tools/PyVRP为距离策略控制，三项神经候选待适配与训练。

新增数据放data、模型放models、运行入口放scripts、配置放configs、说明放docs、实跑输出放results、权重放checkpoints。第三方原代码不修改，原许可证随源码保留。

[完整整理报告](../../C-VRP_Cold-chainVehicleRoutingProblem/results/cc_compare_organization_20261021/README.md)记录原路径、现路径、备份、核验和已知边界。原历史manifest不改写，恢复时需恢复原工作区相对位置。
''',
        'results/README.md': '''# 比较结果入口

[catalog.json](catalog.json)指向现行开发报告、dev04描述性汇总和旧结果的历史归档位置。本次不生成新成绩、不改写原始失败、NO-GO或限定结论。

新结果放新建运行目录，带方法、配置、数据、种子、时间预算和独立路线审计。当前神经优胜与正式测试资格以扩展目录的项目当前状态和分析计划为准。
''',
        'checkpoints/README.md': '''# 已有候选权重

[ASSETS.json](ASSETS.json)记录原路径、新路径、大小与SHA。保留RouteFinder已有4份、DeepACO的2份CVRP100和CaR的3份100客户CVRP/CVRPBLTW权重，共9份。

这些是作者原任务预训练权重，不是当前冷链净效益训练检查点，不能直接当作同目标方法结果。现行MASKCO开发权重继续保存在原运行目录/服务器；不移入此处混淆身份。二进制保持本地，不自动发布。
''',
        '.gitignore': '''__pycache__/
*.py[cod]
/checkpoints/*
!/checkpoints/README.md
!/checkpoints/ASSETS.json
/results/*
!/results/README.md
!/results/catalog.json
''',
        '.gitattributes': '''# Preserve copied source/provenance bytes across platforms.
models/** -text
data/** -text
configs/** -text
checkpoints/ASSETS.json -text
''',
        'scripts/run_classical.py': '''"""Delegate to the existing strict-profit classical baseline entry point."""
from pathlib import Path
import runpy
import sys

workspace = next(p for p in Path(__file__).resolve().parents
                 if (p / 'C-VRP_Cold-chainVehicleRoutingProblem/scripts/project_paths.py').is_file())
entry = workspace / 'C-VRP_Cold-chainVehicleRoutingProblem/scripts/static_profit/run.py'
sys.argv = [str(entry), 'classical', *sys.argv[1:]]
runpy.run_path(str(entry), run_name='__main__')
''',
        'scripts/verify_layout.py': '''"""Check the approved seven-method directory without running experiments."""
from pathlib import Path
import runpy
import sys

workspace = next(p for p in Path(__file__).resolve().parents
                 if (p / 'C-VRP_Cold-chainVehicleRoutingProblem/scripts/project_paths.py').is_file())
entry = workspace / 'C-VRP_Cold-chainVehicleRoutingProblem/tools/maintenance/reorganize_cc_compare_20261021.py'
sys.path.insert(0, str(entry.parent))
sys.argv = [str(entry), 'verify', *sys.argv[1:]]
runpy.run_path(str(entry), run_name='__main__')
''',
    }
    for name, text in docs.items():
        write(STAGED / name, text)


def prepare():
    if ARCHIVE.exists() or REPORT.exists():
        raise ValueError('Refuse to overwrite an existing organization run')
    preserved = preservation_checks()
    nested = git_state(CC / 'MAPT_code')
    inventory = tree_rows(CC)
    REPORT.mkdir(parents=True)
    save(REPORT / 'legacy_inventory.json', inventory)
    document_backups = []
    for path in DOCUMENTS:
        backup = ARCHIVE / 'navigation_before' / relative(path)
        guarded(backup).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, backup)
        document_backups.append(dict(path=relative(path), backup=relative(backup), sha256=sha(path)))
    baseline = []
    for subdir in ['scripts/static_profit', 'scripts/static_delivery']:
        for path in sorted((EXT / subdir).glob('*')):
            if path.is_file():
                baseline.append(dict(path=relative(path), sha256=sha(path)))
    copied, sources, excluded = build_layout()
    staged_inventory = tree_rows(STAGED)
    plan = dict(schema='cc-compare-type-layout-v1', date='2026-10-21', status='PREPARED',
                user_approved_methods=KEEP, old_root=relative(CC), recovery_root=relative(LEGACY),
                legacy_file_count=len(inventory), legacy_bytes=sum(r['bytes'] for r in inventory),
                legacy_inventory_sha256=sha(REPORT / 'legacy_inventory.json'),
                nested_git_before=nested, original_navigation=document_backups,
                scientific_preservation_before=preserved, untouched_algorithm_files=baseline,
                copied_files=copied, staged_inventory=staged_inventory, sources=sources,
                excluded_vendor_assets=len(excluded), root_git_head=subprocess.check_output(['git', '--no-optional-locks', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip(),
                git_commit_or_push=False, servers_modified=False)
    save(PLAN, plan)
    print(json.dumps(dict(status='PREPARED', legacy_files=len(inventory),
          legacy_GiB=round(plan['legacy_bytes']/2**30, 3), active_files=len(staged_inventory),
          active_MiB=round(sum(r['bytes'] for r in staged_inventory)/2**20, 2))), flush=True)


def apply():
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    assert plan['status'] == 'PREPARED'
    assert sha(REPORT / 'legacy_inventory.json') == plan['legacy_inventory_sha256']
    legacy_rows = json.loads((REPORT / 'legacy_inventory.json').read_text(encoding='utf-8'))
    preservation_checks()
    validate_tree(CC, legacy_rows)
    validate_tree(STAGED, plan['staged_inventory'])
    assert git_state(CC / 'MAPT_code') == plan['nested_git_before']
    assert not guarded(LEGACY).exists()
    guarded(LEGACY.parent).mkdir(parents=True, exist_ok=True)
    # Absolute source and target are checked before both non-recursive renames.
    guarded(CC).rename(guarded(LEGACY))
    plan['status'] = 'LEGACY_RENAMED_NEW_TREE_PENDING'
    save(PLAN, plan)
    try:
        guarded(STAGED).rename(guarded(CC))
    except BaseException:
        if not CC.exists():
            guarded(LEGACY).rename(guarded(CC))
            plan['status'] = 'PREPARED_RENAME_ROLLED_BACK'
            save(PLAN, plan)
        raise
    plan['status'] = 'APPLIED_PENDING_NAVIGATION'
    save(PLAN, plan)
    print(json.dumps(dict(status=plan['status'], methods=KEEP)), flush=True)


def verify(full_archive=False):
    plan = json.loads(PLAN.read_text(encoding='utf-8'))
    assert plan['status'] in {'APPLIED_PENDING_NAVIGATION', 'VERIFIED'}
    assert {p.name for p in CC.iterdir() if p.is_dir()} == {'data','models','scripts','configs','docs','results','checkpoints'}
    assert {p.name for p in (CC / 'models').iterdir() if p.is_dir()} == set(KEEP)
    assert sha(REPORT / 'legacy_inventory.json') == plan['legacy_inventory_sha256']
    legacy_rows = json.loads((REPORT / 'legacy_inventory.json').read_text(encoding='utf-8'))
    validate_tree(LEGACY, legacy_rows, hashes=full_archive)
    assert git_state(LEGACY / 'MAPT_code') == plan['nested_git_before']
    validate_tree(CC, plan['staged_inventory'], allow_new_runs=True)
    for row in plan['copied_files']:
        assert sha(CC / row['destination']) == row['sha256']
    for path in (CC / 'scripts').glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'))
    for row in plan['untouched_algorithm_files']:
        assert sha(ROOT / row['path']) == row['sha256'], row['path']
    preserved = preservation_checks()
    result = dict(status='PASS', selected_methods=KEEP, top_level_categories=7,
        original_tree_files=len(legacy_rows), original_tree_bytes=plan['legacy_bytes'],
        original_tree_sha_checked=full_archive, nested_git_history_and_untracked_status_unchanged=True,
        copied_files_sha_checked=len(plan['copied_files']), active_files=len(plan['staged_inventory']),
        unchanged_algorithm_files=len(plan['untouched_algorithm_files']), scientific_preservation=preserved,
        permanent_file_deletions=0, old_methods_removed_from_active_tree=True,
        current_runtime_original_paths_preserved=True, git_commit_or_push=False, servers_modified=False)
    if full_archive:
        plan['full_archive_verification'] = result
    plan['status'] = 'VERIFIED'
    save(PLAN, plan)
    save(REPORT / 'verification.json', result)
    print(json.dumps({k:v for k,v in result.items() if k!='scientific_preservation'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'apply', 'verify'])
    parser.add_argument('--full-archive', action='store_true')
    args = parser.parse_args()
    if args.action == 'prepare': prepare()
    elif args.action == 'apply': apply()
    else: verify(args.full_archive)
