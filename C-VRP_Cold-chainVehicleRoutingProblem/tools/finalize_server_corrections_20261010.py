"""Update only live local documentation; preserve historical protocols/packages."""
from pathlib import Path
import json,datetime,hashlib,csv
E=Path(__file__).resolve().parents[1]
name='三线修正_服务器复验与文献参数版本_2026-10-10.md'
report=E/'docs/当前规划/决策与审计'/name

def prepend(path,text):
 p=E/path;s=p.read_text(encoding='utf-8');assert text not in s
 title,rest=s.split('\n',1);p.write_text(title+'\n\n'+text+'\n'+rest,encoding='utf-8')

brief='**2026-10-10 服务器修正验收**：隔离服务器63项独立核对通过，商品动力学11项回归通过；恢复33项依赖与旧复验日志，14份正式JSON一致。新增商品级一阶/零阶、温区/外推和货物计时/电输入单位版本，参数逐项注明文献或仿真推导。项目内未检出可识别实测校准资料，透明文献仿真不以自有采集为通用门槛。原训练目标hash仍缺，P/D负裁决与暂停状态保持；共享MD未上传。'
for path in ['项目当前状态.md','00_项目导航.md']:
 prepend(path,'> '+brief+' [本轮完整修正记录](docs/当前规划/决策与审计/'+name+')。')
for path in ['docs/当前规划/决策与审计/方向A_总核查与下一阶段路线图_2026-09-30.md',
             'docs/当前规划/结果与进度/执行进度表.md',
             'docs/当前规划/决策与审计/三线审计修正_执行记录与未决项_2026-10-09.md']:
 rel=name if '/决策与审计/' in path else '../决策与审计/'+name
 prepend(path,'> '+brief+' [完整证据]('+rel+')。下文较早的搜索范围和未访问服务器表述按其时间理解。')

p=E/'docs/当前规划/决策与审计/三线审计修正_执行记录与未决项_2026-10-09.md'
s=p.read_text(encoding='utf-8').replace('服务器也无repro_check.txt','当时仅查扩展目录未找到repro_check.txt；最新项目根目录搜索已在B_REPRO_CLEAN找回，见顶部修正记录')
s=s.replace('历史缺失工件没有恢复命中；校准数据、强滚动优化器及新模型目标不能由该工程资格推定存在','当时历史搜索无恢复命中；新一轮已恢复部分依赖/旧日志，并实现独立商品动力学组件，具体范围以顶部新记录为准。原训练身份、强滚动优化器及新模型目标不能由工程资格推定存在')
p.write_text(s,encoding='utf-8')

for line in ['01_取货线','02_配送线','03_负结果线']:
 text='> '+brief+' [完整修正记录](../docs/当前规划/决策与审计/'+name+')。'
 for filename in ['README.md','文档索引.md','实验结果索引.md']:
  prepend(line+'/'+filename,text)

P='01_取货线/文档/协议与审计/取货审计修正_现行解释与后续验收_2026-10-09.md'
prepend(P,'> **2026-10-10**：新增独立商品动力学部件与完整轨迹工程重放已回归通过；旧40日未追补事件日志，原品质参数不变。文献支持的仿真可继续研究，不要求自有实物采集作为通用前提。[完整来源与服务器验收](../../../docs/当前规划/决策与审计/'+name+')。')
p=E/P;s=p.read_text(encoding='utf-8').replace('先校准适用产品及温域','先核定适用商品、文献方程/单位和温域；现实预测主张另需实物验证');p.write_text(s,encoding='utf-8')
prepend('02_配送线/文档/研究设计/配送审计修正_功能合同与启动验收_2026-10-09.md',
 '> **2026-10-10**：商品品质模块与电输入单位检查已独立实现并通过服务器11项回归。旧事件检查器未冒称已整合该物理组件；强滚动优化与迭代收益仍需新科学版本。[商品、来源、范围及假设](../../../docs/当前规划/决策与审计/'+name+')。透明仿真不要求实测校准作为通用门槛。')
prepend('03_负结果线/文档/论文与投稿/负结果审计修正_主张与复现资格_2026-10-09.md',
 '> **2026-10-10**：33项依赖全部恢复、31项字节一致，14正式JSON一致；新隔离固定状态/头腔/图表有限复验通过，原训练目标hash和历史逐事件日志仍缺。[恢复与63项独立裁决](../../../docs/当前规划/决策与审计/'+name+')。下文17项依赖缺口是恢复前静态候选状态。')
p=E/'03_负结果线/文档/论文与投稿/论文工作稿_收口修订_2026-10-09.md'
s=p.read_text(encoding='utf-8').replace('本轮没有重跑实验或访问服务器。','最初静态收口没有运行模型；2026-10-09/10已按新增授权访问服务器进行有限隔离工程复验，未重跑正式主表或C1。')
s=s.replace('本地封存包缺 `repro_check.txt`；发布候选需补可交付的执行日志与两套隔离依赖闭包，不能在当前运行树执行旧人工替换入口。',
 '原封存包仍缺 `repro_check.txt`；包外新恢复证据已找回该日志和33项依赖，隔离固定状态/头腔及图再生通过。原训练身份仍缺，不能因此称完整复现，也不能在当前运行树执行旧人工替换入口。')
s=s.replace('> **定稿与发布状态（作者注，投稿时删除本节）**：',
 '> **2026-10-10 新增复验（作者注）**：[恢复证据与服务器有限复验](../../../docs/当前规划/决策与审计/'+name+')；[当前三图及来源](../../图表工作稿/2026-10-09/FIGURE_PROVENANCE.json)。原包及正式JSON保持。\n>\n> **定稿与发布状态（作者注，投稿时删除本节）**：')
p.write_text(s,encoding='utf-8')
prepend('03_负结果线/文档/论文与投稿/待发布材料与隔离入口说明_2026-10-09.md',
 '> **2026-10-10**：已在包外恢复旧执行日志、33依赖并完成新的有限服务器复验；当前三图在[隔离图表目录](../../图表工作稿/2026-10-09/FIGURE_PROVENANCE.json)重新生成且已检查。[最新复现资格](../../../docs/当前规划/决策与审计/'+name+')覆盖下文“本轮不重画/只做静态核对”的较早范围；完整训练与闭环复现仍未具资格。')
prepend('scripts/correction_engineering/README.md',
 '> **2026-10-10**：新增literature_quality.py与商品参数JSON；封闭形式、变温积分、温域/外推、货物计时和电输入单位11项测试在本地/服务器通过。历史合同保持。新组件独立使用，不声称已集成完整强动态优化。[来源和验收](../../docs/当前规划/决策与审计/'+name+')。')

W=E/'results/three_line_parallel_20261009/WORK_STATUS.json';d=json.loads(W.read_text(encoding='utf-8'))
d['server_authorized_corrections']={'completed_at':datetime.datetime.now().isoformat(),
 'scope':'new isolated engineering QA, read-only recovery, literature commodity module, live local documentation',
 'report':report.relative_to(E).as_posix(),'server_independent_checks':63,'server_independent_pass':True,
 'commodity_tests':11,'commodity_test_server_pass':True,'recovered_dependencies':33,
 'byte_equal_dependencies':31,'formal_json_matched':14,'original_training_hash_recovered':False,
 'own_physical_measurements_identified_in_bounded_search':False,
 'literature_simulation_requires_own_calibration':False,'shared_md_uploaded':False,
 'new_engineering_py_json_uploaded':True,'old_scientific_protocol_modified':False,
 'p_d_core_online_gain_established':False,'formal_full_reproduction_qualified':False}
W.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')

A=E/'results/three_line_corrections_20261009';profiles=E/'scripts/correction_engineering/literature_quality_profiles_v1.json'
search=json.loads((A/'PHYSICAL_CALIBRATION_DATA_SEARCH.json').read_text(encoding='utf-8'))
decision={'date':datetime.datetime.now().isoformat(),'search_file':'PHYSICAL_CALIBRATION_DATA_SEARCH.json',
 'scope':search['scope'],'files_seen':search['files_seen'],'tables_checked':len(search['measurement_table_candidates']),
 'candidate_classification':'simulation performance/outcome tables, file manifests and planning source/schedule registries; no identifiable own temperature/power/quality measurement data found',
 'own_measurements_found':False,'bounded_search_not_global_absence':True,
 'simulation_allowed_without_own_physical_calibration':True,
 'new_profiles':profiles.relative_to(E).as_posix(),'profile_sha256':hashlib.sha256(profiles.read_bytes()).hexdigest(),
 'source_evidence':'3 original food papers with exact table/equation locations in profiles; separate van heat/electric structure paper in report',
 'frozen_rate_is_scenario_derived':True,'old_p_d_b_parameters_changed':False,
 'explicit_assumptions_remain':['vehicle pilot UA/COP/power/door heat','commodity/packaging transfer','frozen rate initial 9-to-5 derivation','objective weights and monetary conversion'],
 'server_module_tests_passed':11,'food_safety_or_business_calibration_claimed':False}
with (A/'LITERATURE_PARAMETER_ADOPTION_20261010.json').open('x',encoding='utf-8') as f:json.dump(decision,f,ensure_ascii=False,indent=2)
print(json.dumps({'live_docs_updated':True,'new_parameter_adoption_saved':True,'remote_md_upload':False}))
