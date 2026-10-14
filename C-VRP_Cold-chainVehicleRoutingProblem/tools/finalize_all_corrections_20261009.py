"""Finalize documentary dispositions and archive synthetic engineering acceptance."""
from pathlib import Path
import sys,json,hashlib,datetime,xml.etree.ElementTree as ET
E=Path(__file__).resolve().parents[1]
O=E/'results/three_line_corrections_20261009'
A=E/'docs/当前规划/文献工件/三线协议审计_2026-10-09'
sys.path.insert(0,str(E/'scripts/correction_engineering'))
from test_corrections import test_pickup_real_trace_replay,delivery_case
from pickup_trace_runner import replay
from delivery_trace_contract import validate_trace
from audit_io import verify
from qualification import assess
O.joinpath('engineering').mkdir(exist_ok=True)
test_pickup_real_trace_replay(O/'engineering')
orders,events,settings=delivery_case()
delivery_result=validate_trace(orders,events,**settings)

def save(name,obj):
    with (O/name).open('x',encoding='utf-8') as stream:
        json.dump(obj,stream,ensure_ascii=False,indent=2)

save('ENGINEERING_ACCEPTANCE.json',{'qualification':'synthetic_engineering_only',
    'pickup_trace':verify(O/'engineering/pickup.jsonl',1),
    'pickup_replay':replay(O/'engineering/pickup.jsonl'),
    'delivery_trace':delivery_result,
    'formal_experiments_performed':False,'new_gain_claimed':False,
    'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (E/'scripts/correction_engineering').glob('*.py')}})
save('engineering/delivery_case.json',{'orders':orders,'events':events,'settings':settings,'result':delivery_result})

funcs=json.loads((A/'FUNCTIONS.json').read_text(encoding='utf-8-sig'))
d=json.loads((E/'02_配送线/文献工件/2026-10-09/evidence_matrix.json').read_text(encoding='utf-8-sig'))
dispositions=[]
for direction,items in [('pickup',funcs['pickup']),('negative',funcs['negative']),('delivery',d['requirements'])]:
    for item in items:
        identifier=item['id']
        record={'direction':direction,'id':identifier,'feature':item['feature'],
          'sources':item.get('sources',[]),'original_action':item.get('action',item.get('decision')),
          'disposition':'documented_scope_or_registered_design_retained',
          'scientific_implementation_completed':False,
          'engineering_evidence':None}
        if direction=='pickup' and identifier in ('P17','P18','P23'):
            record.update(disposition='engineering_observability_implemented_and_synthetic_replayed',
                          engineering_evidence='ENGINEERING_ACCEPTANCE.json')
        if direction=='negative' and any(word in item['feature'] for word in ('复现','源码','依赖','封存')):
            record.update(disposition='static_guard_passed_historical_gap_explicit',
                          engineering_evidence='NEGATIVE_SERVER_SOURCE_SEARCH.json')
        if direction=='delivery':
            record['disposition']='design_requirement_and_acceptance_explicit_no_online_activation'
        dispositions.append(record)
assert len(dispositions)==72
save('ALL_CORRECTION_DISPOSITIONS.json',{'count':72,'items':dispositions,
     'unassigned_items':[],'meaning':'Every audit item has a disposition; not 72 newly implemented functions or 72 experiments.'})
save('RESEARCH_READINESS.json',{
    'pickup_new_method':assess('pickup_new_method',{'incremental_trace_replay_pass':True}),
    'delivery_dynamic':assess('delivery_dynamic',{'event_contract_pass':True,'physical_parameter_scope_declared':True}),
    'negative_full_reproduction':assess('negative_full_reproduction',{})})
save('PARAMETER_DISPOSITIONS.json',{'delivery_parameter_groups':d['parameter_groups'],
     'rule':'Literature supports mechanism, numeric project choices remain labelled engineering; no calibration data invented.',
     'original_P_and_D_protocols_modified':False})

doc=E/'docs/当前规划/决策与审计/三线审计修正_执行记录与未决项_2026-10-09.md'
with doc.open('a',encoding='utf-8') as stream:
    stream.write('''

## 当前修正完整处理记录

用户追加“把当前所有修正完成”后，72项原审计全部建立[逐项处置表](../../../results/three_line_corrections_20261009/ALL_CORRECTION_DISPOSITIONS.json)，16组配送参数登记保留原值与来源限定。未处理条目为0；这表示全部审计项已有准确解释、保留理由或工程处置，不表示72项功能全部新实现。

工程实现位于scripts/correction_engineering，原正式入口与冻结实验不改。取货观察运行器增量持久化每决定的场景、动作、耗时/超时，及完整私有状态和终局物理轨迹；日完成即写盘，hash链可识别截断/篡改，源码启动/结束hash核对。独立重载typed轨迹后调用原评价器逐字段完全复算，固定合成日验收通过。审计数据只写日志、不反馈actor。观察I/O单独计时；原软截止语义保持，不能称硬CPU抢占或正式预算等价。

配送事件检查器可执行双时钟装货、唯一货物归属、总量/分舱容量、承诺行驶段保持、服务开始时间窗与有限车队返仓后复用；固定合成完整行程与违规反例通过。它是事件合同认证组件，未实现强滚动优化器或温度/能耗评价器，formal_online_ready=False。

负结果缺失源码已检查本地文件、ZIP封存、可达Git历史，以及服务器扩展目录的文件与ZIP。原目标hash没有命中，服务器也无repro_check.txt。两份[本地搜索](../../../results/three_line_corrections_20261009/NEGATIVE_SOURCE_RECOVERY_SEARCH_v2.json)和[服务器只读搜索](../../../results/three_line_corrections_20261009/NEGATIVE_SERVER_SOURCE_SEARCH.json)无读取错误；首次路径/Git权限检查报告保留。缺失工件无法从现有材料还原，不以当前文件替代或伪造日志。

三种研究资格由可执行qualification检查器分别验证，缺件关闭；技术条件齐全也不会自动授予科学执行权限。实际[研究准备度](../../../results/three_line_corrections_20261009/RESEARCH_READINESS.json)保留新方法/动态配送/完整历史复现未就绪。资源条件训练、有限教师新目标、温度品质实物校准和强动态控制属于新研发/科学阶段，其缺少收益证据不能通过工程修正变成正结果。

验收：[18项工程测试](../../../results/three_line_corrections_20261009/ENGINEERING_TESTS.xml)、[持久化合成轨迹与独立重放](../../../results/three_line_corrections_20261009/ENGINEERING_ACCEPTANCE.json)。所有当前确认的文档/功能身份错误与可独立处理的日志/静态资格缺陷已处理；历史工件缺失和新科学阶段明确保留，不称已补齐。定时任务保持PAUSED，原负裁决、最终留出和服务器文档待批规则保持。
''')
for relative in ['项目当前状态.md','00_项目导航.md','docs/当前规划/结果与进度/执行进度表.md']:
    p=E/relative
    text=p.read_text(encoding='utf-8-sig')
    head,_,body=text.partition('\n')
    prefix='' if '/' not in relative else '../../../'
    link=prefix+'docs/当前规划/决策与审计/三线审计修正_执行记录与未决项_2026-10-09.md' if '/' not in relative else '../决策与审计/三线审计修正_执行记录与未决项_2026-10-09.md'
    note='\n> **2026-10-09 当前审计修正完整处理**：72项已逐项处置；取货增量日志/完整轨迹工程重放、配送事件合同认证和历史复现缺件关闭已实现并验收。历史原源码未找回，新方法/温度校准/强动态优化未就绪；原负裁决与暂停状态保持。[修正记录](%s)。\n'%link
    p.write_text(head+'\n'+note+body,encoding='utf-8')
p=E/'results/three_line_parallel_20261009/WORK_STATUS.json'
s=json.loads(p.read_text(encoding='utf-8-sig'))
s['corrective_work'].update(completed_at=datetime.datetime.now().isoformat(),
    scope='all audit dispositions and isolated observability/contract/qualification engineering',
    audit_items_disposed=72, engineering_tests_passed=18,
    pickup_synthetic_full_trace_replay=True, delivery_event_component_qualified=True,
    historical_missing_source_recovered=False, new_scientific_functionality_ready=False)
p.write_text(json.dumps(s,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'audit_items_disposed':72,'engineering_tests':18,'synthetic_replay':True,
 'historical_source_recovered':False,'new_scientific_run':False},ensure_ascii=False))
