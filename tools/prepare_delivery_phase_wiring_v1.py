"""Create exclusive source copies; parse AST only, never import the workload."""
import ast
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'research/review_3080_20261010'
OLD = R / 'delivery/timed_actual_request_execution_root_v1/downloaded'
PREP = R / 'delivery/phase_timing_preparation_root_v1'
OUT = R / 'delivery/phase_timing_wiring_preparation_root_v1'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def replace(source, old, new):
    if source.count(old) != 1:
        raise ValueError('Unique frozen source anchor: ' + old[:80])
    return source.replace(old, new)


def create(name, value):
    p = OUT / name
    if isinstance(value, str):
        ast.parse(value) if p.suffix == '.py' else None
        with p.open('x', encoding='utf-8', newline='\n') as f:
            f.write(value)
    else:
        with p.open('x', encoding='utf-8', newline='\n') as f:
            json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


REQUEST = '''    def request(self, raw, directory, stage, cancelled, trace):
        """All spans and mandatory source/parameter gates stay inside request."""
        np,jax,jnp=self.np,self.jax,self.jnp
        from timed_evidence_io_v1 import save
        directory=Path(directory);directory.mkdir(exist_ok=False)
        m=trace.measure
        stage(1,'input_decode')
        def decode_public():
            row=json.loads(raw.decode('utf-8'))
            if self.digest(row)!=self.reg['input_digest']:raise ValueError('Fixed public input identity')
            if row['key']!=[12,3] or len(row['remaining'])!=25:raise ValueError('Fixed first engineering input')
            return row
        row=m(1,0,decode_public)
        def project_public():
            coordinate=np.asarray(self.normalize(jnp.asarray(row['data']['coords'],jnp.float32)[None]))[0]
            return self.make_features(row['data'],self.contract,coordinate,self.units,self.contract.identity())
        features=m(2,0,project_public)
        m(8,0,self.save_npz,directory/'FEATURES.npz',features=features)
        stage(2,'public_features_complete')
        incumbent=m(3,0,self.evaluate,row['data'],row['initial'],self.contract,wave_h=0.)
        m(10,0,save,directory,'INCUMBENT_C0.json',incumbent)
        if not incumbent['hard']:raise ValueError('Incumbent complete C0 barrier')
        def encode_synchronized():
            encoded=self.model.encode(jnp.asarray(features,jnp.float32),attn_options={})
            jax.block_until_ready(encoded)
            return encoded
        encoded=m(4,0,encode_synchronized)
        m(8,0,self.save_npz,directory/'ENCODED.npz',encoded=np.asarray(encoded),features=features)
        stage(3,'encoder_complete')
        routes=copy.deepcopy(row['before']);remaining=row['remaining'][:];steps=[]
        for step in range(25):
            if cancelled():raise TimeoutError('Cancelled before next action')
            actions,groups=m(5,step,self.enumerate,row['data'],routes,remaining,row['partial'],self.contract)
            binding=m(6,step,self.anchor,actions,routes,remaining,len(row['data']['demands']))
            def score_synchronized():
                latent=self.model.decode(encoded,jnp.asarray(binding['timestep']),jnp.asarray(binding['context']),attn_options={},target='embed')
                if self.model.apply_final_norm:latent=self.model.final_norm(latent)
                scores=self.head(latent,binding)
                jax.block_until_ready((scores,latent))
                return np.asarray(scores),np.asarray(latent)
            scores,latent=m(7,step,score_synchronized)
            m(8,step,self.save_npz,directory/('%02d.npz'%step),latent=latent,scores=scores,features=features,**binding)
            if step==0:stage(4,'first_actual_score_complete')
            def bind_consume():
                packet=dict(scores=scores,source='actual_initial_model_untrained',learned_output_qualified=False,
                    binding_digest=self.binding(row['data'],row['key'],step,routes,remaining,actions),source_identity=self.source_identity)
                return self.consume(row['data'],row['key'],step,routes,remaining,actions,groups,packet,self.source_identity)
            transition=m(9,step,bind_consume)
            m(10,step,save,directory,'%02d.json'%step,transition)
            transition_sha=m(11,step,self.sha,directory/('%02d.json'%step))
            steps.append(dict(file='%02d.json'%step,sha256=transition_sha))
            routes,remaining=transition['after'],transition['next_remaining']
            if transition['chosen'] is None:break
        m(12,0,save,directory,'CHAIN_STOP.json',dict(routes=routes,remaining=remaining,steps=steps,
            status='complete' if not remaining else 'no_feasible_action'))
        stage(5,'action_chain_stopped')
        def certify_select():
            candidate=self.evaluate(row['data'],routes,self.contract,wave_h=0.) if not remaining else None
            choose=bool(candidate and candidate['hard'] and candidate['cost']<incumbent['cost'])
            return candidate,choose
        candidate,choose=m(13,0,certify_select)
        m(10,25,save,directory,'C0_SELECTION.json',dict(incumbent=incumbent,candidate=candidate,
            candidate_selected=choose,routes=routes,remaining=remaining))
        stage(6,'complete_C0_and_selection_complete')
        end=m(14,0,self.closure,self.extras,self.trees)
        # Both inventories are evaluated even if the first differs.
        def parameter_gate():
            model_identity=self.inventory(self.model);head_identity=self.inventory(self.head)
            return model_identity==self.parameter_identity and head_identity==self.head_identity
        unchanged=m(15,0,parameter_gate)
        def source_report_gate():
            changes=self.delta(self.start,end)
            qualified=unchanged and not changes['changed'] and not changes['new_files']
            save(directory,'SOURCE_END.json',dict(**end,**changes))
            report=dict(arm=self.arm,key=row['key'],steps=steps,status='complete' if not remaining else 'no_feasible_action',
                routes=routes,remaining=remaining,incumbent=incumbent,candidate=candidate,candidate_selected=choose,
                selected_routes=routes if choose and qualified else row['initial'],parameters_unchanged=unchanged,
                source_qualified=qualified,source_identity=self.source_identity,optimizer_steps=0,learned_output_qualified=False)
            save(directory,'REPORT_BEFORE_PUBLICATION.json',report)
            if not qualified:raise ValueError('Actual loaded source/parameter qualification barrier')
            return report
        return m(16,0,source_report_gate)
'''


PARSER = '''"""Independent byte parser: imports neither writer nor model/engine modules.

Only use archived raw bytes after actual matching writer exit/reap. Engineering
trace validity does not certify completed candidate, scientific source or budget.
"""
import hashlib
import math
import struct
from fractions import Fraction


def parse(raw, sequence, start_clock, end_clock, writer_reaped):
    if type(raw) is not bytes or type(sequence) is not int or not 0<sequence<2**64:
        raise ValueError('Exact byte/sequence input')
    if writer_reaped is not True:raise ValueError('Actual writer must be reaped')
    if not all(math.isfinite(x) for x in (start_clock,end_clock)) or not 0<=start_clock<=end_clock:
        raise ValueError('Outer clock bounds')
    if len(raw)<64:raise ValueError('Short raw trace')
    magic,version,capacity,count,closed=struct.unpack_from('<8sIIII',raw)
    if magic!=b'MCPHT001' or version!=1 or not 1<=capacity<=4096 or not 0<=count<=capacity or closed not in (0,1):
        raise ValueError('Header schema/bounds')
    if len(raw)!=64+64*capacity or any(raw[24:64]):raise ValueError('Exact buffer/header padding')
    records=[];intervals=[];last_began=start_clock
    for index in range(count):
        offset=64+64*index
        seq,phase,step,began,start,end,status=struct.unpack_from('<QII3di',raw,offset)
        if any(raw[offset+44:offset+64]):raise ValueError('Record padding')
        if seq!=sequence or not 1<=phase<=19 or status not in (0,1,2):raise ValueError('Record identity/registered phase/status')
        if not all(math.isfinite(x) for x in (began,start,end)) or not start_clock<=began<=start<=end<=end_clock or began<last_began:
            raise ValueError('Actual outer bounds/monotonic record allocation')
        last_began=began
        if status==0 and (start!=began or end!=began):raise ValueError('Open record must retain initial clocks')
        duration=Fraction.from_float(end)-Fraction.from_float(start) if status else None
        records.append(dict(index=index,sequence=seq,phase=phase,step=step,status=status,
            began_hex=began.hex(),operation_start_hex=start.hex(),operation_end_hex=end.hex(),
            exact_operation_seconds=None if duration is None else [duration.numerator,duration.denominator]))
        if status:intervals.append((Fraction.from_float(start),Fraction.from_float(end)))
    if any(raw[64+64*count:]):raise ValueError('Torn header count or dirty unused tail')
    # Union, not sum: nested operation spans overlap. Untraced gaps and final
    # record writes remain in the separately measured outer request duration.
    union=Fraction(0);right=None
    for left,end in sorted(intervals):
        if right is None or left>right:union+=end-left
        elif end>right:union+=end-right
        right=end if right is None else max(right,end)
    outer=Fraction.from_float(end_clock)-Fraction.from_float(start_clock)
    return dict(raw_sha256=hashlib.sha256(raw).hexdigest(),records=records,closed=bool(closed),
        exact_operation_union_seconds=[union.numerator,union.denominator],
        exact_outer_seconds=[outer.numerator,outer.denominator],
        exact_untraced_outer_seconds=[(outer-union).numerator,(outer-union).denominator],
        operation_union_is_complete_request=False,trace_validity_is_scientific_qualification=False,
        open_records_completed=False)
'''


def main():
    OUT.mkdir(exist_ok=False)
    basis = [OLD / n for n in ('delivery_timed_actual_engine_v3.py', 'delivery_timed_shared_worker_v3.py',
                               'delivery_timed_shared_parent_v3.py')]
    basis += [PREP / 'bounded_phase_trace_v1.py', PREP / 'PHASE_DESIGN.json']
    identities = {str(p.relative_to(ROOT)): sha(p) for p in basis}
    for n in ('bounded_phase_trace_v1.py', 'PHASE_DESIGN.json'):
        shutil.copy2(PREP / n, OUT / n)
    for n in ('action_head_runtime_v1.py', 'timed_evidence_io_v1.py', 'features_objective.py',
              'delivery_actual_action_head_v1.py', 'delivery_actual_score_chain_v1.py',
              'delivery_feasible_action_options_v1.py', 'delivery_retained_edge_action_guard_v1.py', 'CHAIN_INPUT.json'):
        shutil.copy2(OLD / n, OUT / n)
        identities[str((OLD / n).relative_to(ROOT))] = sha(OLD / n)
    engine = (OLD / 'delivery_timed_actual_engine_v3.py').read_text(encoding='utf-8')
    engine = engine[:engine.index('    def request(')] + REQUEST
    create('delivery_phase_actual_engine_v1.py', engine)
    worker = (OLD / 'delivery_timed_shared_worker_v3.py').read_text(encoding='utf-8')
    worker = replace(worker, 'from delivery_timed_actual_engine_v3 import InitialEngine',
                     'from delivery_phase_actual_engine_v1 import InitialEngine\n    from bounded_phase_trace_v1 import PhaseTrace,MAX_RECORDS\n    telemetry=mmap.mmap(-1,64+64*MAX_RECORDS,tagname=spec[\'trace\'])')
    worker = replace(worker, "if not 0<size<=SIZE:raise ValueError('Bounded public input')",
                     "if not 0<size<=SIZE:raise ValueError('Bounded public input')\n    trace=PhaseTrace(telemetry,seq,MAX_RECORDS)")
    worker = replace(worker, 'snapshot=engine.closure(engine.extras,engine.trees)',
                     'snapshot=trace.measure(14,code,engine.closure,engine.extras,engine.trees)')
    worker = replace(worker, "save(root,'FAULT_SOURCE_AT_ENTRY.json',dict(**snapshot,**engine.delta(engine.start,snapshot)))",
                     "trace.measure(16,code,save,root,'FAULT_SOURCE_AT_ENTRY.json',dict(**snapshot,**engine.delta(engine.start,snapshot)))")
    worker = replace(worker, "report=engine.request(bytes(inp[:size]),root/'request',stage,cancelled)",
                     "report=engine.request(bytes(inp[:size]),root/'request',stage,cancelled,trace)")
    # Preserve marker7/8 and source gate order; wrap serialization/publication
    # together. This outer span can contain a nested fault closure span.
    begin = worker.index("        raw=json.dumps(report")
    end = worker.index("        save(root,'NATURAL_FINISH.json'", begin)
    block = worker[begin:end]
    worker = worker[:begin] + '        def serialize_publish():\n' + ''.join('    '+line+'\n' for line in block.splitlines()) + '        trace.measure(17,0,serialize_publish)\n' + worker[end:]
    worker = replace(worker, 'finally:ctrl.close();inp.close();out.close()',
                     'finally:\n        trace.close()\n        trace.storage.release()\n        telemetry.close();ctrl.close();inp.close();out.close()')
    create('delivery_phase_shared_worker_v1.py', worker)
    parent = (OLD / 'delivery_timed_shared_parent_v3.py').read_text(encoding='utf-8')
    parent = replace(parent, 'from timed_evidence_io_v1 import save,save_bytes',
                     'from timed_evidence_io_v1 import save,save_bytes\n    from bounded_phase_trace_v1 import PhaseTrace,MAX_RECORDS')
    parent = replace(parent, "output_directory=tag+'_worker')", "output_directory=tag+'_worker',trace=nonce+'_t')")
    parent = replace(parent, "log=open(base/(tag+'_stderr.txt'),'xb');stdout=open(base/(tag+'_stdout.txt'),'xb')",
                     "telemetry=mmap.mmap(-1,64+64*MAX_RECORDS,tagname=spec['trace'])\n    caller_buffer=bytearray(64+64*MAX_RECORDS)\n    log=open(base/(tag+'_stderr.txt'),'xb');stdout=open(base/(tag+'_stdout.txt'),'xb')")
    parent = replace(parent, 'delivery_timed_shared_worker_v3.py', 'delivery_phase_shared_worker_v1.py')
    parent = replace(parent, 'started=time.perf_counter();deadline=started+8.;total_deadline=started+10.',
                     'started=time.perf_counter();deadline=started+8.;total_deadline=started+10.\n        caller_trace=PhaseTrace(caller_buffer,1,MAX_RECORDS)')
    # One validation attempt consumes one phase18 record. Keep original guard,
    # selection clocks and source/C0 decision intact; no live installation.
    begin = parent.index("                seq,size,h,writer_clock=")
    end = parent.index('            if child.poll() is not None:break', begin)
    block = parent[begin:end]
    # break statements in original validation refer to polling loop. Helper
    # returns after rejection/selection; caller then exits same polling loop.
    block = block.replace(';break', ';return').replace('                break\n', '                return\n')
    function = '                def validate_select():\n                    nonlocal selected,reason,writer_clock,validation_clock,selection_clock\n'
    function += ''.join('    '+line+'\n' for line in block.splitlines())
    function += '                caller_trace.measure(18,0,validate_select)\n                break\n'
    parent = parent[:begin] + function + parent[end:]
    begin = parent.index('        if child.poll() is None:\n            # Selected output')
    end = parent.index('        finished=time.perf_counter()', begin)
    cleanup = parent[begin:end]
    cleanup = cleanup.replace('        stdout.close();log.close();ctrl.close();inp.close();out.close()\n', '')
    # Original selected archive hashing and actual_stage variables are returned
    # from cleanup; preserve actual kill/reap clocks as outer variables.
    function = '        def reap_collect():\n            nonlocal kill_clock,reap_clock\n'
    function += ''.join('    '+line+'\n' for line in cleanup.splitlines())
    function += "            save_bytes(base/(tag+'_WORKER_TRACE.bin'),bytes(telemetry))\n            return archived_hash,actual_stage\n"
    function += '        archived_hash,actual_stage=caller_trace.measure(19,0,reap_collect)\n'
    function += "        caller_trace.close()\n        save_bytes(base/(tag+'_CALLER_TRACE.bin'),bytes(caller_buffer))\n        caller_trace.storage.release()\n        telemetry.close();stdout.close();log.close();ctrl.close();inp.close();out.close()\n"
    parent = parent[:begin] + function + parent[end:]
    parent = replace(parent, 'caller_cleanup_in_elapsed=True,receipt_file_creation_in_elapsed=False,',
                     'caller_cleanup_in_elapsed=True,telemetry_collection_archive_in_elapsed=True,receipt_file_creation_in_elapsed=False,')
    parent = replace(parent, 'for handle in (stdout,log,ctrl,inp,out):',
                     'for handle in (stdout,log,ctrl,inp,out,telemetry):')
    create('delivery_phase_shared_parent_v1.py', parent)
    create('independent_phase_trace_parser_v1.py', PARSER)
    assert all(sha(ROOT / p) == h for p,h in identities.items())
    py = sorted(OUT.glob('*.py'))
    for p in py:ast.parse(p.read_text(encoding='utf-8'))
    create('PREPARATION_REGISTRATION.json', dict(
        registered_utc=datetime.now(timezone.utc).isoformat(),kind='AST_byte_only_phase_wiring_preparation',
        source_tool_sha256=sha(__file__),basis_files=identities,
        package_files={p.name:sha(p) for p in sorted(OUT.iterdir()) if p.suffix in ('.py','.json')},
        fixed_input_key=[12,3],masked_count=25,head_seed=20261012,optimizer_steps=0,
        total_seconds=10,worker_seconds=8,reap_reserve_seconds=2,
        workload_or_trace_imported=False,model_executed=False,source_gate_moved_outside_request=False,
        sample_warmup=False,old_QA_repeated=False,execution_authorized=False,
        full_request_budget_qualified=False,scientific_family3_started=False,
        remaining=['Independent source review of exact wiring and abnormal cleanup',
                   'Independent trace parser actual engineering validation before model execution',
                   'Complete new request replay checker with trace/clock/actual parameter/source gates',
                   'Exact limited execution registration, supervisor, dispatch/download and resource/owner checks'],
        scope='Only source copies; no trace call, model import, C0, server or old QA execution'))
    print('Exclusive phase engine/worker/caller and independent parser AST sealed; execution remains unauthorized.')


if __name__ == '__main__':
    main()
