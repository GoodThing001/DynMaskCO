"""Independent read-only per-step timing/shape mechanism analysis.

Reads old raw trace bytes and saved NPZ shapes; no model, C0 or old auditor.
"""
import hashlib
import json
import statistics
import struct
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'research/review_3080_20261010/delivery/score_split_timed_execution_root_v1/downloaded'
OUT = ROOT/'research/review_3080_20261010/goal_round_20261013_r10_continuous_strong_delivery/DELIVERY_HEAD_SCALING_DIAGNOSIS.json'
HEADER = struct.Struct('<8sIIII40x')
RECORD = struct.Struct('<QII3di20x')

def read_trace(path):
    raw=path.read_bytes()
    magic,version,capacity,count,closed=HEADER.unpack_from(raw)
    if magic!=b'MCPHT001' or version!=1 or count>capacity or len(raw)!=HEADER.size+capacity*RECORD.size:
        raise ValueError('raw trace bounds')
    rows=[]
    for j in range(count):
        seq,phase,step,began,start,end,status=RECORD.unpack_from(raw,HEADER.size+j*RECORD.size)
        if seq<=0 or not 0<=began<=start<=end or status not in (0,1,2):
            raise ValueError('raw record identity or clock')
        if status==1 and phase in (20,21,22,23):
            rows.append((phase,step,end-start))
    return rows,closed,hashlib.sha256(raw).hexdigest()

def main():
    arms=('pretrained_initial','random888_initial')
    out=[]
    for arm in arms:
        tag=arm+'_fault0'
        trace,closed,trace_sha=read_trace(OLD/(tag+'_WORKER_TRACE.bin'))
        by_step={}
        for phase,step,seconds in trace:
            by_step.setdefault(step,{})[phase]=seconds
        rows=[]
        for step in sorted(by_step):
            times=by_step[step]
            if not all(p in times for p in (20,21,22,23)):
                continue
            saved=OLD/(tag+'_worker/request/%.2d.npz'%step)
            with np.load(saved,allow_pickle=False) as z:
                shape=z['actions'].shape
                if z['scores'].shape!=(shape[0],) or shape[1]!=5:
                    raise ValueError('saved action/score shape')
            rows.append(dict(step=step,actions=shape[0],decode_s=times[20],norm_s=times[21],head_s=times[22],host_s=times[23],npz_sha256=hashlib.sha256(saved.read_bytes()).hexdigest()))
        if len(rows)<2:
            raise ValueError('insufficient completed steps')
        # These are descriptive within the previously observed 8-second window.
        head=[r['head_s'] for r in rows]
        actions=[r['actions'] for r in rows]
        out.append(dict(arm=arm,raw_trace_sha256=trace_sha,trace_closed=bool(closed),completed_rows=len(rows),first_actions=actions[0],last_actions=actions[-1],first_head_s=head[0],last_head_s=head[-1],median_head_s=statistics.median(head),min_head_s=min(head),max_head_s=max(head),rows=rows))
    result=dict(scope='previously sealed per-step raw trace and NPZ shapes only',arms=out,checks={'two_arms':len(out)==2,'both_17_steps':all(x['completed_rows']==17 for x in out),'action_space_shrinks':all(x['last_actions']<x['first_actions'] for x in out),'all_positive_head_spans':all(all(r['head_s']>0 for r in x['rows']) for x in out)},model_executed=False,old_QA_rerun=False,full_request_budget_qualified=False,stable_online_gain=False)
    result['failed']=[k for k,v in result['checks'].items() if not v]
    with OUT.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(dict(steps=[x['completed_rows'] for x in out],action_spans=[(x['first_actions'],x['last_actions']) for x in out],head_medians=[x['median_head_s'] for x in out],failed=result['failed'])))
    return int(bool(result['failed']))
if __name__=='__main__':raise SystemExit(main())
