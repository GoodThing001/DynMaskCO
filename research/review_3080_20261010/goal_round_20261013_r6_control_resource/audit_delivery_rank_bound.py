"""Independent saved-score comparison using exact Decimal binary-float conversion."""
import hashlib
import json
from decimal import Decimal
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
PRIOR=HERE.parent/'delivery/compiled_head_identity_root_v1/downloaded'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dec(x):return Decimal.from_float(float(x))
def main():
    source=json.loads((PRIOR/'RESULT.json').read_text(encoding='utf-8'))
    report=json.loads((HERE/'DELIVERY_RANK_BOUND_RESULT.json').read_text(encoding='utf-8'))
    checks=[]
    def ck(name,value):checks.append(dict(name=name,pass_=bool(value)))
    ck('record_count',len(source['records'])==len(report['rows'])==34)
    total=0;robust=0;ties=0
    for orig,row in zip(source['records'],report['rows']):
        p=PRIOR/orig['output_file'];ck('source_sha:'+str(orig['index']),sha(p)==orig['output_sha256'])
        with np.load(p,allow_pickle=False) as z:a=z['scores'].tolist()
        ranks=sorted(enumerate(a),key=lambda t:(-t[1],t[0]))
        error=dec(orig['max_abs_score_difference'])
        condition=dec(ranks[0][1])-error>dec(ranks[1][1])+error
        risk=sum(dec(a)-dec(b)<=error+error for (_,a),(_,b) in zip(ranks,ranks[1:]))
        equal=sum(a==b for (_,a),(_,b) in zip(ranks,ranks[1:]))
        ck('top_identity:'+str(orig['index']),row['top_index']==ranks[0][0] and row['runner_up_index']==ranks[1][0])
        ck('robust_bound:'+str(orig['index']),row['top_index_bound_robust']==condition)
        ck('adjacent_risk:'+str(orig['index']),row['adjacent_rank_pairs_with_gap_le_2error']==risk)
        ck('ties:'+str(orig['index']),row['exact_adjacent_ties']==equal)
        total+=risk;robust+=condition;ties+=equal
    ck('totals',report['top_index_robust_count']==robust and report['vulnerable_adjacent_pairs_total']==total and report['exact_ties_total']==ties)
    ck('qualification_unchanged',not report['strict_output_identity_qualified'] and not report['formal_controls_qualified'] and not report['full_request_budget_qualified'] and not report['scientific_family3_started'] and not report['stable_online_gain'])
    out=dict(checks=checks,checks_count=len(checks),failed=sum(not x['pass_'] for x in checks),
        strict_output_identity_qualified=False,stable_online_gain=False)
    with (HERE/'INDEPENDENT_DELIVERY_RANK_AUDIT.json').open('x',encoding='utf-8') as f:json.dump(out,f,indent=2)
    if out['failed']:raise SystemExit(1)
if __name__=='__main__':main()
