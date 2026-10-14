"""Resolve two checker assumptions with exact registered reduction order.
No tolerance, no new domain replay, no changes to producer or original FAIL.
"""
import json,sys
from pathlib import Path
from fractions import Fraction
import qualify_pickup_joint_policy_transaction_root_v3 as original
ROOT=original.ROOT;OUT=original.P/'private_joint_integration_root_v1'
def read(n):return json.loads((OUT/n).read_text(encoding='utf-8'))
def mean10_pairwise(v):
    # numpy v1.26.4 loops_utils.h.src n=10 pairwise order; Python binary64.
    if len(v)!=10 or any(type(x) is not float for x in v):raise ValueError('10 binary64 values only')
    result=((v[0]+v[1])+(v[2]+v[3]))+((v[4]+v[5])+(v[6]+v[7]))
    result+=v[8];result+=v[9];return result/10.
def main():
    _,decode=original.codecs();raw=decode(read('RESULT.json'));audit=read('INDEPENDENT_PRIVATE_JOINT_REPLAY.json');reg=read('REGISTRATION.json')
    expected=['13:own_incumbent:mean','13:candidate:mean'];checks=[];arithmetic=[]
    def ck(n,v):checks.append({'name':n,'pass':bool(v)})
    ck('original_failures_exact',[c['name'] for c in audit['checks'] if not c['pass']]==expected)
    ck('original_159_check_count',audit['checks_count']==159 and audit['failed']==2)
    ck('original_raw_registration_identity',audit['input_sha256']==original.sha(OUT/'RESULT.json') and audit['registered_sha256']==original.sha(OUT/'REGISTRATION.json'))
    for section in ('inputs','tools'):
        for p,h in reg[section].items():ck('actual_unchanged:'+p,original.sha(p)==h)
    for r in raw['records']:
        for name,s in r['response']['common_scores']['candidate_scores'].items():
            computed=mean10_pairwise(s['branch_values']);saved=s['mean_shadow_utility'];exact=sum((Fraction.from_float(v) for v in s['branch_values']),Fraction())/10
            ck(str((r['index'],name))+':independent_pairwise_bit_exact',computed.hex()==saved.hex())
            arithmetic.append({'index':r['index'],'candidate':name,'registered_numpy_mean':saved,'pure_python_same_pairwise_mean':computed,'sequential_python_mean':sum(s['branch_values'])/10.,'correctly_rounded_exact_rational_mean':float(exact),'registered_hex':saved.hex(),'pairwise_hex':computed.hex(),'rational_error':str(Fraction.from_float(saved)-exact)})
    ck('producer_unchanged_no_reexecution',raw['checks_count']==41 and raw['failed']==0)
    failed=sum(not c['pass'] for c in checks)
    result={'checks':checks,'checks_count':len(checks),'failed':failed,'original_independent_checks':159,'original_failures_preserved':2,'original_other_checks_passed':157,'new_exact_mean_checks':6,'arithmetic':arithmetic,'qualified_component':failed==0,'producer_sha256':original.sha(OUT/'RESULT.json'),'original_audit_sha256':original.sha(OUT/'INDEPENDENT_PRIVATE_JOINT_REPLAY.json'),'current_recovery_source_sha256':original.sha(__file__),'reduction_source':'https://github.com/numpy/numpy/blob/v1.26.4/numpy/core/src/umath/loops_utils.h.src','whole_live_request_qualified':False,'parent_decode_preemptible':False,'stable_online_gain':False,'tolerance_relaxed':False,'domain_algorithms_or_results_changed':False}
    with (OUT/'INDEPENDENT_MEAN_ORDER_RECOVERY_R2.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps({'checks':len(checks),'failed':failed,'original_FAIL_retained':2,'component_qualified':failed==0}));return int(failed>0)
if __name__=='__main__':raise SystemExit(main())
