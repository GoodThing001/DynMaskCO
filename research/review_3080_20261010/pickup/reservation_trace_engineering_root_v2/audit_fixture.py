"""Independent public-field arithmetic audit; imports neither policy nor tracer."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'a1_budget_reserving_admission_root_v4'
REV = {0: 10.0, 1: 20.0, 2: 30.0}


def check(name, passed, records):
    records.append({'name':name,'pass':bool(passed)})


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    run = json.loads((HERE/'FIXTURE_RESULT.json').read_text(encoding='utf-8'))
    source = json.loads((OLD/'RESULT.json').read_text(encoding='utf-8'))
    original_reg = json.loads((OLD/'REGISTRATION.json').read_text(encoding='utf-8'))
    objective = original_reg['objective']
    checks = []
    check('fixed_54_cases',len(run['rows'])==54,checks)
    check('original_policy_sha',run['source_sha256']['original_policy']==sha(OLD/'reserving_admission.py'),checks)
    check('original_registration_sha',run['source_sha256']['original_registration']==sha(OLD/'REGISTRATION.json'),checks)
    check('original_result_sha',run['source_sha256']['original_result']==sha(OLD/'RESULT.json'),checks)
    for index,row in enumerate(run['rows']):
        cls = row['class_id']; q = row['demand']; clock=row['clock']
        e = q*objective['precool_energy_per_unit_per_c']*max(0.0,objective['ambient_c']-objective['target_c'][cls])
        spent = 2.0*objective['precool_energy_per_unit_per_c']*(objective['ambient_c']-objective['target_c'][2]) if row['accepted_before'] else 0.0
        remaining = row['budget']-spent
        bucket = 0
        for i,t in enumerate(source['clock_grid']):
            if t <= clock:bucket=i
        if e > remaining:
            expected = {'allowed':False,'reason':'over_remaining_budget','lambda':None,'clock_bucket':None}
        else:
            lam=source['lambda_grid'][-1]
            for li,x in enumerate(source['lambda_grid']):
                if source['g_table'][li][bucket] <= remaining:
                    lam=x;break
            allow = REV[cls]-lam*e > 0.0
            expected = {'allowed':allow,'reason':'admit' if allow else 'below_shadow_price',
                        'lambda':lam,'clock_bucket':bucket}
        fields = all(row[k]==v for k,v in expected.items())
        fields &= row['sequence']==index and row['order']==1 and row['instance_idx']==0
        fields &= row['estimated_precool_kwh']==e and row['revenue']==REV[cls]
        fields &= row['reserve_decisions_before']==0 and row['reserve_decisions_after']==1
        fields &= row['reserve_rejects_before']==0 and row['reserve_rejects_after']==int(not expected['allowed'])
        copied={k:v for k,v in row.items() if k!='sha256'}
        digest=hashlib.sha256(json.dumps(copied,sort_keys=True,separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()
        check('case_%02d_all_fields'%index,fields and digest==row['sha256'],checks)
    out={'stage':'independent_reservation_trace_public_fixture','checks_count':len(checks),
         'failed':sum(not x['pass'] for x in checks),'failure_names':[x['name'] for x in checks if not x['pass']],
         'checks':checks,'input_sha256':{'fixture':sha(HERE/'FIXTURE_RESULT.json'),
                                        'original_result':sha(OLD/'RESULT.json'),
                                        'original_registration':sha(OLD/'REGISTRATION.json')},
         'original_five_day_audit_failures_preserved':2,'full_live_request_qualified':False,
         'scientific_data_generated':False,'stable_online_gain':False}
    with (HERE/'INDEPENDENT_AUDIT.json').open('x',encoding='utf-8',newline='\n') as f:
        json.dump(out,f,indent=2,ensure_ascii=False);f.write('\n')
    print(json.dumps({'checks_count':out['checks_count'],'failed':out['failed'],'failure_names':out['failure_names']}))
    return int(out['failed']!=0)


if __name__=='__main__':raise SystemExit(main())
