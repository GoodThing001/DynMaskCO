"""Read-only replay of CTRL/OPP records; never execute a controller or evaluator.

Usage: python <this file> --output <a NEW json path in this directory>.
The CSV is emitted only on the first review, not used to backfill OPP receipts.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
WORKSPACE=ROOT.parent
OUT=Path(__file__).resolve().parent
SAA=ROOT/'research/p2_c_opp_01_20261019'
sys.path.insert(0,str(ROOT/'scripts'))
from project_paths import configure_imports
configure_imports(('experimental','simulation','coldchain','evaluation','training'))
import numpy as np
import p2_pos01_reference as ref
from p2_pos01c_gap import view_from_json
from p2_pos01c_control import occupancy
from scenario_saa import CondHistoricalSampler,ScenarioOrder,build_history


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):
            h.update(block)
    return h.hexdigest()


def load(path):
    return json.loads(path.read_text())


def distribution(values):
    return dict(sorted(Counter(values).items()))


def review():
    registration=load(OUT/'registration.json')
    assert all(sha(WORKSPACE/p)==h for p,h in registration['source_sha256'].items())
    assert all(sha(WORKSPACE/p)==h for p,h in registration['preserved_sha256'].items())
    assert sha(WORKSPACE/'MASKCO_code/ckpts/cvrp100.ckpt')==registration['checkpoint_sha256']
    manifest=load(ROOT/'configs/p2_pos01c_r0_20261019_v1/data_identity.json')
    history=[];sources={};history_ids=[]
    source_index=0
    for d in manifest['days']:
        if d['role']=='HIST_TRAIN':
            with np.load(ROOT/d['relative_path'],allow_pickle=False) as z:history.extend(build_history(z))
            history_ids.append(d['day_id'])
        elif d['role']=='SOURCE_TRAIN':
            with np.load(ROOT/d['relative_path'],allow_pickle=False) as z:sources[source_index]={k:z[k] for k in z.files}
            source_index+=1
    sampler=CondHistoricalSampler(history,10)
    comparison=[];csv_rows=[];artifacts={}
    for arm,folder in (('SAA',SAA),('MYOPIC',OUT)):
        result=load(folder/'result.json');phases=load(folder/'phases.json')
        journal=[json.loads(line) for line in (folder/'journal.jsonl').read_text().splitlines()]
        assert result['status']=='COMPLETE' and all(s['complete_root_day'] for s in result['summary'])
        for name in ('result.json','phases.json','journal.jsonl','registration.json'):
            artifacts[arm+'/'+name]=sha(folder/name)
        for p in phases:
            view=view_from_json({'public_view':p['public_view'],'public_identity':p['public_identity']})
            masks=ref.mask_pool(view,ref.plans_from_view(view),seed=0)
            assert [list(m) for m in masks]==p['masks']['masks']
            ds=sources[p['day']]
            visible={i for i in range(1,51) if ds['reveal_time'][0,i]<=view.clock+1e-6}
            assert {o.oid for o in view.orders if o.oid}==visible
            assert set(view.accepted)|set(view.rejected)==visible
            public=[ScenarioOrder(o.oid,o.reveal,o.x,o.y,o.demand,o.temp_class,o.tw_start,o.tw_end,o.service)
                    for o in view.orders if o.oid]
            x=sampler._feat(public,view.clock);h=np.array([sampler._feat(day,view.clock) for day in history])
            mu,sd=h.mean(0),h.std(0)+1e-6
            distances=(((h-mu)/sd-(x-mu)/sd)**2).sum(1)
            nearest=np.argsort(distances)[:10].tolist()
            counts=[sum(o.reveal>view.clock+1e-6 for o in day) for day in history]
            eligible=[i for i in nearest if counts[i]<=50-len(public)]
            assert p['future_support']['nearest']==nearest and p['future_support']['eligible']==eligible
            assert p['future_support']['suffix_counts']==counts
            assert p['future_support']['compatible_day_ids']==[history_ids[i] for i in eligible]
            assert p['multi']==(len(masks)>=2) and p['supported']==(len(eligible)>=2)
            assert p['joint']==(len(masks)>=2 and len(eligible)>=2)
            assert p['P0_reference_check']['ok'] and p['immutable_observer']
            p['_derived_occupancy']=occupancy(view)
            if arm=='MYOPIC':
                assert p['_derived_occupancy']==p['occupancy']
            o=p['_derived_occupancy']
            csv_rows.append([arm,p['day'],p['ordinal'],p['clock'],len(view.accepted),len(view.rejected),
                len(view.served),o['total_mutable_customers'],len(masks),len(eligible),
                p['multi'],p['supported'],p['joint'],json.dumps(o['legal_mask_suffix_lengths'],sort_keys=True)])
        for day in range(2):
            rows=[p for p in phases if p['day']==day];n=len(rows)
            begins=[j for j in journal if j['kind']=='BEGIN' and j['payload']['day']==day]
            assert len(begins)==n==result['days'][day]['phase_attempts']
            assert [p['ordinal'] for p in rows]==list(range(n))
            rates={name:sum(p[field] for p in rows)/n for name,field in
                   (('p_multi','multi'),('p_support','supported'),('p_joint','joint'))}
            assert rates==result['summary'][day]['rates']
            terminal=result['days'][day]
            assert set(terminal['accepted'])==set(terminal['served'])
            dispatch_ids=set();finishes={};admit={}
            for p in rows:
                for v in p['public_view']['vehicles']:
                    cc=v['coldchain_state']
                    if cc and cc['dispatched']:dispatch_ids.add(v['vehicle_id'])
                    if cc:
                        for lot in cc['cargo_manifest']+cc['delivered_to_depot']:
                            oid=lot['order_id'];finish=lot['pickup_finish_time']
                            assert oid not in finishes or finishes[oid]==finish
                            finishes[oid]=finish
            # This closes the upper bound: no accepted service (and hence no
            # first customer dispatch) can occur unseen after the last phase.
            assert set(rows[-1]['ledger']['served'])==set(terminal['served'])
            assert set(finishes)==set(terminal['served'])
            for j in journal:
                if j['kind']=='ADMISSION' and j['payload']['day']==day:
                    for oid in j['payload']['accepted']:
                        assert oid not in admit
                        admit[oid]=j['payload']['clock']
            assert set(admit)==set(finishes)
            if arm=='MYOPIC':
                assert dispatch_ids==set(terminal['dispatched_vehicle_ids'])
                assert {s['order']:s['finish'] for s in terminal['services']}==finishes
                assert terminal['draw_audit']==[]
            lags=[finishes[i]-admit[i] for i in sorted(finishes)]
            comparison.append(dict(arm=arm,day=day,day_id=terminal['day_id'],
                source_sha256=terminal['source_sha256'],root_seed=terminal['root_seed'],
                phases=n,counts=result['summary'][day]['counts'],rates=rates,
                accepted=len(terminal['accepted']),rejected=len(terminal['rejected']),served=len(terminal['served']),
                actual_dispatched_ids=sorted(dispatch_ids),actual_dispatched_count=len(dispatch_ids),
                dispatch_provenance=('new trace + saved actual physical state' if arm=='MYOPIC' else
                    'new read-only derivation from saved actual dispatched flags + no unseen remaining service bound'),
                legal_mutable_total_histogram=distribution(p['_derived_occupancy']['total_mutable_customers'] for p in rows),
                legal_suffix_vehicle_phase_histogram=distribution(v for p in rows for v in
                    p['_derived_occupancy']['legal_mask_suffix_lengths'].values()),
                provisional_suffix_vehicle_phase_histogram=distribution(v for p in rows for v in
                    p['_derived_occupancy']['provisional_suffix_lengths'].values()),
                acceptance_to_service_finish_h=dict(min=min(lags),median=float(np.median(lags)),
                    max=max(lags),mean=float(np.mean(lags))),
                timeouts=terminal['timeouts'],incomplete=0))
    for day in range(2):
        a=next(x for x in comparison if x['arm']=='SAA' and x['day']==day)
        b=next(x for x in comparison if x['arm']=='MYOPIC' and x['day']==day)
        assert a['day_id']==b['day_id'] and a['root_seed']==b['root_seed'] and a['source_sha256']==b['source_sha256']
    csv_path=OUT/'phase_comparison.csv'
    if not csv_path.exists():
        with csv_path.open('x',encoding='utf-8',newline='') as f:
            w=csv.writer(f,lineterminator='\n')
            w.writerow(['arm','day','ordinal','clock_h','accepted','rejected','served','legal_mutable_total',
                'distinct_masks','compatible_of_10','multi','supported','joint','legal_suffix_lengths_by_vehicle'])
            w.writerows(csv_rows)
    return dict(status='PASS',qualification='REFERENCE_ONLY',scope='READ_ONLY_RECOMPUTATION_NO_CONTROLLER_REEXECUTION',
        verified_phases=len(csv_rows),historical_SAA_reruns=0,
        source_files_unchanged=len(registration['source_sha256']),preserved_files_unchanged=len(registration['preserved_sha256']),
        checkpoint_unchanged=True,comparison=comparison,artifact_sha256=artifacts,
        phase_comparison_sha256=sha(csv_path),review_source_sha256=sha(Path(__file__)),
        limitations=['two previously inspected developer roots only; phase/vehicle exposures are not independent days',
                     'SAA dispatch/finish metrics are new derivations, never backfilled original receipts',
                     'mask counts do not establish reconstructed plans, executable action divergence or utility gains'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.resolve().parent!=OUT.resolve() or args.output.exists():
        raise SystemExit('refuse overwrite or output outside this diagnostic directory')
    result=review()
    with args.output.open('x',encoding='utf-8',newline='\n') as f:
        json.dump(result,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
    print('Read-only verification PASS; phases=%d'%result['verified_phases'])
