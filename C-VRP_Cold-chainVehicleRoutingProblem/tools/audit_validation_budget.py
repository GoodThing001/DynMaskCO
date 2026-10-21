"""Reconstruct validation access from saved identities, including functional checks."""
import argparse
from collections import defaultdict
import datetime
import hashlib
import json
from pathlib import Path

EXT=Path(__file__).resolve().parents[1]
RUNS=EXT/'results/static_delivery/algorithm_dev_20261021'
OUT=EXT/'results/static_delivery/algorithm_diagnostics_20261021'


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def key(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=OUT/'historical_validation_budget_v2.json')
    args=parser.parse_args()
    groups={};evidence=[];excluded=[];issues=[]
    for p in sorted(RUNS.rglob('run_identity.json')):
        d=read(p);a=d.get('arguments',{})
        if d.get('role')!='validation':
            excluded.append(dict(path=p.relative_to(EXT).as_posix(),reason='train role; not a validation access'))
            continue
        folder=p.parent
        summary=read(folder/'summary.json') if (folder/'summary.json').exists() else None
        for path in sorted(folder.glob('*_search.json')):
            method=path.name.removesuffix('_search.json')
            row=dict(method=method,fold=d['fold'],seed=d['seed'],
                source=d['native_sources']['profit_search.py'],
                timing='legacy_unaudited_deadline',seconds=float(a['seconds']),iterations=int(a['iterations']),threads=int(a['threads']))
            if method not in ('alns','vns'):
                parent=read(RUNS/'remote/dev09/outputs/upgraded_teachers/validation/c205/summary.json') if summary is None else summary
                cp=parent['variants'][method]
                row.update(checkpoint=cp['verified_weights']['weights.npz'],steps=cp['checkpoint_steps'],
                           predictor=a.get('predictor') or 'eager',precision=a.get('matmul_precision') or 'default',
                           repair=a.get('repair_implementation') or 'classic',remove=10,topk=8)
            k=key(row)
            if k not in groups:groups[k]=dict(configuration_id=k,settings=row,accesses=[])
            cell=dict(identity=p.relative_to(EXT).as_posix(),identity_sha256=sha(p),
                      result=path.relative_to(EXT).as_posix(),result_sha256=sha(path))
            groups[k]['accesses'].append(cell);evidence.append(cell)
            if folder.name=='c206' and a.get('case')=='c205':
                issues.append(dict(path=p.relative_to(EXT).as_posix(),issue='Recovery c206 ALNS identity copied c205 arguments; actual recovery source explicitly runs c206 and independent re-score used c206. Original preserved.'))
    methods=['maskco_base','directed_only','quality_only','quality_directed','alns','vns']
    counts={str(f):{m:sum(g['settings']['fold']==f and g['settings']['method']==m for g in groups.values()) for m in methods} for f in range(4)}
    assert all(n<=12 for c in counts.values() for n in c.values())
    report=dict(status='HISTORICAL_VALIDATION_ACCESS_RECONSTRUCTED',
        recorded_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),cap=12,
        counting_rule='Conservative checkpoint/source/inference/search-budget/thread identity. Cases and repeated executions of exactly one setting are evidence cells, not extra configurations. Functional validation is charged. Train-only profiling/failed runs without validation output are disclosed but not charged. Diagnostic controls will be charged to their parent method, never fresh method namespaces.',
        counts=counts,remaining={f:{m:12-n for m,n in c.items()} for f,c in counts.items()},
        configurations=list(groups.values()),excluded_train_identities=excluded,known_metadata_issues=issues,
        accounting_correction='v1 raw JSON hashed 10 and 10.0 separately, conservatively double-counting recovery of the identical configuration. v2 normalizes numeric types; all 62 evidence cells retained. v1 is preserved, not reset or erased.',
        boundary='Historical frozen-model validation access in algorithm_dev_20261021; old different-objective RRP/scenario studies retain their identities and are not this tuning budget.')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    if args.out.exists():
        old=read(args.out)
        assert old['configurations']==report['configurations'],'Do not reset historical access; investigate added evidence before a new ledger version'
        print(json.dumps(dict(status='EXISTING_LEDGER_MATCHES_EVIDENCE',counts=counts)));return
    args.out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(counts=counts,remaining=report['remaining'],configurations=len(groups),accesses=len(evidence))))


if __name__=='__main__':main()
