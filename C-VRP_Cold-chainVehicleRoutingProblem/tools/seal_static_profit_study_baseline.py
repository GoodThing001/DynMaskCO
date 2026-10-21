"""Seal the recovered dev09/dev10 development comparison without replacing v0."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import zipfile

EXT=Path(__file__).resolve().parents[1]


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--verify',action='store_true')
    args=p.parse_args()
    out=EXT/'results/static_delivery/algorithm_baselines/v1_dev09_dev10_validation'
    if args.verify:
        lock=read(out/'BASELINE_LOCK.json')
        assert sha(out/'BASELINE_LOCK.json')==(out/'BASELINE_LOCK.sha256').read_text().split()[0]
        for name,expected in lock['files'].items():assert sha(out/name)==expected
        print(json.dumps(dict(status='PASS_BASELINE_V1_ARCHIVE_SHA',files=len(lock['files']))))
        return
    if out.exists():raise ValueError('Never replace a baseline; use --verify')
    runs=EXT/'results/static_delivery/algorithm_dev_20261021/remote'
    report=read(runs/'dev10/local_recovered_training_study_verification.json')
    assert report['status']=='PASS_RECOVERED_TRAINING_STUDY' and report['missing']==[]
    assert report['validation_routesets_reaudited']==32 and report['teacher_routesets_reaudited']==456
    assert report['original_exit_code']==124 and report['outer_test_cases_used']==0
    hardware={}
    for label,expected in [('dev09',124),('dev10',0)]:
        root=runs/label;receipt=read(root/'download_verification.json');manifest=read(root/'output_manifest.json')
        assert receipt['exit_code']==expected
        for name in receipt['downloaded_verified']:assert sha(root/name)==manifest[name]
        start=read(root/'started.json')
        hardware[label]={k:start[k] for k in ('gpu_index','gpu_identity','cores','timeout_per_command_s','total_timeout_s')}
    comparison={**report,'baseline_id':'v1_dev09_dev10_validation','hardware':hardware,
        'timing':{arm:{m:dict(mean_elapsed_s=statistics.mean(r['elapsed_s'] for r in report['rows'] if r['arm']==arm and r['method']==m),
                            cases=report['mean_coverage'][arm][m]) for m in report['mean_net_benefit_U'][arm]}
                  for arm in report['mean_net_benefit_U']},
        'limits':['Nominal 10s with closing-audit overshoot; neural training-only compilation/warmup excluded.',
                  'Single seed and validation geometry; shared-server load and GPU0/GPU1 recovery differences.',
                  'ALNS/VNS do not use teacher labels; cross-arm variations reflect timed search execution.',
                  'NPZ weights/optimizer remain on server, recorded by SHA and original paths.',
                  'Development reference only; formal method/checkpoint configuration remains open.']}
    out.mkdir(parents=True)
    (out/'comparison.json').write_text(json.dumps(comparison,indent=2,allow_nan=False),encoding='utf-8')
    with zipfile.ZipFile(out/'evidence.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for label in ('dev09','dev10'):
            for path in sorted((runs/label).rglob('*')):
                if path.is_file() and path.suffix in ('.json','.py','.sh','.log'):
                    z.write(path,label+'/'+path.relative_to(runs/label).as_posix())
        for name in ('verify_static_profit_training_study.py','seal_static_profit_study_baseline.py'):
            z.write(EXT/'tools'/name,'audit_source/'+name)
    lock=dict(baseline_id=comparison['baseline_id'],files={p.name:sha(p) for p in out.iterdir() if p.is_file()})
    (out/'BASELINE_LOCK.json').write_text(json.dumps(lock,indent=2),encoding='utf-8')
    (out/'BASELINE_LOCK.sha256').write_text(sha(out/'BASELINE_LOCK.json')+'  BASELINE_LOCK.json\n',encoding='ascii')
    print(json.dumps(dict(status='SEALED_DEVELOPMENT_BASELINE_V1',output=str(out),lock_sha256=sha(out/'BASELINE_LOCK.json'))))


if __name__=='__main__':main()
