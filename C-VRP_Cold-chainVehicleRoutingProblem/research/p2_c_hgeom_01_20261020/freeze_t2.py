"""Freeze diagnostic bytes after a BLOCKED audit; never claim qualified T2."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import platform
import sys

HERE=Path(__file__).resolve().parent
WORKSPACE=HERE.parents[2]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def freeze():
    import numpy
    import psutil
    audit=json.loads((HERE/'independent_audit.json').read_text(encoding='utf-8'))
    assert audit['status']=='BLOCKED'
    reg=json.loads((HERE/'registration.json').read_text(encoding='utf-8'))
    delivery=reg['seven_elements']['identity_and_delivery']
    frozen={**delivery['contracts'],**delivery['inputs'],**delivery['existing_source_hashes']}
    for path,digest in frozen.items():
        if sha(WORKSPACE/path)!=digest:raise ValueError('IDENTITY_MISMATCH: '+path)
    files=[p for p in HERE.iterdir() if p.is_file() and p.name not in ('run_identity.json',)]
    receipts=[p for p in (HERE/'synthetic_artifacts').rglob('*') if p.suffix in ('.json','.jsonl')]
    data=dict(schema='hgeom_phase1_blocked_diagnostic_freeze_v1',
              freeze_kind='DIAGNOSTIC_ONLY_NOT_QUALIFIED_T2',T2_qualified=False,
              qualification='BLOCKED',formal_evaluation_authorized=False,
              formal_evaluations_consumed=0,
              t1_commit='3cc964a989dc5b7fd9599868086179f10ac036af',registration_sha256=sha(HERE/'registration.json'),
              frozen_original_files=frozen,
              actual_package_files={p.relative_to(WORKSPACE).as_posix():sha(p) for p in sorted(files)},
              synthetic_receipts={p.relative_to(WORKSPACE).as_posix():sha(p) for p in sorted(receipts)},
              environment=dict(python=sys.version,python_executable=sys.executable,
                               python_executable_sha256=sha(Path(sys.executable)),platform=platform.platform(),
                               numpy=numpy.__version__,numpy_init_sha256=sha(Path(numpy.__file__)),
                               psutil=psutil.__version__,psutil_init_sha256=sha(Path(psutil.__file__)),
                               byteorder=sys.byteorder,float_info=dict(mant_dig=sys.float_info.mant_dig,
                                                                      rounds=sys.float_info.rounds)),
              blockers=audit['failed'],
              note='No complete runtime dependency qualification or valid T2 is asserted. These are retained diagnostic byte identities only.')
    out=HERE/'run_identity.json'
    if out.exists():raise ValueError('Refuse overwrite identity')
    out.write_bytes((json.dumps(data,ensure_ascii=False,sort_keys=True,indent=2)+'\n').encode('utf-8'))
    print('DIAGNOSTIC_FREEZE_SHA256',sha(out))


if __name__=='__main__':freeze()
