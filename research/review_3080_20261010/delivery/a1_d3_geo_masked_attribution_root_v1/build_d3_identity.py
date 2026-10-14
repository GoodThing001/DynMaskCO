"""Build the D3-Engineering source identity manifest (per-file SHA + environment)."""
import hashlib
import json
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
EXT = ROOT / 'C-VRP_Cold-chainVehicleRoutingProblem'
OUT = Path(__file__).resolve().parent

FILES = ['scripts/simulation/d3_geo_adapter.py',
         'scripts/tests/test_d3_geo_adapter.py',
         'scripts/simulation/action_contract.py',
         'scripts/simulation/mpre_policy.py',
         'scripts/simulation/dynmaskco_cc_context.py',
         'scripts/simulation/cc_lns_replanner.py']


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for c in iter(lambda: f.read(1 << 20), b''):
            h.update(c)
    return h.hexdigest()


def main():
    import jax
    import numpy
    out = {
        'stage': 'D3_Engineering_source_identity',
        'frozen': False,
        'note': ('per-file SHA at D3-Engineering acceptance. The formal T-G9 freeze acceptance '
                 'must be redone once the source freeze tree exists (user ruling); this file '
                 'is NOT a freeze seal.'),
        'environment': {'platform': platform.platform(), 'python': platform.python_version(),
                        'jax': jax.__version__, 'numpy': numpy.__version__},
        'files': {},
        'training_era_reference': {
            'dynmaskco_cc_context.py':
                '9de8ceb54d088b8afa087948d6469f8cbd6f503678715bbf39e9971f5a91be54',
            'cc_lns_replanner.py':
                '7ea14c07e13943ea41bf9123f8c4f47d40546970b09d0e70777f8d4f8d9437c3',
            'train_mpre_reinforce.py':
                '33c617a3e24bfc4c5d68c2ca4e72f41561999906dcea4795493cc8a2aad1b5fd',
            'note': ('recorded for explicit train/eval identity separation; the training-era '
                     'files are NOT re-instated as production code (user ruling 2)'),
        },
    }
    for rel in FILES:
        p = EXT / rel
        out['files'][rel] = {'sha256': sha(p) if p.is_file() else None,
                             'bytes': p.stat().st_size if p.is_file() else None}
    # explicit comparison against F's training_identity.json src_sha256 (from the server probe)
    f_training = {
        'scripts/simulation/action_contract.py':
            '9d64273aa3df98cb931c0c8c4b0d0112c134c13e080d3b4f1615b8ae06d2b53b',
        'scripts/simulation/mpre_policy.py':
            '5b03e4cae812fa20785e68ce837be3a615aadb44ab29b7dbdc4b95f104a4a191',
        'scripts/simulation/dynmaskco_cc_context.py':
            '9de8ceb54d088b8afa087948d6469f8cbd6f503678715bbf39e9971f5a91be54',
        'scripts/simulation/cc_lns_replanner.py':
            '7ea14c07e13943ea41bf9123f8c4f47d40546970b09d0e70777f8d4f8d9437c3',
    }
    out['vs_F_training_identity'] = {
        rel: {'at_F_training': v, 'current': out['files'][rel]['sha256'],
              'matches': out['files'][rel]['sha256'] == v}
        for rel, v in f_training.items()}
    out['vs_F_training_identity_note'] = (
        'action_contract.py and mpre_policy.py -- the action-enumeration layer that D-Geo must '
        'share with F -- MATCH the training-era identity exactly, so the shared-enumeration '
        'precondition holds without any drift.  dynmaskco_cc_context.py and cc_lns_replanner.py '
        'differ, which is why the B gate is implemented explicitly in the new tree (see '
        'D3_PREREGISTRATION_REV2.md section 2).')
    (OUT / 'D3_SOURCE_IDENTITY.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    for k, v in out['files'].items():
        print((v['sha256'][:16] if v['sha256'] else 'MISSING'), v['bytes'], k)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
