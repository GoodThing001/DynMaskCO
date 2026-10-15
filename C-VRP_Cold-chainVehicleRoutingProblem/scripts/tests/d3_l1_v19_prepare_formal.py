"""Prepare pinned inputs after all reviews pass. Does NOT start a collection."""
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

EXT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(EXT / 'scripts/simulation'))
import d3_l1_collection_runner as R
import d3_l1_p0c_v18 as V

ROOT = EXT.parent / 'research/review_3080_20261010/d3_g0_evidence_root_v1/a1_d3_l1_prereg_root_v1'
E = ROOT / 'v19_evidence'


def main():
    expected = (
        ('V18_INDEPENDENT_REVIEW_BEFORE_CHANGES.json', 'n_fail', 0),
        ('V18_ORIGINAL_BYTES_REVIEW.json', 'ok', True),
        ('engineering_release/ACCEPTANCE.json', 'n_fail', 0),
        ('kill_release/KILL_E2E.json', 'n_fail', 0),
        ('authorization_release/AUTHORIZATION_GATE.json', 'n_fail', 0),
        ('REGRESSION.json', 'n_fail', 0))
    reviews = []
    for name, key, value in expected:
        report = json.loads((E / name).read_text())
        assert type(report[key]) is type(value) and report[key] == value, name
        reviews.append({'path': name, 'sha256': R.sha_file(E / name), 'pass_key': key, 'pass_value': value})
    args = SimpleNamespace(identity=R.DEFAULT_IDENTITY, dataset=R.DEFAULT_DATASET,
                           slot_cap_s=60., day_rollout_cap_s=120., batch_cap_s=3600.)
    identity = V.execution_identity(R, args)
    assert not identity['origin_errors'] and all(identity['frozen_checks'].values())
    engineering = json.loads((E / 'engineering_release/RUNTIME_IDENTITY_V18.json').read_text())
    assert V.comparable(identity) == V.comparable(engineering), 'source changed after qualification'
    identity_path = E / 'RUNTIME_IDENTITY_V19.json'
    with open(identity_path, 'x', encoding='utf-8') as fh:
        json.dump(identity, fh, indent=2, sort_keys=True)
    source_paths = {EXT / rec['path'] for rec in identity['modules'].values()}
    source_paths.update((EXT / 'scripts/tests').glob('d3_l1_v19_*.py'))
    archive = E / 'RUNTIME_SOURCE_V19.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(source_paths):
            z.writestr(p.relative_to(EXT).as_posix(), p.read_bytes())
    auth = {'authority': 'human_user_conditional_instruction',
        'user_quote': '你自己判定核查 通过则继续执行 你需要每一次工作和我汇报当前进度 和剩余工作\n总目标为实现正向稳定增益',
        'prior_request_scope': '上一条明确申请：按V1.8在封存本地环境执行一次16日×4槽F-free采集，未签署前硬阻断。',
        'interpretation': 'Human conditional permission to proceed after independent review; not an agent signature.',
        'authorized_action': 'ONE_F_FREE_16_DAY_4_SLOT_COLLECTION',
        'dataset_sha256': R.DEV_CHECK_SHA, 'parameters': V.FROZEN_PARAMETERS,
        'platform': sys.platform, 'python': sys.version,
        'runtime_identity_sha256': R.sha_file(identity_path),
        'runtime_source_archive_sha256': R.sha_file(archive),
        'output_directory': str((E / 'formal_collection').resolve()),
        'required_reviews': reviews,
        'forbidden_actions': ['F-D-Geo comparison', 'training', 'L2/L3', '40-day science batch', 'final test seeds'],
        'source_base_commit_before_preparation': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=str(EXT.parent), text=True).strip()}
    auth_path = E / 'USER_CONDITIONAL_AUTHORIZATION.json'
    with open(auth_path, 'x', encoding='utf-8') as fh:
        json.dump(auth, fh, indent=2, sort_keys=True)
    proof = {'ok': True, 'required_reviews': reviews,
             'runtime_identity_sha256': R.sha_file(identity_path),
             'source_archive_sha256': R.sha_file(archive), 'authorization_sha256': R.sha_file(auth_path),
             'formal_collection_started': False, 'argv': sys.argv}
    with open(E / 'PRELAUNCH_REVIEW.json', 'x', encoding='utf-8') as fh:
        json.dump(proof, fh, indent=2, sort_keys=True)
    print(json.dumps(proof))
    return 0


if __name__ == '__main__':
    sys.exit(main())
