"""Seal existing V1.8 evidence and exact source bytes; no simulations or policy evaluation."""
import hashlib
import json
import platform
import shutil
import sys
import zipfile
from pathlib import Path

EXT = Path(__file__).resolve().parents[2]
ROOT = EXT.parent / 'research/review_3080_20261010/d3_g0_evidence_root_v1/a1_d3_l1_prereg_root_v1'
E = ROOT / 'v18_evidence'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    gates = ROOT / 'l1_gates'
    originals = {
        'A1_D3_L1_P0C_ACCEPTANCE_V18.json': E / 'acceptance_release/ACCEPTANCE.json',
        'A1_D3_L1_P0C_RUNTIME_IDENTITY_V18.json': E / 'acceptance_release/RUNTIME_IDENTITY_V18.json',
        'A1_D3_L1_P0C_KILL_E2E_V18.json': E / 'kill_release/KILL_E2E.json',
        'A1_D3_L1_P0C_RESUME_PARITY_V18.json': E / 'after/PARITY.json',
    }
    for name, original in originals.items():
        dest = gates / name
        with open(dest, 'xb') as fh:
            fh.write(original.read_bytes())
    ident = json.loads(originals['A1_D3_L1_P0C_RUNTIME_IDENTITY_V18.json'].read_text())
    sources = {EXT / r['path'] for r in ident['modules'].values()}
    sources.update(EXT.joinpath('scripts/tests').glob('d3_l1_v18_*.py'))
    archive = E / 'SOURCE_CLOSURE_V18.zip'
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(sources):
            z.writestr(p.relative_to(EXT).as_posix(), p.read_bytes())
    with zipfile.ZipFile(archive) as z:
        zip_checks = {p.relative_to(EXT).as_posix():
                      hashlib.sha256(z.read(p.relative_to(EXT).as_posix())).hexdigest() == sha(p)
                      for p in sources}
    acceptance = json.loads((E / 'acceptance_release/ACCEPTANCE.json').read_text())
    kill = json.loads((E / 'kill_release/KILL_E2E.json').read_text())
    regression = json.loads((E / 'REGRESSION.json').read_text())
    legacy = json.loads((E / 'legacy_accept_release/P0C_ACCEPTANCE.json').read_text())
    audits = {p.name: json.loads(p.read_text()) for p in
              (E / 'acceptance_release').glob('INDEPENDENT_WORKER_*.json')}
    source_matches = {name: sha(EXT / r['path']) == r['sha256'] for name, r in ident['modules'].items()}
    parity = json.loads((E / 'after/PARITY.json').read_text())
    results = {
        'source_closure': all(source_matches.values()), 'source_archive': all(zip_checks.values()),
        'acceptance_44': acceptance['n_fail'] == 0 and acceptance['n_pass'] == 44,
        'kill_8_real_process_trees': kill['n_fail'] == 0 and len(kill['rows']) == 8,
        'regression_38': regression['n_fail'] == 0 and regression['n_pass'] == 38,
        'legacy_accept_13': legacy['n_fail'] == 0 and legacy['n_pass'] == 13,
        'independent_8_artifacts': len(audits) == 8 and all(a['n_fail'] == 0 for a in audits.values()),
        'four_exact_resumes': len(parity['rows']) == 4 and all(
            r['capture_restore']['verdict'] == 'EQUAL' and r['full_resume_traces']['verdict'] == 'EQUAL'
            and r['full_resume_served']['verdict'] == 'EQUAL' and r['recapture_hash_equal']
            for r in parity['rows']),
    }
    result = {'checks': results, 'all_engineering_checks_pass': all(results.values()),
              'machine': platform.platform(), 'python': sys.version, 'runtime_identity': ident,
              'source_stability': source_matches, 'source_zip_sha256': sha(archive),
              'source_archive_members_verified': zip_checks,
              'acceptance_pass': acceptance['n_pass'], 'kill_assertions_pass':
              sum(sum(r['checks'].values()) for r in kill['rows']),
              'independent_assertions_pass': sum(a['n_pass'] for a in audits.values()),
              'formal_slots_collected': 0, 'formal_collection_authorization': 'NOT_AUTHORIZED',
              'signature': 'REQUEST_ELIGIBLE_ON_PINNED_WINDOWS_RUNTIME' if all(results.values()) else 'HOLD',
              'linux_server_execution_qualification': 'NOT_RUN_NOT_ESTABLISHED',
              'git_baseline': '4ca034dbe9827f750be57ddd22348ecc5e7ec08f',
              'note': 'Record qualification, not a signed authorization. Retain all earlier candidates and FAILs.'}
    with open(gates / 'A1_D3_L1_P0C_ENGINEERING_QUALIFICATION_V18.json', 'x', encoding='utf-8') as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    # All earlier evidence is included, even the genuine failed first kill acceptance.
    manifest = {p.relative_to(ROOT).as_posix(): {'bytes': p.stat().st_size, 'sha256': sha(p)}
                for p in sorted(E.rglob('*')) if p.is_file()}
    for p in sorted(gates.glob('*V18.json')):
        manifest[p.relative_to(ROOT).as_posix()] = {'bytes': p.stat().st_size, 'sha256': sha(p)}
    with open(E / 'ARTIFACT_SHA256_V18.json', 'x', encoding='utf-8') as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    print(json.dumps({'checks': results, 'source_zip_sha256': sha(archive),
                       'independent_pass': result['independent_assertions_pass'],
                       'kill_assertions_pass': result['kill_assertions_pass']}))
    return int(not all(results.values()))


if __name__ == '__main__':
    sys.exit(main())
