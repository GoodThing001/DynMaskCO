"""Probe the 3080 host for the registered shard set and the delivery compute sources.

Read-only. It answers one question with evidence instead of assertion:
can a delivery request running on the 3080 host be identity-verified against the
registered shard set (`SOURCE_HASHES.json`, 46 artifacts)?
"""
import json
import os
import subprocess
from pathlib import Path

ROOT = Path('/home/hzeng/project/MASKCO-Main')
REGISTERED = json.loads(Path(os.environ['DIAG_REGISTERED_SHARDS']).read_text(encoding='utf-8'))


def sha(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


out = {'host_root': str(ROOT), 'registered_shard_count': len(REGISTERED),
       'all_registered_paths_under_research': all(p.startswith('research/')
                                                  for p in REGISTERED),
       'research_tree_present': (ROOT / 'research').is_dir(),
       'present': {}, 'absent': [], 'sources_dir_probe': {}}
for rel, expected in REGISTERED.items():
    path = ROOT / rel
    if path.is_file():
        actual = sha(path)
        out['present'][rel] = {'expected': expected, 'actual': actual,
                               'match': actual == expected}
    else:
        out['absent'].append(rel)

# the compute sources the delivery request actually loads on this host
probe = [
    'C-VRP_Cold-chainVehicleRoutingProblem/scripts/models/mpre.py',
    'C-VRP_Cold-chainVehicleRoutingProblem/scripts/models/cvrptw_utils.py',
    'C-VRP_Cold-chainVehicleRoutingProblem/scripts/project_paths.py',
    'C-VRP_Cold-chainVehicleRoutingProblem/scripts/coldchain/coldchain_contract.py',
    'C-VRP_Cold-chainVehicleRoutingProblem/scripts/coldchain/coldchain_state.py',
    'C-VRP_Cold-chainVehicleRoutingProblem/scripts/evaluation/coldchain_evaluator_a1.py',
    'C-VRP_Cold-chainVehicleRoutingProblem/scripts/evaluation/rolling_optimizer.py',
    'MASKCO_code/ckpts/cvrp100.ckpt',
]
for rel in probe:
    path = ROOT / rel
    out['sources_dir_probe'][rel] = {
        'present': path.is_file(),
        'sha256': sha(path) if path.is_file() else None,
        'bytes': path.stat().st_size if path.is_file() else None}

# a full-tree scan cost measurement on the registered *artifact* tree, for comparison
if (ROOT / 'research').is_dir():
    import time
    t0 = time.perf_counter()
    n = sum(1 for _ in (ROOT / 'research').rglob('*') if _.is_file())
    out['research_tree_scan'] = {'files': n, 'seconds': time.perf_counter() - t0}

(Path(os.environ['DIAG_OUT']) / 'SHARD_PROBE.json').write_text(
    json.dumps(out, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
print(json.dumps({k: v for k, v in out.items() if k != 'present'}, ensure_ascii=False, indent=2))
print('present_count', len(out['present']), 'absent_count', len(out['absent']))
