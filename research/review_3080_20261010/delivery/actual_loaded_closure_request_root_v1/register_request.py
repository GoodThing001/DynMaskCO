"""Register the delivery complete-model closure request and record its identity semantics."""
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(r'D:\PyCharm_\MASKCO-Main')
EXT = ROOT / 'C-VRP_Cold-chainVehicleRoutingProblem'
HERE = ROOT / 'research/review_3080_20261010/delivery/actual_loaded_closure_request_root_v1'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    cand = ROOT / 'tools/actual_loaded_closure_candidate_r20.py'
    r20 = ROOT / 'research/review_3080_20261010/goal_round_20261014_r20_loaded_closure'
    reg = dict(
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        stage='delivery_actual_loaded_closure_request',
        scope='new pre-registered engineering request in the complete model environment',
        purpose=('Qualify the actual loaded closure inside a real complete-model delivery request, and place the '
                 'publication gate where the 8 s + 2 s budget can still absorb it.'),
        identity_semantics=dict(
            candidate='tools/actual_loaded_closure_candidate_r20.py',
            candidate_sha256=sha(cand),
            candidate_covers=['the running interpreter', 'every explicitly listed input file',
                              'every loaded Python module __file__ and __cached__',
                              'every memory-mapped native path',
                              'the module map, native map and explicit-path list'],
            candidate_does_not_cover=['files in the registered tree that the request never loads',
                                      'package trees that are not imported',
                                      'a lazy import that happens after the last capture',
                                      'semantic behaviour of a file whose bytes are unchanged'],
            old_gate='registered whole-tree source gate (scan every registered source/pyc/native file)',
            old_gate_covers=['every file in the registered tree whether or not it is loaded',
                             'mutation of a file the request never loads'],
            old_gate_does_not_cover=['cost: a full tree scan measured 5.618649 s for 8903 files, which cannot fit '
                                     'inside an 8 s request together with inference and cold-chain evaluation'],
            resolution=('the request uses the candidate at three capture points inside the budget and states explicitly '
                        'that whole-tree coverage is NOT provided inside the request; a whole-tree check remains an '
                        'out-of-request publication precondition, run before the timed request, never inside it'),
            capture_points=['P1 after model and explicit files are loaded',
                            'P2 after the first JAX call and the first native (C++) call',
                            'P3 immediately before publication'],
            fail_closed_rule=('publication is refused if any capture differs from the previous capture in file hashes, '
                              'file set, module map, native map or explicit-path list; on refusal the original plan is '
                              'retained, the request is recorded as partial and nothing is published'),
        ),
        budget=dict(execution_seconds=8.0, recovery_reserve_seconds=2.0,
                    counted_inside_execution=['model load', 'inference', 'cold-chain evaluation',
                                              'closure capture at P1/P2/P3', 'sealing', 'publication'],
                    out_of_request=['process start up before the timed section',
                                    'out-of-request whole-tree source check',
                                    'host level resource probing']),
        resource_gates=dict(backbone_ram_gib=8.0, require_cpu_affinity=True,
                            require_no_owned_remote_leftovers=True,
                            gpu_policy=('record GPU utilisation; do not schedule any GPU-dependent stage while either '
                                        'GPU is heavily loaded')),
        preconditions_before_server=[
            'local three-point protocol test passes (THREE_POINT_PROTOCOL_RESULT.json)',
            'resource snapshot recorded from the server (RESOURCE_SNAPSHOT.json)',
            'no owned remote process from a previous round remains',
        ],
        non_goals=['delivery scientific family 3', 'rerun of any old failed request to chase a PASS',
                   'shared MD or 88-mirror upload', 'any GPU-dependent stage while the GPUs are loaded'],
        failure_criteria=['any capture-point difference inside the request',
                          'publication without a stable P2 to P3 comparison',
                          'budget overrun of the 8 s execution or the 2 s recovery reserve',
                          'RAM below the 8 GiB backbone gate or an owned leftover process on the host'],
        qualified_here=['the three-point fail-closed protocol, tested locally with 7 checks',
                        'the identity-semantics mapping between the candidate and the whole-tree gate'],
        not_qualified_here=['the 3080 complete-model loaded closure', 'the real model native lazy-import surface',
                            'the 3080 source identity against the registered shard set',
                            'the full 8 s + 2 s request'],
        scientific_family3_started=False, delivery_science_family_started=False, stable_online_gain=False,
        sources={str(p): sha(p) for p in [
            cand, HERE / 'three_point_protocol.py',
            r20 / 'REGISTRATION.json', r20 / 'RESULT.json', r20 / 'PICKUP_REGISTRATION.json']},
    )
    (HERE / 'IDENTITY_SEMANTICS.md').write_text(
        '# Delivery actual-loaded-closure identity semantics\n\n'
        '## What the round-20 candidate covers\n\n' +
        '\n'.join('- ' + x for x in reg['identity_semantics']['candidate_covers']) +
        '\n\n## What it does not cover\n\n' +
        '\n'.join('- ' + x for x in reg['identity_semantics']['candidate_does_not_cover']) +
        '\n\n## What the old whole-tree gate covers instead\n\n' +
        '\n'.join('- ' + x for x in reg['identity_semantics']['old_gate_covers']) +
        '\n\n## Why the old gate cannot sit inside the request\n\n' +
        '\n'.join('- ' + x for x in reg['identity_semantics']['old_gate_does_not_cover']) +
        '\n\n## Resolution registered for the new request\n\n' +
        reg['identity_semantics']['resolution'] + '\n\n## Capture points\n\n' +
        '\n'.join('- ' + x for x in reg['identity_semantics']['capture_points']) +
        '\n\n## Fail-closed rule\n\n' + reg['identity_semantics']['fail_closed_rule'] + '\n',
        encoding='utf-8', newline='\n')
    reg['sources'][str(HERE / 'IDENTITY_SEMANTICS.md')] = sha(HERE / 'IDENTITY_SEMANTICS.md')

    snap = dict(
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        local=dict(total_ram_gib=None, free_ram_gib=None, cpus=None),
        server=dict(query='free -g; nproc; uptime; nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader',
                    raw='', ram_total_gib=247, ram_available_gib=179, nproc=96,
                    load_average=[29.92, 33.65, 33.37],
                    gpu=[{'index': 0, 'name': 'NVIDIA GeForce RTX 5090', 'utilization_percent': 95,
                          'memory_used_mib': 12369, 'memory_total_mib': 32607},
                         {'index': 1, 'name': 'NVIDIA GeForce RTX 5090', 'utilization_percent': 95,
                          'memory_used_mib': 11716, 'memory_total_mib': 32607}],
                    owned_leftovers_checked=False),
        gate_reading=dict(backbone_ram_gate_gib=8.0, server_ram_pass=True,
                          gpu_heavily_loaded=True, gpu_dependent_stage_allowed=False,
                          cpu_contention_notes='96 cores, load average about 30, so about 30 percent busy'),
    )
    import psutil
    vm = psutil.virtual_memory()
    snap['local'] = dict(total_ram_gib=round(vm.total / 2 ** 30, 2), free_ram_gib=round(vm.available / 2 ** 30, 2),
                         cpus=psutil.cpu_count())
    (HERE / 'RESOURCE_SNAPSHOT.json').write_text(json.dumps(snap, ensure_ascii=False, indent=2) + '\n',
                                                 encoding='utf-8', newline='\n')
    reg['sources'][str(HERE / 'RESOURCE_SNAPSHOT.json')] = sha(HERE / 'RESOURCE_SNAPSHOT.json')
    reg['sources'][str(HERE / 'THREE_POINT_PROTOCOL_RESULT.json')] = sha(HERE / 'THREE_POINT_PROTOCOL_RESULT.json')
    (HERE / 'REGISTRATION.json').write_text(json.dumps(reg, ensure_ascii=False, indent=2) + '\n',
                                            encoding='utf-8', newline='\n')
    print(json.dumps(dict(registration_sha256=sha(HERE / 'REGISTRATION.json'),
                          sources=len(reg['sources']),
                          not_qualified=len(reg['not_qualified_here'])), ensure_ascii=False))


if __name__ == '__main__':
    main()
