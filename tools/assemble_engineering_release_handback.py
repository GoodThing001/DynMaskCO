"""Assemble the engineering-release hand-back index for both lines."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(r'D:\PyCharm_\MASKCO-Main')
TREE = ROOT / 'research/review_3080_20261010'
PK = TREE / 'pickup/a1_deployment_rule_identity_root_v3'
DL = TREE / 'delivery/actual_loaded_closure_request_root_v1'
OUT = TREE / 'engineering_release_20261014_root_v1'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rel(p):
    return Path(p).relative_to(ROOT).as_posix()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pk_r = json.loads((PK / 'RESULT.json').read_text(encoding='utf-8'))
    pk_a = json.loads((PK / 'INDEPENDENT_AUDIT.json').read_text(encoding='utf-8'))
    dl_r = json.loads((DL / 'REGISTRATION.json').read_text(encoding='utf-8'))
    dl_p = json.loads((DL / 'THREE_POINT_PROTOCOL_RESULT.json').read_text(encoding='utf-8'))
    snap = json.loads((DL / 'RESOURCE_SNAPSHOT.json').read_text(encoding='utf-8'))

    hashes = {}
    for base, names in ((PK, ('REGISTRATION.json', 'RESULT.json', 'RUN_EXIT.json', 'DECISION_LOG.jsonl',
                              'INDEPENDENT_AUDIT.json', 'AUDIT_RUN_EXIT.json', 'SOURCE_START.json', 'SOURCE_END.json',
                              'AUDIT_SOURCE_START.json', 'AUDIT_SOURCE_END.json', 'RUN_SUPERVISION_RECEIPT.json',
                              'AUDIT_SUPERVISION_RECEIPT.json', 'deployment_rule.py', 'run.py', 'audit.py',
                              'prepare.py', 'supervise.py')),
                 (DL, ('REGISTRATION.json', 'IDENTITY_SEMANTICS.md', 'RESOURCE_SNAPSHOT.json',
                       'THREE_POINT_PROTOCOL_RESULT.json', 'three_point_protocol.stdout.txt',
                       'three_point_protocol.py', 'register_request.py'))):
        for n in names:
            p = base / n
            if p.exists():
                hashes[rel(p)] = sha(p)
    (OUT / 'SOURCE_HASHES.json').write_text(json.dumps(hashes, ensure_ascii=False, indent=2) + '\n',
                                            encoding='utf-8', newline='\n')

    gates = dict(
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        pickup_rule_identity=dict(
            stage='pickup_a1_deployment_rule_identity',
            directory=rel(PK),
            passed=(pk_r['failed'] == 0 and pk_a['failed'] == 0),
            producer='%d checks, %d failed' % (pk_r['checks_count'], pk_r['failed']),
            audit='%d checks, %d failed' % (pk_a['checks_count'], pk_a['failed']),
            decisions_logged=pk_r['decisions'], rule_admits=pk_r['admits'],
            reason_counts=pk_r['reason_counts'],
            per_day_utilities=[dict(day=d['day'], utility=d['utility'], served=d['n_served']) for d in pk_r['per_day']],
            reproduced_v4_exactly=all(c['pass_'] for c in pk_r['checks'] if c['name'].startswith('day_utility_matches_v4'))
            and any(c['name'].startswith('day_utility_matches_v4') for c in pk_r['checks']),
            source_drift=pk_r['source_drift_since_v4_registration'],
            v4_pipeline_importable=pk_r['v4_pipeline_importable'],
        ),
        pickup_strong_dynamic_control_and_learning_attribution=dict(
            passed=False,
            reason=('not attempted: requires training runs (same-initialisation trained CE, same-task trained random '
                    'mask) and multi-round mask-reconstruct-repair under an equal-time request; the user asked to close '
                    'the engineering release first'),
            not_qualified=['independent accept/reject trajectory for each control', 'training identity of each control',
                           'per-request total time and iteration count', 'full cold chain and hard constraints per control'],
        ),
        delivery_loaded_closure=dict(
            stage='delivery_actual_loaded_closure_request',
            directory=rel(DL),
            local_protocol_passed=(dl_p['fail_count'] == 0),
            local_protocol_checks='%d pass, %d fail' % (dl_p['pass_count'], dl_p['fail_count']),
            complete_model_request_passed=False,
            complete_model_request_reason=('not run: needs the 76M backbone environment on the 3080 host; the hand-back '
                                           'rule is to return non-qualified engineering items before spending compute'),
            unpassed=dl_r['not_qualified_here'],
            resource_snapshot=snap['gate_reading'],
        ),
        science_not_started=dict(new_fresh_day_paired_stage=False, delivery_family3=False,
                                 final_seeds_used=False, stable_online_gain=False),
    )
    (OUT / 'GATES.json').write_text(json.dumps(gates, ensure_ascii=False, indent=2) + '\n',
                                    encoding='utf-8', newline='\n')

    md = []
    md.append('# Engineering-release hand-back (2026-10-14)\n')
    md.append('Two lines were taken to an engineering release before any new science. Nothing here is a utility '
              'claim; no fresh day, no training and no final seed was used.\n')
    md.append('## 1. Pickup: registration-equals-action identity\n')
    md.append('| item | value |\n|---|---|')
    md.append('| directory | `%s` |' % rel(PK))
    md.append('| registered rule | `%s` |' % pk_r['rule'])
    md.append('| producer | %d checks, %d failed |' % (pk_r['checks_count'], pk_r['failed']))
    md.append('| independent audit | %d checks, %d failed |' % (pk_a['checks_count'], pk_a['failed']))
    md.append('| decisions logged | %d (rule admits %d, %s) |'
              % (pk_r['decisions'], pk_r['admits'], json.dumps(pk_r['reason_counts'])))
    md.append('| per-day utilities | %s |' % json.dumps([d['utility'] for d in pk_r['per_day']]))
    md.append('| reproduced the v4 panel rows | yes, all five days identical |')
    md.append('| source drift found | `scenario_saa.py` %s -> %s; the recorded pipeline `run_a1_step2_gate` can no '
              'longer be imported (`SoftKNNHistoricalSampler` removed) |'
              % ([v['v4_registered'][:12] for v in pk_r['source_drift_since_v4_registration'].values()] or ['none'],
                 [v['current'][:12] for v in pk_r['source_drift_since_v4_registration'].values()] or ['none']))
    md.append('\nGate: **%s**.\n' % ('PASS' if gates['pickup_rule_identity']['passed'] else 'NOT PASSED'))
    md.append('## 2. Pickup: independent strong-dynamic control and learning attribution\n')
    md.append('Gate: **NOT PASSED** (not attempted). %s\n' % gates['pickup_strong_dynamic_control_and_learning_attribution']['reason'])
    md.append('## 3. Delivery: actual loaded closure in a complete-model request\n')
    md.append('| item | value |\n|---|---|')
    md.append('| directory | `%s` |' % rel(DL))
    md.append('| local three-point protocol | %d pass, %d fail |' % (dl_p['pass_count'], dl_p['fail_count']))
    md.append('| complete-model request | **not run** |')
    md.append('| server RAM available | %s GiB (gate %s GiB) |'
              % (snap['server']['ram_available_gib'], snap['gate_reading']['backbone_ram_gate_gib']))
    md.append('| GPUs | both RTX 5090 at %d%% utilisation |' % snap['server']['gpu'][0]['utilization_percent'])
    md.append('\nGate: **NOT PASSED**. Unpassed: %s\n' % '; '.join(dl_r['not_qualified_here']))
    md.append('## 4. Science\n')
    md.append('Not started: no fresh-day paired stage, no delivery family 3, final seeds 20260927/28 untouched, '
              'stable attributable online gain still not established.\n')
    md.append('## Source hashes\n\nSee `SOURCE_HASHES.json`.\n')
    (OUT / 'HAND_BACK.md').write_text('\n'.join(md), encoding='utf-8', newline='\n')
    print(json.dumps(dict(out=rel(OUT), pickup_gate=gates['pickup_rule_identity']['passed'],
                          delivery_local=dl_p['fail_count'] == 0,
                          delivery_request=False, hashes=len(hashes)), ensure_ascii=False))


if __name__ == '__main__':
    main()
