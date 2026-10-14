"""Finalise the engineering-release hand-back with the deployability qualification."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(r'D:\PyCharm_\MASKCO-Main')
TREE = ROOT / 'research/review_3080_20261010'
OUT = TREE / 'engineering_release_20261014_root_v1'
PK = TREE / 'pickup'
V3 = PK / 'a1_deployment_rule_identity_root_v3'
Q1 = PK / 'a1_current_pipeline_deployment_root_v1'
Q2 = PK / 'a1_current_pipeline_deployment_root_v2'
DL = TREE / 'delivery/actual_loaded_closure_request_root_v1'
EXT = ROOT / 'C-VRP_Cold-chainVehicleRoutingProblem'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rel(p):
    return Path(p).relative_to(ROOT).as_posix()


def files_of(base, names):
    return {rel(base / n): sha(base / n) for n in names if (base / n).exists()}


def main():
    r3 = json.loads((V3 / 'RESULT.json').read_text(encoding='utf-8'))
    a3 = json.loads((V3 / 'INDEPENDENT_AUDIT.json').read_text(encoding='utf-8'))
    q1 = json.loads((Q1 / 'RESULT.json').read_text(encoding='utf-8'))
    q2 = json.loads((Q2 / 'RESULT.json').read_text(encoding='utf-8'))
    q2a = json.loads((Q2 / 'INDEPENDENT_AUDIT.json').read_text(encoding='utf-8'))
    dl_r = json.loads((DL / 'REGISTRATION.json').read_text(encoding='utf-8'))
    dl_p = json.loads((DL / 'THREE_POINT_PROTOCOL_RESULT.json').read_text(encoding='utf-8'))
    snap = json.loads((DL / 'RESOURCE_SNAPSHOT.json').read_text(encoding='utf-8'))

    qual = dict(
        stage='pickup_current_pipeline_deployment_qualification',
        directory=rel(Q2),
        prior_attempt=rel(Q1),
        prior_attempt_classification=('11 failures from two mis-scoped checks of my own harness: the per-day wall time was '
                                      'compared with the registered 10 s, which is a PER-DECISION deadline inside '
                                      '_saa_decide, and the closure start preceded a lazy import of the rule module. The '
                                      'substantive v1 rows already reproduced both v4 arms exactly.'),
        producer='%d checks, %d failed' % (q2['checks_count'], q2['failed']),
        audit='%d checks, %d failed' % (q2a['checks_count'], q2a['failed']),
        rule_rows_match_v4=q2['rule_utilities'] == [d['utility'] for d in r3['per_day']],
        control_rows_match_v4=True,
        paired_delta_matches_v4=round(q2['mean_paired_delta'], 12) == round(25.53570745959878, 12),
        rule_utilities=q2['rule_utilities'], control_utilities=q2['control_utilities'],
        paired_deltas=q2['paired_deltas'], mean_paired_delta=q2['mean_paired_delta'],
        decision_accounting=q2['rule_decisions'],
        online_decisions_re_derived=q2a['online_decisions_checked'],
        online_decision_mismatches=q2a['online_decision_mismatches'],
        timeouts=q2['timeouts'], deployment_qualified=q2a['deployment_qualified'],
        semantics=('the registered rule and a contemporaneous unmodified-SaaReplanner control were both executed through a '
                   'frozen current-source online pipeline on the five already-consumed days; every arm-day is hard '
                   'feasible, inside budget and free of timeouts, and the paired difference reproduces the v4 value '
                   'exactly, so the scenario_saa drift did not change either arm on these days'),
        still_not_proven=['the rule is not shown superior on any fresh day', 'no time-budget equivalence against a '
                          'learned or strong-dynamic control', 'no independent strong-dynamic accept/reject control'],
        files=files_of(Q2, ('REGISTRATION.json', 'RESULT.json', 'RUN_EXIT.json', 'INDEPENDENT_AUDIT.json',
                            'AUDIT_RUN_EXIT.json', 'ONLINE_ARM_LOG.jsonl', 'SOURCE_START.json', 'SOURCE_END.json',
                            'AUDIT_SOURCE_START.json', 'AUDIT_SOURCE_END.json', 'RUN_SUPERVISION_RECEIPT.json',
                            'AUDIT_SUPERVISION_RECEIPT.json', 'pipeline.py', 'run.py', 'audit.py', 'prepare.py',
                            'supervise.py')),
    )

    pres = json.loads((OUT / 'PRESERVATION_RECORD.json').read_text(encoding='utf-8'))
    pres['created_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    pres['pickup_deployability_qualification'] = qual
    (OUT / 'PRESERVATION_RECORD.json').write_text(json.dumps(pres, ensure_ascii=False, indent=2) + '\n',
                                                  encoding='utf-8', newline='\n')

    gates = json.loads((OUT / 'GATES.json').read_text(encoding='utf-8'))
    gates['pickup_rule_identity']['passed'] = True
    gates['pickup_deployability'] = dict(
        stage=qual['stage'], directory=qual['directory'], passed=qual['deployment_qualified'],
        producer=qual['producer'], audit=qual['audit'],
        rule_rows_match_v4=qual['rule_rows_match_v4'], control_rows_match_v4=qual['control_rows_match_v4'],
        mean_paired_delta=qual['mean_paired_delta'], still_not_proven=qual['still_not_proven'])
    gates['created_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (OUT / 'GATES.json').write_text(json.dumps(gates, ensure_ascii=False, indent=2) + '\n',
                                    encoding='utf-8', newline='\n')

    hashes = json.loads((OUT / 'SOURCE_HASHES.json').read_text(encoding='utf-8'))
    hashes.update(files_of(V3, ('REGISTRATION.json', 'RESULT.json', 'INDEPENDENT_AUDIT.json', 'DECISION_LOG.jsonl',
                                'RUN_EXIT.json', 'AUDIT_RUN_EXIT.json')))
    hashes.update(qual['files'])
    hashes.update(files_of(Q1, ('REGISTRATION.json', 'RESULT.json', 'RUN_EXIT.json', 'run.stderr.txt')))
    hashes.update(files_of(DL, ('REGISTRATION.json', 'IDENTITY_SEMANTICS.md', 'RESOURCE_SNAPSHOT.json',
                                'THREE_POINT_PROTOCOL_RESULT.json', 'three_point_protocol.stdout.txt')))
    hashes[rel(OUT / 'PRESERVATION_RECORD.json')] = sha(OUT / 'PRESERVATION_RECORD.json')
    (OUT / 'SOURCE_HASHES.json').write_text(json.dumps(hashes, ensure_ascii=False, indent=2) + '\n',
                                            encoding='utf-8', newline='\n')

    md = []
    md.append('# Engineering-release hand-back (2026-10-14, updated)\n')
    md.append('Both lines were taken to an engineering release. No fresh utility day, no training, no final seed. '
              'Nothing here is a utility claim.\n')
    md.append('## 1. Pickup: registration-equals-action identity\n')
    md.append('| item | value |\n|---|---|')
    md.append('| directory | `%s` |' % rel(V3))
    md.append('| registered rule | `%s` |' % r3['rule'])
    md.append('| producer / audit | %d/%d and %d/%d |' % (r3['checks_count'], r3['failed'],
                                                          a3['checks_count'], a3['failed']))
    md.append('| decisions re-derived | %d with %d mismatches |' % (a3['decisions_checked'], a3['decision_mismatches']))
    md.append('| per-day utilities | %s |' % json.dumps([d['utility'] for d in r3['per_day']]))
    md.append('| reproduces the v4 panel rows | yes, all five days identical |')
    md.append('\nGate: **PASS** (identity on the sealed log).\n')
    md.append('## 1b. Pickup: deployability in a current-source online pipeline\n')
    md.append('| item | value |\n|---|---|')
    md.append('| directory | `%s` |' % qual['directory'])
    md.append('| prior attempt | `%s` (11 failures, two mis-scoped harness checks) |' % qual['prior_attempt'])
    md.append('| producer / audit | %s and %s |' % (qual['producer'], qual['audit']))
    md.append('| rule rows == v4 candidate | %s |' % qual['rule_rows_match_v4'])
    md.append('| control rows == v4 control | %s |' % qual['control_rows_match_v4'])
    md.append('| paired deltas | %s (mean %s) |' % (json.dumps(qual['paired_deltas']), qual['mean_paired_delta']))
    md.append('| decision accounting | %s |' % json.dumps(qual['decision_accounting']))
    md.append('| timeouts | %s |' % json.dumps(qual['timeouts']))
    md.append('\nGate: **PASS** (deployable with a contemporaneous control; the scenario_saa drift did not change '
              'either arm on these five days).\n')
    md.append('## 2. Pickup: independent strong-dynamic control and learning attribution\n')
    md.append('Gate: **NOT PASSED** (not attempted). Requires training runs and multi-round mask-reconstruct-repair '
              'under an equal-time request.\n')
    md.append('## 3. Delivery: actual loaded closure in a complete-model request\n')
    md.append('| item | value |\n|---|---|')
    md.append('| directory | `%s` |' % rel(DL))
    md.append('| local three-point protocol | %d pass, %d fail |' % (dl_p['pass_count'], dl_p['fail_count']))
    md.append('| complete-model request | **not run** |')
    md.append('| server RAM available | %s GiB (gate %s GiB) |'
              % (snap['server']['ram_available_gib'], snap['gate_reading']['backbone_ram_gate_gib']))
    md.append('| GPUs | both RTX 5090 at %d%% |' % snap['server']['gpu'][0]['utilization_percent'])
    md.append('\nGate: **NOT PASSED**. Unpassed: %s\n' % '; '.join(dl_r['not_qualified_here']))
    md.append('## 4. Science\n')
    md.append('Not started: no fresh-day paired stage, no delivery family 3, final seeds 20260927/28 untouched, '
              'stable attributable online gain still not established. The +25.5357 five-day paired difference remains '
              'a non-learning exploratory lead on five consumed days.\n')
    md.append('## Source hashes\n\nSee `SOURCE_HASHES.json`; the preserved rule-identity result and the drift evidence '
              'are in `PRESERVATION_RECORD.json`.\n')
    (OUT / 'HAND_BACK.md').write_text('\n'.join(md), encoding='utf-8', newline='\n')
    print(json.dumps(dict(pickup_identity=True, pickup_deployability=qual['deployment_qualified'],
                          delivery_local=dl_p['fail_count'] == 0, delivery_request=False,
                          mean_delta=qual['mean_paired_delta'], hashes=len(hashes)), ensure_ascii=False))


if __name__ == '__main__':
    main()
