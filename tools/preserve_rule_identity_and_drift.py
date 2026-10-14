"""Preserve the rule-identity result and the source-drift evidence, and record the hold decision."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(r'D:\PyCharm_\MASKCO-Main')
TREE = ROOT / 'research/review_3080_20261010'
OUT = TREE / 'engineering_release_20261014_root_v1'
PK = TREE / 'pickup'
V1 = PK / 'a1_deployment_rule_identity_root_v1'
V2 = PK / 'a1_deployment_rule_identity_root_v2'
V3 = PK / 'a1_deployment_rule_identity_root_v3'
EXT = ROOT / 'C-VRP_Cold-chainVehicleRoutingProblem'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rel(p):
    return Path(p).relative_to(ROOT).as_posix()


def files_of(base, names):
    out = {}
    for n in names:
        p = base / n
        if p.exists():
            out[rel(p)] = sha(p)
    return out


def main():
    r3 = json.loads((V3 / 'RESULT.json').read_text(encoding='utf-8'))
    a3 = json.loads((V3 / 'INDEPENDENT_AUDIT.json').read_text(encoding='utf-8'))
    recorder = dict(
        created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        purpose=('Preserve the sealed rule-identity result and the source-drift evidence exactly as produced, and '
                 'record the hold decision: no fresh utility day until the deployability gap is closed.'),
        v3_rule_identity=dict(
            directory=rel(V3),
            producer='%d checks, %d failed' % (r3['checks_count'], r3['failed']),
            audit='%d checks, %d failed' % (a3['checks_count'], a3['failed']),
            decisions_logged=r3['decisions'], decision_mismatches=a3['decision_mismatches'],
            rule_identity_established=a3['rule_identity_established'],
            per_day_utilities=[d['utility'] for d in r3['per_day']],
            scope_limit=('this proves rule identity on the sealed five-day log; it does NOT prove the rule is '
                         'deployable in a current-source online pipeline, because the recorded pipeline cannot be '
                         'imported any more'),
            files=files_of(V3, ('REGISTRATION.json', 'RESULT.json', 'RUN_EXIT.json', 'DECISION_LOG.jsonl',
                                'INDEPENDENT_AUDIT.json', 'AUDIT_RUN_EXIT.json', 'SOURCE_START.json', 'SOURCE_END.json',
                                'AUDIT_SOURCE_START.json', 'AUDIT_SOURCE_END.json', 'RUN_SUPERVISION_RECEIPT.json',
                                'AUDIT_SUPERVISION_RECEIPT.json', 'deployment_rule.py', 'run.py', 'audit.py',
                                'prepare.py', 'supervise.py'))),
        source_drift_evidence=dict(
            finding=r3['source_drift_since_v4_registration'],
            current_scenario_saa_sha256=sha(EXT / 'scripts/evaluation/scenario_saa.py'),
            recorded_pipeline_importable=r3['v4_pipeline_importable'],
            recorded_pipeline_import_error=r3['v4_pipeline_import_error'],
            evidence_source=rel(V3 / 'RESULT.json'),
            meaning=('scenario_saa.py changed after the v4 registration; SoftKNNHistoricalSampler and the anytime_vote '
                     'path were removed, so run_a1_step2_gate can no longer be imported. The five-day utilities were '
                     'nevertheless reproduced exactly by the current-source replay, so the drift is API level rather '
                     'than behaviour level for these arms and days.')),
        preserved_prior_attempts={
            'v1': dict(directory=rel(V1), outcome='TypeError on the replanner constructor, exit 1, no day ran',
                       files=files_of(V1, ('SOURCE_START.json', 'run.stdout.txt', 'run.stderr.txt',
                                           'RUN_SUPERVISION_RECEIPT.json'))),
            'v2': dict(directory=rel(V2),
                       outcome=('producer 57/57 reproduced the v4 rows; the audit reported 7 failures because the audit '
                                'itself updated spent on rule admits instead of base acceptances'),
                       files=files_of(V2, ('REGISTRATION.json', 'RESULT.json', 'DECISION_LOG.jsonl',
                                           'INDEPENDENT_AUDIT.json', 'RUN_EXIT.json', 'AUDIT_RUN_EXIT.json'))),
        },
        hold_decision=dict(
            fresh_utility_day=False,
            reason=('the rule has not been shown deployable in a current-source online pipeline; a fresh utility day '
                    'before that would spend compute on a result that cannot be adjudicated'),
            old_failures_modified=False, rerun_to_chase_pass=False,
            delivery_request_deferred=True,
            final_seeds_used=False, stable_online_gain=False),
        next_engineering_step=('qualify the registered rule together with a contemporaneous control through a '
                              'current-source online pipeline on already-consumed days only, and check whether the '
                              'control still reproduces its recorded rows after the scenario_saa drift'),
    )
    out = OUT / 'PRESERVATION_RECORD.json'
    out.write_text(json.dumps(recorder, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(written=rel(out), v3_files=len(recorder['v3_rule_identity']['files']),
                          drift=list(recorder['source_drift_evidence']['finding'].keys())), ensure_ascii=False))


if __name__ == '__main__':
    main()
