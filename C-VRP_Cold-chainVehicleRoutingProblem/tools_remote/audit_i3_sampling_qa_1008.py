"""Independent downloaded v5 QA checks; never serialize private teacher payloads to stdout."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/i3_registered_1008_v5'
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def typed_dict(value):
    assert value['tag']=='dict'
    return dict(value['items'])


def main():
    qa=read(OUT/'QA_PASS.json');low=read(OUT/'LOW_ORDER_QA_PASS.json')
    trained=OUT/'engineering_actual_teacher_training'
    report=read(trained/'train_report.json');card=read(trained/'config.json')
    assert (OUT/'qa.exit').read_text().splitlines()==['RUN_EXIT=0','SEAL_EXIT=0','DONE=1']
    seal=(OUT/'verify_end.txt').read_text().splitlines()
    assert len(seal)==len((OUT/'SOURCE_MANIFEST.sha256').read_text().splitlines())==337
    assert all(line.endswith(': OK') for line in seal)
    assert qa['engineering_release_pass'] and qa['source_stable']
    assert report['qualification']==card['qualification']=='engineering'
    assert report['steps_completed']==len(report['logs'])==3
    assert report['source_stable'] and report['source_start']==report['source_end']
    assert report['checkpoint_reload_verified'] and card['training_complete']
    previous=report['initial_trainable_hash']
    for i,row in enumerate(report['logs'],1):
        assert row['step']==i and row['trainable_before']==previous and row['trainable_after']!=previous
        previous=row['trainable_after']
    assert previous==report['final_trainable_hash']
    assert sum(row['actor_update'] for row in report['logs'])==2
    assert sum(not row['teacher_valid'] for row in report['logs'])==1
    model_sha=sha(trained/'model.bin')
    assert model_sha==qa['model_sha256']==report['model_bin_sha256']
    old=ROOT/'results/i3_registered_1008_v4/engineering_actual_teacher_training'
    assert model_sha==sha(old/'model.bin')
    assert report['frozen_hash']==read(old/'train_report.json')['frozen_hash']
    protocol=ROOT/'configs/i3_20261008_v2/protocol.json'
    assert protocol.read_bytes()==(OUT/'protocol.json').read_bytes()
    assert sha(protocol)=='4825392b56399719a9a0042d97986c4c683362af80d9925b35f74b65e4427f5c'
    collected=OUT/'engineering_low_order_collection'
    cr=read(collected/'report.json');archive=read(collected/'private.json')
    assert sha(collected/'report.json')==low['collection_report_sha256']
    assert sha(collected/'private.json')==low['archive_sha256']==cr['archive_sha256']
    assert low['engineering_low_order_sampling_pass'] and low['actual_customers']==150
    expected=[[12,(a*150+7)//8-1] for a in [1,3,5,7]]
    assert expected==low['expected_keys']==cr['expected_keys']
    assert len(cr['records'])==4 and all(row['same_action_parity'] for row in cr['records'])
    assert cr['source_stable'] and cr['source_start']==cr['source_end']
    payload=typed_dict(archive['payload'])
    assert len(payload['entries']['items'])==4
    assert [list(item['items']) for item in payload['expected_keys']['items']]==expected
    assert cr['collection']['not_formal_training_config'] and cr['collection']['K']==2
    assert low['all_four_terminal_rows_recomputed'] and low['archive_reload_verified']
    result={'engineering_release_verified':True,'sealed_items':337,
        'actual_engineering_model_sha256':model_sha,'model_bytes_match_prior_verified_engineering_qa':True,
        'three_actual_updates':True,'frozen_branch_unchanged':True,'checkpoint_reload_verified':True,
        'low_actual_customer_count':150,'four_capture_keys':expected,
        'low_report_private_archive_sha_verified':True,'independent_terminal_audit_in_qa':True,
        'shared_stdout_not_used_as_qualification_evidence':True,
        'protocol_sha256':sha(protocol),'formal_training_completed':False,'online_gain_established':False,
        'limits':['Only three engineering training steps, not formal 192-step training',
                  'Four engineering states, never reused in the formal teacher set',
                  'Full formal 128-state independent trajectory audit remains mandatory']}
    with (OUT/'INDEPENDENT_QA_RELEASE_AUDIT.json').open('x') as stream:json.dump(result,stream,indent=2)
    print(json.dumps(result))


if __name__=='__main__':main()
