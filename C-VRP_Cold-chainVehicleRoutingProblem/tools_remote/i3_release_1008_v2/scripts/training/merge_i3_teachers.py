"""Merge all sealed day shards; missing days/states cannot become training data."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for folder in ('training', 'evaluation', 'simulation', 'coldchain'):
    sys.path.insert(0, str(ROOT / folder))
from i3_teacher_archive import load_archive, write_archive


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def merge(shards, out):
    out = Path(out)
    if out.exists():
        raise FileExistsError('Do not overwrite an earlier merge attempt')
    context, qualification, config_sha, dataset, entries, days, input_ids = None, None, None, None, [], [], []
    for spec in shards:
        report_path, archive_path = Path(spec['report']), Path(spec['archive'])
        if sha(report_path) != spec['report_sha256']:
            raise ValueError('Shard report identity changed')
        report = json.loads(report_path.read_text(encoding='utf-8'))
        if report['source_stable'] is not True or report['source_start'] != report['source_end']:
            raise ValueError('Unstable shard source')
        if context is None:
            context, qualification, config_sha = report['context'], report['qualification'], report['config_sha256']
        if (report['context'] != context or report['qualification'] != qualification or
                report['config_sha256'] != config_sha):
            raise ValueError('Mixed source/configuration/data/contract shards')
        restored = load_archive(archive_path, archive_sha256=report['archive_sha256'],
            expected_context=context, expected_keys=report['expected_keys'])
        shard_days = {e['day_id'] for e in restored['entries']}
        if len(shard_days) != 1:
            raise ValueError('One source shard must contain exactly one global day')
        days.extend(shard_days)
        dataset = restored['dataset'] if dataset is None else dataset
        entries.extend(restored['entries'])
        input_ids.append({'report_sha256': spec['report_sha256'], 'archive_sha256': report['archive_sha256']})
    if context is None:
        raise ValueError('No teacher shards')
    cc = context['collection_config']
    if len(set(days)) != len(days) or sorted(days) != sorted(cc['days']):
        raise ValueError('Missing, duplicated or unexpected teacher day')
    expected = [(d, i) for d in cc['days'] for i in cc['capture_ordinals']]
    entries.sort(key=lambda e: (e['day_id'], e['ordinal']))
    out.mkdir(parents=True, exist_ok=False)
    digest = write_archive(out / 'private.json', context=context, dataset=dataset,
                           entries=entries, expected_keys=expected)
    load_archive(out / 'private.json', archive_sha256=digest, expected_context=context, expected_keys=expected)
    report = {'qualification': qualification, 'config_sha256': config_sha, 'context': context,
        'expected_keys': expected, 'archive_sha256': digest, 'inputs': input_ids,
        'merged_states': len(entries), 'merged_days': len(days),
        'merge_source_sha256': sha(__file__), 'formal_training': False, 'online_gain_established': False}
    out.joinpath('report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-config', required=True)
    parser.add_argument('--input-sha256', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if sha(args.input_config) != args.input_sha256:
        raise ValueError('Merge input configuration identity mismatch')
    result = merge(json.loads(Path(args.input_config).read_text(encoding='utf-8'))['shards'], args.out)
    print('MERGE_COMPLETE days=%s states=%s qualification=%s' %
          (result['merged_days'], result['merged_states'], result['qualification']))
