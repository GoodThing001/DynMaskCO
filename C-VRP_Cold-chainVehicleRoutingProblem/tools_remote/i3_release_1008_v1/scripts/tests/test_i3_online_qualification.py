"""Prevent a real engineering-trained checkpoint entering a 40-day paper gate."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'evaluation'))
from run_a1_step3_compare import main, require_online_model_qualification


def test_guard(engineering_model):
    card = json.loads((Path(engineering_model).parent / 'config.json').read_text(encoding='utf-8'))
    assert card['qualification'] == 'engineering' and card['training_complete'] is True
    with tempfile.TemporaryDirectory() as temp:
        out = Path(temp) / 'must_not_start'
        try:
            main(['--model-bin', str(engineering_model), '--arm', card['args']['arm'],
                  '--out', str(out)])
        except ValueError as exc:
            assert 'not eligible' in str(exc)
        else:
            raise AssertionError('Engineering model entered the paper experiment')
        assert not out.exists()
        try:
            require_online_model_qualification(Path(temp) / 'missing' / 'model.bin')
        except ValueError:
            pass
        else:
            raise AssertionError('Missing lineage entered the paper experiment')
        other = Path(temp) / 'incomplete'
        other.mkdir()
        (other / 'config.json').write_text(json.dumps({'qualification': 'formal_development',
                                                      'training_complete': False}))
        try:
            require_online_model_qualification(other / 'model.bin')
        except ValueError:
            pass
        else:
            raise AssertionError('Incomplete training entered the paper experiment')
        for name, fields in (
                ('unknown', {'qualification': 'unknown', 'training_complete': True}),
                ('i3', {'qualification': 'formal_development', 'training_complete': True,
                        'i3_training': {'checkpoint_reload_verified': True}})):
            folder = Path(temp) / name
            folder.mkdir()
            (folder / 'config.json').write_text(json.dumps(fields))
            try:
                require_online_model_qualification(folder / 'model.bin')
            except ValueError:
                pass
            else:
                raise AssertionError('Unknown qualification or I3 pool mismatch entered legacy driver')
    print('ALL PASS: engineering/incomplete/missing/unknown lineage rejected before paper gate; I3 decoder-pool mismatch rejected by legacy driver')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--engineering-model', required=True)
    args = parser.parse_args()
    test_guard(args.engineering_model)
