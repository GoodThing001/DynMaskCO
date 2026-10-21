"""Pinned teacher loading and balanced epoch scheduling must not leak roles."""
from pathlib import Path
import sys
import unittest
import importlib.util
import tempfile
import json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'static_profit'))
from profit_core import EXTENSION_ROOT,fold_definition


class TrainingDataIsolation(unittest.TestCase):
    def test_balanced_cycles_are_deterministic(self):
        self.assertIsNotNone(importlib.util.find_spec('training_data'),'explicit balanced scheduling absent')
        from training_data import sample_schedule
        a=sample_schedule(228,1440,1000,'shuffled')
        self.assertEqual(a,sample_schedule(228,1440,1000,'shuffled'))
        self.assertNotEqual(a,sample_schedule(228,1440,1001,'shuffled'))
        for start in range(0,1368,228):self.assertEqual(sorted(a[start:start+228]),list(range(228)))
        self.assertEqual(sample_schedule(3,5,1000,'ordered'),[0,1,2,0,1])

    def test_pinned_teachers_validated_and_validation_role_rejected(self):
        self.assertIsNotNone(importlib.util.find_spec('training_data'),'pinned teacher loader absent')
        from training_data import load_teachers
        source=EXTENSION_ROOT/'results/static_delivery/algorithm_dev_20261021/remote/dev04/outputs/train/teachers.json'
        selected=load_teachers(source,0,['r101'])
        self.assertEqual(len(selected),6)
        rows=json.loads(source.read_text(encoding='utf-8'))
        rows[0]['task']['role']='validation'
        workspace_temp=EXTENSION_ROOT/'results/static_delivery/algorithm_dev_20261021'
        with tempfile.TemporaryDirectory(dir=workspace_temp) as directory:
            self.assertTrue(Path(directory).resolve().is_relative_to(workspace_temp.resolve()))
            path=Path(directory)/'bad.json';path.write_text(json.dumps(rows),encoding='utf-8')
            with self.assertRaises(ValueError):load_teachers(path,0,['r101'])


if __name__=='__main__':unittest.main()
